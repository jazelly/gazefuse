// SimTuner.swift — parameter search for the fusion / arbiter tunables.
//
// Objective: mean Webgrid BPS over a set of training seeds (and noise levels) of a
// preset, evaluated with the real pipeline inside the simulator. The seeds fix the
// bias field, the target sequence and the sensor noise streams (common random
// numbers), so differences between candidates are mostly due to the parameters.
// Search: random search in the (log-)ranges of `FusionTunables`, then coordinate
// descent around the incumbent with shrinking steps. The result is reported on
// held-out test seeds to show how much of the gain generalises.
import Foundation

public struct SimTunerOptions: Codable, Equatable {
    public var preset: SimPreset
    public var noises: [SimNoiseLevel]
    public var trainSeeds: [UInt64]
    public var testSeeds: [UInt64]
    public var sessionSeconds: Double
    public var randomIterations: Int
    public var coordinatePasses: Int
    /// Names of the tunables to search (default: all that affect the preset).
    public var parameters: [String]
    public var clickSource: SimClickSource
    public var webgrid: WebgridConfig

    public init(preset: SimPreset = .gazeHeadEKF, noises: [SimNoiseLevel] = [.typical],
                trainSeeds: [UInt64] = Array(1...6), testSeeds: [UInt64] = Array(101...106),
                sessionSeconds: Double = 60, randomIterations: Int = 60, coordinatePasses: Int = 2,
                parameters: [String] = [], clickSource: SimClickSource = .pinch, webgrid: WebgridConfig = WebgridConfig()) {
        self.preset = preset; self.noises = noises; self.trainSeeds = trainSeeds; self.testSeeds = testSeeds
        self.sessionSeconds = sessionSeconds; self.randomIterations = randomIterations; self.coordinatePasses = coordinatePasses
        self.parameters = parameters; self.clickSource = clickSource; self.webgrid = webgrid
    }

    /// Tunables that matter for the preset (e.g. no head parameters without a head channel).
    public func effectiveParameters() -> [FusionTunable] {
        if !parameters.isEmpty { return parameters.compactMap { FusionTunables.tunable(named: $0) } }
        let direct = ["directMinCutoff", "directBeta"]
        let head = ["headAccelerationSigma", "headDeadbandRadPerSec", "headPointsPerRadian", "headAccelExponent", "headVelocitySigma"]
        return FusionTunables.all.filter { p in
            switch preset {
            case .mouseLike: return p.name == "onsetLookback"
            case .gazeOnly: return direct.contains(p.name) || p.name == "onsetLookback"
            case .gazeEKF, .gazeHandEKF: return !direct.contains(p.name) && !head.contains(p.name)
            case .gazeHeadEKF, .gazeHeadHandEKF: return !direct.contains(p.name)
            }
        }
    }
}

public struct SimTunerReport: Codable, Equatable {
    public struct Step: Codable, Equatable {
        public var iteration: Int
        public var phase: String
        public var parameter: String?
        public var value: Double?
        public var trainBPS: Double
        public var accepted: Bool
    }
    public var options: SimTunerOptions
    public var initial: FusionTuningBundle
    public var best: FusionTuningBundle
    public var initialTrainBPS: Double
    public var bestTrainBPS: Double
    public var initialTestBPS: Double
    public var bestTestBPS: Double
    public var evaluations: Int
    public var steps: [Step]
}

public final class SimTuner {
    public let options: SimTunerOptions
    public var logger: ((String) -> Void)?
    private var evaluations = 0

    public init(options: SimTunerOptions, logger: ((String) -> Void)? = nil) { self.options = options; self.logger = logger }

    /// Mean BPS of `bundle` over seeds × noises, evaluated serially.
    public func evaluate(_ bundle: FusionTuningBundle, seeds: [UInt64]) -> Double {
        evaluateMetrics(bundle, seeds: seeds).bps
    }

    private func evaluateMetrics(_ bundle: FusionTuningBundle, seeds: [UInt64]) -> (bps: Double, ntpm: Double) {
        evaluations += 1
        var configs: [SimConfig] = []
        var wg = options.webgrid
        wg.sessionSeconds = options.sessionSeconds
        for n in options.noises {
            configs.append(SimConfig.make(preset: options.preset, noise: n, tuning: bundle, webgrid: wg, clickSource: options.clickSource))
        }
        var total = 0.0, netTargets = 0.0
        for config in configs {
            for seed in seeds {
                let result = WebgridSimulator(config: config, seed: seed).run()
                total += result.bps
                netTargets += result.ntpm
            }
        }
        let count = Double(max(configs.count * seeds.count, 1))
        return (total / count, netTargets / count)
    }

    public func run(initial: FusionTuningBundle, seed: UInt64 = 7) -> SimTunerReport {
        precondition(!options.trainSeeds.isEmpty && !options.testSeeds.isEmpty && !options.noises.isEmpty,
                     "Tuning requires training seeds, held-out seeds, and at least one noise level")
        precondition(Set(options.trainSeeds).isDisjoint(with: options.testSeeds),
                     "Held-out seeds must be separate from training seeds")
        precondition(options.randomIterations >= 0 && options.coordinatePasses >= 0,
                     "Search iteration counts must be non-negative")
        evaluations = 0
        let params = options.effectiveParameters()
        let rng = SimRNG(seed: seed)
        var steps: [SimTunerReport.Step] = []
        var best = initial
        let initialMetrics = evaluateMetrics(initial, seeds: options.trainSeeds)
        var bestScore = initialMetrics.bps
        var bestNetTargets = initialMetrics.ntpm
        let initialTrain = bestScore
        logger?("initial train BPS \(fmt(bestScore)) (\(params.count) parameters)")
        var iteration = 0

        // Phase 1: random search.
        for _ in 0..<options.randomIterations {
            iteration += 1
            var cand = best
            // Perturb a random subset (≈ half) of the parameters around the incumbent,
            // mixing global samples with local ones.
            for p in params where rng.bernoulli(0.5) {
                let v: Double
                if rng.bernoulli(0.5) {
                    v = sample(p, rng: rng)
                } else {
                    v = perturb(p.get(best), p, factor: 0.5, rng: rng)
                }
                p.set(&cand, p.clamp(v))
            }
            let metrics = evaluateMetrics(cand, seeds: options.trainSeeds)
            let s = metrics.bps
            // Webgrid clamps negative throughput to zero. Net targets break ties
            // on that observed plateau, without accepting any decrease in BPS.
            let accepted = s > bestScore || (s == bestScore && metrics.ntpm > bestNetTargets)
            steps.append(.init(iteration: iteration, phase: "random", parameter: nil, value: nil, trainBPS: s, accepted: accepted))
            if accepted { best = cand; bestScore = s; bestNetTargets = metrics.ntpm; logger?("[random \(iteration)] train BPS \(fmt(s)), NTPM \(fmt(metrics.ntpm)) ✓") }
        }

        // Phase 2: coordinate descent with shrinking steps.
        for pass in 0..<options.coordinatePasses {
            let factor = 0.6 / Double(pass + 1)
            for p in params {
                let base = p.get(best)
                let candidates: [Double]
                if p.integer {
                    candidates = [base - 1, base + 1]
                } else if p.logScale {
                    candidates = [base * (1 - factor), base * (1 + factor)]
                } else {
                    let span = (p.range.upperBound - p.range.lowerBound) * 0.25 * factor
                    candidates = [base - span, base + span]
                }
                for v in candidates {
                    let cv = p.clamp(v)
                    if cv == base { continue }
                    iteration += 1
                    var cand = best
                    p.set(&cand, cv)
                    let metrics = evaluateMetrics(cand, seeds: options.trainSeeds)
                    let s = metrics.bps
                    let accepted = s > bestScore || (s == bestScore && metrics.ntpm > bestNetTargets)
                    steps.append(.init(iteration: iteration, phase: "coordinate\(pass + 1)", parameter: p.name, value: cv, trainBPS: s, accepted: accepted))
                    if accepted { best = cand; bestScore = s; bestNetTargets = metrics.ntpm; logger?("[coord \(pass + 1)] \(p.name)=\(fmt(cv)) train BPS \(fmt(s)), NTPM \(fmt(metrics.ntpm)) ✓") }
                }
            }
        }

        let initialTest = evaluate(initial, seeds: options.testSeeds)
        let bestTest = evaluate(best, seeds: options.testSeeds)
        logger?("train BPS \(fmt(initialTrain)) → \(fmt(bestScore)); test BPS \(fmt(initialTest)) → \(fmt(bestTest)) (\(evaluations) evaluations)")
        return SimTunerReport(options: options, initial: initial, best: best, initialTrainBPS: initialTrain, bestTrainBPS: bestScore,
                              initialTestBPS: initialTest, bestTestBPS: bestTest, evaluations: evaluations, steps: steps)
    }

    private func sample(_ p: FusionTunable, rng: SimRNG) -> Double {
        if p.logScale {
            return exp(rng.uniform(log(p.range.lowerBound), log(p.range.upperBound)))
        }
        return rng.uniform(p.range.lowerBound, p.range.upperBound)
    }

    private func perturb(_ v: Double, _ p: FusionTunable, factor: Double, rng: SimRNG) -> Double {
        if p.integer { return v + Double(rng.int(3) - 1) }
        if p.logScale { return v * exp(rng.gaussian() * factor) }
        return v + rng.gaussian() * factor * (p.range.upperBound - p.range.lowerBound) * 0.25
    }

    private func fmt(_ v: Double) -> String { String(format: "%.3f", v) }
}
