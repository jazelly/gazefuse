import Foundation

public struct MotionJawCalibrationReport: Codable, Equatable {
    public let restMedianRMS: Double
    /// Unscaled median absolute deviation of the resting RMS, in g.
    public let restMAD: Double
    /// Lower quartile of the observed burst peaks, in g.
    public let clenchPeakRMS: Double
    public let thresholdRMS: Double
}

public enum MotionJawCalibrationError: Error {
    case insufficientSamples
    case noSeparatedClenchSignal
}

/// Experimental low-rate earbud IMU detector, not a bruxism/medical classifier.
///
/// Removes slow acceleration with a one-pole high pass, then measures time-weighted
/// RMS over ~3 frames at 25 Hz. An isolated, short excursion above median + k*MAD
/// is accepted only after release. Rotation/large acceleration veto the entire
/// candidate. A quiet interval and refractory period reject sustained activity.
/// Talking or chewing with an identical IMU signature is inherently ambiguous.
/// Call from one serial queue. All accelerations are in g, rotations in rad/s.
public final class JawClenchDetector: JawClenchDetecting {
    public struct Config: Codable, Equatable {
        public var highPassTimeConstant: Double = 0.12
        public var energyWindow: Double = 0.12
        public var noiseWindow: Double = 6
        public var noiseWarmup: Double = 0.8
        public var noiseUpdateInterval: Double = 0.2
        /// Robust standard deviation floor, preventing sensitivity in near-zero noise.
        public var noiseScaleFloor: Double = 0.00015
        public var minimumRMS: Double = 0.006
        public var thresholdZ: Double = 6
        /// Release threshold as a fraction of the excess above the noise median.
        public var releaseRatio: Double = 0.55
        /// Minimum span of above-release filtered energy; raw support is also required.
        public var minimumBurstDuration: Double = 0.06
        /// Maximum span of the supported raw excursion, excluding the filter tail.
        public var maximumBurstDuration: Double = 0.42
        public var minimumSupportingSamples: Int = 2
        public var supportRatio: Double = 0.65
        public var quietBefore: Double = 0.12
        public var quietAfter: Double = 0.10
        public var maximumRotationRate: Double = 0.35
        public var maximumAcceleration: Double = 0.12
        public var motionQuietPeriod: Double = 0.24
        public var refractoryPeriod: Double = 0.28
        /// Longer gaps cancel unfinished gestures; no movement is inferred in a gap.
        public var maximumSampleGap: Double = 0.16
        public var doubleClenchEnabled: Bool = false
        /// Maximum onset-to-onset interval; single clicks wait for this deadline.
        public var doubleClenchInterval: Double = 0.5
        public var calibrationSeparationZ: Double = 3
        public var calibrationPeakRatio: Double = 0.45
        public init() {}
    }

    public private(set) var config: Config
    public private(set) var currentRMS: Double = 0
    public private(set) var currentThresholdRMS: Double = 0

    private struct Candidate {
        var onset: Double
        var lastActive: Double
        var lastSupported: Double
        var peak: Double
        var threshold: Double
        var median: Double
        var baseline: Vector3
        var supportCount: Int
        var releaseSince: Double?
    }
    private var features: MotionJawFeatures
    private var previousTime: Double?
    private var sessionStart: Double?
    private var quietSince: Double?
    private var blockedUntil = -Double.infinity
    private var lastAcceptedOnset = -Double.infinity
    private var candidate: Candidate?
    private var pendingSingle: InputEvent?
    private var noise: [(t: Double, value: Double)] = []
    private var sortedNoise: [Double] = []
    private var deviations: [Double] = []
    private var noiseMedian = 0.0
    private var noiseSigma = 0.0
    private var lastNoiseUpdate = -Double.infinity

    public init(config: Config = Config()) {
        // Invalid persisted settings fail at construction rather than changing meaning.
        let positive = [config.highPassTimeConstant, config.energyWindow, config.noiseWindow,
                        config.noiseWarmup, config.noiseUpdateInterval, config.noiseScaleFloor,
                        config.minimumRMS, config.thresholdZ, config.minimumBurstDuration,
                        config.maximumBurstDuration, config.supportRatio, config.maximumRotationRate,
                        config.maximumAcceleration, config.maximumSampleGap, config.doubleClenchInterval,
                        config.calibrationSeparationZ]
        precondition(positive.allSatisfy { $0.isFinite && $0 > 0 })
        precondition(config.releaseRatio > 0 && config.releaseRatio < 1)
        precondition(config.calibrationPeakRatio > 0 && config.calibrationPeakRatio < 1)
        precondition(config.minimumBurstDuration < config.maximumBurstDuration)
        precondition(config.minimumSupportingSamples >= 2)
        precondition([config.quietBefore, config.quietAfter, config.motionQuietPeriod,
                      config.refractoryPeriod].allSatisfy { $0.isFinite && $0 >= 0 })
        self.config = config
        features = MotionJawFeatures(config: config)
        noise.reserveCapacity(180)
        sortedNoise.reserveCapacity(180)
        deviations.reserveCapacity(180)
        currentThresholdRMS = config.minimumRMS
    }

    /// Clears stream state and pending clicks, retaining the calibrated Config.
    public func reset() {
        features.reset()
        previousTime = nil
        sessionStart = nil
        quietSince = nil
        blockedUntil = -.infinity
        lastAcceptedOnset = -.infinity
        candidate = nil
        pendingSingle = nil
        noise.removeAll(keepingCapacity: true)
        noiseMedian = 0
        noiseSigma = 0
        lastNoiseUpdate = -.infinity
        currentRMS = 0
        currentThresholdRMS = config.minimumRMS
    }

    public func process(_ sample: HeadMotionSample) -> [InputEvent] {
        guard MotionJawFeatures.isValid(sample) else {
            candidate = nil
            pendingSingle = nil
            return []
        }
        if let previousTime, sample.t <= previousTime { return [] }
        if let previousTime, sample.t - previousTime > config.maximumSampleGap {
            reset()
        }
        previousTime = sample.t
        if sessionStart == nil { sessionStart = sample.t }
        let feature = features.process(sample)
        currentRMS = feature.rms
        let threshold = max(config.minimumRMS, noiseMedian + config.thresholdZ * max(noiseSigma, config.noiseScaleFloor))
        currentThresholdRMS = threshold

        // A real head movement cancels even a not-yet-delivered double/single click.
        if sample.rotationRate.length > config.maximumRotationRate || sample.userAcceleration.length > config.maximumAcceleration {
            candidate = nil
            pendingSingle = nil
            blockedUntil = sample.t + config.motionQuietPeriod
            quietSince = nil
            features.clearEnergy()
            return []
        }
        guard sample.t >= blockedUntil else { return [] }

        var events: [InputEvent] = []
        if let pending = pendingSingle, sample.t > pending.t + config.doubleClenchInterval,
           candidate == nil || candidate!.onset > pending.t + config.doubleClenchInterval {
            events.append(pending)
            pendingSingle = nil
        }

        if var active = candidate {
            let support = MotionJawFeatures.distance(sample.userAcceleration, active.baseline)
            if support >= active.threshold * config.supportRatio {
                active.supportCount += 1
                active.lastSupported = sample.t
            }
            active.peak = max(active.peak, feature.rms)
            let release = active.median + config.releaseRatio * (active.threshold - active.median)
            if feature.rms >= release {
                active.lastActive = sample.t
                active.releaseSince = nil
                quietSince = nil
            } else if active.releaseSince == nil {
                active.releaseSince = sample.t
                quietSince = sample.t
            }
            // Bound the physical excursion, not the causal filter's decay tail.
            if active.lastSupported - active.onset > config.maximumBurstDuration {
                candidate = nil
                quietSince = nil
            } else if let releaseSince = active.releaseSince, sample.t - releaseSince >= config.quietAfter {
                candidate = nil
                let duration = active.lastActive - active.onset
                if duration >= config.minimumBurstDuration &&
                    active.lastSupported - active.onset <= config.maximumBurstDuration &&
                    active.supportCount >= config.minimumSupportingSamples {
                    // Evidence strength, not a calibrated probability of intent.
                    let confidence = min(1, max(0, 0.5 + 0.5 * (active.peak / active.threshold - 1)))
                    let event = InputEvent(t: active.onset, source: .jaw, kind: .click, confidence: confidence)
                    lastAcceptedOnset = active.onset
                    if config.doubleClenchEnabled {
                        if let first = pendingSingle, active.onset - first.t <= config.doubleClenchInterval {
                            events.append(InputEvent(t: first.t, source: .jaw, kind: .click, button: .right,
                                                     confidence: min(first.confidence, confidence)))
                            pendingSingle = nil
                        } else {
                            pendingSingle = event
                        }
                    } else {
                        events.append(event)
                    }
                }
            } else {
                candidate = active
            }
        } else if feature.rms >= threshold {
            let wasQuiet = quietSince.map { sample.t - $0 >= config.quietBefore } ?? false
            if sample.t - sessionStart! >= config.noiseWarmup && wasQuiet &&
                sample.t - lastAcceptedOnset >= config.refractoryPeriod {
                candidate = Candidate(onset: sample.t, lastActive: sample.t, lastSupported: sample.t, peak: feature.rms,
                                      threshold: threshold, median: noiseMedian, baseline: feature.baseline,
                                      supportCount: features.supportingSampleCount(threshold: threshold * config.supportRatio,
                                                                                 baseline: feature.baseline),
                                      releaseSince: nil)
            }
            quietSince = nil
        } else {
            if quietSince == nil { quietSince = sample.t }
            addNoise(feature.rms, at: sample.t)
        }
        return events
    }

    /// Guided rest and isolated gentle clenches, sampled using the runtime filters.
    /// Throws if no burst separates from rest. No learning model or hardware claim.
    /// On success updates minimumRMS, resets the stream and returns measured levels.
    @discardableResult
    public func calibrate(restSamples: [HeadMotionSample], clenchSamples: [HeadMotionSample]) throws -> MotionJawCalibrationReport {
        let rest = calibrationLevels(restSamples)
        let clenches = calibrationLevels(clenchSamples)
        guard rest.count >= 20, clenches.count >= 3 else { throw MotionJawCalibrationError.insufficientSamples }
        let median = MotionJawFeatures.median(rest.map(\.value))
        let mad = MotionJawFeatures.median(rest.map { abs($0.value - median) })
        let sigma = max(config.noiseScaleFloor, 1.4826 * mad)
        let separation = median + config.calibrationSeparationZ * sigma
        var peaks: [Double] = []
        var peak = 0.0
        var onset: Double?
        var lastTime = 0.0
        for level in clenches {
            if level.value > separation {
                if onset == nil { onset = level.t }
                peak = max(peak, level.value)
                lastTime = level.t
            } else if let start = onset {
                if lastTime - start >= config.minimumBurstDuration { peaks.append(peak) }
                onset = nil
                peak = 0
            }
        }
        if let start = onset, lastTime - start >= config.minimumBurstDuration { peaks.append(peak) }
        peaks.sort()
        guard !peaks.isEmpty else { throw MotionJawCalibrationError.noSeparatedClenchSignal }
        let representativePeak = peaks[(peaks.count - 1) / 4]
        let learnedThreshold = max(median + config.thresholdZ * sigma, representativePeak * config.calibrationPeakRatio)
        guard learnedThreshold < representativePeak else { throw MotionJawCalibrationError.noSeparatedClenchSignal }
        config.minimumRMS = learnedThreshold
        reset()
        return MotionJawCalibrationReport(restMedianRMS: median, restMAD: mad,
                                          clenchPeakRMS: representativePeak, thresholdRMS: learnedThreshold)
    }

    private func calibrationLevels(_ samples: [HeadMotionSample]) -> [(t: Double, value: Double)] {
        var extractor = MotionJawFeatures(config: config)
        var levels: [(t: Double, value: Double)] = []
        var last: Double?
        for sample in samples where MotionJawFeatures.isValid(sample) {
            if let last, sample.t <= last { continue }
            if let last, sample.t - last > config.maximumSampleGap { extractor.reset() }
            last = sample.t
            let value = extractor.process(sample).rms
            if sample.rotationRate.length <= config.maximumRotationRate && sample.userAcceleration.length <= config.maximumAcceleration {
                levels.append((sample.t, value))
            }
        }
        return levels
    }

    private func addNoise(_ value: Double, at t: Double) {
        noise.append((t, value))
        noise.removeAll { $0.t < t - config.noiseWindow }
        guard t - lastNoiseUpdate >= config.noiseUpdateInterval else { return }
        lastNoiseUpdate = t
        sortedNoise.removeAll(keepingCapacity: true)
        for point in noise { sortedNoise.append(point.value) }
        sortedNoise.sort()
        noiseMedian = sortedNoise[sortedNoise.count / 2]
        deviations.removeAll(keepingCapacity: true)
        for value in sortedNoise { deviations.append(abs(value - noiseMedian)) }
        deviations.sort()
        noiseSigma = 1.4826 * deviations[deviations.count / 2]
    }
}

private struct MotionJawFeatures {
    let config: JawClenchDetector.Config
    private var baseline: Vector3?
    private var previousTime: Double?
    private var window: [(t: Double, start: Double, power: Double, acceleration: Vector3, baseline: Vector3)] = []

    init(config: JawClenchDetector.Config) {
        self.config = config
        window.reserveCapacity(8)
    }

    mutating func reset() {
        baseline = nil
        previousTime = nil
        clearEnergy()
    }

    mutating func clearEnergy() { window.removeAll(keepingCapacity: true) }

    mutating func process(_ sample: HeadMotionSample) -> (rms: Double, baseline: Vector3) {
        guard let oldBaseline = baseline, let previousTime else {
            baseline = sample.userAcceleration
            previousTime = sample.t
            return (0, sample.userAcceleration)
        }
        let dt = sample.t - previousTime
        let decay = exp(-dt / config.highPassTimeConstant)
        let a = sample.userAcceleration
        let nextBaseline = Vector3(decay * oldBaseline.x + (1 - decay) * a.x,
                                   decay * oldBaseline.y + (1 - decay) * a.y,
                                   decay * oldBaseline.z + (1 - decay) * a.z)
        baseline = nextBaseline
        self.previousTime = sample.t
        let residual = Self.distance(a, nextBaseline)
        window.append((sample.t, sample.t - dt, residual * residual, a, oldBaseline))
        let cutoff = sample.t - config.energyWindow
        window.removeAll { $0.t <= cutoff }
        var energy = 0.0
        var duration = 0.0
        for point in window {
            let weight = point.t - max(cutoff, point.start)
            energy += weight * point.power
            duration += weight
        }
        // RMS crosses its threshold after the physical onset. Retain preceding
        // raw samples and their quiet baseline so a 2-frame clench is not counted
        // as a single impulse merely because the energy window introduced delay.
        return ((energy / max(duration, Double.leastNonzeroMagnitude)).squareRoot(), window.first?.baseline ?? oldBaseline)
    }

    func supportingSampleCount(threshold: Double, baseline: Vector3) -> Int {
        var count = 0
        for point in window where Self.distance(point.acceleration, baseline) >= threshold { count += 1 }
        return count
    }

    static func distance(_ a: Vector3, _ b: Vector3) -> Double {
        let x = a.x - b.x, y = a.y - b.y, z = a.z - b.z
        return (x * x + y * y + z * z).squareRoot()
    }

    static func isValid(_ sample: HeadMotionSample) -> Bool {
        sample.t.isFinite && sample.userAcceleration.x.isFinite &&
        sample.userAcceleration.y.isFinite && sample.userAcceleration.z.isFinite &&
        sample.rotationRate.x.isFinite && sample.rotationRate.y.isFinite && sample.rotationRate.z.isFinite
    }

    static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        return sorted[sorted.count / 2]
    }
}
