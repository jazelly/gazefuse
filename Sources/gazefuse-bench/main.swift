// gazefuse-bench — small command-line harness for simulation, tuning and replay.
//
// The executable intentionally keeps argument handling local and synchronous. It
// drives the same GazeFuseCore simulator/fusion/arbiter path used by tests and does
// not add a second benchmark implementation.
import Foundation
import GazeFuseCore

#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

private enum BenchCLIError: Error, CustomStringConvertible {
    case usage(String)
    case invalidOption(String)
    case missingValue(String)
    case invalidValue(String, String)

    var description: String {
        switch self {
        case .usage(let text): return text
        case .invalidOption(let name): return "unknown option \(name)"
        case .missingValue(let name): return "missing value for \(name)"
        case .invalidValue(let name, let value): return "invalid value for \(name): \(value)"
        }
    }
}

private struct BenchArguments {
    let command: String
    let positionals: [String]
    let options: [String: String]

    init(_ raw: [String]) throws {
        guard let command = raw.first, !command.hasPrefix("-") else {
            throw BenchCLIError.usage(Self.usage)
        }
        self.command = command.lowercased()
        var positionals: [String] = []
        var options: [String: String] = [:]
        var i = 1
        while i < raw.count {
            let token = raw[i]
            guard token.hasPrefix("-") else {
                positionals.append(token); i += 1; continue
            }
            let pieces = token.dropFirst(2).split(separator: "=", maxSplits: 1).map(String.init)
            guard token.hasPrefix("--"), let key = pieces.first, !key.isEmpty else {
                throw BenchCLIError.invalidOption(token)
            }
            if pieces.count == 2 {
                options[key] = pieces[1]
            } else if key == "write-defaults" || key == "help" || key == "h" {
                options[key] = "true"
            } else {
                guard i + 1 < raw.count, !raw[i + 1].hasPrefix("-") else {
                    throw BenchCLIError.missingValue("--\(key)")
                }
                options[key] = raw[i + 1]
                i += 1
            }
            i += 1
        }
        self.positionals = positionals
        self.options = options
    }

    static let usage = """
    Usage:
      gazefuse-bench sim [--preset NAME] [--noise LEVEL] [--grid N] [--seconds S] [--seeds N] [--seed-start N] [--config PATH]
      gazefuse-bench tune [--preset NAME] [--noise LEVEL] [--seconds S] [--train-seeds N] [--test-seeds N]
                          [--iterations N] [--passes N] [--parameters a,b,c] [--config PATH] [--output PATH]
                          [--write-defaults]
      gazefuse-bench all [--grid N] [--seconds S] [--seeds N] [--seed-start N] [--config PATH] [--output PATH]
      gazefuse-bench replay FILE.jsonl [--display-hz H]

    Presets: mouseLike, gazeOnly, gazeEKF, gazeHeadEKF, gazeHandEKF, gazeHeadHandEKF.
    Noise: optimistic, typical, pessimistic (tune also accepts all).
    """

    func string(_ key: String, default value: String? = nil) throws -> String? {
        guard let v = options[key] else { return value }
        guard !v.isEmpty else { throw BenchCLIError.invalidValue("--\(key)", v) }
        return v
    }

    func int(_ key: String, default value: Int) throws -> Int {
        guard let s = options[key] else { return value }
        guard let n = Int(s), n > 0 else { throw BenchCLIError.invalidValue("--\(key)", s) }
        return n
    }

    func nonNegativeInt(_ key: String, default value: Int) throws -> Int {
        guard let s = options[key] else { return value }
        guard let n = Int(s), n >= 0 else { throw BenchCLIError.invalidValue("--\(key)", s) }
        return n
    }

    func double(_ key: String, default value: Double) throws -> Double {
        guard let s = options[key] else { return value }
        guard let n = Double(s), n > 0, n.isFinite else { throw BenchCLIError.invalidValue("--\(key)", s) }
        return n
    }

    func uint64(_ key: String, default value: UInt64) throws -> UInt64 {
        guard let s = options[key] else { return value }
        guard let n = UInt64(s) else { throw BenchCLIError.invalidValue("--\(key)", s) }
        return n
    }

    func preset(_ key: String = "preset", default value: SimPreset = .gazeHeadEKF) throws -> SimPreset {
        guard let s = try string(key, default: value.rawValue), let p = SimPreset(rawValue: s) else {
            throw BenchCLIError.invalidValue("--\(key)", options[key] ?? "")
        }
        return p
    }

    func noise(_ key: String = "noise", default value: SimNoiseLevel = .typical) throws -> SimNoiseLevel {
        guard let s = try string(key, default: value.rawValue), let n = SimNoiseLevel(rawValue: s) else {
            throw BenchCLIError.invalidValue("--\(key)", options[key] ?? "")
        }
        return n
    }
}

private func makeSeeds(count: Int, offset: UInt64 = 1) -> [UInt64] {
    (0..<count).map { offset &+ UInt64($0) }
}

private func validateOptions(_ args: BenchArguments) throws {
    let common: Set<String> = ["help", "h"]
    let allowed: Set<String>
    switch args.command {
    case "sim": allowed = common.union(["preset", "noise", "grid", "seconds", "seeds", "seed-start", "config"])
    case "tune": allowed = common.union(["preset", "noise", "seconds", "train-seeds", "test-seeds", "iterations", "passes", "parameters", "config", "output", "write-defaults"])
    case "all": allowed = common.union(["grid", "seconds", "seeds", "seed-start", "config", "output"])
    case "replay": allowed = common.union(["display-hz"])
    default: return
    }
    if let unknown = args.options.keys.first(where: { !allowed.contains($0) }) {
        throw BenchCLIError.invalidOption("--\(unknown)")
    }
}

private func loadTuning(_ args: BenchArguments) throws -> FusionTuningBundle {
    guard let path = try args.string("config") else { return FusionTuningBundle() }
    do {
        return try JSONDecoder().decode(FusionTuningBundle.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
    } catch {
        throw BenchCLIError.invalidValue("--config", "\(path) (\(error))")
    }
}

private func fmt(_ value: Double, digits: Int = 3) -> String {
    String(format: "%.*f", digits, value)
}

private func runSimulation(_ args: BenchArguments) throws -> SimSummary {
    let preset = try args.preset()
    let noise = try args.noise()
    let grid = try args.int("grid", default: 30)
    guard grid >= 2 else { throw BenchCLIError.invalidValue("--grid", "\(grid) (must be at least 2)") }
    let seconds = try args.double("seconds", default: 60)
    let seedCount = try args.int("seeds", default: 10)
    let seedStart = try args.uint64("seed-start", default: 1)
    var webgrid = WebgridConfig(gridSize: grid)
    webgrid.sessionSeconds = seconds
    let config = SimConfig.make(preset: preset, noise: noise, tuning: try loadTuning(args), webgrid: webgrid)
    return SimSummary.run(config: config, seeds: makeSeeds(count: seedCount, offset: seedStart))
}

private func printSimulation(_ summary: SimSummary) {
    print("preset=\(summary.preset.rawValue) noise=\(summary.noise.rawValue) sessions=\(summary.sessions)")
    print("BPS \(fmt(summary.bpsMean)) ± \(fmt(summary.bpsSD)) | NTPM \(fmt(summary.ntpmMean, digits: 2)) | accuracy \(fmt(summary.accuracyMean * 100, digits: 1))% | mean time/target \(fmt(summary.timePerTargetMean, digits: 3))s")
}

private func runTune(_ args: BenchArguments) throws {
    let preset = try args.preset()
    let noise = args.options["noise"] == "all" ? SimNoiseLevel.typical : try args.noise()
    let seconds = try args.double("seconds", default: 60)
    let trainCount = try args.int("train-seeds", default: 6)
    let testCount = try args.int("test-seeds", default: 6)
    let iterations = try args.nonNegativeInt("iterations", default: 20)
    let passes = try args.nonNegativeInt("passes", default: 1)
    let parameterNames = try args.string("parameters")?.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty } ?? []
    var webgrid = WebgridConfig()
    webgrid.sessionSeconds = seconds
    let noises: [SimNoiseLevel]
    if args.options["noise"] == "all" { noises = SimNoiseLevel.allCases } else { noises = [noise] }
    let options = SimTunerOptions(preset: preset, noises: noises, trainSeeds: makeSeeds(count: trainCount),
                                  testSeeds: makeSeeds(count: testCount, offset: 101), sessionSeconds: seconds,
                                  randomIterations: iterations, coordinatePasses: passes, parameters: parameterNames, webgrid: webgrid)
    let tuner = SimTuner(options: options) { print($0) }
    let report = tuner.run(initial: try loadTuning(args))
    let writeDefaults = args.options["write-defaults"] == "true"
    let output = try args.string("output", default: writeDefaults ? "Sources/GazeFuseCore/Fusion/DefaultTuning.swift" : "docs/tuned-fusion-config.json") ?? "docs/tuned-fusion-config.json"
    let url = URL(fileURLWithPath: output, relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    if writeDefaults {
        let note = "Generated by gazefuse-bench tune; held-out BPS \(fmt(report.bestTestBPS))"
        try FusionTunables.renderDefaultTuningSwift(report.best, note: note).write(to: url, atomically: true, encoding: .utf8)
    } else {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report.best).write(to: url)
    }
    print("best config: \(url.path)")
    print("train BPS \(fmt(report.initialTrainBPS)) → \(fmt(report.bestTrainBPS)); held-out BPS \(fmt(report.initialTestBPS)) → \(fmt(report.bestTestBPS)) (\(report.evaluations) evaluations)")
}

private func runAll(_ args: BenchArguments) throws {
    let grid = try args.int("grid", default: 30)
    guard grid >= 2 else { throw BenchCLIError.invalidValue("--grid", "\(grid) (must be at least 2)") }
    let seconds = try args.double("seconds", default: 60)
    let seedCount = try args.int("seeds", default: 3)
    let seedStart = try args.uint64("seed-start", default: 1)
    let tuning = try loadTuning(args)
    let output = try args.string("output", default: "docs/benchmarks.md") ?? "docs/benchmarks.md"
    var rows = ["| Preset | Noise | BPS (mean ± SD) | NTPM | Accuracy | Mean time/target |", "|---|---|---:|---:|---:|---:|"]
    for preset in SimPreset.allCases {
        for noise in SimNoiseLevel.allCases {
            var webgrid = WebgridConfig(gridSize: grid)
            webgrid.sessionSeconds = seconds
            let config = SimConfig.make(preset: preset, noise: noise, tuning: tuning, webgrid: webgrid)
            let summary = SimSummary.run(config: config, seeds: makeSeeds(count: seedCount, offset: seedStart))
            rows.append("| \(preset.rawValue) | \(noise.rawValue) | \(fmt(summary.bpsMean)) ± \(fmt(summary.bpsSD)) | \(fmt(summary.ntpmMean, digits: 2)) | \(fmt(summary.accuracyMean * 100, digits: 1))% | \(fmt(summary.timePerTargetMean))s |")
            print("\(preset.rawValue)/\(noise.rawValue): BPS \(fmt(summary.bpsMean)) ± \(fmt(summary.bpsSD))")
        }
    }
    let text = "# GazeFuse simulated Webgrid benchmarks\n\nGenerated by `gazefuse-bench all` (grid \(grid)×\(grid), \(fmt(seconds, digits: 1)) s/session, \(seedCount) seeds).\n\n" + rows.joined(separator: "\n") + "\n"
    let url = URL(fileURLWithPath: output, relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try text.write(to: url, atomically: true, encoding: .utf8)
    print("wrote \(url.path)")
}

private func runReplay(_ args: BenchArguments) throws {
    guard let path = args.positionals.first else { throw BenchCLIError.usage("replay requires a JSONL file path\n\n\(BenchArguments.usage)") }
    let displayHz = try args.double("display-hz", default: 60)
    let url = URL(fileURLWithPath: path)
    let parsed = try RecordingReader.read(url: url)
    let screen: ScreenRect = parsed.events.compactMap { event in
        if case .header(let header) = event { return header.screen }
        return nil
    }.first ?? ScreenRect(x: 0, y: 0, width: 1512, height: 982)
    let fusion = EKFCursorFusion(screen: screen)
    let arbiter = ClickArbiter()
    let result = ReplayEngine(fusion: fusion, arbiter: arbiter, displayHz: displayHz).run(parsed.events)
    print("events=\(result.events) skippedLines=\(parsed.skipped) duration=\(fmt(result.duration, digits: 3))s")
    print("gazeEstimates=\(result.gazeEstimates) headSamples=\(result.headSamples) inputEvents=\(result.inputEvents) actions=\(result.actions.count)")
    print("rawGaze=\(result.rawGazeSamples) mapped=\(result.rawGazeMapped) skipped=\(result.rawGazeSkipped) rawHand=\(result.rawHandSamples) classified=\(result.rawHandClassified) skipped=\(result.rawHandSkipped)")
    print("cursorSamples=\(result.cursorSamples) meanSpeed=\(fmt(result.meanCursorSpeed, digits: 2)) maxSpeed=\(fmt(result.maxCursorSpeed, digits: 2)) fixationJitter=\(fmt(result.fixationJitter, digits: 2))")
    if let deviation = result.recordedCursorDeviation { print("recordedCursorDeviation=\(fmt(deviation, digits: 2))") }
    if !result.markers.isEmpty { print("markers=\(result.markers.count)") }
    for action in result.actions {
        let p = action.position
        var line = "action t=\(fmt(action.t, digits: 3)) kind=\(action.kind.rawValue) button=\(action.button.rawValue) source=\(action.source.rawValue) position=(\(fmt(p.x, digits: 1)),\(fmt(p.y, digits: 1)))"
        if let scroll = action.scroll { line += " scroll=(\(fmt(scroll.x, digits: 1)),\(fmt(scroll.y, digits: 1)))" }
        print(line)
    }
}

private func runCLI() throws {
    let args = try BenchArguments(Array(CommandLine.arguments.dropFirst()))
    try validateOptions(args)
    if args.options["help"] == "true" || args.options["h"] == "true" {
        print(BenchArguments.usage); return
    }
    switch args.command {
    case "sim": printSimulation(try runSimulation(args))
    case "tune": try runTune(args)
    case "all": try runAll(args)
    case "replay": try runReplay(args)
    default: throw BenchCLIError.usage("unknown command \(args.command)\n\n\(BenchArguments.usage)")
    }
}

do {
    try runCLI()
} catch {
    FileHandle.standardError.write(Data("gazefuse-bench: \(error)\n".utf8))
    exit(2)
}
