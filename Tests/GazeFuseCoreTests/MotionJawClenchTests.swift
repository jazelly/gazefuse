import XCTest
@testable import GazeFuseCore

/// Deterministic 25 Hz-ish IMU scenarios for the jaw detector. The generator
/// produces one continuous stream so warmup, quiet windows, and gap handling are
/// exercised together rather than by concatenating independent clips.
final class MotionJawClenchTests: XCTestCase {
    private struct MotionJawTestRNG {
        private var state: UInt64

        init(seed: UInt64) { state = seed == 0 ? 0x9E3779B97F4A7C15 : seed }

        mutating func nextUnit() -> Double {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            z ^= z >> 31
            return Double(z >> 11) / Double(1 << 53)
        }

        mutating func normal() -> Double {
            let u1 = max(nextUnit(), 1e-12)
            return (-2 * log(u1)).squareRoot() * cos(2 * Double.pi * nextUnit())
        }
    }

    private struct MotionJawBurst {
        let start: Double
        let duration: Double
        let amplitude: Double
    }

    private struct MotionJawBlock {
        let name: String
        let start: Double
        let end: Double
    }

    private struct MotionJawDelivered {
        let event: InputEvent
        let deliveredAt: Double
    }

    private func sample(
        t: Double,
        acceleration: Vector3,
        rotation: Vector3 = .zero,
        yaw: Double = 0,
        pitch: Double = 0
    ) -> HeadMotionSample {
        HeadMotionSample(t: t, yaw: yaw, pitch: pitch, roll: 0,
                         rotationRate: rotation, userAcceleration: acceleration,
                         gravity: Vector3(0, 0, -1))
    }

    /// Build one monotonic stream. The optional dropout is a single missing
    /// sample at a time, keeping every interval below the detector's .16 s gap.
    private func stream(
        duration: Double,
        seed: UInt64,
        bursts: [MotionJawBurst],
        blocks: [MotionJawBlock] = [],
        dropoutEvery: Int? = nil,
        impulseAt: Double? = nil
    ) -> [HeadMotionSample] {
        var rng = MotionJawTestRNG(seed: seed)
        var output: [HeadMotionSample] = []
        output.reserveCapacity(Int(duration * 25))
        var t = 0.0
        var index = 0
        var impulseEmitted = false
        while t <= duration {
            let dt = 0.035 + 0.010 * rng.nextUnit()
            let dropped = dropoutEvery.map { $0 > 0 && index > 0 && index % $0 == 0 } ?? false
            var acceleration = Vector3(
                0.00065 * rng.normal() + 0.00055 * sin(0.31 * t),
                0.00065 * rng.normal() + 0.00042 * cos(0.23 * t),
                0.00065 * rng.normal() + 0.00030 * sin(0.17 * t)
            )
            var rotation = Vector3.zero
            var yaw = 0.0
            var pitch = 0.0

            if let burst = bursts.first(where: { t >= $0.start && t <= $0.start + $0.duration }) {
                let phase = (t - burst.start) / burst.duration
                let envelope = sin(Double.pi * min(max(phase, 0), 1))
                acceleration.x += burst.amplitude * envelope
                acceleration.y += burst.amplitude * 0.33 * envelope
            }

            for block in blocks where t >= block.start && t < block.end {
                let phase = (t - block.start) * 2 * Double.pi
                switch block.name {
                case "head-turn":
                    rotation = Vector3(0.05 * sin(phase), 0.72 * sin(0.7 * phase), 0.04 * cos(phase))
                    acceleration.x += 0.025 * sin(phase)
                    acceleration.y += 0.015 * cos(0.7 * phase)
                    yaw = 0.35 * sin(0.7 * phase)
                    pitch = 0.10 * cos(phase)
                case "nod":
                    rotation = Vector3(0.55 * sin(1.3 * phase), 0.06 * cos(phase), 0.04 * sin(phase))
                    acceleration.z += 0.08 * sin(1.3 * phase)
                    pitch = 0.22 * sin(1.3 * phase)
                case "walking":
                    let p = 1.8 * phase
                    acceleration.x += 0.072 * sin(p)
                    acceleration.y += 0.035 * sin(p + 0.8)
                    acceleration.z += 0.020 * cos(p)
                    rotation = Vector3(0.10 * sin(p), 0.08 * cos(p), 0)
                case "talking":
                    let p = 2.4 * phase
                    acceleration.x += 0.009 * max(0, sin(p))
                    acceleration.y += 0.004 * sin(p + 0.5)
                    acceleration.z += 0.003 * cos(p)
                    rotation = Vector3(0.035 * sin(p), 0.025 * cos(p), 0)
                case "chewing":
                    let p = 2.0 * phase
                    acceleration.x += 0.013 * max(0, sin(p))
                    acceleration.y += 0.005 * sin(p + 0.2)
                    acceleration.z += 0.004 * cos(p)
                default:
                    break
                }
            }

            // One .08 g impulse is below the .12 g gross-motion veto and should
            // still be rejected by support count/duration evidence.
            if let impulseAt, t >= impulseAt, !impulseEmitted {
                acceleration.x += 0.08
                impulseEmitted = true
            }
            if !dropped { output.append(sample(t: t, acceleration: acceleration, rotation: rotation, yaw: yaw, pitch: pitch)) }
            index += 1
            t += dt
        }
        return output
    }

    private func feed(_ samples: [HeadMotionSample], into detector: JawClenchDetector) -> [MotionJawDelivered] {
        var output: [MotionJawDelivered] = []
        for sample in samples {
            for event in detector.process(sample) {
                output.append(MotionJawDelivered(event: event, deliveredAt: sample.t))
            }
        }
        return output
    }

    private func matches(_ event: InputEvent, to burst: MotionJawBurst) -> Bool {
        // event.t is estimated onset; the first active frame can be one tick late.
        event.source == .jaw && event.kind == .click &&
            event.t >= burst.start - 0.10 && event.t <= burst.start + 0.12
    }

    private func score(_ events: [MotionJawDelivered], truths: [MotionJawBurst])
        -> (precision: Double, recall: Double, f1: Double, matched: Int) {
        var remaining = truths
        var matched = 0
        for delivered in events where delivered.event.source == .jaw && delivered.event.kind == .click {
            if let index = remaining.firstIndex(where: { matches(delivered.event, to: $0) }) {
                remaining.remove(at: index)
                matched += 1
            }
        }
        let clickCount = events.filter { $0.event.source == .jaw && $0.event.kind == .click }.count
        let precision = clickCount == 0 ? 0 : Double(matched) / Double(clickCount)
        let recall = truths.isEmpty ? 0 : Double(matched) / Double(truths.count)
        let f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall)
        return (precision, recall, f1, matched)
    }

    private struct MotionJawBench {
        var precision = 0.0, recall = 0.0, f1 = 0.0, meanLatency = 0.0, p90Latency = 0.0
        var matched = 0, events = 0, truths = 0
        var fpPerMinute: [String: Double] = [:]
        var missed: [Double] = []
    }

    private func mixedBenchmark(config: JawClenchDetector.Config, seed: UInt64 = 0xC1E4CE) -> MotionJawBench {
        let bursts = [
            MotionJawBurst(start: 3.0, duration: 0.14, amplitude: 0.022),
            MotionJawBurst(start: 7.2, duration: 0.20, amplitude: 0.031),
            MotionJawBurst(start: 11.3, duration: 0.11, amplitude: 0.017),
            MotionJawBurst(start: 15.9, duration: 0.27, amplitude: 0.037),
            MotionJawBurst(start: 21.0, duration: 0.16, amplitude: 0.026),
            MotionJawBurst(start: 25.8, duration: 0.12, amplitude: 0.014),
            MotionJawBurst(start: 30.1, duration: 0.22, amplitude: 0.034),
            MotionJawBurst(start: 35.0, duration: 0.18, amplitude: 0.019),
            MotionJawBurst(start: 40.2, duration: 0.29, amplitude: 0.040),
            MotionJawBurst(start: 45.3, duration: 0.13, amplitude: 0.018),
            MotionJawBurst(start: 50.5, duration: 0.24, amplitude: 0.028),
            MotionJawBurst(start: 56.2, duration: 0.15, amplitude: 0.021),
            MotionJawBurst(start: 63.0, duration: 0.20, amplitude: 0.030),
            MotionJawBurst(start: 70.1, duration: 0.10, amplitude: 0.016),
            MotionJawBurst(start: 76.0, duration: 0.26, amplitude: 0.035),
            MotionJawBurst(start: 83.4, duration: 0.17, amplitude: 0.023),
            MotionJawBurst(start: 90.0, duration: 0.21, amplitude: 0.032),
            MotionJawBurst(start: 98.2, duration: 0.12, amplitude: 0.015),
            MotionJawBurst(start: 106.1, duration: 0.28, amplitude: 0.039),
            MotionJawBurst(start: 114.4, duration: 0.16, amplitude: 0.020),
            MotionJawBurst(start: 123.0, duration: 0.23, amplitude: 0.027),
            MotionJawBurst(start: 132.1, duration: 0.14, amplitude: 0.018),
            MotionJawBurst(start: 141.0, duration: 0.25, amplitude: 0.033)
        ]
        let blocks = [
            MotionJawBlock(name: "head-turn", start: 67.0, end: 69.5),
            MotionJawBlock(name: "nod", start: 72.0, end: 74.8),
            MotionJawBlock(name: "walking", start: 78.0, end: 88.0),
            MotionJawBlock(name: "talking", start: 92.0, end: 103.0),
            MotionJawBlock(name: "chewing", start: 116.0, end: 121.5)
        ]
        let samples = stream(duration: 150, seed: seed, bursts: bursts,
                             blocks: blocks, dropoutEvery: 127, impulseAt: 18.4)
        let delivered = feed(samples, into: JawClenchDetector(config: config))
        let metrics = score(delivered, truths: bursts)
        let clicks = delivered.filter { $0.event.source == .jaw && $0.event.kind == .click }
        let latencies = clicks.map { $0.deliveredAt - $0.event.t }.sorted()
        var out = MotionJawBench()
        out.precision = metrics.precision; out.recall = metrics.recall; out.f1 = metrics.f1
        out.matched = metrics.matched; out.events = clicks.count; out.truths = bursts.count
        out.meanLatency = latencies.isEmpty ? 0 : latencies.reduce(0, +) / Double(latencies.count)
        out.p90Latency = latencies.isEmpty ? 0 : latencies[min(latencies.count - 1, Int(0.9 * Double(latencies.count)))]
        out.missed = bursts.filter { burst in !clicks.contains { matches($0.event, to: burst) } }.map(\.start)
        for block in blocks {
            let fp = clicks.filter { d in d.event.t >= block.start && d.event.t < block.end && !bursts.contains { matches(d.event, to: $0) } }.count
            out.fpPerMinute[block.name] = Double(fp) / ((block.end - block.start) / 60)
        }
        return out
    }

    private func report(_ label: String, _ m: MotionJawBench) {
        print(String(format: "MotionJaw %@: TP=%d/%d events=%d precision=%.3f recall=%.3f F1=%.3f meanLatency=%.3fs p90=%.3fs",
                     label, m.matched, m.truths, m.events, m.precision, m.recall, m.f1, m.meanLatency, m.p90Latency)
              + " FP/min " + m.fpPerMinute.sorted { $0.key < $1.key }.map { "\($0.key)=\(String(format: "%.2f", $0.value))" }.joined(separator: " "))
    }

    func testLongContinuousSyntheticBenchmarkReportsUsefulMetrics() {
        var classic = JawClenchDetector.Config(); classic.earlyCommitEnabled = false
        let before = mixedBenchmark(config: classic)
        let after = mixedBenchmark(config: JawClenchDetector.Config())
        report("classic (release+quietAfter)", before)
        report("early-commit (default)", after)
        print("MotionJaw missed burst starts: \(after.missed)")
        XCTAssertGreaterThanOrEqual(after.precision, 0.90)
        XCTAssertGreaterThanOrEqual(after.recall, 0.80)
        XCTAssertGreaterThanOrEqual(after.f1, max(0.84, before.f1 - 0.005), "early commit must not hurt F1")
        XCTAssertLessThan(after.meanLatency, 0.30, "early commit should cut mean delivery latency")
        for (name, fp) in after.fpPerMinute { XCTAssertLessThanOrEqual(fp, 12.0, "nuisance \(name) FP/min \(fp)") }
        // Robustness across noise seeds: early commit no worse than classic on F1 on average.
        var fb = 0.0, fa = 0.0
        for seed: UInt64 in [1, 2, 3, 4, 5, 6, 7, 8] {
            fb += mixedBenchmark(config: classic, seed: seed).f1; fa += mixedBenchmark(config: JawClenchDetector.Config(), seed: seed).f1
        }
        print(String(format: "MotionJaw 8-seed mean F1 classic=%.3f early=%.3f", fb / 8, fa / 8))
        XCTAssertGreaterThanOrEqual(fa / 8, fb / 8 - 0.01)
    }

    func testZZJawSweep() throws {
        guard ProcessInfo.processInfo.environment["GF_SWEEP"] != nil else { throw XCTSkip("sweep") }
        var classic = JawClenchDetector.Config(); classic.earlyCommitEnabled = false
        report("classic", mixedBenchmark(config: classic))
        for d in [0.04, 0.06, 0.08, 0.10, 0.12] { for r in [1.0, 1.25] { for q in [0.3, 0.6] {
            var c = JawClenchDetector.Config(); c.earlyCommitDuration = d; c.earlyCommitPeakRatio = r; c.earlyCommitQuietBefore = q
            var f = 0.0, l = 0.0
            for seed: UInt64 in [0xC1E4CE, 1, 2, 3, 4, 5] { let m = mixedBenchmark(config: c, seed: seed); f += m.f1; l += m.meanLatency }
            report(String(format: "d=%.2f r=%.2f q=%.2f meanF1(6)=%.3f meanLat(6)=%.3f", d, r, q, f / 6, l / 6), mixedBenchmark(config: c))
        } } }
        var fc = 0.0, lc = 0.0
        for seed: UInt64 in [0xC1E4CE, 1, 2, 3, 4, 5] { let m = mixedBenchmark(config: classic, seed: seed); fc += m.f1; lc += m.meanLatency }
        print(String(format: "MotionJaw classic meanF1(6)=%.3f meanLat(6)=%.3f", fc / 6, lc / 6))
    }

    func testSingleOutlierAndLongGapDoNotBecomeClicks() {
        let samples = stream(duration: 8, seed: 0xBAD5EED, bursts: [], impulseAt: 3.0)
        let delivered = feed(samples, into: JawClenchDetector())
        XCTAssertTrue(delivered.filter { $0.event.source == .jaw && $0.event.kind == .click }.isEmpty,
                      "a one-frame below-gate impulse produced a jaw click: \(delivered)")

        // Interrupt a real candidate before release. Its missing tail must not
        // complete a click when ordinary rest resumes after the gap.
        var withGap = stream(duration: 8, seed: 0xBAD5EED, bursts: [
            MotionJawBurst(start: 3.0, duration: 0.20, amplitude: 0.030)
        ])
        withGap.removeAll { $0.t > 3.12 && $0.t < 3.50 }
        let gapClicks = feed(withGap, into: JawClenchDetector())
            .filter { $0.event.source == .jaw && $0.event.kind == .click }
        XCTAssertTrue(gapClicks.isEmpty, "gap cancellation leaked a click: \(gapClicks)")
    }

    func testDoubleClenchProducesRightClickAndSingleSettlesLeft() {
        var config = JawClenchDetector.Config()
        config.doubleClenchEnabled = true
        config.doubleClenchInterval = 0.70
        let detector = JawClenchDetector(config: config)
        let pair = [MotionJawBurst(start: 2.0, duration: 0.16, amplitude: 0.030),
                    MotionJawBurst(start: 2.60, duration: 0.16, amplitude: 0.030)]
        let pairOutput = feed(stream(duration: 5, seed: 0xD0B1E, bursts: pair), into: detector)
            .filter { $0.event.source == .jaw && $0.event.kind == .click }
        XCTAssertEqual(pairOutput.count, 1, "double clench should emit exactly one event: \(pairOutput)")
        XCTAssertEqual(pairOutput.first?.event.button, MouseButton.right, "double clench must map to right click")
        XCTAssertGreaterThanOrEqual(pairOutput.first?.event.confidence ?? 0, 0.5)
        XCTAssertGreaterThanOrEqual(pairOutput.first?.event.t ?? 0, 1.9)

        detector.reset()
        let singleOutput = feed(stream(duration: 7, seed: 0xD0B1E,
                                       bursts: [MotionJawBurst(start: 2.0, duration: 0.16, amplitude: 0.030)]),
                                into: detector)
            .filter { $0.event.source == .jaw && $0.event.kind == .click }
        XCTAssertEqual(singleOutput.count, 1, "single candidate should settle to one click: \(singleOutput)")
        XCTAssertEqual(singleOutput.first?.event.button, MouseButton.left)
        XCTAssertGreaterThanOrEqual(singleOutput.first?.deliveredAt ?? 0,
                                    (singleOutput.first?.event.t ?? 0) + config.doubleClenchInterval)
    }

    func testCalibrationReportImprovesWeakClenchDetection() throws {
        let rest = stream(duration: 3, seed: 0x51A, bursts: [])
        let training = stream(duration: 8, seed: 0xCA11B, bursts: [
            MotionJawBurst(start: 4.0, duration: 0.20, amplitude: 0.016),
            MotionJawBurst(start: 6.0, duration: 0.22, amplitude: 0.016)
        ])
        let weak = stream(duration: 5, seed: 0xCA11C, bursts: [
            MotionJawBurst(start: 2.0, duration: 0.20, amplitude: 0.010)
        ])

        let defaultClicks = feed(weak, into: JawClenchDetector())
            .filter { $0.event.source == .jaw && $0.event.kind == .click }
        let calibratedDetector = JawClenchDetector()
        let report = try calibratedDetector.calibrate(restSamples: rest, clenchSamples: training)
        let calibratedClicks = feed(weak, into: calibratedDetector)
            .filter { $0.event.source == .jaw && $0.event.kind == .click }

        print(String(format: "MotionJaw calibration: restMedian=%.5f MAD=%.5f clenchQ1=%.5f threshold=%.5f defaultWeak=%d calibratedWeak=%d",
                     report.restMedianRMS, report.restMAD, report.clenchPeakRMS,
                     report.thresholdRMS, defaultClicks.count, calibratedClicks.count))
        XCTAssertGreaterThan(report.restMedianRMS, 0)
        XCTAssertGreaterThanOrEqual(report.restMAD, 0)
        XCTAssertGreaterThan(report.clenchPeakRMS, report.restMedianRMS)
        XCTAssertGreaterThan(report.thresholdRMS, report.restMedianRMS)
        XCTAssertLessThanOrEqual(report.thresholdRMS, report.clenchPeakRMS)
        XCTAssertEqual(defaultClicks.count, 0,
                       "default threshold unexpectedly accepted weak clench: \(defaultClicks)")
        XCTAssertEqual(calibratedClicks.count, 1,
                       "calibrated threshold should recover one held-out weak clench: \(calibratedClicks)")
    }
}
