import XCTest
@testable import GazeFuseCore

/// Tests for the integrator-added fusion features: head→gaze leak compensation,
/// adaptive gaze noise, and an end-to-end simulator regression gate.
final class FusionLeakAdaptiveTests: XCTestCase {
    private struct RNG { var s: UInt64
        mutating func next() -> Double { s &+= 0x9E3779B97F4A7C15; var z = s
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9; z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return Double((z ^ (z >> 31)) >> 11) / Double(1 << 53) }
        mutating func gauss() -> Double { let u = max(next(), 1e-12), v = next(); return (-2 * log(u)).squareRoot() * cos(2 * .pi * v) } }

    func testLeakCompensatorLearnsTrueLeakAndRemovesIt() {
        let trueLeak = Point2(800, 500)
        let c = GazeHeadLeakCompensator(pointerGain: 1500)
        var rng = RNG(s: 7)
        var t = 0.0
        var theta = Point2.zero
        for fix in 0..<40 {
            c.newFixation(at: t)
            let target = Point2(300 + Double(fix % 5) * 200, 300)
            let rate = Point2(rng.gauss() * 0.1, rng.gauss() * 0.1)
            for _ in 0..<30 {    // 1 s fixation: AirPods 25 Hz-ish + camera 30 Hz interleaved
                t += 1.0 / 30
                theta = theta + rate * (1.0 / 30)
                c.ingestHead(yawRate: rate.x, pitchRate: rate.y, at: t)
                let z = target + Point2(trueLeak.x * theta.x, trueLeak.y * theta.y) + Point2(rng.gauss() * 10, rng.gauss() * 10)
                c.learn(z, at: t)
            }
        }
        let L = c.clampedLeak
        XCTAssertEqual(L.x, trueLeak.x, accuracy: 120, "yaw leak estimate \(L.x)")
        XCTAssertEqual(L.y, trueLeak.y, accuracy: 120, "pitch leak estimate \(L.y)")
    }

    func testLeakIsClampedWhenUserTracksCursorWithEyes() {
        // Gaze follows the head-driven cursor exactly (leak == pointer gain): clamp at 0.9·gain.
        let c = GazeHeadLeakCompensator(pointerGain: 1000)
        var t = 0.0, theta = 0.0
        for _ in 0..<600 { t += 0.033; theta += 0.003; c.ingestHead(yawRate: 0.003 / 0.033, pitchRate: 0, at: t)
            c.learn(Point2(1000 * theta, 0), at: t) }
        XCTAssertLessThanOrEqual(c.clampedLeak.x, 900 + 1e-6)
    }

    func testAdaptiveGazeNoiseGrowsWithNoisyTracker() {
        let f = EKFCursorFusion()
        var rng = RNG(s: 11)
        var t = 0.0
        for _ in 0..<600 {
            t += 1.0 / 30
            f.ingest(gaze: GazeEstimate(t: t, point: Point2(700 + rng.gauss() * 60, 500 + rng.gauss() * 60), confidence: 1))
            _ = f.cursor(at: t)
        }
        XCTAssertGreaterThan(f.effectiveGazeSigma, 40, "σ̂ = \(f.effectiveGazeSigma)")
        XCTAssertLessThan(f.effectiveGazeSigma, 90)
        var cfg = EKFCursorFusionConfig(); cfg.adaptiveGazeNoise = false
        XCTAssertEqual(EKFCursorFusion(config: cfg).effectiveGazeSigma, cfg.gazeSigmaPoints)
    }

    func testLowConfidenceJumpNeverResets() {
        let f = EKFCursorFusion()
        var t = 0.0
        for _ in 0..<60 { t += 1.0 / 30; f.ingest(gaze: GazeEstimate(t: t, point: Point2(700, 500), confidence: 1)) }
        let resets = f.stats.resets
        for _ in 0..<3 { t += 1.0 / 30; f.ingest(gaze: GazeEstimate(t: t, point: Point2(700, 650), confidence: 0.4)) }
        XCTAssertEqual(f.stats.resets, resets, "blink-edge-like low-confidence jumps must not reset")
        XCTAssertLessThan((f.cursor(at: t).position - Point2(700, 500)).length, 10)
    }

    /// End-to-end regression gate on the simulated Webgrid (fixed seeds, 30×30, 60 s).
    func testSimulatedWebgridRegressionGate() {
        let seeds: [UInt64] = [501, 502, 503, 504]
        let head = SimSummary.run(config: .make(preset: .gazeHeadEKF, noise: .typical), seeds: seeds)
        let gazeOnly = SimSummary.run(config: .make(preset: .gazeOnly, noise: .typical), seeds: seeds)
        print(String(format: "regression gate: gazeHeadEKF typical %.2f BPS, gazeOnly %.2f BPS", head.bpsMean, gazeOnly.bpsMean))
        XCTAssertGreaterThan(head.bpsMean, 4.0)
        XCTAssertGreaterThan(head.accuracyMean, 0.9)
        XCTAssertGreaterThan(head.bpsMean, gazeOnly.bpsMean + 3)
    }
}
