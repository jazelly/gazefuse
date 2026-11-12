import Foundation
import XCTest
@testable import GazeFuseCore

private struct MotionPoseTestRNG {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func unit() -> Double {
        // SplitMix64: compact deterministic generator suitable for synthetic
        // sensor jitter; no global RNG state is shared between tests.
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        z ^= z >> 31
        return Double(z >> 11) / Double(1 << 53)
    }

    mutating func signed(amplitude: Double) -> Double {
        (unit() * 2 - 1) * amplitude
    }
}

final class MotionHeadPoseTests: XCTestCase {
    private func sample(t: Double, yaw: Double, pitch: Double = 0,
                        rotation: Vector3 = .zero, acceleration: Vector3 = .zero) -> HeadMotionSample {
        HeadMotionSample(t: t, yaw: yaw, pitch: pitch, roll: 0,
                         rotationRate: rotation, userAcceleration: acceleration)
    }

    func testWrapsAcrossPiWithoutLargeJump() {
        let tracker = HeadPoseTracker(config: .init(smoothingTimeConstant: 0.001, deadbandRadians: 0,
                                                    maximumGap: 1, driftRemovalEnabled: false))
        _ = tracker.process(sample(t: 0, yaw: 0))
        let before = tracker.process(sample(t: 0.04, yaw: Double.pi - 0.03))
        let after = tracker.process(sample(t: 0.08, yaw: -Double.pi + 0.03))
        XCTAssertLessThan(abs(after.deltaYaw), 0.1)
        XCTAssertGreaterThan(before.yaw, 3.0)
        XCTAssertLessThan(after.yaw, -3.0)

        let stale = tracker.process(sample(t: 0.08, yaw: 0.5))
        XCTAssertFalse(stale.isValid)
        XCTAssertFalse(stale.crossedGap)
        XCTAssertEqual(stale.deltaYaw, 0, accuracy: 1e-12)
        XCTAssertEqual(stale.yaw, after.yaw, accuracy: 1e-12)
        let resumed = tracker.process(sample(t: 0.12, yaw: -Double.pi + 0.02))
        XCTAssertLessThan(abs(resumed.deltaYaw), 0.1)
    }

    func testRecenterAndDeadband() {
        let tracker = HeadPoseTracker(config: .init(smoothingTimeConstant: 0.01, deadbandRadians: 0.05,
                                                    maximumGap: 1, driftRemovalEnabled: false))
        _ = tracker.process(sample(t: 1, yaw: 1.0, pitch: -0.4))
        let small = tracker.process(sample(t: 1.04, yaw: 1.03, pitch: -0.43))
        XCTAssertEqual(small.yaw, 0, accuracy: 1e-8)
        XCTAssertEqual(small.pitch, 0, accuracy: 1e-8)
        tracker.recenter()
        let zero = tracker.process(sample(t: 1.08, yaw: 1.03, pitch: -0.43))
        XCTAssertEqual(zero.yaw, 0, accuracy: 1e-8)
        XCTAssertEqual(zero.pitch, 0, accuracy: 1e-8)
        let move = tracker.process(sample(t: 1.12, yaw: 1.20, pitch: -0.43))
        XCTAssertGreaterThan(move.yaw, 0.1)
    }

    func testSlowDriftOnlyNearZero() {
        let config = HeadPoseTracker.Config(smoothingTimeConstant: 0.01, deadbandRadians: 0,
                                            maximumGap: 1, driftRemovalEnabled: true,
                                            driftTimeConstant: 0.5, driftCaptureWindowRadians: 0.08,
                                            minimumStillDuration: 0.1,
                                            stillRotationRateThreshold: 0.02,
                                            stillAccelerationThreshold: 0.02)
        let tracker = HeadPoseTracker(config: config)
        _ = tracker.process(sample(t: 0, yaw: 0))
        // A slow sensor zero drift near the origin is allowed to settle.
        var nearZero = 0.0
        for i in 1...100 {
            nearZero += 0.0005
            _ = tracker.process(sample(t: Double(i) * 0.02, yaw: nearZero))
        }
        let settled = tracker.process(sample(t: 2.1, yaw: nearZero))
        XCTAssertLessThan(abs(settled.yaw), 0.01)

        tracker.reset()
        _ = tracker.process(sample(t: 0, yaw: 0))
        // A deliberate, sustained offset is outside the capture window and stays.
        var held: HeadPoseEstimate?
        for i in 1...100 {
            held = tracker.process(sample(t: Double(i) * 0.02, yaw: 0.35))
        }
        XCTAssertNotNil(held)
        XCTAssertGreaterThan(abs(held!.yaw), 0.25)
    }

    func testIrregularIntervalsAndDropoutDoNotInventDelta() {
        let tracker = HeadPoseTracker(config: .init(smoothingTimeConstant: 0.01, deadbandRadians: 0,
                                                    maximumGap: 0.2, driftRemovalEnabled: false))
        _ = tracker.process(sample(t: 0, yaw: 0))
        let regular = tracker.process(sample(t: 0.04, yaw: 0.2))
        XCTAssertGreaterThan(regular.deltaYaw, 0)
        let gap = tracker.process(sample(t: 1.2, yaw: 1.4))
        XCTAssertTrue(gap.crossedGap)
        XCTAssertEqual(gap.deltaYaw, 0, accuracy: 1e-12)
        let next = tracker.process(sample(t: 1.24, yaw: 1.45))
        XCTAssertLessThan(abs(next.deltaYaw), 0.2)
        // A duplicate/out-of-order timestamp is ignored as a continuous motion.
        let duplicate = tracker.process(sample(t: 1.24, yaw: 2.4))
        XCTAssertFalse(duplicate.isValid)
        XCTAssertFalse(duplicate.crossedGap)
        XCTAssertEqual(duplicate.deltaYaw, 0, accuracy: 1e-12)
    }

    func testJitteredConstantPoseSettlesNearExpectedRelativeAngle() {
        let tracker = HeadPoseTracker(config: .init(smoothingTimeConstant: 0.10, deadbandRadians: 0,
                                                    maximumGap: 0.5, driftRemovalEnabled: false))
        _ = tracker.process(sample(t: 0, yaw: 0.30, pitch: -0.10))

        var rng = MotionPoseTestRNG(seed: 0x6A09E667F3BCC909)
        var t = 0.0
        var errors: [Double] = []
        var last: HeadPoseEstimate?
        for i in 1...240 {
            t += 0.035 + rng.signed(amplitude: 0.012)
            let result = tracker.process(sample(t: t,
                                               yaw: 0.72 + rng.signed(amplitude: 0.012),
                                               pitch: 0.08 + rng.signed(amplitude: 0.010)))
            last = result
            if i > 190 {
                let yawError = result.yaw - 0.42
                let pitchError = result.pitch - 0.18
                errors.append((yawError * yawError + pitchError * pitchError).squareRoot())
            }
        }

        let meanError = errors.reduce(0, +) / Double(errors.count)
        print(String(format: "MotionHeadPoseTests.meanErrorRad=%.6f", meanError))
        XCTAssertLessThan(meanError, 0.015)
        XCTAssertLessThan(abs(last!.yaw - 0.42), 0.03)
        XCTAssertLessThan(abs(last!.pitch - 0.18), 0.03)
    }
}
