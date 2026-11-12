import XCTest
@testable import GazeFuseCore

final class FusionTests: XCTestCase {
    private let screen = ScreenRect(x: 0, y: 0, width: 1000, height: 700)

    func testFixationSmoothingRejectsSingleOutlierAndConfirmsSaccade() {
        var config = EKFCursorFusionConfig(
            gazeSigmaPoints: 12,
            minGazeConfidence: 0.2,
            positionProcessSigma: 0.5,
            biasProcessSigma: 0.2,
            biasPriorSigma: 20,
            saccadeMinJumpPoints: 40,
            saccadeConfirmSamples: 2,
            saccadeConfirmWindow: 0.2,
            saccadeMaxWindow: 0,
            head: HeadPointerConfig(enabled: false))
        config.saccadeGateNIS = 9.21
        let fusion = EKFCursorFusion(config: config, screen: screen)
        let base = Point2(400, 300)
        var rawSquared = 0.0
        var filteredSquared = 0.0
        var count = 0

        for i in 0..<120 {
            let t = Double(i) / 60
            let jitter = 10 * sin(Double(i) * 1.7) + 4 * cos(Double(i) * 0.43)
            let sample = Point2(base.x + jitter, base.y - 0.6 * jitter)
            rawSquared += (sample - base).length * (sample - base).length
            fusion.ingest(gaze: GazeEstimate(t: t, point: sample, confidence: 1))
            if i >= 20 {
                let output = fusion.cursor(at: t).position
                filteredSquared += (output - base).length * (output - base).length
                count += 1
            }
        }
        let rawRMS = (rawSquared / 120).squareRoot()
        let filteredRMS = (filteredSquared / Double(count)).squareRoot()
        print(String(format: "Fusion fixation metric: rawRMS=%.3f filteredRMS=%.3f ratio=%.3f", rawRMS, filteredRMS, filteredRMS / rawRMS))
        XCTAssertLessThan(filteredRMS, 0.75 * rawRMS)

        let before = fusion.cursor(at: 2).position
        fusion.ingest(gaze: GazeEstimate(t: 2.01, point: Point2(700, 500), confidence: 1))
        _ = fusion.cursor(at: 2.01)
        XCTAssertGreaterThanOrEqual(fusion.stats.gazeRejected, 1)
        XCTAssertLessThan(fusion.cursor(at: 2.01).position.distance(to: before), 60)

        fusion.ingest(gaze: GazeEstimate(t: 2.02, point: Point2(700, 500), confidence: 1))
        fusion.ingest(gaze: GazeEstimate(t: 2.03, point: Point2(700, 500), confidence: 1))
        let saccade = fusion.cursor(at: 2.03)
        print(String(format: "Fusion saccade metric: resetError=%.3f", saccade.position.distance(to: Point2(700, 500))))
        XCTAssertEqual(saccade.mode, .saccade)
        XCTAssertLessThan(saccade.position.distance(to: Point2(700, 500)), 2)
        XCTAssertNotNil(fusion.lastReset)

        let acceptedBeforeBlink = fusion.stats.gazeAccepted
        fusion.ingest(gaze: GazeEstimate(t: 2.04, point: Point2(100, 100), confidence: 0))
        XCTAssertEqual(fusion.stats.gazeAccepted, acceptedBeforeBlink)
        XCTAssertGreaterThan(fusion.stats.gazeLowConfidence, 0)
    }

    func testHandDeltaClampsAndHistoryHandlesOutOfOrderGaze() {
        var config = EKFCursorFusionConfig(
            gazeSigmaPoints: 4,
            positionProcessSigma: 0,
            biasProcessSigma: 0,
            biasPriorSigma: 1,
            saccadeMinJumpPoints: 200,
            saccadeMaxWindow: 0,
            head: HeadPointerConfig(enabled: false))
        config.reorderWindow = 0.1
        let fusion = EKFCursorFusion(config: config, screen: screen)
        fusion.ingest(gaze: GazeEstimate(t: 0, point: Point2(200, 200), confidence: 1))
        XCTAssertEqual(fusion.cursor(at: 0).position, Point2(200, 200))

        fusion.ingest(relativeDelta: Point2(100, -50), at: 0.1)
        XCTAssertEqual(fusion.cursor(at: 0.1).position, Point2(300, 150))
        XCTAssertEqual(fusion.position(at: 0.1), Point2(300, 150))

        fusion.ingest(gaze: GazeEstimate(t: -1, point: Point2(20, 20), confidence: 1))
        XCTAssertEqual(fusion.stats.gazeDropped, 1)

        fusion.ingest(gaze: GazeEstimate(t: 0.11, point: Point2(2000, -50), confidence: 1))
        let clamped = fusion.cursor(at: 0.11).position
        XCTAssertGreaterThanOrEqual(clamped.x, screen.origin.x)
        XCTAssertLessThanOrEqual(clamped.x, screen.origin.x + screen.width - 1)
        XCTAssertGreaterThanOrEqual(clamped.y, screen.origin.y)
        XCTAssertLessThanOrEqual(clamped.y, screen.origin.y + screen.height - 1)
    }

    func testZeroConfidenceIsIgnoredEvenWhenThresholdIsZeroAndDirectDropsLateGaze() {
        let ekf = EKFCursorFusion(config: EKFCursorFusionConfig(
            minGazeConfidence: 0,
            head: HeadPointerConfig(enabled: false)), screen: screen)
        ekf.ingest(gaze: GazeEstimate(t: 0, point: Point2(100, 100), confidence: 1))
        ekf.ingest(gaze: GazeEstimate(t: 0.1, point: Point2(900, 600), confidence: 0))
        XCTAssertEqual(ekf.stats.gazeLowConfidence, 1)
        XCTAssertLessThan(ekf.cursor(at: 0.1).position.distance(to: Point2(100, 100)), 1)

        let direct = DirectGazeCursor(config: DirectGazeCursorConfig(minGazeConfidence: 0), screen: screen)
        direct.ingest(gaze: GazeEstimate(t: 0, point: Point2(100, 100), confidence: 1))
        direct.ingest(gaze: GazeEstimate(t: 0.1, point: Point2(900, 600), confidence: 0))
        direct.ingest(gaze: GazeEstimate(t: -0.1, point: Point2(700, 500), confidence: 1))
        XCTAssertEqual(direct.cursor(at: 0.1).position, Point2(100, 100))
    }

    func testDelayedGazeDoesNotAmplifyContinuousHeadMotion() {
        let head = HeadPointerConfig(
            enabled: true,
            source: .rotationRate,
            yawSign: 1,
            pitchSign: 1,
            deadbandRadPerSec: 0,
            pointsPerRadian: 500,
            accelExponent: 0,
            velocitySigma: 2,
            timeout: 0.5)
        let config = EKFCursorFusionConfig(
            gazeSigmaPoints: 20,
            positionProcessSigma: 0,
            headAccelerationSigma: 100,
            velocityDecayTau: 10,
            biasProcessSigma: 0,
            biasPriorSigma: 1,
            saccadeMinJumpPoints: 1000,
            saccadeMaxWindow: 0,
            head: head)
        let synchronous = EKFCursorFusion(config: config, screen: screen)
        let delayed = EKFCursorFusion(config: config, screen: screen)
        let gazePoint = screen.center
        synchronous.ingest(gaze: GazeEstimate(t: 0, point: gazePoint, confidence: 1))
        delayed.ingest(gaze: GazeEstimate(t: 0, point: gazePoint, confidence: 1))

        let dt = 0.02
        let latency = 0.06
        for i in 1...60 {
            let t = Double(i) * dt
            // 0.8 rad/s around the configured yaw axis = 400 pt/s.
            let measuredHead = HeadMotionSample(t: t, yaw: 0, pitch: 0, roll: 0,
                                                rotationRate: Vector3(0, 0.8, 0), userAcceleration: .zero)
            synchronous.ingest(gaze: GazeEstimate(t: t, point: gazePoint, confidence: 1))
            synchronous.ingest(head: measuredHead)
            _ = synchronous.cursor(at: t)

            delayed.ingest(head: measuredHead)
            if t >= latency {
                delayed.ingest(gaze: GazeEstimate(t: t - latency, point: gazePoint, confidence: 1))
            }
            _ = delayed.cursor(at: t)
        }

        let syncSpeed = abs(synchronous.stateVelocity.x)
        let delayedSpeed = abs(delayed.stateVelocity.x)
        let positionGap = synchronous.statePosition.distance(to: delayed.statePosition)
        print(String(format: "Fusion delayed-head metric: syncSpeed=%.2f delayedSpeed=%.2f positionGap=%.2f syncX=%.2f delayedX=%.2f syncBias=%.2f delayedBias=%.2f", syncSpeed, delayedSpeed, positionGap, synchronous.statePosition.x, delayed.statePosition.x, synchronous.bias.x, delayed.bias.x))
        XCTAssertLessThan(delayedSpeed, syncSpeed * 1.5 + 100)
        XCTAssertLessThan(positionGap, 140)
    }

    func testOutOfOrderAttitudeSampleDoesNotPolluteDerivative() {
        let headConfig = HeadPointerConfig(
            enabled: true,
            source: .attitudeDerivative,
            yawSign: 1,
            pitchSign: 1,
            deadbandRadPerSec: 0,
            pointsPerRadian: 100,
            accelExponent: 0,
            velocitySigma: 1,
            timeout: 1)
        let config = EKFCursorFusionConfig(
            gazeSigmaPoints: 10,
            positionProcessSigma: 0,
            biasProcessSigma: 0,
            biasPriorSigma: 1,
            head: headConfig)
        let fusion = EKFCursorFusion(config: config, screen: screen)
        fusion.ingest(gaze: GazeEstimate(t: 0, point: screen.center, confidence: 1))
        fusion.ingest(head: Self.head(t: 1.0, yaw: 0))
        fusion.ingest(head: Self.head(t: 1.1, yaw: 0.1))
        fusion.ingest(head: Self.head(t: 1.05, yaw: -0.5)) // late packet: must be ignored
        fusion.ingest(head: Self.head(t: 1.2, yaw: 0.2))

        XCTAssertGreaterThan(fusion.stateVelocity.x, 70)
        XCTAssertLessThan(fusion.stateVelocity.x, 150)
    }

    func testHeadOnlyStreamProducesRelativeCursorAndConfidence() {
        let headConfig = HeadPointerConfig(
            enabled: true,
            source: .rotationRate,
            yawSign: 1,
            pitchSign: 1,
            deadbandRadPerSec: 0,
            pointsPerRadian: 100,
            accelExponent: 0,
            velocitySigma: 2,
            timeout: 0.25)
        let fusion = EKFCursorFusion(config: EKFCursorFusionConfig(head: headConfig), screen: screen)
        fusion.ingest(head: HeadMotionSample(t: 0, yaw: 0, pitch: 0, roll: 0,
                                             rotationRate: Vector3(0, 0.2, 0), userAcceleration: .zero))
        let active = fusion.cursor(at: 0.2)
        print(String(format: "Fusion head-only metric: position=(%.2f,%.2f) speed=(%.2f,%.2f) confidence=%.3f", active.position.x, active.position.y, active.velocity.x, active.velocity.y, active.confidence))
        XCTAssertEqual(active.mode, .fixation)
        XCTAssertGreaterThan(active.confidence, 0)
        XCTAssertGreaterThan(active.position.x, screen.center.x)

        XCTAssertEqual(fusion.cursor(at: 0.4).mode, .idle)
    }

    func testClickArbiterLocksOnsetRejectsDuplicateReleaseAndAcceptsOwnerLowConfidenceUp() {
        let cursor = FusionTestCursor(samples: [
            (0, Point2(10, 10)),
            (0.1, Point2(20, 20)),
            (0.2, Point2(30, 30)),
            (0.5, Point2(40, 40)),
            (0.6, Point2(50, 50)),
        ])
        let arbiter = ClickArbiter(config: ClickArbiterConfig(
            onsetLookback: 0.1,
            minInterval: 0.05,
            duplicateWindow: 0.2,
            handMinConfidence: 0.5,
            jawMinConfidence: 0.5,
            clickMaxDuration: 0.3,
            clickMaxTravel: 20))

        let down = arbiter.ingest(InputEvent(t: 0.2, source: .hand, kind: .down, confidence: 1), cursor: cursor)
        XCTAssertEqual(down.count, 1)
        XCTAssertEqual(down[0].position, Point2(20, 20))
        XCTAssertEqual(down[0].source, .hand)

        XCTAssertTrue(arbiter.ingest(InputEvent(t: 0.21, source: .jaw, kind: .down, confidence: 1), cursor: cursor).isEmpty)
        XCTAssertTrue(arbiter.ingest(InputEvent(t: 0.22, source: .jaw, kind: .up, confidence: 1), cursor: cursor).isEmpty)
        XCTAssertTrue(arbiter.activeButtons.contains(.left))

        let up = arbiter.ingest(InputEvent(t: 0.3, source: .hand, kind: .up, confidence: 0.05), cursor: cursor)
        XCTAssertEqual(up.count, 1)
        XCTAssertEqual(up[0].kind, .up)
        XCTAssertEqual(up[0].position, Point2(20, 20))
        XCTAssertTrue(arbiter.activeButtons.isEmpty)

        XCTAssertTrue(arbiter.ingest(InputEvent(t: 0.31, source: .hand, kind: .click, confidence: 1), cursor: cursor).isEmpty)
        let right = arbiter.ingest(InputEvent(t: 0.6, source: .jaw, kind: .click, button: .right, confidence: 1), cursor: cursor)
        XCTAssertEqual(right.count, 1)
        XCTAssertEqual(right[0].button, .right)
        XCTAssertEqual(right[0].position, Point2(40, 40))
    }

    private static func head(t: TimeSec, yaw: Double) -> HeadMotionSample {
        HeadMotionSample(t: t, yaw: yaw, pitch: 0, roll: 0,
                         rotationRate: .zero, userAcceleration: .zero)
    }
}

private final class FusionTestCursor: CursorFusing {
    var screen = ScreenRect(x: 0, y: 0, width: 1000, height: 700)
    private let samples: [(TimeSec, Point2)]

    init(samples: [(TimeSec, Point2)]) { self.samples = samples.sorted { $0.0 < $1.0 } }

    func ingest(gaze: GazeEstimate) {}
    func ingest(head: HeadMotionSample) {}
    func ingest(relativeDelta: Point2, at t: TimeSec) {}

    func cursor(at t: TimeSec) -> CursorState {
        CursorState(t: t, position: position(at: t) ?? screen.center, mode: .fixation, confidence: 1)
    }

    func position(at t: TimeSec) -> Point2? {
        samples.last(where: { $0.0 <= t })?.1
    }

    func reset() {}
}
