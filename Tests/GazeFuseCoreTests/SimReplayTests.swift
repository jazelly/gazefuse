import XCTest
@testable import GazeFuseCore

final class SimReplayTests: XCTestCase {
    private let screen = ScreenRect(x: 0, y: 0, width: 800, height: 600)

    func testReplaySkipsRawGazeAndHandWithoutHooks() {
        let events: [RecordedEvent] = [
            .header(RecordingHeader(createdAt: "test", screen: screen)),
            .gaze(GazeSample(t: 0, leftPupil: Point2(0.1, 0.2), rightPupil: Point2(0.2, 0.2),
                             headYaw: 0, headPitch: 0, headRoll: 0, faceCenter: Point2(0.5, 0.5), faceSize: 0.3, confidence: 0.9)),
            .hand(HandSample(t: 0.01, chirality: "right", joints: [:])),
            .handLost(0.02),
            .marker(t: 0.03, text: "raw-only")
        ]
        let replay = ReplayEngine(fusion: EKFCursorFusion(screen: screen), arbiter: ClickArbiter())
        let result = replay.run(events)
        XCTAssertEqual(result.events, events.count)
        XCTAssertEqual(result.rawGazeSamples, 1)
        XCTAssertEqual(result.rawGazeMapped, 0)
        XCTAssertEqual(result.rawGazeSkipped, 1)
        XCTAssertEqual(result.rawHandSamples, 1)
        XCTAssertEqual(result.rawHandClassified, 0)
        XCTAssertEqual(result.rawHandSkipped, 1)
        XCTAssertEqual(result.actions.count, 0)
        XCTAssertEqual(result.markers, ["raw-only"])
    }

    func testReplayDrivesDerivedGazeAndInputThroughArbiter() {
        let events: [RecordedEvent] = [
            .header(RecordingHeader(createdAt: "test", screen: screen)),
            .gazeEstimate(GazeEstimate(t: 0, point: Point2(200, 150), confidence: 1)),
            .cursor(CursorState(t: 0.05, position: Point2(200, 150))),
            .input(InputEvent(t: 0.20, source: .jaw, kind: .click, confidence: 1)),
            .action(ClickAction(t: 0.21, kind: .click, button: .left, position: Point2(200, 150), source: .jaw))
        ]
        let replay = ReplayEngine(fusion: EKFCursorFusion(screen: screen), arbiter: ClickArbiter())
        let result = replay.run(events)
        XCTAssertEqual(result.gazeEstimates, 1)
        XCTAssertEqual(result.inputEvents, 1)
        XCTAssertEqual(result.actions.count, 1)
        XCTAssertEqual(result.actions.first?.kind, .click)
        XCTAssertEqual(result.actions.first?.button, .left)
        XCTAssertEqual(result.recordedActionCount, 1)
        XCTAssertNotNil(result.recordedCursorDeviation)
        XCTAssertGreaterThan(result.cursorSamples, 1)
    }

    func testReplayRawHooksMapAndClassify() {
        let raw = GazeSample(t: 0.1, leftPupil: nil, rightPupil: nil, headYaw: nil, headPitch: nil,
                             headRoll: nil, faceCenter: Point2(0.5, 0.5), faceSize: 0.2, confidence: 1)
        let hand = HandSample(t: 0.1, chirality: "right", joints: [:])
        var gazeCalls = 0
        var handCalls = 0
        var hooks = ReplayHooks()
        hooks.gazeMapper = { sample in
            gazeCalls += 1
            return GazeEstimate(t: sample.t, point: Point2(300, 250), confidence: 1)
        }
        hooks.handClassifier = { _, _ in
            handCalls += 1
            return [InputEvent(t: 0.2, source: .hand, kind: .click, confidence: 1)]
        }
        let events: [RecordedEvent] = [
            .header(RecordingHeader(createdAt: "test", screen: screen)),
            .gaze(raw), .hand(hand)
        ]
        let replay = ReplayEngine(fusion: EKFCursorFusion(screen: screen), arbiter: ClickArbiter(), hooks: hooks)
        let result = replay.run(events)
        XCTAssertEqual(gazeCalls, 1)
        XCTAssertEqual(handCalls, 1)
        XCTAssertEqual(result.rawGazeMapped, 1)
        XCTAssertEqual(result.rawGazeSkipped, 0)
        XCTAssertEqual(result.rawHandClassified, 1)
        XCTAssertEqual(result.actions.count, 1)
    }

    func testReplayReaderCountsMalformedLines() {
        let valid = RecordingReader.encode([.marker(t: 0.2, text: "ok")])
        let parsed = RecordingReader.parse(valid + "{bad json}\n")
        XCTAssertEqual(parsed.events.count, 1)
        XCTAssertEqual(parsed.skipped, 1)
    }
}
