import XCTest
@testable import GazeFuseCore

final class MatrixRecordingTests: XCTestCase {
    func testInverseAndCholesky() {
        let a = Matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
        let inv = a.inverse!
        let id = a * inv
        for i in 0..<3 { for j in 0..<3 { XCTAssertEqual(id[i, j], i == j ? 1 : 0, accuracy: 1e-12) } }
        let b = Matrix(column: [1, 2, 3])
        let x = a.choleskySolve(b)!
        let r = a * x
        for i in 0..<3 { XCTAssertEqual(r[i, 0], b[i, 0], accuracy: 1e-12) }
        XCTAssertNil(Matrix([[1, 2], [2, 4]]).inverse)
    }

    func testRecordingRoundTrip() {
        let events: [RecordedEvent] = [
            .header(RecordingHeader(createdAt: "x", screen: ScreenRect(x: 0, y: 0, width: 1512, height: 982))),
            .gaze(GazeSample(t: 1, leftPupil: Point2(0.1, 0), rightPupil: nil, headYaw: 0.1, headPitch: nil, headRoll: 0,
                             faceCenter: Point2(0.5, 0.5), faceSize: 0.3, confidence: 0.9)),
            .hand(HandSample(t: 1.1, chirality: "right", joints: [.thumbTip: HandPoint(x: 0.4, y: 0.5, confidence: 0.9)])),
            .handLost(1.2),
            .head(HeadMotionSample(t: 1.3, yaw: 0, pitch: 0, roll: 0, rotationRate: .zero, userAcceleration: Vector3(0, 0, 0.01))),
            .input(InputEvent(t: 1.4, source: .hand, kind: .down)),
            .cursor(CursorState(t: 1.5, position: Point2(10, 20))),
            .action(ClickAction(t: 1.6, kind: .click, button: .left, position: Point2(1, 2), source: .jaw)),
            .calibrationTarget(t: 1.7, target: Point2(100, 100)),
            .marker(t: 1.8, text: "hello"),
        ]
        let text = RecordingReader.encode(events) + "garbage line\n"
        let (back, skipped) = RecordingReader.parse(text)
        XCTAssertEqual(skipped, 1)
        XCTAssertEqual(back, events)
    }
}
