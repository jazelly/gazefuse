import XCTest
@testable import GazeFuseCore

/// Unit tests for the synthetic model itself, `HandGestureFeatures` and `HandOneEuroFilter`.
final class HandFeaturesTests: XCTestCase {

    private func cleanSample(_ pose: HandTestPose, _ transform: HandTestTransform = HandTestTransform()) -> HandSample {
        var joints: [HandJoint: HandPoint] = [:]
        for (j, p) in HandTestModel.joints(pose) {
            let q = transform.apply(p)
            joints[j] = HandPoint(x: q.x, y: q.y, confidence: 0.95)
        }
        return HandSample(t: 0, chirality: "right", joints: joints)
    }

    func testSyntheticModelGeometry() {
        let open = HandGestureFeatures(sample: cleanSample(HandTestPose()))!
        XCTAssertEqual(open.handSize, 0.16, accuracy: 0.01, "hand size ≈ palm length (scale)")
        XCTAssertGreaterThan(open.index!.distance, 0.9)
        XCTAssertLessThan(open.index!.distance, 1.8)
        XCTAssertGreaterThan(open.middle!.distance, 0.9)

        var pinch = HandTestPose(); pinch.indexPinch = 1
        let closed = HandGestureFeatures(sample: cleanSample(pinch))!
        XCTAssertEqual(closed.index!.distance, pinch.contactGap, accuracy: 0.02, "thumb IK lands on the index tip")
        XCTAssertGreaterThan(closed.middle!.distance, 0.3, "middle tip stays away during an index pinch")
        XCTAssertNotEqual(closed.indexState, .curled, "a pinching index finger must not be vetoed as curled")

        var mid = HandTestPose(); mid.middlePinch = 1
        let closedMid = HandGestureFeatures(sample: cleanSample(mid))!
        XCTAssertEqual(closedMid.middle!.distance, mid.contactGap, accuracy: 0.02)
        XCTAssertGreaterThan(closedMid.index!.distance, 0.3)

        // Half way the distance is clearly in between (monotone closing trajectory).
        var half = HandTestPose(); half.indexPinch = 0.5
        let h = HandGestureFeatures(sample: cleanSample(half))!
        XCTAssertGreaterThan(h.index!.distance, 0.3)
        XCTAssertLessThan(h.index!.distance, open.index!.distance)

        var fist = HandTestPose()
        fist.indexCurl = 1; fist.middleCurl = 1; fist.ringCurl = 1; fist.littleCurl = 1; fist.thumbWrapped = true
        let f = HandGestureFeatures(sample: cleanSample(fist))!
        XCTAssertTrue(f.isFist)
        XCTAssertEqual(f.indexState, .curled)
        XCTAssertLessThan(f.index!.distance, 0.35, "fist must be a real challenge: thumb tip close to the curled index tip")
        print("hand model: open dIdx \(open.index!.distance) dMid \(open.middle!.distance); closed \(closed.index!.distance); fist dIdx \(f.index!.distance) indexCurl \(f.indexCurl!)")
    }

    func testScaleRotationTranslationInvariance() {
        var pose = HandTestPose(); pose.indexPinch = 0.6; pose.ringCurl = 0.5
        let base = HandGestureFeatures(sample: cleanSample(pose))!
        var rng = HandTestRNG(seed: 7)
        for _ in 0..<20 {
            var tr = HandTestTransform()
            tr.scale = rng.uniform(0.05, 0.3)
            tr.center = Point2(rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9))
            tr.rotation = rng.uniform(-.pi, .pi)
            let f = HandGestureFeatures(sample: cleanSample(pose, tr))!
            XCTAssertEqual(f.index!.distance, base.index!.distance, accuracy: 1e-9)
            XCTAssertEqual(f.middle!.distance, base.middle!.distance, accuracy: 1e-9)
            XCTAssertEqual(f.indexCurl!, base.indexCurl!, accuracy: 1e-9)
            XCTAssertEqual(f.handSize, tr.scale * base.handSize / 0.16, accuracy: 1e-9)
        }
    }

    func testTiltRobustness() {
        var pose = HandTestPose(); pose.indexPinch = 1
        let flat = HandGestureFeatures(sample: cleanSample(pose))!
        for deg in [20.0, 40.0, 55.0] {
            var tr = HandTestTransform(); tr.yaw = deg * .pi / 180
            let f = HandGestureFeatures(sample: cleanSample(pose, tr))!
            // Tilt shrinks everything; the max-of-two-axes hand size keeps the normalized
            // closed distance well under the on-threshold.
            XCTAssertLessThan(f.index!.distance, 0.2, "yaw \(deg)")
            XCTAssertGreaterThan(f.handSize, flat.handSize * 0.75, "yaw \(deg)")
            var tp = HandTestTransform(); tp.pitch = deg * .pi / 180
            let g = HandGestureFeatures(sample: cleanSample(pose, tp))!
            XCTAssertLessThan(g.index!.distance, 0.2, "pitch \(deg)")
            XCTAssertGreaterThan(g.handSize, flat.handSize * 0.75, "pitch \(deg)")
        }
    }

    func testFingerStatesOpenPalmAndFist() {
        let open = HandGestureFeatures(sample: cleanSample(HandTestPose()))!
        for finger in HandFinger.allCases { XCTAssertEqual(open.state(of: finger), .extended, "\(finger)") }
        XCTAssertTrue(open.isOpenPalm)
        XCTAssertFalse(open.isFist)

        var curled = HandTestPose(); curled.indexCurl = 1; curled.middleCurl = 1; curled.ringCurl = 1; curled.littleCurl = 1
        let c = HandGestureFeatures(sample: cleanSample(curled))!
        XCTAssertTrue(c.isFist)
        XCTAssertFalse(c.isOpenPalm)
        XCTAssertEqual(c.state(of: .thumb), .extended)

        var pinch = HandTestPose(); pinch.indexPinch = 1
        let p = HandGestureFeatures(sample: cleanSample(pinch))!
        XCTAssertFalse(p.isOpenPalm, "thumb near a fingertip is not an open palm")
        XCTAssertEqual(p.state(of: .middle), .extended)
    }

    func testMissingJoints() {
        var s = cleanSample(HandTestPose())
        s.joints[.indexTip]!.confidence = 0.1
        let f = HandGestureFeatures(sample: s)!
        XCTAssertNil(f.index)
        XCTAssertNotNil(f.middle)
        XCTAssertEqual(f.indexState, .unknown)
        XCTAssertFalse(f.isOpenPalm)

        // Wrist gone (hand close to the camera): palm width carries the hand size.
        var w = cleanSample(HandTestPose())
        w.joints[.wrist] = nil
        let g = HandGestureFeatures(sample: w)!
        XCTAssertEqual(g.handSize, 0.16, accuracy: 0.02)

        // No palm at all -> nil.
        var none = cleanSample(HandTestPose())
        for j in [HandJoint.wrist, .indexMCP, .middleMCP, .ringMCP, .littleMCP] { none.joints[j]!.confidence = 0.05 }
        XCTAssertNil(HandGestureFeatures(sample: none))
        XCTAssertNil(HandGestureFeatures(sample: HandSample(t: 0, chirality: nil, joints: [:])))
    }

    func testOneEuroFilterStepAndNoise() {
        // Step response: a fast step passes with little lag thanks to beta.
        var f = HandOneEuroFilter(minCutoff: 4, beta: 1, dCutoff: 1)
        let dt = 1.0 / 30
        f.filter(1, dt: dt)
        var crossed = -1
        for i in 1...10 {
            let v = f.filter(0.06, dt: dt)
            if crossed < 0 && v < 0.28 { crossed = i }
        }
        XCTAssertGreaterThan(crossed, 0)
        XCTAssertLessThanOrEqual(crossed, 2, "filtered distance should cross the on-threshold within 2 frames of a step")

        // Noise attenuation at rest.
        var g = HandOneEuroFilter(minCutoff: 4, beta: 1, dCutoff: 1)
        var rng = HandTestRNG(seed: 3)
        var rawVar = 0.0, filtVar = 0.0
        for _ in 0..<2000 {
            let x = 0.5 + 0.03 * rng.gaussian()
            let y = g.filter(x, dt: dt)
            rawVar += (x - 0.5) * (x - 0.5); filtVar += (y - 0.5) * (y - 0.5)
        }
        XCTAssertLessThan(filtVar, rawVar * 0.8)

        var p = HandOneEuroFilter2D(minCutoff: 1.5, beta: 5)
        p.filter(Point2(0, 0), dt: dt)
        let v = p.filter(Point2(1, 1), dt: dt)
        XCTAssertGreaterThan(v.x, 0.5)
        XCTAssertEqual(v.x, v.y, accuracy: 1e-12)
        p.reset()
        XCTAssertNil(p.value)
    }

    func testConfigCodableRoundTrip() throws {
        var c = PinchGestureClassifier.Config()
        c.pinchOnDistance = 0.3; c.scrollEnabled = true; c.motionReference = .palmCenter
        c.features.minJointConfidence = 0.4
        let data = try JSONEncoder().encode(c)
        let back = try JSONDecoder().decode(PinchGestureClassifier.Config.self, from: data)
        XCTAssertEqual(back, c)
    }
}
