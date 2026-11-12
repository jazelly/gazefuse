import XCTest
@testable import GazeFuseCore

/// Metric tests for `PinchGestureClassifier` on synthetic hand kinematics (see
/// HandSyntheticHand.swift). Every scenario is seeded, so the numbers printed here are the
/// ones reported in docs/agent-notes/hand.md.
final class HandPinchClassifierTests: XCTestCase {

    // MARK: - Scenario builders

    /// Taps / short holds with random placement, relaxed-finger idle periods, re-entries,
    /// dropouts and outliers. `buttons` cycles through the pinch types.
    private func tapScenario(seed: UInt64, fps: Double = 30, count: Int = 40, buttons: [MouseButton] = [.left],
                             contactGap: Double = 0.06, dropProb: Double = 0.02) -> HandTestScenario {
        let s = HandTestScenario(seed: seed, fps: fps)
        s.dropProb = dropProb
        s.pose.contactGap = contactGap
        let f = fps / 30   // keep durations in seconds when the frame rate changes
        func frames(_ n: Double) -> Int { Int((n * f).rounded()) }
        s.randomizePlacement()
        s.absent(frames: frames(5))
        for i in 0..<count {
            if i % 5 == 0 {
                s.absent(frames: frames(10))
                s.randomizePlacement()
            }
            let relax = s.rng.uniform(0, 0.3)
            s.pose.indexCurl = relax; s.pose.middleCurl = relax; s.pose.ringCurl = relax * 1.5; s.pose.littleCurl = relax * 1.5
            s.idle(frames: frames(s.rng.uniform(10, 40)))
            // `f` scales the number of frames so each episode keeps the same wall-clock
            // duration at 60 fps. Per-frame translation must therefore shrink by `f`,
            // otherwise the 60 fps scenario travels four times as far as the 30 fps one.
            let drift = Point2(s.rng.uniform(-0.002, 0.002), s.rng.uniform(-0.002, 0.002)) / f
            s.pinch(buttons[i % buttons.count],
                    closeFrames: frames(s.rng.uniform(2, 5).rounded()),
                    holdFrames: frames(s.rng.uniform(3, 14).rounded()),
                    openFrames: frames(s.rng.uniform(2, 5).rounded()),
                    translation: drift)
            s.idle(frames: frames(s.rng.uniform(5, 15)))
        }
        s.idle(frames: frames(20))
        return s
    }

    /// Everything except pinches: waving / wiggling fingers, fast sweeps with motion blur,
    /// fists, leaving and re-entering, partial occlusion, relaxed hands.
    private func nonPinchScenario(seed: UInt64, rounds: Int = 40, appendingTo existing: HandTestScenario? = nil) -> HandTestScenario {
        let s = existing ?? HandTestScenario(seed: seed)
        s.dropProb = 0.02
        s.randomizePlacement()
        for r in 0..<rounds {
            if r % 4 == 0 { s.absent(frames: 8); s.randomizePlacement() }
            s.idle(frames: 10)
            s.wave(frames: Int(s.rng.uniform(20, 50)), minAperture: 0.45)
            s.fastSweep(frames: Int(s.rng.uniform(6, 15)), speed: s.rng.uniform(0.8, 2.0))
            s.idle(frames: 5)
            s.fist(frames: Int(s.rng.uniform(8, 25)))
            s.idle(frames: 5)
            let occl: Set<HandJoint> = r % 2 == 0 ? [.indexTip, .indexDIP, .thumbTip] : [.wrist, .thumbCMC, .middleTip]
            s.occluded(frames: Int(s.rng.uniform(5, 15)), joints: occl)
            // Hand entering from the edge: only part of it visible for a few frames.
            s.occluded(frames: 3, joints: [.wrist, .littleMCP, .littlePIP, .littleDIP, .littleTip, .ringTip])
            s.idle(frames: 10)
        }
        return s
    }

    // MARK: - Click precision / recall / latency

    func testLeftTapsAt30fps() {
        let s = tapScenario(seed: 1)
        let r = HandTestEvaluator.evaluate(s, classifier: PinchGestureClassifier())
        print("taps 30fps: \(r)  frames \(s.count)")
        XCTAssertEqual(r.episodes, 40)
        XCTAssertGreaterThanOrEqual(r.recall, 0.975)
        XCTAssertGreaterThanOrEqual(r.precision, 0.975)
        XCTAssertEqual(r.wrongButton, 0)
        XCTAssertEqual(r.missingUps, 0)
        XCTAssertEqual(r.extraUps, 0)
        XCTAssertLessThanOrEqual(r.meanDownLatency, 2.5, "frames from first contact to .down")
        XCTAssertLessThanOrEqual(r.maxDownLatency, 4)
        XCTAssertLessThanOrEqual(r.meanUpLatency, 2.5, "frames from first non-contact frame to .up")
    }

    func testLeftTapsAt60fps() {
        let s = tapScenario(seed: 2, fps: 60)
        let r = HandTestEvaluator.evaluate(s, classifier: PinchGestureClassifier(), lateSlack: 6, upSlack: 6)
        print("taps 60fps: \(r)  frames \(s.count)")
        XCTAssertEqual(r.episodes, 40)
        XCTAssertGreaterThanOrEqual(r.recall, 0.975)
        XCTAssertGreaterThanOrEqual(r.precision, 0.975)
        XCTAssertEqual(r.missingUps, 0)
        XCTAssertLessThanOrEqual(r.meanDownLatency, 3.5)
        XCTAssertLessThanOrEqual(r.meanUpLatency, 3.5)
    }

    func testLeftRightDiscrimination() {
        let s = tapScenario(seed: 3, count: 60, buttons: [.left, .right, .right, .left])
        let r = HandTestEvaluator.evaluate(s, classifier: PinchGestureClassifier())
        print("left/right: \(r)")
        XCTAssertEqual(r.episodes, 60)
        XCTAssertEqual(r.wrongButton, 0)
        XCTAssertGreaterThanOrEqual(r.recall, 0.975)
        XCTAssertGreaterThanOrEqual(r.precision, 0.975)
        XCTAssertEqual(r.missingUps, 0)
    }

    func testManySeedsAggregate() {
        var tp = 0, fp = 0, fn = 0, wrong = 0, lat: [Int] = [], ups: [Int] = [], missingUps = 0
        var fpSegments: [String: Int] = [:]
        for seed in 10..<20 {
            let s = tapScenario(seed: UInt64(seed), count: 30, buttons: [.left, .left, .right])
            let r = HandTestEvaluator.evaluate(s, classifier: PinchGestureClassifier())
            tp += r.truePositives; fp += r.falsePositives; fn += r.falseNegatives; wrong += r.wrongButton
            lat += r.downLatencies; ups += r.upLatencies; missingUps += r.missingUps
            for (k, v) in r.falsePositiveSegments { fpSegments[k, default: 0] += v }
            if r.falsePositives > 0 { HandTestDiagnostics.dumpFalsePositives(s, r, label: "seed \(seed)") }
        }
        let precision = Double(tp) / Double(tp + fp), recall = Double(tp) / Double(tp + fn)
        let meanLat = Double(lat.reduce(0, +)) / Double(lat.count)
        let meanUp = Double(ups.reduce(0, +)) / Double(ups.count)
        print(String(format: "aggregate 10 seeds x 30 taps: TP %d FP %d FN %d wrong %d precision %.4f recall %.4f downLatency mean %.2f max %d upLatency mean %.2f max %d missingUps %d",
                     tp, fp, fn, wrong, precision, recall, meanLat, lat.max() ?? 0, meanUp, ups.max() ?? 0, missingUps) + " FP by segment \(fpSegments)")
        XCTAssertGreaterThanOrEqual(precision, 0.98)
        XCTAssertGreaterThanOrEqual(recall, 0.98)
        XCTAssertEqual(wrong, 0)
        XCTAssertEqual(missingUps, 0)
        XCTAssertLessThanOrEqual(meanLat, 2.5)
    }

    // MARK: - False positives

    func testNoFalsePositivesDuringNonPinchMotion() {
        var downs = 0, frames = 0
        var bySegment: [String: Int] = [:]
        for seed in 100..<105 {
            let s = nonPinchScenario(seed: UInt64(seed))
            let r = HandTestEvaluator.evaluate(s, classifier: PinchGestureClassifier())
            downs += r.falsePositives
            frames += s.count
            for (k, v) in r.falsePositiveSegments { bySegment[k, default: 0] += v }
            if r.falsePositives > 0 { HandTestDiagnostics.dumpFalsePositives(s, r, label: "non-pinch seed \(seed)") }
            XCTAssertEqual(r.episodes, 0)
        }
        print("non-pinch motion: \(frames) frames, false-positive downs \(downs) by segment \(bySegment)")
        XCTAssertEqual(downs, 0)
    }

    func testFistAloneNeverClicks() {
        // The fist pose puts the thumb tip ~0.15 hand sizes from the curled index tip:
        // only the curled-finger veto keeps it from clicking.
        let s = HandTestScenario(seed: 200)
        s.randomizePlacement()
        for _ in 0..<30 { s.idle(frames: 10); s.fist(frames: 20) }
        let r = HandTestEvaluator.evaluate(s, classifier: PinchGestureClassifier())
        XCTAssertEqual(r.falsePositives, 0)
        var noVeto = PinchGestureClassifier.Config(); noVeto.vetoCurledFinger = false
        let r2 = HandTestEvaluator.evaluate(s, classifier: PinchGestureClassifier(config: noVeto))
        print("fist: downs with veto \(r.falsePositives), without veto \(r2.falsePositives)")
        XCTAssertGreaterThan(r2.falsePositives, 0, "sanity: the fist scenario is a real challenge")
    }

    func testFastTranslationWithBlurDoesNotClick() {
        let s = HandTestScenario(seed: 300)
        s.randomizePlacement()
        for _ in 0..<60 {
            s.idle(frames: 8)
            s.pose.indexPinch = 0.5    // half-closed fingers while the hand flies across the frame
            s.fastSweep(frames: 12, speed: s.rng.uniform(1.0, 2.5))
            s.pose.indexPinch = 0
            if s.transform.center.x < 0.1 || s.transform.center.x > 0.9 || s.transform.center.y < 0.1 || s.transform.center.y > 0.9 {
                s.randomizePlacement()
            }
        }
        let r = HandTestEvaluator.evaluate(s, classifier: PinchGestureClassifier())
        print("fast sweeps: frames \(s.count) false positives \(r.falsePositives)")
        XCTAssertEqual(r.falsePositives, 0)
    }

    // MARK: - Drag

    func testDragContinuityAndMotionDelta() {
        var totalFrames = 0, deltaFrames = 0
        var maxIntegratedError = 0.0
        for seed in 400..<406 {
            let s = HandTestScenario(seed: UInt64(seed))
            s.dropProb = 0.03
            s.randomizePlacement()
            s.transform.center = Point2(0.5, 0.5)
            s.idle(frames: 15)
            let ang = s.rng.uniform(0, 2 * .pi)
            let v = Point2(cos(ang), sin(ang)) * 0.004          // 0.12 image units / s at 30 fps
            s.pinch(.left, closeFrames: 3, holdFrames: 90, openFrames: 3, translation: v)
            s.idle(frames: 15)
            let c = PinchGestureClassifier()
            let r = HandTestEvaluator.evaluate(s, classifier: c)
            XCTAssertEqual(r.truePositives, 1, "seed \(seed)")
            XCTAssertEqual(r.falsePositives, 0, "seed \(seed)")
            XCTAssertEqual(r.missingUps, 0, "seed \(seed)")
            XCTAssertEqual(r.extraUps, 0, "seed \(seed)")

            let down = r.events.first { $0.event.kind == .down }!, up = r.events.first { $0.event.kind == .up }!
            var sum = Point2.zero, n = 0
            for i in down.frame...up.frame { if let d = r.deltas[i] { sum = sum + d; n += 1 } }
            let held = up.frame - down.frame
            totalFrames += held; deltaFrames += n
            XCTAssertGreaterThanOrEqual(Double(n), 0.85 * Double(held), "deltas available on most held frames (seed \(seed))")
            // Integrated delta ≈ true translation over the held interval (filter lag ≈ 1–2 frames).
            let truth = v * Double(held)
            let err = (sum - truth).length / truth.length
            maxIntegratedError = max(maxIntegratedError, err)
            XCTAssertLessThan(err, 0.12, "integrated motion error \(err) (seed \(seed))")
            // Direction agrees.
            let dot = (sum.x * truth.x + sum.y * truth.y) / (sum.length * truth.length)
            XCTAssertGreaterThan(dot, 0.99, "seed \(seed)")
            XCTAssertTrue(r.isPinching[down.frame + 1])
            XCTAssertFalse(r.isPinching[up.frame])
        }
        print(String(format: "drag: delta available on %.1f%% of held frames; max integrated error %.4f",
                     100 * Double(deltaFrames) / Double(totalFrames), maxIntegratedError))
    }

    func testPalmCenterMotionReference() {
        let s = HandTestScenario(seed: 410)
        s.randomizePlacement(); s.transform.center = Point2(0.5, 0.5)
        s.idle(frames: 15)
        let v = Point2(0.003, -0.002)
        s.pinch(.left, closeFrames: 3, holdFrames: 60, openFrames: 3, translation: v)
        s.idle(frames: 10)
        var cfg = PinchGestureClassifier.Config(); cfg.motionReference = .palmCenter
        let r = HandTestEvaluator.evaluate(s, classifier: PinchGestureClassifier(config: cfg))
        XCTAssertEqual(r.truePositives, 1)
        let down = r.events.first { $0.event.kind == .down }!, up = r.events.first { $0.event.kind == .up }!
        var sum = Point2.zero
        for i in down.frame...up.frame { if let d = r.deltas[i] { sum = sum + d } }
        let truth = v * Double(up.frame - down.frame)
        XCTAssertLessThan((sum - truth).length / truth.length, 0.12)
    }

    // MARK: - Robustness to the hand disappearing / tips occluded

    func testHandLostReleasesButton() {
        let s = HandTestScenario(seed: 500)
        s.randomizePlacement()
        s.idle(frames: 15)
        s.pose.indexPinch = 1
        for _ in 0..<12 { s.step() }
        let lostAt = s.count
        s.absent(frames: 25)                      // hand leaves while pinched
        s.pose.indexPinch = 0
        s.idle(frames: 15)
        s.pinch(.left, closeFrames: 3, holdFrames: 6, openFrames: 3)
        s.idle(frames: 10)
        let c = PinchGestureClassifier()
        let r = HandTestEvaluator.evaluate(s, classifier: c)
        let ups = r.events.filter { $0.event.kind == .up }
        let downs = r.events.filter { $0.event.kind == .down }
        XCTAssertEqual(downs.count, 2)
        XCTAssertEqual(ups.count, 2)
        XCTAssertGreaterThanOrEqual(ups[0].frame, lostAt)
        XCTAssertLessThanOrEqual(ups[0].frame, lostAt + c.config.maxDropoutFrames + 1, "release promptly after the hand is lost")
        XCTAssertFalse(r.isPinching[lostAt + c.config.maxDropoutFrames + 1])
        XCTAssertGreaterThan(downs[1].frame, lostAt + 25, "re-entry pinch is detected again")
    }

    func testOccludedTipsAreBridgedThenReleased() {
        let s = HandTestScenario(seed: 510)
        s.randomizePlacement()
        s.idle(frames: 15)
        s.pose.indexPinch = 1
        for _ in 0..<8 { s.step() }
        s.occluded(frames: 2, joints: [.indexTip, .thumbTip])      // short: must be bridged
        for _ in 0..<8 { s.step() }
        let longOcclusionAt = s.count
        s.occluded(frames: 10, joints: [.indexTip, .thumbTip])     // long: must release
        for _ in 0..<8 { s.step() }
        s.pose.indexPinch = 0
        s.idle(frames: 10)
        let c = PinchGestureClassifier()
        let r = HandTestEvaluator.evaluate(s, classifier: c)
        let downs = r.events.filter { $0.event.kind == .down }
        let ups = r.events.filter { $0.event.kind == .up }
        XCTAssertEqual(downs.count, 1, "long occlusion while still closed must not re-arm a second pinch")
        XCTAssertEqual(ups.count, 1)
        XCTAssertGreaterThanOrEqual(ups[0].frame, longOcclusionAt)
        XCTAssertLessThanOrEqual(ups[0].frame, longOcclusionAt + c.config.maxDropoutFrames + 1)
        XCTAssertFalse(downs.contains { $0.frame >= longOcclusionAt }, "no onset is allowed until an open hand is observed")
    }

    func testClosedHandEnteringFrameDoesNotClickUntilOpenThenPinch() {
        let s = HandTestScenario(seed: 511)
        s.randomizePlacement()
        s.pose.indexPinch = 1
        for _ in 0..<18 { s.step() }

        // A hand can enter the camera already pinched. The classifier must stay disarmed
        // until it has observed an open hand; otherwise the first visible frame is a click.
        let c = PinchGestureClassifier()
        var eventsBeforeOpen: [InputEvent] = []
        for f in s.frames { eventsBeforeOpen += c.process(f.sample, at: f.t) }
        XCTAssertTrue(eventsBeforeOpen.isEmpty)

        s.pose.indexPinch = 0
        s.idle(frames: 10)
        s.pinch(.left, closeFrames: 3, holdFrames: 5, openFrames: 3)
        let newFrames = s.frames.dropFirst(18)
        var eventsAfterOpen: [InputEvent] = []
        for f in newFrames { eventsAfterOpen += c.process(f.sample, at: f.t) }
        XCTAssertEqual(eventsAfterOpen.map(\.kind), [.down, .up])
    }

    func testLongOcclusionWhileClosedRequiresOpenBeforeRearm() {
        let s = HandTestScenario(seed: 512)
        s.randomizePlacement()
        s.idle(frames: 12)
        s.pose.indexPinch = 1
        for _ in 0..<10 { s.step() }
        let c = PinchGestureClassifier()
        var allEvents: [InputEvent] = []
        for f in s.frames { allEvents += c.process(f.sample, at: f.t) }
        XCTAssertEqual(allEvents.filter { $0.kind == .down }.count, 1)

        // Long loss while the fingers remain closed releases the held gesture and disarms
        // onset. Visible closed frames after re-entry must not create a second click.
        s.occluded(frames: 2, joints: [.indexTip, .thumbTip])
        s.absent(frames: c.config.maxDropoutFrames + 3)
        for _ in 0..<8 { s.step() } // still closed, now visible again
        let beforeOpenCount = allEvents.count
        for f in s.frames.dropFirst(22) { allEvents += c.process(f.sample, at: f.t) }
        XCTAssertEqual(allEvents.count, beforeOpenCount + 1, "only the release from the original held pinch is emitted")
        XCTAssertEqual(allEvents.last?.kind, .up)

        s.pose.indexPinch = 0
        s.idle(frames: 10)
        s.pinch(.left, closeFrames: 3, holdFrames: 5, openFrames: 3)
        for f in s.frames.dropFirst(22 + 2 + c.config.maxDropoutFrames + 3 + 8) {
            allEvents += c.process(f.sample, at: f.t)
        }
        XCTAssertEqual(allEvents.filter { $0.kind == .down }.count, 2)
        XCTAssertEqual(allEvents.filter { $0.kind == .up }.count, 2)
    }

    func testOnsetDebounceClearsAcrossMissingFrame() {
        let s = HandTestScenario(seed: 513)
        s.randomizePlacement()
        s.noise.sigma = 0
        s.noise.outlierProb = 0
        s.dropProb = 0
        s.idle(frames: 8)
        var cfg = PinchGestureClassifier.Config()
        cfg.minPresenceFrames = 1
        cfg.minOnFrames = 3
        cfg.distanceMinCutoff = 100
        cfg.distanceBeta = 0
        let c = PinchGestureClassifier(config: cfg)
        for f in s.frames { XCTAssertTrue(c.process(f.sample, at: f.t).isEmpty) }

        // Two consecutive contact candidates are intentionally below minOnFrames. A
        // missing frame must clear that partial onset instead of allowing the next single
        // contact frame to complete it.
        s.pose.indexPinch = 1
        for _ in 0..<2 {
            s.step()
            XCTAssertTrue(c.process(s.frames.last!.sample, at: s.frames.last!.t).isEmpty)
        }
        s.absent(frames: 1)
        XCTAssertTrue(c.process(s.frames.last!.sample, at: s.frames.last!.t).isEmpty)

        // Two more contact frames still do not emit; the third consecutive frame after the
        // dropout is the first valid onset.
        for _ in 0..<2 {
            s.step()
            XCTAssertTrue(c.process(s.frames.last!.sample, at: s.frames.last!.t).isEmpty)
        }
        s.step()
        let down = c.process(s.frames.last!.sample, at: s.frames.last!.t)
        XCTAssertEqual(down.map(\.kind), [.down])

        s.pose.indexPinch = 0
        s.idle(frames: 4)
        var release: [InputEvent] = []
        for f in s.frames.suffix(4) { release += c.process(f.sample, at: f.t) }
        XCTAssertEqual(release.map(\.kind), [.up])
    }

    func testTwoFrameDeliveryDelayAppearsInOnsetLatency() {
        let s = HandTestScenario(seed: 515)
        s.randomizePlacement()
        s.noise.sigma = 0
        s.noise.outlierProb = 0
        s.dropProb = 0
        s.idle(frames: 12)
        s.pinch(.left, closeFrames: 3, holdFrames: 8, openFrames: 4)

        var cfg = PinchGestureClassifier.Config()
        cfg.minPresenceFrames = 1
        cfg.minOnFrames = 2
        cfg.distanceMinCutoff = 100
        cfg.distanceBeta = 0
        let c = PinchGestureClassifier(config: cfg)
        let deliveryDelay = 2
        var buffer: [HandSample?] = []
        var delivered: [HandTestEvent] = []
        for (frame, source) in s.frames.enumerated() {
            buffer.append(source.sample)
            guard buffer.count > deliveryDelay else { continue }
            let sample = buffer.removeFirst()
            let events = c.process(sample, at: source.t)
            delivered.append(contentsOf: events.map { HandTestEvent(frame: frame, event: $0) })
        }
        var nextTime = s.frames.last!.t + s.dt
        let flushFrame = s.frames.count + deliveryDelay
        for sample in buffer {
            let events = c.process(sample, at: nextTime)
            delivered.append(contentsOf: events.map { HandTestEvent(frame: flushFrame, event: $0) })
            nextTime += s.dt
        }

        let down = try! XCTUnwrap(delivered.first { $0.event.kind == .down })
        let firstContact = try! XCTUnwrap(s.frames.firstIndex { $0.indexContact })
        let latencyFrames = down.frame - firstContact
        print("two-frame delivery delay: contact frame \(firstContact), down frame \(down.frame), observed latency \(latencyFrames) frames")
        XCTAssertGreaterThanOrEqual(latencyFrames, deliveryDelay)
        XCTAssertLessThanOrEqual(latencyFrames, deliveryDelay + 4)
        XCTAssertGreaterThanOrEqual(down.event.t, s.frames[firstContact].t + Double(deliveryDelay) * s.dt)
        XCTAssertEqual(delivered.filter { $0.event.kind == .down }.count, 1)
    }

    func testAmbiguousDualPinchDoesNotChooseAButton() {
        let s = HandTestScenario(seed: 514)
        s.randomizePlacement()
        s.noise.sigma = 0
        s.noise.outlierProb = 0
        s.idle(frames: 12)
        // This fixture moves the middle tip independently to make two pairs overlap;
        // disable the anatomical curled-finger veto so the test isolates the documented
        // left/right discrimination margin rather than rejecting the synthetic pose first.
        var dualConfig = PinchGestureClassifier.Config()
        dualConfig.vetoCurledFinger = false
        let c = PinchGestureClassifier(config: dualConfig)
        for f in s.frames { _ = c.process(f.sample, at: f.t) }

        var pose = HandTestPose(); pose.indexPinch = 1
        let base = HandTestSynth.sample(pose: pose, transform: s.transform, noise: s.noise,
                                        t: 0, rng: &s.rng)
        var joints = base.joints
        let thumb = joints[.thumbTip]!
        let size = HandGestureFeatures(sample: base)!.handSize
        // Put both tips at nearly identical, valid contact distances. The middle finger
        // remains extended, so the only ambiguity is button arbitration itself.
        joints[.middleTip] = HandPoint(x: thumb.x - 0.07 * size, y: thumb.y, confidence: 0.95)
        let dual = HandSample(t: 1, chirality: "right", joints: joints)
        for i in 0..<5 {
            let e = c.process(HandSample(t: 1 + Double(i) / 30, chirality: "right", joints: dual.joints),
                              at: 1 + Double(i) / 30)
            XCTAssertTrue(e.isEmpty, "ambiguous dual pinch must be rejected")
        }
    }

    func testFlushReleasesHeldButton() {
        let s = HandTestScenario(seed: 520)
        s.randomizePlacement()
        s.idle(frames: 10)
        s.pose.indexPinch = 1
        for _ in 0..<10 { s.step() }
        let c = PinchGestureClassifier()
        var downs = 0
        for f in s.frames { downs += c.process(f.sample, at: f.t).filter { $0.kind == .down }.count }
        XCTAssertEqual(downs, 1)
        XCTAssertTrue(c.isPinching)
        let flushed = c.flush(at: s.frames.last!.t + 0.01)
        XCTAssertEqual(flushed.map { $0.kind }, [.up])
        XCTAssertFalse(c.isPinching)
        XCTAssertNil(c.pinchMotionDelta)
        XCTAssertTrue(c.flush(at: 1).isEmpty)
    }

    // MARK: - Scroll mode

    private func scrollScenario(seed: UInt64, translation: Point2, holdFrames: Int) -> HandTestScenario {
        let s = HandTestScenario(seed: seed)
        s.randomizePlacement(); s.transform.center = Point2(0.5, 0.5); s.transform.rotation = 0
        s.idle(frames: 15)
        s.pinch(.left, closeFrames: 3, holdFrames: holdFrames, openFrames: 3, translation: translation)
        s.idle(frames: 10)
        return s
    }

    func testScrollModeVerticalMoveScrolls() {
        var cfg = PinchGestureClassifier.Config(); cfg.scrollEnabled = true
        let s = scrollScenario(seed: 600, translation: Point2(0.0005, 0.006), holdFrames: 40)
        let r = HandTestEvaluator.evaluate(s, classifier: PinchGestureClassifier(config: cfg))
        XCTAssertEqual(r.events.filter { $0.event.kind == .down }.count, 0, "vertical pinch-move is a scroll, not a click")
        XCTAssertEqual(r.events.filter { $0.event.kind == .up }.count, 0)
        let scrolls = r.events.filter { $0.event.kind == .scroll }
        XCTAssertGreaterThan(scrolls.count, 20)
        let dy = scrolls.map { $0.event.scroll!.y }.reduce(0, +)
        let dx = scrolls.map { $0.event.scroll!.x }.reduce(0, +)
        XCTAssertGreaterThan(dy, 0, "hand moving up (+y image) scrolls with +dy")
        XCTAssertGreaterThan(abs(dy), 5 * abs(dx))
        print("scroll: \(scrolls.count) events, total dy \(dy) points")
    }

    func testScrollModeHorizontalMoveDrags() {
        var cfg = PinchGestureClassifier.Config(); cfg.scrollEnabled = true
        let s = scrollScenario(seed: 601, translation: Point2(0.006, 0.0005), holdFrames: 40)
        let r = HandTestEvaluator.evaluate(s, classifier: PinchGestureClassifier(config: cfg))
        XCTAssertEqual(r.scrollEvents, 0)
        let downs = r.events.filter { $0.event.kind == .down }
        XCTAssertEqual(downs.count, 1)
        XCTAssertEqual(r.events.filter { $0.event.kind == .up }.count, 1)
        // The deferred down carries the onset time, before the hand started moving.
        let down = downs[0]
        XCTAssertLessThan(down.event.t, s.frames[down.frame].t - 0.02)
        XCTAssertGreaterThanOrEqual(r.deltas[down.frame + 2]?.x ?? 0, 0.002)
    }

    func testScrollModeTapAndLongPress() {
        var cfg = PinchGestureClassifier.Config(); cfg.scrollEnabled = true
        // Quick tap: down + up at release, both after the pinch opened, down stamped at onset.
        let tap = scrollScenario(seed: 602, translation: .zero, holdFrames: 4)
        let r = HandTestEvaluator.evaluate(tap, classifier: PinchGestureClassifier(config: cfg))
        let kinds = r.events.map { $0.event.kind }
        print("scroll tap events: \(r.events.map { "\($0.event.kind)@\($0.frame) t=\($0.event.t)" })")
        XCTAssertEqual(kinds, [.down, .up])
        XCTAssertEqual(r.events[0].frame, r.events[1].frame)
        XCTAssertLessThan(r.events[0].event.t, r.events[1].event.t)
        XCTAssertEqual(r.truePositives, 1)
        // Long still press: down after scrollDeferMaxDuration, up at release.
        let press = scrollScenario(seed: 603, translation: .zero, holdFrames: 40)
        let p = HandTestEvaluator.evaluate(press, classifier: PinchGestureClassifier(config: cfg))
        let pk = p.events.map { $0.event.kind }
        XCTAssertEqual(pk, [.down, .up])
        let gap = press.frames[p.events[0].frame].t - p.events[0].event.t
        XCTAssertGreaterThanOrEqual(gap, cfg.scrollDeferMaxDuration - 0.001)
        XCTAssertLessThan(gap, cfg.scrollDeferMaxDuration + 0.1)
        XCTAssertGreaterThan(p.events[1].frame - p.events[0].frame, 10)
    }

    // MARK: - Adaptive thresholds

    func testAdaptiveThresholdsRecoverWideContactUser() {
        // Camera angle makes touching fingertips look 0.32 hand-sizes apart: above the fixed
        // on-threshold, so fixed thresholds mostly miss. The adaptive mode learns the user's
        // pinched distance from the pinches it does catch and recovers.
        let fixed = HandTestEvaluator.evaluate(tapScenario(seed: 700, count: 60, contactGap: 0.32),
                                               classifier: PinchGestureClassifier())
        var cfg = PinchGestureClassifier.Config(); cfg.adaptiveThresholds = true
        let adaptiveScenario = tapScenario(seed: 700, count: 60, contactGap: 0.32)
        let adaptiveClassifier = PinchGestureClassifier(config: cfg)
        let adaptive = HandTestEvaluator.evaluate(adaptiveScenario, classifier: adaptiveClassifier)
        print("wide contact (0.32): fixed \(fixed)")
        print("wide contact (0.32): adaptive \(adaptive)")

        _ = adaptiveScenario
        XCTAssertLessThan(fixed.recall, 0.7)
        // Near-miss learning lets adaptive mode learn from the pinches it *missed*.
        XCTAssertGreaterThanOrEqual(adaptive.recall, 0.85)
        XCTAssertGreaterThanOrEqual(adaptive.precision, 0.95)
        // Normal user: adaptive mode must not degrade.
        let normal = HandTestEvaluator.evaluate(tapScenario(seed: 701, count: 40), classifier: PinchGestureClassifier(config: cfg))
        print("normal contact: adaptive \(normal)")
        XCTAssertGreaterThanOrEqual(normal.recall, 0.975)
        XCTAssertGreaterThanOrEqual(normal.precision, 0.975)
        let np = HandTestEvaluator.evaluate(nonPinchScenario(seed: 702, rounds: 20), classifier: PinchGestureClassifier(config: cfg))
        XCTAssertEqual(np.falsePositives, 0, "adaptive mode must not add false positives on non-pinch motion")
    }

    func testAdaptiveNearMissLearningAcrossUsers() {
        // Several "wide contact" users (camera angle hides the contact) and seeds.
        var lines: [String] = []
        var fixedTP = 0, adaptTP = 0, total = 0, adaptFP = 0
        for gap in [0.30, 0.34, 0.38] {
            for seed: UInt64 in [710, 711, 712] {
                let fixed = HandTestEvaluator.evaluate(tapScenario(seed: seed, count: 40, contactGap: gap), classifier: PinchGestureClassifier())
                var cfg = PinchGestureClassifier.Config(); cfg.adaptiveThresholds = true
                // Same classifier keeps adapting, then sees a long block of non-pinch motion.
                let sc = nonPinchScenario(seed: seed, rounds: 12, appendingTo: tapScenario(seed: seed, count: 40, contactGap: gap))
                let adaptive = HandTestEvaluator.evaluate(sc, classifier: PinchGestureClassifier(config: cfg))
                fixedTP += fixed.truePositives; adaptTP += adaptive.truePositives; total += fixed.episodes
                adaptFP += adaptive.falsePositives
                lines.append(String(format: "gap %.2f seed %d: fixed recall %.3f | adaptive recall %.3f precision %.3f FP %d",
                                    gap, seed, fixed.recall, adaptive.recall, adaptive.precision, adaptive.falsePositives))
            }
        }
        lines.forEach { print("near-miss: " + $0) }
        print(String(format: "near-miss aggregate: fixed recall %.3f -> adaptive recall %.3f over %d taps, adaptive FP %d (incl. non-pinch blocks)",
                     Double(fixedTP) / Double(total), Double(adaptTP) / Double(total), total, adaptFP))
        XCTAssertGreaterThan(Double(adaptTP) / Double(total), 0.8)
        // A wave that opens only to 0.45 is close to a 0.38 "contact" — inherently ambiguous for
        // a widened threshold. Keep the budget explicit: ≤ 3 FP over 9 × 12 non-pinch rounds.
        XCTAssertLessThanOrEqual(adaptFP, 3)
    }

    // MARK: - Open-palm clutch

    func testOpenPalmClutchSignal() {
        let s = HandTestScenario(seed: 800)
        s.randomizePlacement()
        s.idle(frames: 10)                        // open hand
        let c = PinchGestureClassifier()
        var openFrames = 0
        for f in s.frames { _ = c.process(f.sample, at: f.t); if c.isOpenPalm { openFrames += 1 } }
        XCTAssertTrue(c.isOpenPalm)
        XCTAssertGreaterThanOrEqual(openFrames, 6)
        s.pinch(.left, closeFrames: 3, holdFrames: 5, openFrames: 3)
        for f in s.frames[10...] { _ = c.process(f.sample, at: f.t); if f.indexContact { XCTAssertFalse(c.isOpenPalm) } }
        _ = c.process(nil, at: 100)
        XCTAssertFalse(c.isOpenPalm)
    }

    // MARK: - Throughput

    func testThroughput() {
        let s = tapScenario(seed: 900, count: 100)
        let c = PinchGestureClassifier()
        let t0 = Date()
        var n = 0
        for f in s.frames { n += c.process(f.sample, at: f.t).count }
        let us = Date().timeIntervalSince(t0) / Double(s.count) * 1e6
        print(String(format: "throughput: %.1f µs per frame (%d frames, %d events)", us, s.count, n))
        XCTAssertLessThan(us, 200, "far below a 33 ms frame budget even in debug builds")
    }
}
