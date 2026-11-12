// PinchGestureClassifier.swift — turns a stream of `HandSample`s into mouse-button
// `InputEvent`s:
//
//   thumb–index pinch   -> left  `.down` on contact, `.up` on release (tap = short, hold = drag)
//   thumb–middle pinch  -> right `.down` / `.up`
//   pinch + move        -> `pinchMotionDelta` (normalized image units) for drag / fine adjust
//   optional scroll     -> pinch + mostly-vertical move emits `.scroll` instead of a drag
//
// Pipeline per frame
//   1. `HandGestureFeatures`: hand size, tip-to-tip distances in hand-size units, joint
//      confidences, finger curl states.
//   2. Single-frame outlier rejection on each raw distance (a jump faster than
//      `maxDistanceRate` is ignored once; if it persists it is accepted on the next frame —
//      Vision occasionally swaps or teleports a fingertip for one frame).
//   3. One-Euro smoothing of each distance (jitter-free when still, fast when closing).
//   4. State machine with hysteresis (`pinchOnDistance` < `pinchOffDistance`), hold frames
//      (`minOnFrames` / `minOffFrames`), confidence gating (`minOnsetConfidence`),
//      presence gating (`minPresenceFrames` valid frames after the hand (re)appears),
//      velocity gating (no onset while the whole hand moves faster than
//      `maxOnsetHandSpeed` hand-sizes/s — a real pinch is fingertips converging relative
//      to the hand, not the hand translating; fast translation also means motion blur),
//      a curled-finger veto (a fist can bring the thumb near a curled index finger),
//      dropout tolerance (`maxDropoutFrames` missing / low-confidence frames are bridged;
//      longer gaps release the button so a drag can never get stuck), and a short
//      refractory period after release.
//      A new pinch must follow an observed open pair. Entering the frame already
//      pinched, or returning closed after a long occlusion, never starts a click.
//   5. Left vs right: both pairs are evaluated; when both are under the on-threshold the
//      closer pair must win by `discriminationMargin`, otherwise nothing starts. Once a
//      button is down the other one is locked out until release.
//
// Event timestamps: `.down` carries the time of the *first* frame under the on-threshold
// (the onset), which is up to `minOnFrames - 1` frames before it is emitted (more when
// scroll mode defers the decision). The click arbiter can therefore use the cursor
// position *at* the pinch onset, before the gesture itself disturbs gaze/head tracking
// `.up` and `.scroll` carry the current time. This timestamp is inferred from
// filtered landmarks; it is not a measurement of physical fingertip contact.
//
// Scroll mode (`scrollEnabled`, default off) uses the classic touch disambiguation: the
// left `.down` is deferred while the pinch is "pending"; moving mostly vertically beyond
// `scrollActivationDistance` turns the pinch into a scroll (only `.scroll` events, no
// click), moving otherwise turns it into a drag (`.down` with the onset timestamp), and
// releasing without moving yields a tap (`.down` + `.up`). A pinch held still for longer
// than `scrollDeferMaxDuration` becomes a press/drag.
//
// Optional per-user adaptation (`adaptiveThresholds`, default off) keeps a slow running
// estimate of the open-hand distance (while idle and clearly open) and of the pinched
// distance (while held) and places the on/off thresholds at fixed fractions of that gap,
// clamped to a safe range. Useful when the camera angle makes touching fingertips look
// further apart than the fixed defaults expect.
//
// All thresholds in hand-size units are relative to `HandGestureFeatures.handSize`
// (≈ palm length), so they are independent of the hand's distance to the camera.
import Foundation

/// Stateful, single-serial-queue classifier. Feed every camera frame, including nil
/// detections. Timestamps must be monotonic seconds. Call `flush(at:)` and deliver its
/// events before stopping tracking; `reset()` only clears internal state.
public final class PinchGestureClassifier: HandGestureClassifying {

    /// Which point the drag / scroll motion is measured from.
    public enum MotionReference: String, Codable {
        /// Midpoint of the thumb tip and the pinched fingertip (what the user "holds").
        case pinchPoint
        /// Mean of the palm joints — less fingertip noise, unaffected by finger opening/closing.
        case palmCenter
    }

    public struct Config: Codable, Equatable {
        public var features = HandGestureFeatures.Config()

        // MARK: thresholds (hand-size units) and debouncing
        /// Smoothed tip-to-tip distance below which a pinch may start.
        public var pinchOnDistance: Double = 0.28
        /// Smoothed distance above which a held pinch is released (hysteresis).
        public var pinchOffDistance: Double = 0.42
        /// When both pairs are under the on-threshold the closer pair must be closer by this much.
        public var discriminationMargin: Double = 0.06
        /// Consecutive frames under the on-threshold before `.down` is emitted.
        public var minOnFrames: Int = 2
        /// Consecutive frames above the off-threshold before `.up` is emitted.
        public var minOffFrames: Int = 2
        /// min(thumbTip, fingerTip) confidence required at onset (holding only needs `features.minJointConfidence`).
        public var minOnsetConfidence: Double = 0.45
        /// Valid hand frames required after the hand (re)appears before any pinch may start.
        public var minPresenceFrames: Int = 3
        /// Missing / invalid frames bridged while a pinch is held; one more releases the button.
        public var maxDropoutFrames: Int = 2
        /// Seconds after a release during which no new pinch may start.
        public var refractoryPeriod: Double = 0.08
        /// Whole-hand speed (hand-sizes / s, EMA-smoothed) above which no pinch may start.
        public var maxOnsetHandSpeed: Double = 3.0
        /// Per-frame EMA coefficient for palm speed (0...1).
        public var handSpeedSmoothing: Double = 0.5
        /// Raw distance change (hand-sizes / s) above which a frame is treated as a one-frame outlier.
        public var maxDistanceRate: Double = 30
        /// Refuse a pinch when the candidate finger is fully curled (fist-like pose): its raw
        /// or EMA-smoothed curl ratio is at or below `features.curledCurlRatio`.
        public var vetoCurledFinger: Bool = true
        /// EMA coefficient for the curl ratios used by the veto (per frame).
        public var curlSmoothing: Double = 0.35
        /// Refuse a pinch unless the fingertip is the thumb tip's nearest joint of that finger
        /// (smoothed tip distance ≤ smoothed DIP distance and ≤ PIP distance) — a thumb
        /// wrapped over a curled finger is not a pinch.
        public var requireTipNearest: Bool = true

        // MARK: smoothing
        public var distanceMinCutoff: Double = 4     // Hz
        public var distanceBeta: Double = 1.0        // per (hand-size / s)
        /// Derivative low-pass cutoff shared by the distance and motion filters (Hz).
        public var derivativeCutoff: Double = 1
        public var motionMinCutoff: Double = 1.5     // Hz
        public var motionBeta: Double = 5.0          // per (image-unit / s)
        public var motionReference: MotionReference = .pinchPoint

        // MARK: scroll mode
        public var scrollEnabled: Bool = false
        /// Palm displacement since onset (hand-size units) that decides drag vs scroll.
        public var scrollActivationDistance: Double = 0.15
        /// |dy| must exceed this multiple of |dx| for the motion to count as a scroll.
        public var scrollAxisRatio: Double = 1.5
        /// A pending pinch held still this long becomes a press / drag.
        public var scrollDeferMaxDuration: Double = 0.4
        /// Scroll points emitted per hand-size of motion.
        public var scrollGainPoints: Double = 400

        // MARK: per-user adaptation
        public var adaptiveThresholds: Bool = false
        public var adaptiveOpenRate: Double = 0.02
        public var adaptivePinchedRate: Double = 0.1
        /// on = pinched + onFraction * (open - pinched); off likewise with offFraction
        /// (on is clamped to 0.12...0.42, off to on + 0.08 ... 0.7).
        public var adaptiveOnFraction: Double = 0.2
        public var adaptiveOffFraction: Double = 0.35

        // MARK: open-palm clutch
        /// Consecutive open-palm frames before `isOpenPalm` becomes true.
        public var openPalmMinFrames: Int = 3

        public init() {}
    }

    /// Use positive cutoff frequencies, 0 <= on < off, and nonnegative gains/rates.
    /// Change configuration between gestures, or flush before replacing it.
    public var config: Config

    // MARK: - Outputs

    /// Smoothed motion of the reference point since the previous frame (normalized image
    /// units) while a pinch is held (drag), nil otherwise and on the onset frame.
    public private(set) var pinchMotionDelta: Point2?
    public var isPinching: Bool { phase != .idle }
    /// True while a pinch has been resolved as a scroll gesture (scroll mode only).
    public var isScrolling: Bool { phase == .scrolling }
    /// The button currently held / pending / scrolling, nil when idle.
    public var pinchButton: MouseButton? { phase == .idle ? nil : button }
    /// True after `openPalmMinFrames` consecutive open-palm frames (clutch / pause signal).
    public var isOpenPalm: Bool { openPalmFrames >= max(1, config.openPalmMinFrames) }
    /// Features of the last valid frame (debugging / UI).
    public private(set) var lastFeatures: HandGestureFeatures?
    public private(set) var filteredIndexDistance: Double?
    public private(set) var filteredMiddleDistance: Double?
    /// EMA-smoothed whole-hand speed, hand-sizes / s.
    public private(set) var handSpeed: Double = 0
    /// Thresholds in effect (fixed, or adapted when `adaptiveThresholds`).
    public var activeThresholds: (on: Double, off: Double) {
        guard config.adaptiveThresholds else { return (config.pinchOnDistance, config.pinchOffDistance) }
        let gap = max(openEstimate - pinchedEstimate, 0.2)
        let on = min(max(pinchedEstimate + config.adaptiveOnFraction * gap, 0.12), 0.42)
        let off = min(max(pinchedEstimate + config.adaptiveOffFraction * gap, on + 0.08), 0.7)
        return (on, off)
    }
    /// Smoothed curl ratios used by the curled-finger veto (debugging / UI).
    public private(set) var smoothedIndexCurl: Double?
    public private(set) var smoothedMiddleCurl: Double?

    // MARK: - State

    private enum Phase { case idle, pending, held, scrolling }
    private var phase: Phase = .idle
    private var button: MouseButton = .left
    private var onsetT: TimeSec = 0
    private var onsetConfidence: Double = 1
    /// Palm centre at onset: the drag-vs-scroll decision uses whole-hand motion so the
    /// fingers finishing their closing movement cannot bias it.
    private var onsetPalm: Point2 = .zero

    private var lastT: TimeSec?
    private var presenceFrames = 0
    private var dropoutFrames = 0
    private var releaseFrames = 0
    private var refractoryUntil: TimeSec = -.infinity
    private var candidateButton: MouseButton?
    private var candidateFrames = 0
    private var candidateStartT: TimeSec = 0
    private var indexArmed = false
    private var middleArmed = false
    private var indexMissingFrames = 0
    private var middleMissingFrames = 0

    /// Outlier-rejected One-Euro smoothing of one scalar distance stream.
    private struct Smoother {
        var filter = HandOneEuroFilter()
        var lastRaw: Double?
        var rejected = 0
        var value: Double? { filter.value }
        mutating func reset() { filter.reset(); lastRaw = nil; rejected = 0 }
        /// nil when the input is missing this frame or rejected as a one-frame outlier
        /// (filter state is kept so the next sample continues smoothly).
        mutating func feed(_ x: Double?, dt: Double, maxRate: Double) -> Double? {
            guard let x = x else { return nil }
            if let lr = lastRaw, rejected == 0, abs(x - lr) / dt > maxRate {
                rejected += 1
                return nil
            }
            rejected = 0
            lastRaw = x
            return filter.filter(x, dt: dt)
        }
    }
    private var indexSmoother = Smoother()
    private var middleSmoother = Smoother()
    private var indexDipSmoother = Smoother()
    private var middleDipSmoother = Smoother()
    private var motionFilter = HandOneEuroFilter2D()
    private var lastMotionRef: Point2?
    /// Smoothed palm centre for the drag-vs-scroll decision.
    private var palmFilter = HandOneEuroFilter2D()
    private var lastPalmCenter: Point2?
    private var openEstimate = 1.0
    private var pinchedEstimate = 0.1
    private var openPalmFrames = 0

    public init(config: Config = Config()) {
        self.config = config
    }

    // MARK: - HandGestureClassifying

    public func process(_ sample: HandSample?, at t: TimeSec) -> [InputEvent] {
        var events: [InputEvent] = []
        let dt = lastT.map { min(max(t - $0, 1.0 / 240), 0.25) } ?? (1.0 / 30)
        lastT = t
        pinchMotionDelta = nil
        syncFilterParameters()

        guard let sample = sample, let f = HandGestureFeatures(sample: sample, config: config.features) else {
            lastFeatures = nil
            openPalmFrames = 0
            filteredIndexDistance = nil
            filteredMiddleDistance = nil
            dropout(at: t, events: &events)
            return events
        }
        lastFeatures = f
        openPalmFrames = f.isOpenPalm ? openPalmFrames + 1 : 0
        presenceFrames += 1
        let a = min(max(config.curlSmoothing, 0.01), 1)
        if let c = f.indexCurl { smoothedIndexCurl = smoothedIndexCurl.map { $0 + a * (c - $0) } ?? c }
        if let c = f.middleCurl { smoothedMiddleCurl = smoothedMiddleCurl.map { $0 + a * (c - $0) } ?? c }

        if let lp = lastPalmCenter {
            let v = f.palmCenter.distance(to: lp) / (f.handSize * dt)
            handSpeed += min(max(config.handSpeedSmoothing, 0), 1) * (v - handSpeed)
        } else {
            handSpeed = 0
        }
        lastPalmCenter = f.palmCenter
        let palm = palmFilter.filter(f.palmCenter, dt: dt)

        let maxRate = config.maxDistanceRate
        let dIdx = indexSmoother.feed(f.index?.distance, dt: dt, maxRate: maxRate)
        let dMid = middleSmoother.feed(f.middle?.distance, dt: dt, maxRate: maxRate)
        _ = indexDipSmoother.feed(f.index?.dipDistance, dt: dt, maxRate: maxRate)
        _ = middleDipSmoother.feed(f.middle?.dipDistance, dt: dt, maxRate: maxRate)
        filteredIndexDistance = dIdx
        filteredMiddleDistance = dMid
        indexMissingFrames = dIdx == nil ? indexMissingFrames + 1 : 0
        middleMissingFrames = dMid == nil ? middleMissingFrames + 1 : 0
        if indexMissingFrames > config.maxDropoutFrames { indexArmed = false }
        if middleMissingFrames > config.maxDropoutFrames { middleArmed = false }

        if phase == .idle {
            dropoutFrames = 0
            idleStep(features: f, palm: palm, dIdx: dIdx, dMid: dMid, dt: dt, t: t, events: &events)
        } else {
            heldStep(features: f, palm: palm, dIdx: dIdx, dMid: dMid, dt: dt, t: t, events: &events)
        }
        return events
    }

    /// Release any held button (emits `.up` if needed) and go idle — call before pausing
    /// tracking (e.g. open-palm clutch, app pause) so the mouse button never stays stuck.
    public func flush(at t: TimeSec) -> [InputEvent] {
        var events: [InputEvent] = []
        endPinch(at: t, cancelled: true, events: &events)
        resetHandTracking()
        lastFeatures = nil
        filteredIndexDistance = nil
        filteredMiddleDistance = nil
        openPalmFrames = 0
        return events
    }

    public func reset() {
        phase = .idle
        pinchMotionDelta = nil
        lastFeatures = nil
        filteredIndexDistance = nil
        filteredMiddleDistance = nil
        handSpeed = 0
        lastT = nil
        presenceFrames = 0
        dropoutFrames = 0
        releaseFrames = 0
        refractoryUntil = -.infinity
        clearCandidate()
        resetHandTracking()
        motionFilter.reset()
        lastMotionRef = nil
        openEstimate = 1.0
        pinchedEstimate = 0.1
        openPalmFrames = 0
    }

    private func resetHandTracking() {
        presenceFrames = 0
        indexArmed = false; middleArmed = false
        indexMissingFrames = 0; middleMissingFrames = 0
        indexSmoother.reset(); middleSmoother.reset(); indexDipSmoother.reset(); middleDipSmoother.reset()
        smoothedIndexCurl = nil; smoothedMiddleCurl = nil
        palmFilter.reset()
        lastPalmCenter = nil
        handSpeed = 0
    }

    // MARK: - Steps

    private func idleStep(features f: HandGestureFeatures, palm: Point2, dIdx: Double?, dMid: Double?,
                          dt: Double, t: TimeSec, events: inout [InputEvent]) {
        let (on, off) = activeThresholds
        if let d = dIdx, let p = f.index, min(d, p.distance) > off,
           p.confidence >= config.minOnsetConfidence { indexArmed = true }
        if let d = dMid, let p = f.middle, min(d, p.distance) > off,
           p.confidence >= config.minOnsetConfidence { middleArmed = true }
        if config.adaptiveThresholds, let d = dIdx, d > off + 0.15 {
            openEstimate += config.adaptiveOpenRate * (d - openEstimate)
        }
        guard presenceFrames >= config.minPresenceFrames, t >= refractoryUntil,
              handSpeed <= config.maxOnsetHandSpeed else {
            clearCandidate()
            return
        }

        func qualifies(_ d: Double?, _ pair: HandGestureFeatures.Pinch?, dip: Double?,
                       rawCurl: Double?, smoothedCurl: Double?) -> Bool {
            guard let d = d, let p = pair, d < on, p.confidence >= config.minOnsetConfidence else { return false }
            if config.requireTipNearest {
                if let dip = dip ?? p.dipDistance, dip < d { return false }
                if let pip = p.pipDistance, pip < d { return false }
            }
            if config.vetoCurledFinger {
                let curl = min(rawCurl ?? 1, smoothedCurl ?? 1)
                if curl <= config.features.curledCurlRatio { return false }
            }
            return true
        }
        // Both tips can already be close even though their filters cross the on
        // threshold on different frames. Resolve raw contact ambiguity first so
        // filter lag or a curl veto cannot accidentally select the other button.
        func winsContactMargin(_ pair: HandGestureFeatures.Pinch?, over other: HandGestureFeatures.Pinch?) -> Bool {
            guard let p = pair, let q = other, q.confidence >= config.minOnsetConfidence,
                  q.distance < on else { return true }
            return p.distance + config.discriminationMargin <= q.distance
        }
        let idxOK = indexArmed && winsContactMargin(f.index, over: f.middle)
            && qualifies(dIdx, f.index, dip: indexDipSmoother.value, rawCurl: f.indexCurl, smoothedCurl: smoothedIndexCurl)
        let midOK = middleArmed && winsContactMargin(f.middle, over: f.index)
            && qualifies(dMid, f.middle, dip: middleDipSmoother.value, rawCurl: f.middleCurl, smoothedCurl: smoothedMiddleCurl)
        var candidate: MouseButton?
        if idxOK && midOK, let a = dIdx, let b = dMid {
            if a + config.discriminationMargin <= b { candidate = .left }
            else if b + config.discriminationMargin <= a { candidate = .right }
        } else if idxOK {
            candidate = .left
        } else if midOK {
            candidate = .right
        }

        if let c = candidate, c == candidateButton {
            candidateFrames += 1
        } else {
            candidateButton = candidate
            candidateFrames = candidate == nil ? 0 : 1
            candidateStartT = t
        }
        guard let b = candidate, candidateFrames >= max(1, config.minOnFrames) else { return }

        // Start the pinch.
        indexArmed = false; middleArmed = false
        let pair = (b == .left ? f.index : f.middle)!
        let d = (b == .left ? dIdx : dMid)!
        let depth = min(max((off - d) / max(off - on, 1e-6), 0.3), 1)
        button = b
        onsetT = candidateStartT
        onsetConfidence = min(max(pair.confidence * depth, 0.05), 1)
        motionFilter.reset()
        lastMotionRef = motionFilter.filter(reference(f, pair: pair), dt: dt)
        onsetPalm = palm
        releaseFrames = 0
        dropoutFrames = 0
        clearCandidate()
        if config.scrollEnabled && b == .left {
            phase = .pending
        } else {
            phase = .held
            events.append(InputEvent(t: onsetT, source: .hand, kind: .down, button: b, confidence: onsetConfidence))
        }
    }

    private func heldStep(features f: HandGestureFeatures, palm: Point2, dIdx: Double?, dMid: Double?,
                          dt: Double, t: TimeSec, events: inout [InputEvent]) {
        let pair = button == .left ? f.index : f.middle
        guard let d = (button == .left ? dIdx : dMid), let p = pair else {
            dropout(at: t, events: &events)
            return
        }
        dropoutFrames = 0
        let off = activeThresholds.off
        if d > off { releaseFrames += 1 } else { releaseFrames = 0 }
        if releaseFrames >= max(1, config.minOffFrames) {
            endPinch(at: t, cancelled: false, events: &events)
            return
        }
        if config.adaptiveThresholds, d <= off {
            pinchedEstimate += config.adaptivePinchedRate * (d - pinchedEstimate)
        }

        let ref = motionFilter.filter(reference(f, pair: p), dt: dt)
        let delta = lastMotionRef.map { ref - $0 }
        lastMotionRef = ref

        switch phase {
        case .held:
            pinchMotionDelta = delta
        case .pending:
            let disp = (palm - onsetPalm) / f.handSize
            if disp.length >= config.scrollActivationDistance {
                if abs(disp.y) >= config.scrollAxisRatio * abs(disp.x) {
                    phase = .scrolling
                } else {
                    phase = .held
                    events.append(InputEvent(t: onsetT, source: .hand, kind: .down, button: button, confidence: onsetConfidence))
                    pinchMotionDelta = delta
                }
            } else if t - onsetT >= config.scrollDeferMaxDuration {
                phase = .held
                events.append(InputEvent(t: onsetT, source: .hand, kind: .down, button: button, confidence: onsetConfidence))
            }
        case .scrolling:
            if let delta = delta, delta.length > 1e-6 {
                let s = delta * (config.scrollGainPoints / f.handSize)
                events.append(InputEvent(t: t, source: .hand, kind: .scroll, button: button,
                                         confidence: onsetConfidence, scroll: s))
            }
        case .idle:
            break
        }
    }

    /// A frame without the hand or its active pair. Brief gaps retain presence;
    /// long gaps release and disarm before restarting the filters.
    private func dropout(at t: TimeSec, events: inout [InputEvent]) {
        clearCandidate()
        releaseFrames = 0
        dropoutFrames += 1
        guard dropoutFrames > config.maxDropoutFrames else { return }
        endPinch(at: t, cancelled: true, events: &events)
        // A long loss of either the hand or active tips requires opening again.
        resetHandTracking()
    }

    /// Leave any non-idle phase. `cancelled` = not a deliberate release (hand lost, flush):
    /// a held button is still released (never leave the mouse button stuck), but a pending
    /// pinch does not turn into a tap.
    private func endPinch(at t: TimeSec, cancelled: Bool, events: inout [InputEvent]) {
        switch phase {
        case .idle:
            return
        case .held:
            events.append(InputEvent(t: t, source: .hand, kind: .up, button: button, confidence: onsetConfidence))
        case .pending:
            if !cancelled {
                events.append(InputEvent(t: onsetT, source: .hand, kind: .down, button: button, confidence: onsetConfidence))
                events.append(InputEvent(t: t, source: .hand, kind: .up, button: button, confidence: onsetConfidence))
            }
        case .scrolling:
            break
        }
        phase = .idle
        refractoryUntil = t + config.refractoryPeriod
        releaseFrames = 0
        dropoutFrames = 0
        pinchMotionDelta = nil
        motionFilter.reset()
        lastMotionRef = nil
        clearCandidate()
    }

    // MARK: - Helpers

    private func clearCandidate() {
        candidateButton = nil
        candidateFrames = 0
    }

    private func reference(_ f: HandGestureFeatures, pair: HandGestureFeatures.Pinch) -> Point2 {
        config.motionReference == .palmCenter ? f.palmCenter : pair.point
    }

    private func syncFilterParameters() {
        indexSmoother.filter.minCutoff = config.distanceMinCutoff; indexSmoother.filter.beta = config.distanceBeta
        middleSmoother.filter.minCutoff = config.distanceMinCutoff; middleSmoother.filter.beta = config.distanceBeta
        indexDipSmoother.filter.minCutoff = config.distanceMinCutoff; indexDipSmoother.filter.beta = config.distanceBeta
        middleDipSmoother.filter.minCutoff = config.distanceMinCutoff; middleDipSmoother.filter.beta = config.distanceBeta
        motionFilter.minCutoff = config.motionMinCutoff; motionFilter.beta = config.motionBeta
        palmFilter.minCutoff = config.motionMinCutoff; palmFilter.beta = config.motionBeta
        indexSmoother.filter.dCutoff = config.derivativeCutoff; middleSmoother.filter.dCutoff = config.derivativeCutoff
        indexDipSmoother.filter.dCutoff = config.derivativeCutoff; middleDipSmoother.filter.dCutoff = config.derivativeCutoff
        motionFilter.dCutoff = config.derivativeCutoff; palmFilter.dCutoff = config.derivativeCutoff
    }
}
