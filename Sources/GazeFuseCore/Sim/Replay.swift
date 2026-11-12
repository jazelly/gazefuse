// Replay.swift — runs a recording (Recording.swift JSONL) through the cursor
// pipeline (fusion + arbiter) and reports cursor statistics and actions.
//
// Events are processed in file order (= arrival order, which preserves the
// out-of-order gaze timestamps the app saw). Between events a synthetic 60 Hz
// display clock calls `cursor(at:)`, as the app does, so the history ring buffer the
// arbiter relies on is populated the same way. Raw `gaze` / `hand` samples are only
// used when the corresponding hook is provided (the integrator wires the real gaze
// mapper / hand classifier there); otherwise they are counted and skipped.
import Foundation

public struct ReplayHooks {
    /// Raw face features → calibrated gaze (e.g. a fitted `GazeMapper`).
    public var gazeMapper: ((GazeSample) -> GazeEstimate?)?
    /// Raw hand frame (nil = hand lost) → input events (e.g. `HandGestureClassifying`).
    public var handClassifier: ((HandSample?, TimeSec) -> [InputEvent])?
    /// Pinch-drag relative delta in screen points for the current hand frame, if any.
    public var handRelativeDelta: ((HandSample?, TimeSec) -> Point2?)?
    /// AirPods sample → jaw-clench events (e.g. `JawClenchDetecting`).
    public var jawDetector: ((HeadMotionSample) -> [InputEvent])?
    public var onCursor: ((CursorState) -> Void)?
    public var onAction: ((ClickAction) -> Void)?
    public init() {}
}

public struct ReplayResult: Codable, Equatable {
    public var events = 0
    public var gazeEstimates = 0, headSamples = 0, inputEvents = 0
    public var rawGazeSamples = 0, rawGazeMapped = 0, rawGazeSkipped = 0
    public var rawHandSamples = 0, rawHandClassified = 0, rawHandSkipped = 0
    public var markers: [String] = []
    public var calibrationTargets = 0
    public var duration: Double = 0
    public var cursorSamples = 0
    public var meanCursorSpeed: Double = 0
    public var maxCursorSpeed: Double = 0
    /// RMS frame-to-frame displacement while in `.fixation` mode.
    public var fixationJitter: Double = 0
    public var modeSeconds: [String: Double] = [:]
    public var actions: [ClickAction] = []
    /// If the recording contains `cursor` events: mean |replayed − recorded| at those times.
    public var recordedCursorDeviation: Double?
    public var recordedActionCount: Int?
    public var fusionStats: EKFCursorFusion.Stats?
    public init() {}
}

public final class ReplayEngine {
    public let fusion: CursorFusing
    public let arbiter: ClickArbitrating
    public var hooks: ReplayHooks
    public let displayHz: Double

    public init(fusion: CursorFusing, arbiter: ClickArbitrating, hooks: ReplayHooks = ReplayHooks(), displayHz: Double = 60) {
        self.fusion = fusion; self.arbiter = arbiter; self.hooks = hooks; self.displayHz = displayHz
    }

    public func run(_ events: [RecordedEvent]) -> ReplayResult {
        var r = ReplayResult()
        var clock: TimeSec? = nil
        let period = 1 / displayHz
        var speedSum = 0.0, lastPos: Point2?, lastT: TimeSec?
        var jitterSum = 0.0, jitterN = 0
        var devSum = 0.0, devN = 0
        var firstT: TimeSec?, lastEventT: TimeSec?
        var lastMode = CursorMode.idle, lastModeT: TimeSec?

        func display(at t: TimeSec) {
            let cs = fusion.cursor(at: t)
            hooks.onCursor?(cs)
            r.cursorSamples += 1
            if let lp = lastPos, let lt = lastT, t > lt {
                let d = (cs.position - lp).length
                let v = d / (t - lt)
                speedSum += v; r.maxCursorSpeed = max(r.maxCursorSpeed, v)
                if cs.mode == .fixation && lastMode == .fixation { jitterSum += d * d; jitterN += 1 }
            }
            if let lmt = lastModeT { r.modeSeconds[lastMode.rawValue, default: 0] += t - lmt }
            lastMode = cs.mode; lastModeT = t
            lastPos = cs.position; lastT = t
        }

        func advance(to t: TimeSec) {
            if clock == nil { clock = t; display(at: t); return }
            while let c = clock, c + period <= t { clock = c + period; display(at: c + period) }
        }

        for e in events {
            r.events += 1
            if case .header = e { continue }
            let t = e.t
            if firstT == nil { firstT = t }
            lastEventT = max(lastEventT ?? t, t)
            advance(to: t)
            switch e {
            case .header: break
            case .gaze(let s):
                r.rawGazeSamples += 1
                if let m = hooks.gazeMapper, let g = m(s) { r.rawGazeMapped += 1; fusion.ingest(gaze: g) } else { r.rawGazeSkipped += 1 }
            case .gazeEstimate(let g):
                r.gazeEstimates += 1; fusion.ingest(gaze: g)
            case .hand(let h):
                r.rawHandSamples += 1
                var used = false
                if let c = hooks.handClassifier { used = true; for ev in c(h, t) { r.inputEvents += 1; emit(ev) } }
                if let d = hooks.handRelativeDelta, let delta = d(h, t) { used = true; fusion.ingest(relativeDelta: delta, at: t) }
                if used { r.rawHandClassified += 1 } else { r.rawHandSkipped += 1 }
            case .handLost(let t):
                if let c = hooks.handClassifier { for ev in c(nil, t) { r.inputEvents += 1; emit(ev) } }
                _ = hooks.handRelativeDelta?(nil, t)
            case .head(let h):
                r.headSamples += 1; fusion.ingest(head: h)
                if let j = hooks.jawDetector { for ev in j(h) { r.inputEvents += 1; emit(ev) } }
            case .input(let ev):
                r.inputEvents += 1; emit(ev)
            case .cursor(let rec):
                let cs = fusion.cursor(at: rec.t)
                devSum += (cs.position - rec.position).length; devN += 1
            case .action:
                r.recordedActionCount = (r.recordedActionCount ?? 0) + 1
            case .calibrationTarget:
                r.calibrationTargets += 1
            case .marker(_, let text):
                r.markers.append(text)
            }
        }
        func emit(_ ev: InputEvent) {
            for a in arbiter.ingest(ev, cursor: fusion) { r.actions.append(a); hooks.onAction?(a) }
        }
        if let f = firstT, let l = lastEventT { r.duration = l - f }
        if r.cursorSamples > 1 { r.meanCursorSpeed = speedSum / Double(r.cursorSamples - 1) }
        if jitterN > 0 { r.fixationJitter = (jitterSum / Double(jitterN)).squareRoot() }
        if devN > 0 { r.recordedCursorDeviation = devSum / Double(devN) }
        r.fusionStats = (fusion as? EKFCursorFusion)?.stats
        return r
    }
}
