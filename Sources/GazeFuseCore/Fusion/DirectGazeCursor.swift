// DirectGazeCursor.swift — baseline mapper: gaze → One Euro filter → cursor.
//
// Used by the `gazeOnly` benchmark preset and as the reference the EKF is compared
// against. Head rotation (if enabled) and hand relative deltas are integrated into a
// plain additive offset that is discarded whenever the raw gaze jumps by more than
// `offsetResetJump` (a new fixation) — i.e. nothing is learnt across saccades.
import Foundation

public struct DirectGazeCursorConfig: Codable, Equatable {
    public var oneEuro: OneEuroConfig
    public var minGazeConfidence: Double
    public var head: HeadPointerConfig
    /// A raw gaze jump larger than this (pt) resets the filter and the head/hand offset.
    public var offsetResetJump: Double
    public var gazeTimeout: TimeSec
    public var historyCapacity: Int

    public init(oneEuro: OneEuroConfig = OneEuroConfig(minCutoff: FusionDefaultTuning.directMinCutoff,
                                                      beta: FusionDefaultTuning.directBeta, dCutoff: 1.0),
                minGazeConfidence: Double = 0.2,
                head: HeadPointerConfig = HeadPointerConfig(enabled: false),
                offsetResetJump: Double = 120,
                gazeTimeout: TimeSec = 0.5,
                historyCapacity: Int = 512) {
        self.oneEuro = oneEuro; self.minGazeConfidence = minGazeConfidence; self.head = head
        self.offsetResetJump = offsetResetJump; self.gazeTimeout = gazeTimeout; self.historyCapacity = historyCapacity
    }
}

public final class DirectGazeCursor: CursorFusing {
    public var config: DirectGazeCursorConfig { didSet { filter.config = config.oneEuro } }
    public var screen: ScreenRect

    private var filter: OneEuroFilter2D
    private var lastRaw: Point2?
    private var lastGazeTime: TimeSec?
    private var lastJumpTime: TimeSec?
    private var offset = Point2.zero
    private var headVelocity = Point2.zero
    private var lastHeadTime: TimeSec?
    private var lastHeadSample: HeadMotionSample?
    private var lastHandTime: TimeSec?
    private var history: FusionCursorHistory

    public init(config: DirectGazeCursorConfig = DirectGazeCursorConfig(),
                screen: ScreenRect = ScreenRect(x: 0, y: 0, width: 1512, height: 982)) {
        self.config = config; self.screen = screen
        filter = OneEuroFilter2D(config: config.oneEuro)
        history = FusionCursorHistory(capacity: config.historyCapacity)
    }

    public func reset() {
        filter.reset(); lastRaw = nil; lastGazeTime = nil; lastJumpTime = nil
        offset = .zero; headVelocity = .zero; lastHeadTime = nil; lastHeadSample = nil; lastHandTime = nil
        history.removeAll()
    }

    public func ingest(gaze: GazeEstimate) {
        // Confidence zero is the shared blink/unusable sentinel even when a
        // caller intentionally sets `minGazeConfidence` to zero. Ignore captures
        // that arrive out of order so a late packet cannot become the next
        // One-Euro reference sample.
        guard gaze.confidence > 0, gaze.confidence >= config.minGazeConfidence else { return }
        if let last = lastGazeTime, gaze.t <= last { return }
        if let lr = lastRaw, (gaze.point - lr).length > config.offsetResetJump {
            filter.reset(); offset = .zero; lastJumpTime = gaze.t
        }
        lastRaw = gaze.point
        filter.filter(gaze.point, at: gaze.t)
        lastGazeTime = max(lastGazeTime ?? gaze.t, gaze.t)
    }

    public func ingest(head: HeadMotionSample) {
        guard config.head.enabled else { return }
        let prev = lastHeadSample
        lastHeadSample = head
        guard let rates = HeadPointerGain.rates(from: head, previous: prev, config: config.head) else { return }
        if let lt = lastHeadTime, head.t > lt {
            offset = offset + headVelocity * min(head.t - lt, config.head.timeout)
        }
        headVelocity = HeadPointerGain.velocity(yawRate: rates.yaw, pitchRate: rates.pitch, config: config.head)
        lastHeadTime = max(lastHeadTime ?? head.t, head.t)
    }

    public func ingest(relativeDelta: Point2, at t: TimeSec) {
        offset = offset + relativeDelta
        lastHandTime = max(lastHandTime ?? t, t)
    }

    public func cursor(at t: TimeSec) -> CursorState {
        var pos = (filter.value ?? screen.center) + offset
        var v = Point2.zero
        if let lt = lastHeadTime, t > lt, t - lt < config.head.timeout {
            pos = pos + headVelocity * (t - lt); v = headVelocity
        }
        let display = screen.clamp(pos)
        history.append(FusionCursorHistory.Entry(t: t, state: pos, display: display))
        let mode: CursorMode
        if filter.value == nil { mode = .idle }
        else if let h = lastHandTime, t - h < 0.15 { mode = .drag }
        else if let j = lastJumpTime, t - j < 0.15 { mode = .saccade }
        else if let g = lastGazeTime, t - g > config.gazeTimeout { mode = .idle }
        else { mode = .fixation }
        return CursorState(t: t, position: display, velocity: v, mode: mode, confidence: filter.value == nil ? 0 : 0.7)
    }

    public func position(at t: TimeSec) -> Point2? { history.lookup(at: t)?.display }
}
