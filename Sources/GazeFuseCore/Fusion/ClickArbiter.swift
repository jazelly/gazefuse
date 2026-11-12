// ClickArbiter.swift — merges click intents from several sources into mouse actions.
//
// Rules, in order:
//  1. Confidence: events below the per-source threshold are ignored.
//  2. Button state: a `.down` while the button is already down, an `.up` without a
//     preceding `.down`, and a `.click` while the button is held are ignored
//     (the second source expressed the same intent, or the event is stale).
//  3. Debounce / duplicate suppression per button: a new press/click is ignored if
//     the last action on that button was less than `minInterval` ago from the same
//     source, or less than `duplicateWindow` ago from a *different* source (e.g. the
//     user both pinched and clenched for one click).
//  4. Onset position lock: pinching or clenching jitters the gaze and the head, and
//     the classifier reports the gesture ~50–150 ms after it started. The action
//     therefore lands at the fused cursor position `onsetLookback` seconds *before*
//     the event time (from the fusion's history ring buffer).
//  5. Drag: a `.down`…`.up` pair longer than `clickMaxDuration` or with the cursor
//     having travelled more than `clickMaxTravel` is a drag — the `.up` lands at the
//     current cursor position. Otherwise the `.up` lands exactly where the `.down`
//     did so the OS sees a clean click. While a button is held, `activeButtons` /
//     `isDragging` tell the event emitter to post drag events as the cursor moves.
//  6. `.scroll` passes through at the current cursor position.
import Foundation

public struct ClickArbiterConfig: Codable, Equatable {
    /// Seconds before the event time at which the action position is sampled.
    public var onsetLookback: TimeSec
    /// Minimum interval between actions on one button from the same source.
    public var minInterval: TimeSec
    /// Window in which a second source's press/click on the same button is a duplicate.
    public var duplicateWindow: TimeSec
    public var handMinConfidence: Double
    public var jawMinConfidence: Double
    public var dwellMinConfidence: Double
    public var keyboardMinConfidence: Double
    public var simulatedMinConfidence: Double
    /// A down/up pair shorter than this and within `clickMaxTravel` is a click, not a drag.
    public var clickMaxDuration: TimeSec
    public var clickMaxTravel: Double

    public init(onsetLookback: TimeSec = FusionDefaultTuning.onsetLookback,
                minInterval: TimeSec = 0.15,
                duplicateWindow: TimeSec = 0.3,
                handMinConfidence: Double = 0.5,
                jawMinConfidence: Double = 0.6,
                dwellMinConfidence: Double = 0.5,
                keyboardMinConfidence: Double = 0,
                simulatedMinConfidence: Double = 0,
                clickMaxDuration: TimeSec = 0.3,
                clickMaxTravel: Double = 12) {
        self.onsetLookback = onsetLookback; self.minInterval = minInterval; self.duplicateWindow = duplicateWindow
        self.handMinConfidence = handMinConfidence; self.jawMinConfidence = jawMinConfidence
        self.dwellMinConfidence = dwellMinConfidence; self.keyboardMinConfidence = keyboardMinConfidence
        self.simulatedMinConfidence = simulatedMinConfidence
        self.clickMaxDuration = clickMaxDuration; self.clickMaxTravel = clickMaxTravel
    }

    public func minConfidence(for source: InputSource) -> Double {
        switch source {
        case .hand: return handMinConfidence
        case .jaw: return jawMinConfidence
        case .dwell: return dwellMinConfidence
        case .keyboard: return keyboardMinConfidence
        case .simulated: return simulatedMinConfidence
        }
    }
}

public final class ClickArbiter: ClickArbitrating {
    public struct Stats: Codable, Equatable {
        public var emitted = 0, lowConfidence = 0, duplicates = 0, debounced = 0, orphanUps = 0
        public init() {}
    }

    public var config: ClickArbiterConfig
    public private(set) var stats = Stats()

    private struct Pressed { var source: InputSource; var t: TimeSec; var position: Point2 }
    private struct LastAction { var t: TimeSec; var source: InputSource }
    private var pressed: [MouseButton: Pressed] = [:]
    private var lastAction: [MouseButton: LastAction] = [:]

    public init(config: ClickArbiterConfig = ClickArbiterConfig()) { self.config = config }

    /// Buttons currently held (the emitter should post drag events while non-empty).
    public var activeButtons: Set<MouseButton> { Set(pressed.keys) }
    public var isDragging: Bool { !pressed.isEmpty }

    public func reset() { pressed.removeAll(); lastAction.removeAll(); stats = Stats() }

    public func ingest(_ event: InputEvent, cursor: CursorFusing) -> [ClickAction] {
        // A classifier can become uncertain while a pinch is being released. Once
        // the same source owns the button, accept its up event regardless of the
        // instantaneous confidence so a drag cannot remain stuck. Other sources
        // still have to pass their normal confidence threshold and are rejected
        // below unless they own the matching press.
        let ownsRelease = event.kind == .up && pressed[event.button]?.source == event.source
        if !ownsRelease && event.confidence < config.minConfidence(for: event.source) {
            stats.lowConfidence += 1
            return []
        }
        let t = event.t
        let button = event.button
        switch event.kind {
        case .scroll:
            stats.emitted += 1
            return [ClickAction(t: t, kind: .scroll, button: button, position: currentPosition(cursor, at: t),
                                scroll: event.scroll ?? .zero, source: event.source)]
        case .down:
            if pressed[button] != nil { stats.duplicates += 1; return [] }
            if isSuppressed(button: button, source: event.source, at: t) { return [] }
            let p = onsetPosition(cursor, at: t)
            pressed[button] = Pressed(source: event.source, t: t, position: p)
            lastAction[button] = LastAction(t: t, source: event.source)
            stats.emitted += 1
            return [ClickAction(t: t, kind: .down, button: button, position: p, source: event.source)]
        case .up:
            guard let down = pressed[button] else { stats.orphanUps += 1; return [] }
            guard down.source == event.source else { stats.duplicates += 1; return [] }
            pressed[button] = nil
            let cur = currentPosition(cursor, at: t)
            let isClick = (t - down.t) <= config.clickMaxDuration && cur.distance(to: down.position) <= config.clickMaxTravel
            let p = isClick ? down.position : cur
            lastAction[button] = LastAction(t: t, source: event.source)
            stats.emitted += 1
            return [ClickAction(t: t, kind: .up, button: button, position: p, source: event.source)]
        case .click:
            if pressed[button] != nil { stats.duplicates += 1; return [] }
            if isSuppressed(button: button, source: event.source, at: t) { return [] }
            let p = onsetPosition(cursor, at: t)
            lastAction[button] = LastAction(t: t, source: event.source)
            stats.emitted += 1
            return [ClickAction(t: t, kind: .click, button: button, position: p, source: event.source)]
        }
    }

    private func isSuppressed(button: MouseButton, source: InputSource, at t: TimeSec) -> Bool {
        guard let last = lastAction[button] else { return false }
        let dt = t - last.t
        if last.source == source {
            if dt < config.minInterval { stats.debounced += 1; return true }
        } else if dt < config.duplicateWindow {
            stats.duplicates += 1; return true
        }
        return false
    }

    private func onsetPosition(_ cursor: CursorFusing, at t: TimeSec) -> Point2 {
        cursor.position(at: t - config.onsetLookback) ?? currentPosition(cursor, at: t)
    }

    private func currentPosition(_ cursor: CursorFusing, at t: TimeSec) -> Point2 {
        cursor.position(at: t) ?? cursor.cursor(at: t).position
    }
}
