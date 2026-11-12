import AppKit
import CoreGraphics
import GazeFuseCore

/// Moves the real macOS cursor and synthesises clicks with CGEvent.
/// Requires the Accessibility permission (System Settings → Privacy & Security → Accessibility).
final class CursorOutput {
    private let source = CGEventSource(stateID: .hidSystemState)
    private var buttonsDown: Set<MouseButton> = []
    private var lastClickT: TimeSec = 0
    private var clickCount: Int64 = 1
    private(set) var lastPosition: CGPoint?
    var enabled = false
    /// Tag stamped on every synthetic event so the app can tell them apart from the physical mouse.
    static let syntheticTag: Int64 = 0x47465A45 // 'GFZE'

    static func isTrusted(prompt: Bool) -> Bool {
        let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        return AXIsProcessTrustedWithOptions([key: prompt] as CFDictionary)
    }

    /// Current hardware cursor location in CG global coordinates (top-left origin).
    static func currentLocation() -> CGPoint { CGEvent(source: nil)?.location ?? .zero }

    func move(to p: Point2) {
        guard enabled else { return }
        let pt = CGPoint(x: p.x, y: p.y)
        let type: CGEventType
        if buttonsDown.contains(.left) { type = .leftMouseDragged }
        else if buttonsDown.contains(.right) { type = .rightMouseDragged }
        else { type = .mouseMoved }
        let button: CGMouseButton = buttonsDown.contains(.right) ? .right : .left
        if let e = CGEvent(mouseEventSource: source, mouseType: type, mouseCursorPosition: pt, mouseButton: button) {
            e.setIntegerValueField(.eventSourceUserData, value: CursorOutput.syntheticTag)
            e.post(tap: .cghidEventTap)
        }
        lastPosition = pt
    }

    func perform(_ a: ClickAction) {
        guard enabled else { return }
        let pt = CGPoint(x: a.position.x, y: a.position.y)
        switch a.kind {
        case .down: post(down: true, button: a.button, at: pt, t: a.t)
        case .up: post(down: false, button: a.button, at: pt, t: a.t)
        case .click:
            post(down: true, button: a.button, at: pt, t: a.t)
            post(down: false, button: a.button, at: pt, t: a.t)
        case .scroll:
            let d = a.scroll ?? .zero
            CGEvent(scrollWheelEvent2Source: source, units: .pixel, wheelCount: 2,
                    wheel1: Int32(-d.y), wheel2: Int32(-d.x), wheel3: 0).map {
                $0.setIntegerValueField(.eventSourceUserData, value: CursorOutput.syntheticTag); $0.post(tap: .cghidEventTap) }
        }
    }

    /// Releases any held buttons (e.g. when tracking stops mid-drag).
    func releaseAll() {
        let pt = lastPosition ?? CursorOutput.currentLocation()
        for b in buttonsDown { post(down: false, button: b, at: pt, t: 0) }
        buttonsDown.removeAll()
    }

    private func post(down: Bool, button: MouseButton, at pt: CGPoint, t: TimeSec) {
        let (type, cgb): (CGEventType, CGMouseButton) = {
            switch button {
            case .left: return (down ? .leftMouseDown : .leftMouseUp, .left)
            case .right: return (down ? .rightMouseDown : .rightMouseUp, .right)
            case .middle: return (down ? .otherMouseDown : .otherMouseUp, .center)
            }
        }()
        if down {
            // Double-click support: a second press within 0.35 s increments clickState.
            clickCount = (t - lastClickT < 0.35) ? min(clickCount + 1, 3) : 1
            lastClickT = t
            buttonsDown.insert(button)
        } else {
            buttonsDown.remove(button)
        }
        guard let e = CGEvent(mouseEventSource: source, mouseType: type, mouseCursorPosition: pt, mouseButton: cgb) else { return }
        e.setIntegerValueField(.mouseEventClickState, value: clickCount)
        e.setIntegerValueField(.eventSourceUserData, value: CursorOutput.syntheticTag)
        e.post(tap: .cghidEventTap)
        lastPosition = pt
    }
}
