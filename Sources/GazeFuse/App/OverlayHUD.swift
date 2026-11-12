import AppKit
import GazeFuseCore

/// Transparent, click-through, always-on-top overlay that draws the raw gaze
/// estimate (blue ring) and the fused cursor (green dot). Debug aid; toggle from the menu.
final class OverlayHUD {
    private var window: NSWindow?
    private let view = HUDView()

    var isVisible: Bool { window?.isVisible ?? false }

    func toggle() { isVisible ? hide() : show() }

    func show() {
        guard let screen = NSScreen.main else { return }
        if window == nil {
            let w = NSWindow(contentRect: screen.frame, styleMask: .borderless, backing: .buffered, defer: false, screen: screen)
            w.isOpaque = false; w.backgroundColor = .clear; w.hasShadow = false
            w.ignoresMouseEvents = true
            w.level = .screenSaver
            w.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary, .ignoresCycle]
            view.frame = NSRect(origin: .zero, size: screen.frame.size)
            w.contentView = view
            window = w
        }
        window?.orderFrontRegardless()
    }
    func hide() { window?.orderOut(nil) }

    /// Points are CG global (top-left origin); converted to view coordinates here.
    func update(gaze: Point2?, cursor: Point2?, text: String) {
        guard isVisible, let screen = window?.screen ?? NSScreen.main else { return }
        let h = screen.frame.height
        func conv(_ p: Point2?) -> NSPoint? { p.map { NSPoint(x: $0.x - screen.frame.origin.x, y: h - $0.y) } }
        view.gaze = conv(gaze); view.cursor = conv(cursor); view.text = text
        view.needsDisplay = true
    }
}

private final class HUDView: NSView {
    var gaze: NSPoint?
    var cursor: NSPoint?
    var text = ""
    override func draw(_ dirtyRect: NSRect) {
        if let g = gaze {
            NSColor.systemBlue.withAlphaComponent(0.8).setStroke()
            let p = NSBezierPath(ovalIn: NSRect(x: g.x - 18, y: g.y - 18, width: 36, height: 36)); p.lineWidth = 3; p.stroke()
        }
        if let c = cursor {
            NSColor.systemGreen.withAlphaComponent(0.9).setFill()
            NSBezierPath(ovalIn: NSRect(x: c.x - 6, y: c.y - 6, width: 12, height: 12)).fill()
        }
        let attrs: [NSAttributedString.Key: Any] = [.font: NSFont.monospacedSystemFont(ofSize: 13, weight: .medium),
                                                    .foregroundColor: NSColor.white,
                                                    .backgroundColor: NSColor.black.withAlphaComponent(0.55)]
        (text as NSString).draw(at: NSPoint(x: 16, y: bounds.height - 60), withAttributes: attrs)
    }
}
