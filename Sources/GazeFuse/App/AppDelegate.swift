import AppKit
import GazeFuseCore

/// Menu-bar UI. Global shortcuts (need Accessibility):
///   ⌃⌥G  start / stop cursor control      ⌃⌥C  calibrate
///   ⌃⌥R  start / stop recording           ⌃⌥H  toggle HUD
///   Esc during calibration cancels it. Touching the real mouse/trackpad pauses control for 1 s.
final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    private var statusItem: NSStatusItem!
    private let engine = Engine()
    private let hud = OverlayHUD()
    private var calibration: CalibrationWindowController?
    private var monitors: [Any] = []
    private var statusLine = NSMenuItem(title: "Starting…", action: nil, keyEquivalent: "")
    private var trackingItem: NSMenuItem!
    private var recordItem: NSMenuItem!
    private var lastStats = Engine.Stats()
    private let args = CommandLine.arguments

    func applicationDidFinishLaunching(_ notification: Notification) {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.title = "◉ GF"
        statusItem.button?.toolTip = "GazeFuse — gaze + pinch + AirPods cursor control"
        buildMenu()

        engine.onLog = { msg in NSLog("[GazeFuse] %@", msg) }
        engine.onStats = { [weak self] s in self?.updateStats(s) }

        if args.contains("--no-sensors") {
            NSLog("[GazeFuse] --no-sensors: UI only")
        } else {
            CameraPipeline.requestAccess { [weak self] ok in
                guard let self else { return }
                if !ok { self.alert("Camera access denied", "Enable GazeFuse in System Settings → Privacy & Security → Camera.") }
                self.engine.startSensors()
            }
            _ = CursorOutput.isTrusted(prompt: true)
        }
        installMonitors()
        if let secs = args.firstIndex(of: "--quit-after").flatMap({ Double(args[safe: $0 + 1] ?? "") }) {
            DispatchQueue.main.asyncAfter(deadline: .now() + secs) { NSApp.terminate(nil) }
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        engine.setTracking(false)
        if engine.isRecording { _ = engine.toggleRecording() }
    }

    // MARK: menu

    private func buildMenu() {
        let m = NSMenu(); m.delegate = self
        statusLine.isEnabled = false
        m.addItem(statusLine)
        m.addItem(.separator())
        trackingItem = item("Start Cursor Control   ⌃⌥G", #selector(toggleTracking))
        m.addItem(trackingItem)
        m.addItem(item("Calibrate Gaze…   ⌃⌥C", #selector(calibrate)))
        let ch = NSMenu()
        for (title, key) in [("Eye gaze (camera)", "gaze"), ("Hand pinch click (camera)", "hand"),
                             ("Pinch-drag fine adjust", "handFineAdjust"), ("AirPods head fine-correct", "head"),
                             ("AirPods jaw-clench click", "jaw")] {
            let it = item(title, #selector(toggleChannel(_:))); it.representedObject = key; it.state = .on; ch.addItem(it)
        }
        ch.addItem(.separator())
        let take = item("Pinch: hold = adjust, release = click", #selector(setPinchMode(_:))); take.representedObject = "takeOff"; take.state = .on
        let drag = item("Pinch: press / drag (mouse down-up)", #selector(setPinchMode(_:))); drag.representedObject = "pressDrag"
        ch.addItem(take); ch.addItem(drag)
        let chItem = NSMenuItem(title: "Input Channels", action: nil, keyEquivalent: ""); chItem.submenu = ch
        m.addItem(chItem)
        m.addItem(item("Show / Hide HUD   ⌃⌥H", #selector(toggleHUD)))
        m.addItem(.separator())
        recordItem = item("Start Recording   ⌃⌥R", #selector(toggleRecording))
        m.addItem(recordItem)
        m.addItem(item("Open Neuralink Webgrid", #selector(openWebgrid)))
        m.addItem(item("Mark: Webgrid run start", #selector(markStart)))
        m.addItem(item("Mark: Webgrid run end", #selector(markEnd)))
        m.addItem(item("Open Data Folder", #selector(openData)))
        m.addItem(.separator())
        m.addItem(item("Open Accessibility Settings", #selector(openAX)))
        m.addItem(item("Open Camera Settings", #selector(openCam)))
        m.addItem(.separator())
        m.addItem(item("Quit GazeFuse", #selector(quit), key: "q"))
        statusItem.menu = m
    }

    private func item(_ t: String, _ s: Selector, key: String = "") -> NSMenuItem {
        let i = NSMenuItem(title: t, action: s, keyEquivalent: key); i.target = self; return i
    }

    func menuWillOpen(_ menu: NSMenu) { refreshStatusLine() }

    private func refreshStatusLine() {
        let s = lastStats
        let ax = CursorOutput.isTrusted(prompt: false) ? "AX ✓" : "AX ✗"
        statusLine.title = String(format: "cam %.0f fps · gaze %@ · hand %@ · AirPods %@ · %@ · cal %@",
                                  s.fps, s.gazeValid ? "✓" : "–", s.handVisible ? "✓" : "–",
                                  s.headConnected ? "✓" : "–", ax, engine.mapper.isCalibrated ? "✓" : "✗")
    }

    private func updateStats(_ s: Engine.Stats) {
        lastStats = s
        hud.update(gaze: s.lastGaze, cursor: s.cursor,
                   text: String(format: " %@ | cam %.0ffps | gaze %@ | hand %@ | AirPods %@ | clicks %d ",
                                s.mode, s.fps, s.gazeValid ? "on" : "off", s.handVisible ? "on" : "off",
                                s.headConnected ? "on" : "off", s.clicks))
    }

    // MARK: actions

    @objc private func toggleTracking() {
        let on = !engine.tracking
        if on && !CursorOutput.isTrusted(prompt: true) {
            alert("Accessibility permission needed",
                  "GazeFuse moves the cursor with CGEvent. Enable it in System Settings → Privacy & Security → Accessibility, then try again.")
            return
        }
        if on && !engine.mapper.isCalibrated { alert("Not calibrated yet", "Run Calibrate Gaze… first (⌃⌥C)."); return }
        engine.setTracking(on)
        trackingItem.title = on ? "Stop Cursor Control   ⌃⌥G" : "Start Cursor Control   ⌃⌥G"
        statusItem.button?.title = on ? "◉ GF•" : "◉ GF"
    }

    @objc private func calibrate() {
        guard calibration == nil, let screen = NSScreen.main else { return }
        if engine.tracking { toggleTracking() }
        engine.queue.sync { engine.mapper.reset() }
        let rect = Engine.mainScreenRect()
        let c = CalibrationWindowController(screen: screen, mapper: engine.mapper, plan: CalibrationPlan(screen: rect, layout: .thirteen))
        engine.queue.sync { engine.calibrationSink = { [weak c] s in c?.feed(s) } }
        c.onFinish = { [weak self] result in
            guard let self else { return }
            self.engine.queue.sync { self.engine.calibrationSink = nil }
            self.calibration = nil
            switch result {
            case .success(let r):
                self.engine.saveCalibration()
                self.engine.mark(String(format: "calibration mean=%.1fpt p95=%.1fpt n=%d", r.meanErrorPoints, r.p95ErrorPoints, r.pointCount))
                self.alert("Calibration done",
                           String(format: "Mean error %.0f pt, 95%% %.0f pt over %d targets.\n%@\n\nPress ⌃⌥G to start cursor control.",
                                  r.meanErrorPoints, r.p95ErrorPoints, r.pointCount, r.notes))
            case .failure(let e):
                self.alert("Calibration failed", e.localizedDescription)
            }
        }
        calibration = c
        c.start()
    }

    @objc private func toggleChannel(_ sender: NSMenuItem) {
        guard let key = sender.representedObject as? String else { return }
        sender.state = sender.state == .on ? .off : .on
        let on = sender.state == .on
        engine.queue.async {
            switch key {
            case "gaze": self.engine.channels.gaze = on
            case "hand": self.engine.channels.hand = on
            case "handFineAdjust": self.engine.channels.handFineAdjust = on
            case "head": self.engine.channels.head = on
            case "jaw": self.engine.channels.jaw = on
            default: break
            }
        }
    }

    @objc private func setPinchMode(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String, let mode = Engine.PinchMode(rawValue: raw) else { return }
        sender.menu?.items.filter { ($0.representedObject as? String).flatMap(Engine.PinchMode.init) != nil }.forEach { $0.state = .off }
        sender.state = .on
        engine.queue.async { self.engine.pinchMode = mode; self.engine.output.releaseAll() }
    }

    @objc private func toggleHUD() { hud.toggle() }

    @objc private func toggleRecording() {
        let url = engine.toggleRecording()
        recordItem.title = url != nil ? "Stop Recording   ⌃⌥R" : "Start Recording   ⌃⌥R"
        statusItem.button?.title = url != nil ? "◉ GF REC" : (engine.tracking ? "◉ GF•" : "◉ GF")
    }

    @objc private func openWebgrid() { NSWorkspace.shared.open(URL(string: "https://neuralink.com/webgrid/")!) }
    @objc private func markStart() { engine.mark("webgrid-start") }
    @objc private func markEnd() { engine.mark("webgrid-end") }
    @objc private func openData() { NSWorkspace.shared.open(Engine.appSupport) }
    @objc private func openAX() {
        NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!)
    }
    @objc private func openCam() {
        NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera")!)
    }
    @objc private func quit() { NSApp.terminate(nil) }

    // MARK: global shortcuts + physical-mouse detection

    private func installMonitors() {
        let keys: (NSEvent) -> Void = { [weak self] e in
            guard let self, e.modifierFlags.contains([.control, .option]) else { return }
            switch e.charactersIgnoringModifiers?.lowercased() {
            case "g": self.toggleTracking()
            case "c": self.calibrate()
            case "r": self.toggleRecording()
            case "h": self.toggleHUD()
            default: break
            }
        }
        if let m = NSEvent.addGlobalMonitorForEvents(matching: .keyDown, handler: keys) { monitors.append(m) }
        if let m = NSEvent.addLocalMonitorForEvents(matching: .keyDown, handler: { e in keys(e); return e }) { monitors.append(m) }
        // Physical mouse/trackpad activity pauses gaze control briefly so the two never fight.
        if let m = NSEvent.addGlobalMonitorForEvents(matching: [.mouseMoved, .leftMouseDragged, .scrollWheel], handler: { [weak self] e in
            if e.cgEvent?.getIntegerValueField(.eventSourceUserData) != CursorOutput.syntheticTag { self?.engine.notePhysicalMouse() }
        }) { monitors.append(m) }
    }

    private func alert(_ title: String, _ text: String) {
        let a = NSAlert(); a.messageText = title; a.informativeText = text
        NSApp.activate(ignoringOtherApps: true)
        a.runModal()
    }
}

extension Array {
    subscript(safe i: Int) -> Element? { indices.contains(i) ? self[i] : nil }
}
