import AppKit
import QuartzCore
import GazeFuseCore

/// Wires sensors → estimators/classifiers → fusion → arbitration → cursor output.
/// All algorithm objects are touched only on `queue` (serial).
final class Engine {
    struct Channels: Codable { var gaze = true; var hand = true; var head = true; var jaw = true; var handFineAdjust = true }

    let queue = DispatchQueue(label: "gazefuse.engine", qos: .userInteractive)
    let camera = CameraPipeline()
    let headphones = HeadphoneMotionSource()
    let output = CursorOutput()

    private(set) var mapper = RidgeGazeCalibrator()
    let handClassifier = PinchGestureClassifier()
    let clench = JawClenchDetector()
    let fusion = EKFCursorFusion()
    let arbiter = ClickArbiter()

    var channels = Channels()
    /// How a thumb–index pinch maps to the mouse:
    /// - takeOff: pinch-and-hold = clutch (hand motion fine-adjusts the cursor), release = left click
    ///   ("take-off" selection, Potter et al. 1988). A quick tap is simply a click on release. Default.
    /// - pressDrag: pinch = mouse down, release = mouse up (enables drag-and-drop).
    enum PinchMode: String { case takeOff, pressDrag }
    var pinchMode: PinchMode = .takeOff
    /// Points of cursor motion per unit of normalized-image pinch motion (pinch-drag fine adjust).
    var handFineGain: Double = 900

    private(set) var tracking = false
    private(set) var recorder: SessionRecorder?
    private var timer: DispatchSourceTimer?
    private var physicalMousePauseUntil: TimeSec = 0
    private var lastCursorLogT: TimeSec = 0
    private var lastGazeSample: GazeSample?

    /// Calibration hook: while non-nil, raw gaze samples are routed here instead of the cursor.
    var calibrationSink: ((GazeSample) -> Void)?

    // Live stats for the menu / HUD.
    struct Stats { var fps = 0.0; var gazeValid = false; var handVisible = false; var headConnected = false
                   var lastGaze: Point2?; var cursor: Point2?; var clicks = 0; var headStatus = "-"; var mode = "idle" }
    private(set) var stats = Stats()
    var onStats: ((Stats) -> Void)?
    var onLog: ((String) -> Void)?

    static var appSupport: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("GazeFuse")
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }
    static var calibrationURL: URL { appSupport.appendingPathComponent("calibration.json") }
    static var recordingsDir: URL { appSupport.appendingPathComponent("recordings") }

    static func mainScreenRect() -> ScreenRect {
        let b = CGDisplayBounds(CGMainDisplayID())
        return ScreenRect(x: b.origin.x, y: b.origin.y, width: b.width, height: b.height)
    }

    init() {
        fusion.screen = Engine.mainScreenRect()
        camera.onGaze = { [weak self] s, t in self?.queue.async { self?.handleGaze(s, t) } }
        camera.onHand = { [weak self] s, t in self?.queue.async { self?.handleHand(s, t) } }
        camera.onStatus = { [weak self] m in self?.onLog?(m) }
        headphones.onSample = { [weak self] s in self?.queue.async { self?.handleHead(s) } }
        headphones.onStatusChange = { [weak self] st in
            self?.queue.async {
                self?.stats.headStatus = "\(st)"
                if case .connected = st { self?.stats.headConnected = true } else { self?.stats.headConnected = false }
            }
        }
        loadCalibration()
    }

    // MARK: lifecycle

    func startSensors() {
        do { try camera.configure() } catch { onLog?("Camera: \(error.localizedDescription)") }
        camera.start()
        headphones.start()
        startTimer()
    }

    func setTracking(_ on: Bool) {
        queue.async {
            self.tracking = on
            self.output.enabled = on
            if !on { self.output.releaseAll(); self.arbiter.reset() }
            self.fusion.reset()
            self.recorder?.write(.marker(t: CACurrentMediaTime(), text: on ? "tracking-on" : "tracking-off"))
        }
    }

    func notePhysicalMouse() { queue.async { self.physicalMousePauseUntil = CACurrentMediaTime() + 1.0 } }

    private func startTimer() {
        let tm = DispatchSource.makeTimerSource(queue: queue)
        tm.schedule(deadline: .now(), repeating: .milliseconds(8), leeway: .milliseconds(1))   // ~120 Hz
        tm.setEventHandler { [weak self] in self?.tick() }
        tm.resume(); timer = tm
    }

    private func tick() {
        let t = CACurrentMediaTime()
        stats.fps = camera.fps
        let st = fusion.cursor(at: t)
        stats.cursor = st.position; stats.mode = st.mode.rawValue
        if tracking && calibrationSink == nil && t >= physicalMousePauseUntil && st.mode != .idle {
            output.move(to: st.position)
        }
        if let r = recorder, t - lastCursorLogT > 1.0 / 30 { r.write(.cursor(st)); lastCursorLogT = t }
        if Int(t * 10) % 3 == 0 { let s = stats; DispatchQueue.main.async { self.onStats?(s) } }
    }

    // MARK: sensor handlers (engine queue)

    private func handleGaze(_ s: GazeSample?, _ t: TimeSec) {
        guard let s else { stats.gazeValid = false; return }
        recorder?.write(.gaze(s))
        lastGazeSample = s
        if let sink = calibrationSink { sink(s); return }
        guard channels.gaze, let est = mapper.estimate(s) else { stats.gazeValid = false; return }
        stats.gazeValid = est.confidence > 0; stats.lastGaze = est.point
        recorder?.write(.gazeEstimate(est))
        fusion.ingest(gaze: est)
    }

    private func handleHand(_ s: HandSample?, _ t: TimeSec) {
        stats.handVisible = s != nil
        if let s { recorder?.write(.hand(s)) } else { recorder?.write(.handLost(t)) }
        guard channels.hand, calibrationSink == nil else { return }
        let events = handClassifier.process(s, at: t)
        if channels.handFineAdjust, handClassifier.isPinching, let d = handClassifier.pinchMotionDelta {
            // Camera image is not mirrored and Vision y points up: flip both axes to screen space.
            fusion.ingest(relativeDelta: Point2(-d.x * handFineGain, -d.y * handFineGain), at: t)
        }
        for e in events {
            if pinchMode == .takeOff, e.button == .left, e.kind == .down || e.kind == .up {
                recorder?.write(.input(e))
                if e.kind == .up {
                    handle(event: InputEvent(t: e.t, source: .hand, kind: .click, button: .left, confidence: e.confidence))
                }
                continue
            }
            handle(event: e)
        }
    }

    private func handleHead(_ s: HeadMotionSample) {
        recorder?.write(.head(s))
        guard calibrationSink == nil else { return }
        if channels.head { fusion.ingest(head: s) }
        if channels.jaw { for e in clench.process(s) { handle(event: e) } }
    }

    private func handle(event e: InputEvent) {
        recorder?.write(.input(e))
        guard tracking else { return }
        for a in arbiter.ingest(e, cursor: fusion) {
            recorder?.write(.action(a))
            output.perform(a)
            if a.kind == .click || a.kind == .down { stats.clicks += 1 }
        }
    }

    // MARK: calibration persistence

    func saveCalibration() {
        queue.async {
            do { try self.mapper.encodedModel().write(to: Engine.calibrationURL) }
            catch { self.onLog?("Save calibration failed: \(error)") }
        }
    }
    func loadCalibration() {
        guard let d = try? Data(contentsOf: Engine.calibrationURL),
              let m = try? RidgeGazeCalibrator.loadModel(from: d) else { return }
        queue.async { self.mapper = m }
    }

    // MARK: recording

    func toggleRecording() -> URL? {
        var url: URL?
        queue.sync {
            if let r = recorder { r.close(); onLog?("Recording saved: \(r.url.path)"); recorder = nil; return }
            let f = DateFormatter(); f.dateFormat = "yyyyMMdd-HHmmss"
            let u = Engine.recordingsDir.appendingPathComponent("session-\(f.string(from: Date())).jsonl")
            recorder = try? SessionRecorder(url: u, header: RecordingHeader(screen: fusion.screen, notes: "GazeFuse live session"))
            url = u
        }
        return url
    }
    func mark(_ text: String) { queue.async { self.recorder?.write(.marker(t: CACurrentMediaTime(), text: text)) } }
    var isRecording: Bool { queue.sync { recorder != nil } }
}
