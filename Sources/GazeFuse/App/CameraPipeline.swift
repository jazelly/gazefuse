import AVFoundation
import CoreMedia
import QuartzCore
import Vision
import GazeFuseCore

/// Captures the built-in camera and runs the face + hand Vision requests on every
/// frame with ONE VNImageRequestHandler (shared image preprocessing).
final class CameraPipeline: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "gazefuse.camera", qos: .userInteractive)
    let face = FaceGazeExtractor()
    let hand = HandPoseExtractor()

    /// Optional appearance-based gaze model (nil when none is installed).
    var appearance: AppearanceGazeEstimator?
    var appearanceEnabled = true
    var faceEnabled = true
    var handEnabled = true
    var onGaze: ((GazeSample?, TimeSec) -> Void)?
    var onHand: ((HandSample?, TimeSec) -> Void)?
    var onStatus: ((String) -> Void)?

    private(set) var fps: Double = 0
    private var lastFrameT: TimeSec = 0
    private(set) var deviceName = "-"

    static func requestAccess(_ done: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: done(true)
        case .notDetermined: AVCaptureDevice.requestAccess(for: .video) { ok in DispatchQueue.main.async { done(ok) } }
        default: done(false)
        }
    }

    func configure() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        if session.canSetSessionPreset(.hd1280x720) { session.sessionPreset = .hd1280x720 }
        let discovery = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera, .external],
                                                         mediaType: .video, position: .unspecified)
        guard let device = discovery.devices.first(where: { $0.deviceType == .builtInWideAngleCamera }) ?? discovery.devices.first
                ?? AVCaptureDevice.default(for: .video) else {
            throw NSError(domain: "GazeFuse", code: 1, userInfo: [NSLocalizedDescriptionKey: "No camera found"])
        }
        deviceName = device.localizedName
        let input = try AVCaptureDeviceInput(device: device)
        if session.canAddInput(input) { session.addInput(input) }
        // Prefer the highest frame rate available at 720p.
        try? device.lockForConfiguration()
        if let range = device.activeFormat.videoSupportedFrameRateRanges.max(by: { $0.maxFrameRate < $1.maxFrameRate }) {
            device.activeVideoMinFrameDuration = range.minFrameDuration
        }
        device.unlockForConfiguration()
        let out = AVCaptureVideoDataOutput()
        out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        out.alwaysDiscardsLateVideoFrames = true
        out.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(out) { session.addOutput(out) }
    }

    /// Toggled from the menu; applied on the camera queue (which owns the extractor).
    func setPupilRefinement(_ on: Bool) { queue.async { self.face.config.refinePupil = on } }
    func setAppearanceEnabled(_ on: Bool) { queue.async { self.appearanceEnabled = on } }

    func start() { queue.async { if !self.session.isRunning { self.session.startRunning() } } }
    func stop() { queue.async { if self.session.isRunning { self.session.stopRunning() } } }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        // Convert the presentation time to the host clock == CACurrentMediaTime().
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        var t = CACurrentMediaTime()
        if let clock = session.synchronizationClock {
            let host = CMSyncConvertTime(pts, from: clock, to: CMClockGetHostTimeClock())
            if host.isValid { t = CMTimeGetSeconds(host) }
        }
        if lastFrameT > 0 { let dt = t - lastFrameT; if dt > 0 { fps = fps * 0.9 + (1 / dt) * 0.1 } }
        lastFrameT = t

        var reqs: [VNRequest] = []
        if faceEnabled { reqs += face.requests }
        if handEnabled { reqs += hand.requests }
        guard !reqs.isEmpty else { return }
        if faceEnabled { face.prepare(pixelBuffer: pb) }
        let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up, options: [:])
        do { try handler.perform(reqs) } catch { face.clearPreparedFrame(); onStatus?("Vision error: \(error.localizedDescription)"); return }
        let size = CGSize(width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))
        if faceEnabled {
            var sample = face.extract(t: t, imageSize: size)
            if appearanceEnabled, sample != nil, let model = appearance, let box = face.lastFaceBox {
                sample?.appearanceGaze = model.estimate(pixelBuffer: pb, faceBox: box)
            }
            onGaze?(sample, t)
        }
        if handEnabled { onHand?(hand.extract(t: t), t) }
    }
}
