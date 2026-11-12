// Interfaces.swift — the shared contract between GazeFuse modules.
//
// Every module (gaze, hand, AirPods motion, fusion, simulator, app shell) talks
// to the others ONLY through the types and protocols in this file. Keep it small,
// Codable and hardware-free so everything can be recorded, replayed and tested.
//
// Coordinate conventions
// - Screen space: CoreGraphics *global display* coordinates in points, origin at the
//   top-left of the main display, +x right, +y DOWN (same space CGEvent uses).
// - Image space: Vision normalized coordinates 0...1, origin BOTTOM-left, +y up
//   (as returned by VNFaceObservation / VNHumanHandPoseObservation), in the
//   *un-mirrored* camera image.
// - Time: seconds, monotonic (CACurrentMediaTime / ProcessInfo.systemUptime).

import Foundation

public typealias TimeSec = Double

// MARK: - Basic geometry

public struct Point2: Codable, Equatable, Hashable, CustomStringConvertible {
    public var x: Double
    public var y: Double
    public init(_ x: Double, _ y: Double) { self.x = x; self.y = y }
    public init(x: Double, y: Double) { self.x = x; self.y = y }
    public static let zero = Point2(0, 0)
    public static func + (a: Point2, b: Point2) -> Point2 { Point2(a.x + b.x, a.y + b.y) }
    public static func - (a: Point2, b: Point2) -> Point2 { Point2(a.x - b.x, a.y - b.y) }
    public static func * (a: Point2, s: Double) -> Point2 { Point2(a.x * s, a.y * s) }
    public static func / (a: Point2, s: Double) -> Point2 { Point2(a.x / s, a.y / s) }
    public var length: Double { (x * x + y * y).squareRoot() }
    public func distance(to o: Point2) -> Double { (self - o).length }
    public var description: String { String(format: "(%.1f, %.1f)", x, y) }
}

public struct Vector3: Codable, Equatable {
    public var x: Double, y: Double, z: Double
    public init(_ x: Double, _ y: Double, _ z: Double) { self.x = x; self.y = y; self.z = z }
    public static let zero = Vector3(0, 0, 0)
    public var length: Double { (x * x + y * y + z * z).squareRoot() }
}

/// Rectangle in screen space (top-left origin).
public struct ScreenRect: Codable, Equatable {
    public var origin: Point2
    public var width: Double
    public var height: Double
    public init(x: Double, y: Double, width: Double, height: Double) {
        origin = Point2(x, y); self.width = width; self.height = height
    }
    public var center: Point2 { Point2(origin.x + width / 2, origin.y + height / 2) }
    public func clamp(_ p: Point2) -> Point2 {
        Point2(min(max(p.x, origin.x), origin.x + width - 1), min(max(p.y, origin.y), origin.y + height - 1))
    }
    public func contains(_ p: Point2) -> Bool {
        p.x >= origin.x && p.x < origin.x + width && p.y >= origin.y && p.y < origin.y + height
    }
}

// MARK: - Raw sensor samples (what gets recorded)

/// One camera frame's worth of eye/face features (produced by the app's Vision face
/// extractor, consumed by a `GazeMapper`). All optional fields may be nil when the
/// detector could not produce them.
public struct GazeSample: Codable, Equatable {
    public var t: TimeSec
    /// Pupil centre relative to its eye-contour centre, divided by eye width (≈ -0.5...0.5).
    /// Image space orientation (+x = image right, +y = image up).
    public var leftPupil: Point2?
    public var rightPupil: Point2?
    /// Eye openness = eye-contour height / width (blink detection), per eye.
    public var leftEyeOpenness: Double?
    public var rightEyeOpenness: Double?
    /// Head pose from Vision (radians). yaw: + = turn to image left; pitch: + = down; roll.
    public var headYaw: Double?
    public var headPitch: Double?
    public var headRoll: Double?
    /// Face bounding-box centre (normalized image space) and size (bbox width, normalized).
    public var faceCenter: Point2
    public var faceSize: Double
    /// Detector confidence 0...1.
    public var confidence: Double
    /// Optional extra landmark-derived features (e.g. inner/outer eye-corner positions),
    /// free for an extractor to fill and a mapper to consume. Keep length stable per session.
    public var extra: [Double]?
    /// Optional appearance-based (CNN) gaze direction for the face crop, radians:
    /// x = yaw, y = pitch, in the model's own convention (the calibration absorbs signs/offsets).
    public var appearanceGaze: Point2?

    public init(t: TimeSec, leftPupil: Point2?, rightPupil: Point2?, leftEyeOpenness: Double? = nil,
                rightEyeOpenness: Double? = nil, headYaw: Double?, headPitch: Double?, headRoll: Double?,
                faceCenter: Point2, faceSize: Double, confidence: Double, extra: [Double]? = nil,
                appearanceGaze: Point2? = nil) {
        self.appearanceGaze = appearanceGaze
        self.t = t; self.leftPupil = leftPupil; self.rightPupil = rightPupil
        self.leftEyeOpenness = leftEyeOpenness; self.rightEyeOpenness = rightEyeOpenness
        self.headYaw = headYaw; self.headPitch = headPitch; self.headRoll = headRoll
        self.faceCenter = faceCenter; self.faceSize = faceSize; self.confidence = confidence; self.extra = extra
    }
}

/// The 21 hand joints, named like Vision's `VNHumanHandPoseObservation.JointName`.
public enum HandJoint: String, Codable, CaseIterable, CodingKeyRepresentable {
    case wrist
    case thumbCMC, thumbMP, thumbIP, thumbTip
    case indexMCP, indexPIP, indexDIP, indexTip
    case middleMCP, middlePIP, middleDIP, middleTip
    case ringMCP, ringPIP, ringDIP, ringTip
    case littleMCP, littlePIP, littleDIP, littleTip
}

public struct HandPoint: Codable, Equatable {
    /// Normalized image space (Vision convention, origin bottom-left).
    public var x: Double
    public var y: Double
    public var confidence: Double
    public init(x: Double, y: Double, confidence: Double) { self.x = x; self.y = y; self.confidence = confidence }
    public var point: Point2 { Point2(x, y) }
}

/// One detected hand in one camera frame.
public struct HandSample: Codable, Equatable {
    public var t: TimeSec
    /// "left" / "right" as reported by Vision (may be nil).
    public var chirality: String?
    public var joints: [HandJoint: HandPoint]
    public init(t: TimeSec, chirality: String?, joints: [HandJoint: HandPoint]) {
        self.t = t; self.chirality = chirality; self.joints = joints
    }
}

/// One CMHeadphoneMotionManager update (AirPods). Units as CoreMotion: radians,
/// rad/s, g. ~25 Hz on AirPods 3 / Pro.
public struct HeadMotionSample: Codable, Equatable {
    public var t: TimeSec
    public var yaw: Double
    public var pitch: Double
    public var roll: Double
    /// Attitude quaternion (x, y, z, w).
    public var quaternion: [Double]
    public var rotationRate: Vector3
    public var userAcceleration: Vector3
    public var gravity: Vector3
    public init(t: TimeSec, yaw: Double, pitch: Double, roll: Double, quaternion: [Double] = [0, 0, 0, 1],
                rotationRate: Vector3, userAcceleration: Vector3, gravity: Vector3 = Vector3(0, 0, -1)) {
        self.t = t; self.yaw = yaw; self.pitch = pitch; self.roll = roll; self.quaternion = quaternion
        self.rotationRate = rotationRate; self.userAcceleration = userAcceleration; self.gravity = gravity
    }
}

// MARK: - Derived signals

/// A calibrated gaze point on screen.
public struct GazeEstimate: Codable, Equatable {
    public var t: TimeSec
    public var point: Point2
    /// 0...1; 0 = unusable (e.g. blink), 1 = best.
    public var confidence: Double
    public init(t: TimeSec, point: Point2, confidence: Double) { self.t = t; self.point = point; self.confidence = confidence }
}

public enum InputSource: String, Codable { case hand, jaw, dwell, keyboard, simulated }
public enum MouseButton: String, Codable, CaseIterable { case left, right, middle }
public enum InputEventKind: String, Codable {
    /// Button pressed (start of pinch). Followed later by `.up`. Enables drag.
    case down
    /// Button released.
    case up
    /// Instantaneous click (down+up), e.g. a jaw clench.
    case click
    /// Scroll by `scroll` (points) — e.g. pinch-and-move vertically.
    case scroll
}

/// A discrete intent from a classifier (pinch, clench, ...).
public struct InputEvent: Codable, Equatable {
    public var t: TimeSec
    public var source: InputSource
    public var kind: InputEventKind
    public var button: MouseButton
    public var confidence: Double
    /// For `.scroll`: (dx, dy) in points. Otherwise nil.
    public var scroll: Point2?
    public init(t: TimeSec, source: InputSource, kind: InputEventKind, button: MouseButton = .left,
                confidence: Double = 1, scroll: Point2? = nil) {
        self.t = t; self.source = source; self.kind = kind; self.button = button
        self.confidence = confidence; self.scroll = scroll
    }
}

public enum CursorMode: String, Codable { case idle, saccade, fixation, drag }

/// Output of the fusion layer.
public struct CursorState: Codable, Equatable {
    public var t: TimeSec
    public var position: Point2
    public var velocity: Point2
    public var mode: CursorMode
    /// 0...1 overall confidence of the estimate.
    public var confidence: Double
    public init(t: TimeSec, position: Point2, velocity: Point2 = .zero, mode: CursorMode = .idle, confidence: Double = 1) {
        self.t = t; self.position = position; self.velocity = velocity; self.mode = mode; self.confidence = confidence
    }
}

/// A final, arbitrated mouse action to emit via CGEvent.
public struct ClickAction: Codable, Equatable {
    public enum Kind: String, Codable { case down, up, click, scroll }
    public var t: TimeSec
    public var kind: Kind
    public var button: MouseButton
    /// Where the action should land (may differ from the live cursor: the arbiter
    /// can use the position from just *before* the gesture onset to cancel the
    /// cursor jitter that the gesture itself causes).
    public var position: Point2
    public var scroll: Point2?
    public var source: InputSource
    public init(t: TimeSec, kind: Kind, button: MouseButton, position: Point2, scroll: Point2? = nil, source: InputSource) {
        self.t = t; self.kind = kind; self.button = button; self.position = position; self.scroll = scroll; self.source = source
    }
}

// MARK: - Module protocols

/// Maps raw eye features to a screen point. Implemented by the gaze module
/// (`RidgeGazeCalibrator`). Calibration = collecting samples while the user
/// fixates known targets, then `fit`.
public protocol GazeMapper: AnyObject {
    var isCalibrated: Bool { get }
    /// Add samples recorded while the user looked at `target` (screen space).
    func addCalibrationSamples(_ samples: [GazeSample], target: Point2)
    /// Fit the model. Returns a quality report (e.g. leave-one-out error in points).
    @discardableResult func fit() throws -> CalibrationReport
    func reset()
    /// nil when the sample is unusable (blink, missing pupils, not calibrated).
    func estimate(_ sample: GazeSample) -> GazeEstimate?
}

public struct CalibrationReport: Codable, Equatable {
    public var pointCount: Int
    public var sampleCount: Int
    /// Mean / 95th-percentile cross-validated error in screen points.
    public var meanErrorPoints: Double
    public var p95ErrorPoints: Double
    public var notes: String
    public init(pointCount: Int, sampleCount: Int, meanErrorPoints: Double, p95ErrorPoints: Double, notes: String = "") {
        self.pointCount = pointCount; self.sampleCount = sampleCount
        self.meanErrorPoints = meanErrorPoints; self.p95ErrorPoints = p95ErrorPoints; self.notes = notes
    }
}

/// Hand-gesture classifier (`PinchGestureClassifier`). Feed every hand frame
/// (or `nil` when no hand is visible); get zero or more events back.
public protocol HandGestureClassifying: AnyObject {
    func process(_ sample: HandSample?, at t: TimeSec) -> [InputEvent]
    /// While a pinch is held, a smoothed hand-motion delta in *normalized image*
    /// units since the previous frame (for drag / fine adjust). nil otherwise.
    var pinchMotionDelta: Point2? { get }
    var isPinching: Bool { get }
    func reset()
}

/// AirPods jaw-clench detector (`JawClenchDetector`).
public protocol JawClenchDetecting: AnyObject {
    func process(_ sample: HeadMotionSample) -> [InputEvent]
    func reset()
}

/// Sensor fusion → cursor (`EKFCursorFusion`).
public protocol CursorFusing: AnyObject {
    var screen: ScreenRect { get set }
    func ingest(gaze: GazeEstimate)
    func ingest(head: HeadMotionSample)
    /// Relative fine-adjust input in screen points (e.g. from pinch-drag hand motion).
    func ingest(relativeDelta: Point2, at t: TimeSec)
    /// Predict/advance to time t and return the cursor to display.
    func cursor(at t: TimeSec) -> CursorState
    /// Recent cursor history lookup (for click arbitration at gesture onset).
    func position(at t: TimeSec) -> Point2?
    func reset()
}

/// Merges events from multiple click sources into final mouse actions (`ClickArbiter`).
public protocol ClickArbitrating: AnyObject {
    func ingest(_ event: InputEvent, cursor: CursorFusing) -> [ClickAction]
    func reset()
}
