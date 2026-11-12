// HeadPoseTracker.swift — a small, hardware-free utility for AirPods attitude data.
//
// The cursor fusion module consumes rotationRate directly. This type is intended
// for UI affordances that need a stable, relative yaw/pitch value (for example a
// setup indicator or an optional accessibility gesture), and deliberately does
// not turn head pose into cursor velocity.

import Foundation

/// Smoothed relative head pose at one input timestamp.
public struct HeadPoseEstimate: Codable, Equatable {
    public let t: TimeSec
    /// Relative yaw in radians after smoothing and deadband.
    public let yaw: Double
    /// Relative pitch in radians after smoothing and deadband.
    public let pitch: Double
    /// Change in the reported relative yaw since the previous accepted sample.
    /// This is zero across an initialization or an unobserved time gap.
    public let deltaYaw: Double
    /// Change in the reported relative pitch since the previous accepted sample.
    /// This is zero across an initialization or an unobserved time gap.
    public let deltaPitch: Double
    /// True when the input was accepted and finite.
    public let isValid: Bool
    /// True when smoothing was intentionally reset because the elapsed timestamp
    /// gap exceeded `Config.maximumGap`. Non-increasing timestamps are rejected
    /// without changing tracker state and have `isValid == false`.
    public let crossedGap: Bool

    public init(t: TimeSec, yaw: Double, pitch: Double, deltaYaw: Double = 0,
                deltaPitch: Double = 0, isValid: Bool = true, crossedGap: Bool = false) {
        self.t = t
        self.yaw = yaw
        self.pitch = pitch
        self.deltaYaw = deltaYaw
        self.deltaPitch = deltaPitch
        self.isValid = isValid
        self.crossedGap = crossedGap
    }
}

/// Relative yaw/pitch tracker for `HeadMotionSample` streams.
///
/// The first finite sample establishes the zero. Subsequent angles are wrapped
/// through ±π before an exponential smoother is applied, so a turn from +π to
/// -π is a small change. `recenter()` establishes a new zero on demand. Optional
/// drift removal only moves that zero while the headset is still *and* the
/// reported pose is already close to zero; this prevents a deliberate sustained
/// head offset from slowly being erased.
public final class HeadPoseTracker {
    /// Runtime tunables. All time values are seconds and angles are radians.
    public struct Config: Codable, Equatable {
        /// Exponential smoothing time constant. Smaller values follow motion
        /// faster but expose more sensor noise.
        public var smoothingTimeConstant: Double
        /// Output values inside this radius are reported as zero. Values outside
        /// it are reduced by the radius, avoiding a sudden jump at the boundary.
        public var deadbandRadians: Double
        /// Maximum elapsed time for which a sample is treated as continuous.
        /// A longer dropout cannot reveal the path taken between endpoints.
        public var maximumGap: Double
        /// Enables the guarded, slow baseline drift update described above.
        public var driftRemovalEnabled: Bool
        /// Time constant for baseline drift while the guarded update is active.
        public var driftTimeConstant: Double
        /// Pose must be within this angle of zero before baseline drift is allowed.
        public var driftCaptureWindowRadians: Double
        /// Minimum continuous still time before drift correction is allowed.
        public var minimumStillDuration: Double
        /// Maximum gyro magnitude considered still (rad/s).
        public var stillRotationRateThreshold: Double
        /// Maximum user acceleration considered still (g).
        public var stillAccelerationThreshold: Double

        public init(smoothingTimeConstant: Double = 0.12,
                    deadbandRadians: Double = 0.012,
                    maximumGap: Double = 0.75,
                    driftRemovalEnabled: Bool = true,
                    driftTimeConstant: Double = 45,
                    driftCaptureWindowRadians: Double = 0.10,
                    minimumStillDuration: Double = 1.0,
                    stillRotationRateThreshold: Double = 0.12,
                    stillAccelerationThreshold: Double = 0.06) {
            self.smoothingTimeConstant = smoothingTimeConstant
            self.deadbandRadians = deadbandRadians
            self.maximumGap = maximumGap
            self.driftRemovalEnabled = driftRemovalEnabled
            self.driftTimeConstant = driftTimeConstant
            self.driftCaptureWindowRadians = driftCaptureWindowRadians
            self.minimumStillDuration = minimumStillDuration
            self.stillRotationRateThreshold = stillRotationRateThreshold
            self.stillAccelerationThreshold = stillAccelerationThreshold
        }

        public static let `default` = Config()
    }

    public let config: Config

    private var hasReference = false
    private var baselineYaw = 0.0
    private var baselinePitch = 0.0
    private var filteredYaw = 0.0
    private var filteredPitch = 0.0
    private var previousOutputYaw = 0.0
    private var previousOutputPitch = 0.0
    private var lastTimestamp: TimeSec?
    private var lastRawYaw = 0.0
    private var lastRawPitch = 0.0
    private var stillDuration = 0.0

    public init(config: Config = .default) {
        // Codable decoding assigns stored properties directly, so validate once
        // more at the tracker boundary in case a persisted config was malformed.
        precondition(config.smoothingTimeConstant.isFinite && config.smoothingTimeConstant > 0,
                     "smoothingTimeConstant must be finite and positive")
        precondition(config.deadbandRadians.isFinite && config.deadbandRadians >= 0,
                     "deadbandRadians must be finite and non-negative")
        precondition(config.maximumGap.isFinite && config.maximumGap > 0,
                     "maximumGap must be finite and positive")
        precondition(config.driftTimeConstant.isFinite && config.driftTimeConstant > 0,
                     "driftTimeConstant must be finite and positive")
        precondition(config.driftCaptureWindowRadians.isFinite && config.driftCaptureWindowRadians >= 0,
                     "driftCaptureWindowRadians must be finite and non-negative")
        precondition(config.minimumStillDuration.isFinite && config.minimumStillDuration >= 0,
                     "minimumStillDuration must be finite and non-negative")
        precondition(config.stillRotationRateThreshold.isFinite && config.stillRotationRateThreshold >= 0,
                     "stillRotationRateThreshold must be finite and non-negative")
        precondition(config.stillAccelerationThreshold.isFinite && config.stillAccelerationThreshold >= 0,
                     "stillAccelerationThreshold must be finite and non-negative")
        self.config = config
    }

    /// Processes one sample. A non-finite attitude or timestamp is ignored and
    /// returns an invalid, zero-delta estimate; the previous valid state remains
    /// intact. The caller can therefore treat dropouts as missing observations.
    @discardableResult
    public func process(_ sample: HeadMotionSample) -> HeadPoseEstimate {
        guard sample.t.isFinite, sample.yaw.isFinite, sample.pitch.isFinite else {
            return HeadPoseEstimate(t: sample.t, yaw: previousOutputYaw, pitch: previousOutputPitch,
                                    isValid: false, crossedGap: true)
        }

        guard hasReference else {
            hasReference = true
            lastRawYaw = wrap(sample.yaw)
            lastRawPitch = wrap(sample.pitch)
            baselineYaw = lastRawYaw
            baselinePitch = lastRawPitch
            filteredYaw = 0
            filteredPitch = 0
            previousOutputYaw = 0
            previousOutputPitch = 0
            lastTimestamp = sample.t
            stillDuration = 0
            return HeadPoseEstimate(t: sample.t, yaw: 0, pitch: 0)
        }

        guard let previousTimestamp = lastTimestamp, sample.t > previousTimestamp else {
            // A replayed or out-of-order sample is not a dropout: do not let its
            // attitude or timestamp poison the next valid delta.
            return HeadPoseEstimate(t: sample.t, yaw: previousOutputYaw, pitch: previousOutputPitch,
                                    isValid: false)
        }

        let dt = sample.t - previousTimestamp
        lastTimestamp = sample.t
        lastRawYaw = wrap(sample.yaw)
        lastRawPitch = wrap(sample.pitch)
        if !dt.isFinite || dt > config.maximumGap {
            let relativeYaw = shortestDelta(from: baselineYaw, to: lastRawYaw)
            let relativePitch = shortestDelta(from: baselinePitch, to: lastRawPitch)
            filteredYaw = relativeYaw
            filteredPitch = relativePitch
            let outputYaw = applyDeadband(relativeYaw)
            let outputPitch = applyDeadband(relativePitch)
            previousOutputYaw = outputYaw
            previousOutputPitch = outputPitch
            stillDuration = 0
            return HeadPoseEstimate(t: sample.t, yaw: outputYaw, pitch: outputPitch,
                                    crossedGap: true)
        }

        let still = sample.rotationRate.length <= config.stillRotationRateThreshold &&
            sample.userAcceleration.length <= config.stillAccelerationThreshold
        stillDuration = still ? stillDuration + dt : 0

        // Only adapt the reference around the existing zero. This is the guard
        // that keeps a held, intentional head offset from disappearing.
        if config.driftRemovalEnabled && stillDuration >= config.minimumStillDuration {
            let beforeYaw = shortestDelta(from: baselineYaw, to: lastRawYaw)
            let beforePitch = shortestDelta(from: baselinePitch, to: lastRawPitch)
            if abs(beforeYaw) <= config.driftCaptureWindowRadians &&
                abs(beforePitch) <= config.driftCaptureWindowRadians {
                let beta = 1 - exp(-dt / config.driftTimeConstant)
                baselineYaw = wrap(baselineYaw + shortestDelta(from: baselineYaw, to: lastRawYaw) * beta)
                baselinePitch = wrap(baselinePitch + shortestDelta(from: baselinePitch, to: lastRawPitch) * beta)
            }
        }

        let targetYaw = shortestDelta(from: baselineYaw, to: lastRawYaw)
        let targetPitch = shortestDelta(from: baselinePitch, to: lastRawPitch)
        let alpha = 1 - exp(-dt / config.smoothingTimeConstant)
        filteredYaw = filteredYaw + shortestDelta(from: filteredYaw, to: targetYaw) * alpha
        filteredPitch = filteredPitch + shortestDelta(from: filteredPitch, to: targetPitch) * alpha
        filteredYaw = wrap(filteredYaw)
        filteredPitch = wrap(filteredPitch)

        let outputYaw = applyDeadband(filteredYaw)
        let outputPitch = applyDeadband(filteredPitch)
        let deltaYaw = shortestDelta(from: previousOutputYaw, to: outputYaw)
        let deltaPitch = shortestDelta(from: previousOutputPitch, to: outputPitch)
        previousOutputYaw = outputYaw
        previousOutputPitch = outputPitch
        return HeadPoseEstimate(t: sample.t, yaw: outputYaw, pitch: outputPitch,
                                deltaYaw: deltaYaw, deltaPitch: deltaPitch)
    }

    /// Sets the current finite attitude as the new zero. If no sample has been
    /// accepted yet, the next finite sample establishes the reference instead.
    public func recenter() {
        guard hasReference else { return }
        baselineYaw = lastRawYaw
        baselinePitch = lastRawPitch
        filteredYaw = 0
        filteredPitch = 0
        previousOutputYaw = 0
        previousOutputPitch = 0
        stillDuration = 0
    }

    /// Clears the reference and all smoothing history.
    public func reset() {
        hasReference = false
        baselineYaw = 0
        baselinePitch = 0
        filteredYaw = 0
        filteredPitch = 0
        previousOutputYaw = 0
        previousOutputPitch = 0
        lastTimestamp = nil
        lastRawYaw = 0
        lastRawPitch = 0
        stillDuration = 0
    }

    private func applyDeadband(_ value: Double) -> Double {
        let magnitude = abs(value)
        guard magnitude > config.deadbandRadians else { return 0 }
        return value - (value.sign == .minus ? -config.deadbandRadians : config.deadbandRadians)
    }

    @inline(__always)
    private func shortestDelta(from: Double, to: Double) -> Double {
        wrap(to - from)
    }

    @inline(__always)
    private func wrap(_ angle: Double) -> Double {
        var value = angle.truncatingRemainder(dividingBy: 2 * .pi)
        if value > .pi { value -= 2 * .pi }
        if value <= -.pi { value += 2 * .pi }
        return value
    }
}
