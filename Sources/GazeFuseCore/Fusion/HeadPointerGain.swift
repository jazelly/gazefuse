// HeadPointerGain.swift — maps AirPods head rotation rate to cursor velocity.
//
// Head motion is a *relative*, precise channel (like a mouse): rotation rate in
// rad/s → cursor velocity in pt/s through a deadband and a pointer-acceleration
// curve similar in spirit to macOS pointer acceleration:
//
//     m      = |(yawRate, pitchRate)|              (isotropic: diagonal = axis-aligned)
//     mEff   = max(m − deadband, 0)
//     speed  = pointsPerRadian · mEff · (mEff / accelReferenceRate)^accelExponent
//     v      = min(speed, maxSpeed) · (yawRate, pitchRate) / m
//
// With accelExponent = 0 the mapping is linear; with 0.5 (default) slow head motion
// (< accelReferenceRate) is attenuated for fine control and fast motion is boosted.
//
// Axis conventions (VERIFY ON HARDWARE — the motion module owns the AirPods):
// `HeadMotionSample.rotationRate` is in the headphone device frame. The defaults
// assume the CoreMotion headphone frame where rotation about +y is yaw (head turn,
// positive = turning left) and rotation about +x is pitch (positive = looking up),
// hence both default signs are −1 so that turning right / nodding down move the
// cursor +x (right) / +y (down). If the assumption is wrong, flip `yawSign` /
// `pitchSign` or set `source = .attitudeDerivative`, which differentiates the
// sensor-fused Euler angles `yaw`/`pitch` (unambiguous semantics, slightly noisier).
import Foundation

public enum HeadRateSource: String, Codable {
    /// Use the gyroscope `rotationRate` components selected by `yawAxis` / `pitchAxis`.
    case rotationRate
    /// Differentiate the attitude Euler angles `yaw` / `pitch` between consecutive samples.
    case attitudeDerivative
}

public struct HeadPointerConfig: Codable, Equatable {
    public var enabled: Bool
    public var source: HeadRateSource
    /// rotationRate component (0 = x, 1 = y, 2 = z) that drives horizontal cursor motion, and its sign.
    public var yawAxis: Int
    public var yawSign: Double
    /// rotationRate component that drives vertical cursor motion, and its sign.
    public var pitchAxis: Int
    public var pitchSign: Double
    /// Rotation rates below this (rad/s) produce no motion (gyro noise, postural tremor).
    public var deadbandRadPerSec: Double
    /// Base gain, screen points per radian of head rotation (before acceleration).
    public var pointsPerRadian: Double
    /// Acceleration curve exponent (0 = linear).
    public var accelExponent: Double
    /// Rate (rad/s) at which the acceleration factor equals 1.
    public var accelReferenceRate: Double
    /// Cap on the commanded cursor speed (pt/s).
    public var maxSpeedPointsPerSec: Double
    /// Measurement noise (pt/s) of the derived cursor velocity, used as R by the EKF.
    public var velocitySigma: Double
    /// Without a head sample for this long the head velocity is treated as zero.
    public var timeout: TimeSec

    public init(enabled: Bool = true, source: HeadRateSource = .rotationRate,
                yawAxis: Int = 1, yawSign: Double = -1, pitchAxis: Int = 0, pitchSign: Double = -1,
                deadbandRadPerSec: Double = FusionDefaultTuning.headDeadbandRadPerSec,
                pointsPerRadian: Double = FusionDefaultTuning.headPointsPerRadian,
                accelExponent: Double = FusionDefaultTuning.headAccelExponent,
                accelReferenceRate: Double = 0.15,
                maxSpeedPointsPerSec: Double = 3000,
                velocitySigma: Double = FusionDefaultTuning.headVelocitySigma,
                timeout: TimeSec = 0.25) {
        self.enabled = enabled; self.source = source
        self.yawAxis = yawAxis; self.yawSign = yawSign; self.pitchAxis = pitchAxis; self.pitchSign = pitchSign
        self.deadbandRadPerSec = deadbandRadPerSec; self.pointsPerRadian = pointsPerRadian
        self.accelExponent = accelExponent; self.accelReferenceRate = accelReferenceRate
        self.maxSpeedPointsPerSec = maxSpeedPointsPerSec; self.velocitySigma = velocitySigma; self.timeout = timeout
    }
}

public enum HeadPointerGain {
    /// Cursor velocity (pt/s, screen axes: +x right, +y down) for the given signed
    /// yaw/pitch rates (rad/s, already sign-corrected so + = right / down).
    public static func velocity(yawRate: Double, pitchRate: Double, config c: HeadPointerConfig) -> Point2 {
        let m = (yawRate * yawRate + pitchRate * pitchRate).squareRoot()
        let mEff = m - c.deadbandRadPerSec
        guard mEff > 0, m > 0 else { return .zero }
        let speed = min(speedFor(effectiveRate: mEff, config: c), c.maxSpeedPointsPerSec)
        return Point2(speed * yawRate / m, speed * pitchRate / m)
    }

    /// The acceleration curve: effective (post-deadband) rate → speed in pt/s.
    public static func speedFor(effectiveRate mEff: Double, config c: HeadPointerConfig) -> Double {
        guard mEff > 0 else { return 0 }
        let accel = c.accelExponent == 0 ? 1 : pow(mEff / max(c.accelReferenceRate, 1e-6), c.accelExponent)
        return c.pointsPerRadian * mEff * accel
    }

    /// Inverse of the curve: the (pre-deadband) rotation rate needed for `speed` pt/s.
    /// Used by the simulated human as feed-forward knowledge of the mapping.
    public static func rate(forSpeed speed: Double, config c: HeadPointerConfig) -> Double {
        guard speed > 0 else { return 0 }
        let s = min(speed, c.maxSpeedPointsPerSec)
        let mEff = pow(s * pow(max(c.accelReferenceRate, 1e-6), c.accelExponent) / c.pointsPerRadian, 1.0 / (1.0 + c.accelExponent))
        return mEff + c.deadbandRadPerSec
    }

    /// Signed (yaw, pitch) rates in rad/s from a sample, per `config.source`.
    /// Returns nil when `.attitudeDerivative` has no previous sample to difference.
    public static func rates(from s: HeadMotionSample, previous: HeadMotionSample?, config c: HeadPointerConfig) -> (yaw: Double, pitch: Double)? {
        switch c.source {
        case .rotationRate:
            return (c.yawSign * component(s.rotationRate, c.yawAxis), c.pitchSign * component(s.rotationRate, c.pitchAxis))
        case .attitudeDerivative:
            guard let p = previous, s.t > p.t else { return nil }
            let dt = s.t - p.t
            return (c.yawSign * wrapAngle(s.yaw - p.yaw) / dt, c.pitchSign * wrapAngle(s.pitch - p.pitch) / dt)
        }
    }

    static func component(_ v: Vector3, _ axis: Int) -> Double {
        switch axis { case 0: return v.x; case 1: return v.y; default: return v.z }
    }

    static func wrapAngle(_ a: Double) -> Double {
        var r = a.truncatingRemainder(dividingBy: 2 * .pi)
        if r > .pi { r -= 2 * .pi } else if r < -.pi { r += 2 * .pi }
        return r
    }
}
