// GazeHeadLeakCompensator.swift — online estimate of how much head rotation leaks
// into the webcam gaze estimate, so the head channel is not fought by the gaze channel.
//
// Problem: a personal ridge calibration compensates head pose only partly. When the
// user keeps the eyes on a target and turns the head to nudge the cursor (exactly
// what the head channel asks for), the *estimated* gaze moves too:
//     z(t) = p_target + b + L ⊙ θ(t) + n
// with θ the (yaw, pitch) head angle and L (pt/rad per axis) the leak gain. Without
// compensation the fusion sees that as the eyes moving, and part of every head
// correction is undone (or amplified) by the next gaze samples.
//
// Estimator: within one fixation (between saccade resets) p_target + b is constant,
// so differences relative to the fixation's first sample obey Δz = L·Δθ + n.
// L is estimated per axis by exponentially-forgotten least squares across fixations,
// with a ridge prior towards 0 and a clamp, and the gaze sample is pre-compensated:
//     z' = z − L̂ ⊙ (θ − θ_ref)
// where θ_ref is a slowly moving reference so a constant offset is left to the bias.
// θ is obtained by integrating the same sign-corrected head rates the pointer uses,
// so no assumption about the AirPods attitude axes is needed; slow gyro drift only
// changes θ_ref.
//
// Assumption: during a fixation the user looks at the target, not at the cursor.
// If the user instead tracks the cursor with the eyes, L̂ grows toward the pointer
// gain; the clamp (`maxLeakFraction` · pointer gain) bounds that failure mode.
import Foundation

public struct GazeHeadLeakConfig: Codable, Equatable {
    public var enabled: Bool = true
    /// Forgetting factor per accepted sample pair (closer to 1 = longer memory).
    public var forgetting: Double = 0.998
    /// Ridge prior strength in rad² (pulls L̂ toward 0 until enough head motion was seen).
    public var priorRad2: Double = 0.002
    /// |L̂| is clamped to this fraction of the head pointer gain.
    public var maxLeakFraction: Double = 0.9
    /// Only fixation samples at least this long after a reset are used for learning.
    public var settleTime: TimeSec = 0.15
    /// Ignore |Δz| larger than this (pt) when learning (residual saccades, artefacts).
    public var maxInnovation: Double = 250
    /// Time constant (s) of the slowly moving head-angle reference.
    public var referenceTau: TimeSec = 20
    public init() {}
}

public final class GazeHeadLeakCompensator {
    public var config: GazeHeadLeakConfig
    /// Current leak estimate (pt/rad, x from yaw, y from pitch).
    public var leak: Point2 { Point2(sxy.x / (sxx.x + config.priorRad2), sxy.y / (sxx.y + config.priorRad2)) }
    public private(set) var theta = Point2.zero
    private var thetaRef = Point2.zero
    private var lastHeadT: TimeSec?
    private var history: [(t: TimeSec, theta: Point2)] = []
    // Fixation anchor.
    private var anchor: (z: Point2, theta: Point2)?
    private var fixationStart: TimeSec = 0
    // LS accumulators.
    private var sxx = Point2.zero
    private var sxy = Point2.zero
    private var maxLeak: Double

    public init(config: GazeHeadLeakConfig = GazeHeadLeakConfig(), pointerGain: Double) {
        self.config = config
        maxLeak = config.maxLeakFraction * pointerGain
    }

    public func reset() {
        theta = .zero; thetaRef = .zero; lastHeadT = nil; history.removeAll()
        anchor = nil; sxx = .zero; sxy = .zero
    }

    /// Integrate sign-corrected head rates (rad/s, + = cursor right / down).
    public func ingestHead(yawRate: Double, pitchRate: Double, at t: TimeSec) {
        if let lt = lastHeadT, t > lt {
            let dt = min(t - lt, 0.2)
            theta = theta + Point2(yawRate, pitchRate) * dt
            let a = 1 - exp(-dt / max(config.referenceTau, 1e-3))
            thetaRef = thetaRef + (theta - thetaRef) * a
        }
        if lastHeadT == nil || t > lastHeadT! { lastHeadT = t }
        history.append((t, theta))
        if history.count > 128 { history.removeFirst(64) }
    }

    /// Head angle at (or just before) time t.
    func theta(at t: TimeSec) -> Point2 {
        guard let last = history.last else { return theta }
        if t >= last.t { return last.theta }
        var i = history.count - 1
        while i > 0 && history[i].t > t { i -= 1 }
        return history[i].theta
    }

    /// Call when the fusion confirmed a saccade (new fixation).
    public func newFixation(at t: TimeSec) { anchor = nil; fixationStart = t }

    /// Learns from a raw gaze sample that the fusion accepted as an in-fixation update.
    public func learn(_ z: Point2, at t: TimeSec) {
        guard config.enabled, t - fixationStart >= config.settleTime else { return }
        let th = theta(at: t)
        guard let a = anchor else { anchor = (z, th); return }
        let dth = th - a.theta, dz = z - a.z
        guard abs(dz.x) < config.maxInnovation && abs(dz.y) < config.maxInnovation else { return }
        let f = config.forgetting
        sxx = Point2(sxx.x * f + dth.x * dth.x, sxx.y * f + dth.y * dth.y)
        sxy = Point2(sxy.x * f + dth.x * dz.x, sxy.y * f + dth.y * dz.y)
    }

    /// The gaze sample with the estimated head leak removed.
    public func compensate(_ z: Point2, at t: TimeSec) -> Point2 {
        guard config.enabled else { return z }
        let L = clampedLeak
        let d = theta(at: t) - thetaRef
        return Point2(z.x - L.x * d.x, z.y - L.y * d.y)
    }

    public var clampedLeak: Point2 {
        let L = leak
        return Point2(min(max(L.x, -maxLeak), maxLeak), min(max(L.y, -maxLeak), maxLeak))
    }
}
