// SimHuman.swift — the simulated user: eyes, head, hand and click decisions.
//
// The human sees the *displayed* cursor (perception delay), looks at the target
// (saccade latency, main-sequence saccade with undershoot + corrective saccade,
// fixational drift, microsaccades, blinks), corrects the residual cursor offset with
// the head (intermittent visuomotor controller with reaction time, neuromuscular
// lag and signal-dependent motor noise) or with a hand clutch, and clicks only when
// the perceived cursor has dwelt inside the target — or when patience runs out.
//
// Coordinate convention for the head: `headAngle` / `headRate` are (yaw, pitch) in
// radians with +yaw = turned right (cursor +x) and +pitch = nodded down (cursor +y);
// the sensor layer converts to the CoreMotion frame per `HeadPointerConfig`. The
// controller uses that same configured gain in reverse, modelling a trained user
// adapted to the pointer mapping; this is an explicit simulator assumption rather
// than a claim about untrained human behaviour.
import Foundation

/// Physical actions the human performs during a tick.
public enum SimHumanAction: Equatable {
    /// A pinch / clench gesture physically started now.
    case clickOnset
    /// The clutched hand moved by Δ points (one camera frame's worth).
    case handDelta(Point2)
}

/// Time-stamped samples with lookup "value at or before t" (for delayed perception).
struct SimDelayLine {
    private var ts: [TimeSec] = []
    private var vs: [Point2] = []
    mutating func append(_ t: TimeSec, _ v: Point2) {
        ts.append(t); vs.append(v)
        if ts.count > 512 { ts.removeFirst(256); vs.removeFirst(256) }
    }
    func value(at t: TimeSec) -> Point2? {
        guard let first = ts.first, t >= first else { return nil }
        var lo = 0, hi = ts.count - 1
        while lo < hi {
            let mid = (lo + hi + 1) / 2
            if ts[mid] <= t { lo = mid } else { hi = mid - 1 }
        }
        return vs[lo]
    }
    mutating func removeAll() { ts.removeAll(); vs.removeAll() }
}

/// One exact Ornstein–Uhlenbeck step: x ← x·e^(−dt/τ) + σ·√(1 − e^(−2dt/τ))·N(0,1).
@inline(__always) func simOUStep(_ x: Point2, sigma: Double, tau: Double, dt: TimeSec, rng: SimRNG) -> Point2 {
    guard tau > 0 else { return rng.gaussian2D(sd: sigma) }
    let a = exp(-dt / tau)
    return x * a + rng.gaussian2D(sd: sigma * (1 - a * a).squareRoot())
}

public final class SimHuman {
    let cfg: SimConfig
    let rng: SimRNG
    let headGain: HeadPointerConfig
    let ppd: Double

    // Target
    public private(set) var targetCenter = Point2.zero
    public private(set) var targetRect = ScreenRect(x: 0, y: 0, width: 1, height: 1)
    private var targetOnset: TimeSec = 0
    private var hasTarget = false

    // Eye
    public private(set) var eyePosition: Point2
    private var fixationPoint: Point2
    private struct Saccade { var from: Point2; var to: Point2; var t0: TimeSec; var duration: TimeSec; var primary: Bool }
    private var saccade: Saccade?
    private var plannedSaccade: (to: Point2, at: TimeSec, primary: Bool)?
    private var landedAt: TimeSec?
    private var drift = Point2.zero
    private var nextMicrosaccade: TimeSec = 0
    private var blinkStart: TimeSec = -1
    private var blinkEnd: TimeSec = -1
    private var nextBlink: TimeSec = 0
    private var lastCompensation: TimeSec = -1

    // Perception
    private var perceived = SimDelayLine()
    private var insideSince: TimeSec?

    // Head
    public private(set) var headAngle = Point2.zero
    public private(set) var headRate = Point2.zero
    private var headCmd = Point2.zero
    private var headSmooth = Point2.zero
    private var motorNoise = Point2.zero
    private var tremor = Point2.zero
    private var nextHeadDecision: TimeSec = 0
    private struct Submovement { var t0: TimeSec; var duration: TimeSec; var amplitude: Point2; var plannedCursor: Point2; var startCursor: Point2 }
    private var headSub: Submovement?
    private var headGainScale = 1.0
    private var handSub: Submovement?
    private var clenchUntil: TimeSec = -1
    private var clenchDir = Point2.zero

    // Hand
    public private(set) var isHandClutched = false
    private var handEngageAt: TimeSec?
    private var handReleaseAt: TimeSec?
    private var nextHandDecision: TimeSec = 0
    private var handVelocity = Point2.zero
    private var lastHandFrame: TimeSec = 0
    private var bestError = Double.infinity
    private var bestErrorTime: TimeSec = 0

    // Click
    private var clickOnsetAt: TimeSec?
    private var awaitingUntil: TimeSec = -1
    private var lastClickDecision: TimeSec = -1
    private var holdUntil: TimeSec = -1
    public private(set) var clickAttempts = 0
    public private(set) var patienceClicks = 0

    public init(config: SimConfig, rng: SimRNG, headGain: HeadPointerConfig, startGaze: Point2) {
        cfg = config; self.rng = rng; self.headGain = headGain; ppd = config.pointsPerDegree
        eyePosition = startGaze; fixationPoint = startGaze
        nextBlink = rng.exponential(rate: config.eye.blinkRatePerMin / 60)
        nextMicrosaccade = rng.exponential(rate: config.eye.microsaccadeRatePerSec)
    }

    // MARK: - Inputs from the world

    public func newTarget(center: Point2, rect: ScreenRect, at t: TimeSec) {
        targetCenter = center; targetRect = rect; targetOnset = t; hasTarget = true
        landedAt = nil; insideSince = nil
        lastClickDecision = t; awaitingUntil = -1
        bestError = .infinity; bestErrorTime = t
        if isHandClutched && handReleaseAt == nil { handReleaseAt = t + cfg.hand.releaseLatency }
        handEngageAt = nil
        let latency = rng.lognormal(mean: cfg.eye.saccadeLatencyMean, cv: cfg.eye.saccadeLatencyCV)
        plannedSaccade = (to: landingPoint(for: center), at: t + latency, primary: true)
    }

    private func landingPoint(for target: Point2) -> Point2 {
        let a = target - fixationPoint
        return fixationPoint + a * (1 - cfg.eye.saccadeUndershootFraction) + rng.gaussian2D(sd: cfg.eye.saccadeLandingSigmaDeg * ppd)
    }

    /// The displayed cursor at a display frame.
    public func perceive(cursor: Point2, at t: TimeSec) { perceived.append(t, cursor) }

    public func noteOutcome(_ outcome: WebgridTask.ClickOutcome, at t: TimeSec) {
        awaitingUntil = -1
        if outcome != .correct { insideSince = nil; lastClickDecision = t }
    }

    // MARK: - Observable truth for the sensors

    public func isBlinking(at t: TimeSec) -> Bool { t >= blinkStart && t < blinkEnd }
    public func isBlinkEdge(at t: TimeSec, window w: Double) -> Bool {
        (t >= blinkStart - w && t < blinkStart) || (t >= blinkEnd && t < blinkEnd + w)
    }
    public var isSaccading: Bool { saccade != nil }
    public var hasLanded: Bool { landedAt != nil }
    /// Angular rate the IMU experiences: voluntary + tremor + clench transient (rad/s).
    public func headTrueRate(at t: TimeSec) -> Point2 { headRate + tremor + clenchImpulse(at: t) }

    private func clenchImpulse(at t: TimeSec) -> Point2 {
        let d = cfg.head.clenchImpulseDuration
        guard t < clenchUntil, d > 0 else { return .zero }
        let phase = (t - (clenchUntil - d)) / d
        return clenchDir * (cfg.head.clenchImpulseRate * sin(.pi * phase))
    }

    // MARK: - Dynamics

    public func step(t: TimeSec, dt: TimeSec) -> [SimHumanAction] {
        var actions: [SimHumanAction] = []
        stepEye(t: t, dt: dt)
        if cfg.headEnabled { stepHead(t: t, dt: dt) }
        if cfg.handAdjustEnabled { stepHand(t: t, dt: dt, actions: &actions) }
        stepClick(t: t, actions: &actions)
        return actions
    }

    private func stepEye(t: TimeSec, dt: TimeSec) {
        let e = cfg.eye
        if t >= nextBlink && t >= blinkEnd {
            blinkStart = t; blinkEnd = t + e.blinkDuration
            nextBlink = blinkEnd + rng.exponential(rate: e.blinkRatePerMin / 60)
        }
        if saccade == nil, let p = plannedSaccade, t >= p.at {
            let amplitudeDeg = (p.to - eyePosition).length / ppd
            saccade = Saccade(from: eyePosition, to: p.to, t0: t,
                              duration: e.saccadeDurationBase + e.saccadeDurationPerDeg * amplitudeDeg, primary: p.primary)
            fixationPoint = p.to
            plannedSaccade = nil
        }
        if let s = saccade {
            let tau = (t - s.t0) / max(s.duration, 1e-3)
            if tau >= 1 {
                eyePosition = s.to; saccade = nil; drift = .zero
                if s.primary {
                    landedAt = t
                    if (targetCenter - s.to).length > e.correctiveThresholdDeg * ppd, plannedSaccade == nil {
                        plannedSaccade = (to: targetCenter + rng.gaussian2D(sd: e.saccadeLandingSigmaDeg * ppd * e.correctiveSaccadeSigmaFraction),
                                          at: t + e.correctiveSaccadeLatency, primary: false)
                    }
                }
            } else {
                let sj = tau * tau * tau * (10 - 15 * tau + 6 * tau * tau)
                eyePosition = s.from + (s.to - s.from) * sj
            }
            return
        }
        drift = simOUStep(drift, sigma: e.fixationDriftSigmaDeg * ppd, tau: e.fixationDriftTau, dt: dt, rng: rng)
        if t >= nextMicrosaccade {
            drift = rng.direction() * (e.microsaccadeAmplitudeDeg * ppd * rng.uniform(e.microsaccadeAmplitudeMinFraction,
                                                                                       e.microsaccadeAmplitudeMaxFraction))
            nextMicrosaccade = t + rng.exponential(rate: e.microsaccadeRatePerSec)
        }
        eyePosition = fixationPoint + drift

        // Compensatory gazing: re-aim by the perceived offset when nothing else can correct.
        let d = cfg.decision
        if cfg.compensatoryGaze, hasTarget, landedAt != nil, plannedSaccade == nil, t > holdUntil,
           t - lastCompensation > d.compensatoryReaction,
           let c = perceived.value(at: t - d.compensatoryReaction) {
            let err = targetCenter - c
            if err.length > d.compensatoryMinOffset {
                plannedSaccade = (to: fixationPoint + err * d.compensatoryGain + rng.gaussian2D(sd: d.compensatoryScatterDeg * ppd),
                                  at: t, primary: false)
                lastCompensation = t
            }
        }
    }

    /// Minimum-jerk speed profile ds/dτ for τ ∈ [0, 1].
    @inline(__always) private static func minJerkRate(_ tau: Double) -> Double {
        guard tau > 0, tau < 1 else { return 0 }
        return 30 * tau * tau - 60 * tau * tau * tau + 30 * tau * tau * tau * tau
    }

    private func submovementDuration(error: Double, base: Double, perBit: Double) -> Double {
        base + perBit * log2(error / max(cfg.webgrid.cellSize, 1) + 1)
    }

    private func stepHead(t: TimeSec, dt: TimeSec) {
        let h = cfg.head
        tremor = simOUStep(tremor, sigma: h.tremorSigma, tau: h.tremorTau, dt: dt, rng: rng)
        guard h.useSubmovements else { stepHeadContinuous(t: t, dt: dt); return }
        var voluntary = Point2.zero
        if let sub = headSub {
            let tau = (t - sub.t0) / max(sub.duration, 1e-3)
            if tau >= 1 {
                headSub = nil
                nextHeadDecision = t + h.reactionTime
            } else {
                voluntary = sub.amplitude * (Self.minJerkRate(tau) / sub.duration)
            }
        } else if t >= nextHeadDecision {
            nextHeadDecision = t + h.controlInterval
            if hasTarget, let landed = landedAt, !isHandClutched, handEngageAt == nil, t > holdUntil,
               t > landed + h.postSaccadeAssessDelay,
               let c = perceived.value(at: t - cfg.decision.perceptionDelay) {
                let err = targetCenter - c
                let e = err.length
                if e > max(h.stopDeadzone, cfg.decision.perceptualThreshold) {
                    let d = submovementDuration(error: e, base: h.submovementBase, perBit: h.submovementPerBit)
                    // Feed-forward inverse of the learned pointer mapping: the average rate
                    // that would cover the distance at constant speed, times the learned scale.
                    let want = e * h.submovementGain
                    let rate = min(HeadPointerGain.rate(forSpeed: want / d, config: headGain) * headGainScale, h.maxRate)
                    var amp = err * (rate * d / e)
                    amp = amp * (1 + rng.gaussian() * h.motorNoiseFraction) + rng.gaussian2D(sd: h.motorNoiseFraction * 0.3 * amp.length)
                    if abs(headAngle.x) > h.maxExcursion && amp.x * headAngle.x > 0 { amp.x = 0 }
                    if abs(headAngle.y) > h.maxExcursion && amp.y * headAngle.y > 0 { amp.y = 0 }
                    // Gain adaptation from the previous submovement's perceived outcome.
                    headSub = Submovement(t0: t, duration: d, amplitude: amp, plannedCursor: c + err * h.submovementGain, startCursor: c)
                }
            }
        }
        let noiseSd = h.motorNoiseFloor
        motorNoise = simOUStep(motorNoise, sigma: noiseSd, tau: h.motorTau, dt: dt, rng: rng)
        headRate = voluntary + motorNoise
        headAngle = headAngle + (headRate + tremor) * dt
        adaptHeadGain(t: t)
    }

    private var pendingAdaptation: (start: Point2, planned: Point2, at: TimeSec)?
    private func adaptHeadGain(t: TimeSec) {
        // When a submovement has just finished and its effect is visible, compare the
        // achieved with the planned cursor displacement and nudge the internal gain.
        if let sub = headSub, pendingAdaptation == nil || pendingAdaptation!.at < sub.t0 {
            pendingAdaptation = (sub.startCursor, sub.plannedCursor, sub.t0 + sub.duration + cfg.decision.perceptionDelay + 0.05)
        }
        guard let pa = pendingAdaptation, headSub == nil, t >= pa.at,
              let now = perceived.value(at: t - cfg.decision.perceptionDelay) else { return }
        pendingAdaptation = nil
        let planned = pa.planned - pa.start, achieved = now - pa.start
        let pl = planned.length
        guard pl > 15 else { return }
        let along = (achieved.x * planned.x + achieved.y * planned.y) / pl
        guard along > 1 else { return }
        let ratio = min(max(pl / along, 0.5), 2.0)
        headGainScale = min(max(headGainScale * pow(ratio, cfg.head.gainAdaptationRate), 0.3), 3.0)
    }

    private func stepHeadContinuous(t: TimeSec, dt: TimeSec) {
        let h = cfg.head
        if t >= nextHeadDecision {
            nextHeadDecision = t + h.controlInterval
            var desired = Point2.zero
            if hasTarget, let landed = landedAt, !isHandClutched, handEngageAt == nil, t > holdUntil,
               t - h.reactionTime > landed + h.postSaccadeAssessDelay,
               let c = perceived.value(at: t - h.reactionTime) {
                let err = targetCenter - c
                if err.length > max(h.stopDeadzone, cfg.decision.perceptualThreshold) {
                    let speed = min(h.loopGain * err.length, h.maxCursorSpeed)
                    let rate = min(HeadPointerGain.rate(forSpeed: speed, config: headGain), h.maxRate)
                    desired = err * (rate / err.length)
                    if abs(headAngle.x) > h.maxExcursion && desired.x * headAngle.x > 0 { desired.x = 0 }
                    if abs(headAngle.y) > h.maxExcursion && desired.y * headAngle.y > 0 { desired.y = 0 }
                }
            }
            headCmd = desired
        }
        let k = min(dt / max(h.motorTau, 1e-3), 1)
        headSmooth = headSmooth + (headCmd - headSmooth) * k
        let noiseSd = h.motorNoiseFraction * headCmd.length + h.motorNoiseFloor
        motorNoise = simOUStep(motorNoise, sigma: noiseSd, tau: h.motorTau, dt: dt, rng: rng)
        headRate = headSmooth + motorNoise
        headAngle = headAngle + (headRate + tremor) * dt
    }

    private func stepHand(t: TimeSec, dt: TimeSec, actions: inout [SimHumanAction]) {
        let hc = cfg.hand
        if !isHandClutched, handEngageAt == nil, hasTarget, let landed = landedAt, t > holdUntil, clickOnsetAt == nil,
           headSub == nil, t - hc.reactionTime > landed + cfg.head.postSaccadeAssessDelay,
           let c = perceived.value(at: t - hc.reactionTime) {
            let err = (targetCenter - c).length
            if err < bestError * hc.stallImprovementFraction { bestError = err; bestErrorTime = t }
            // Engage only if the cursor is visibly off target (otherwise just click).
            if !targetRect.contains(c) && err > hc.stopDeadzone + cfg.decision.perceptualThreshold {
                let stalled = !cfg.headEnabled || (t - bestErrorTime > hc.engageAfterStall)
                if stalled { handEngageAt = t + hc.engageLatency }
            }
        }
        if let ea = handEngageAt, t >= ea {
            isHandClutched = true; handEngageAt = nil; nextHandDecision = t; lastHandFrame = t; handVelocity = .zero
            handSub = nil
        }
        guard isHandClutched else { return }
        if hc.useSubmovements {
            if let sub = handSub {
                let tau = (t - sub.t0) / max(sub.duration, 1e-3)
                if tau >= 1 { handSub = nil; handVelocity = .zero; nextHandDecision = t + hc.reactionTime }
                else { handVelocity = sub.amplitude * (Self.minJerkRate(tau) / sub.duration) }
            } else if t >= nextHandDecision, handReleaseAt == nil {
                nextHandDecision = t + hc.controlInterval
                if let c = perceived.value(at: t - cfg.decision.perceptionDelay) {
                    let err = targetCenter - c
                    // Release once the cursor is comfortably inside the target.
                    let inner = ScreenRect(x: targetRect.origin.x + targetRect.width * 0.15, y: targetRect.origin.y + targetRect.height * 0.15,
                                           width: targetRect.width * 0.7, height: targetRect.height * 0.7)
                    if inner.contains(c) || err.length < hc.stopDeadzone {
                        handReleaseAt = t + hc.releaseLatency
                    } else {
                        let d = submovementDuration(error: err.length, base: hc.submovementBase, perBit: hc.submovementPerBit)
                        var amp = err * hc.submovementGain
                        amp = amp * (1 + rng.gaussian() * hc.motorNoiseFraction) + rng.gaussian2D(sd: hc.motorNoiseFraction * 0.3 * amp.length)
                        handSub = Submovement(t0: t, duration: d, amplitude: amp, plannedCursor: c + amp, startCursor: c)
                    }
                }
            }
        } else if t >= nextHandDecision {
            nextHandDecision = t + hc.controlInterval
            if let c = perceived.value(at: t - hc.reactionTime) {
                let err = targetCenter - c
                if err.length < hc.stopDeadzone || targetRect.contains(c) {
                    handVelocity = .zero
                    if handReleaseAt == nil { handReleaseAt = t + hc.releaseLatency }
                } else {
                    let speed = min(hc.loopGain * err.length, hc.maxSpeed) * (1 + rng.gaussian() * hc.motorNoiseFraction)
                    handVelocity = err * (speed / err.length)
                }
            }
        }
        let frame = 1 / hc.frameRate
        if t - lastHandFrame >= frame - 1e-9 {
            actions.append(.handDelta(handVelocity * (t - lastHandFrame)))
            lastHandFrame = t
        }
        if let r = handReleaseAt, t >= r {
            isHandClutched = false; handReleaseAt = nil; handVelocity = .zero; handSub = nil
            bestError = .infinity; bestErrorTime = t
            // Take-off selection: the release itself is the click when on target.
            if hc.releaseClicks, hasTarget, clickOnsetAt == nil, t > awaitingUntil,
               let c = perceived.value(at: t - cfg.decision.perceptionDelay), targetRect.contains(c) {
                clickOnsetAt = t
                lastClickDecision = t
                insideSince = nil
            }
        }
    }

    private func stepClick(t: TimeSec, actions: inout [SimHumanAction]) {
        let c = cfg.click, d = cfg.decision
        if let on = clickOnsetAt, t >= on {
            clickOnsetAt = nil
            actions.append(.clickOnset)
            clickAttempts += 1
            awaitingUntil = t + c.retryTimeout
            switch cfg.clickSource {
            case .pinch:
                holdUntil = t + c.pinchHoldDuration + c.postClickHold
            case .jaw:
                holdUntil = t + cfg.head.clenchImpulseDuration + c.postClickHold
                clenchUntil = t + cfg.head.clenchImpulseDuration
                clenchDir = rng.direction()
            }
        }
        guard hasTarget, clickOnsetAt == nil, t > awaitingUntil, !isHandClutched, handEngageAt == nil,
              headSub == nil, landedAt != nil, saccade == nil else { return }
        let tp = t - d.perceptionDelay
        let f = min(max(d.clickInnerFraction, 0.05), 1)
        let inner = ScreenRect(x: targetRect.origin.x + targetRect.width * (1 - f) / 2, y: targetRect.origin.y + targetRect.height * (1 - f) / 2,
                               width: targetRect.width * f, height: targetRect.height * f)
        if let cur = perceived.value(at: tp), inner.contains(cur) {
            if insideSince == nil { insideSince = tp }
        } else {
            insideSince = nil
        }
        // Click only once the cursor has visibly settled (it barely moved over the dwell).
        var settled = true
        if let cur = perceived.value(at: tp), let before = perceived.value(at: tp - d.clickDwell) {
            settled = (cur - before).length < cfg.webgrid.cellSize * d.settleFraction
        }
        let dwellOK = settled && (insideSince.map { tp - $0 >= d.clickDwell } ?? false)
        let patienceOK = t - max(targetOnset, lastClickDecision) > d.patience
        if dwellOK || patienceOK {
            if !dwellOK { patienceClicks += 1 }
            clickOnsetAt = t + c.motorDelay
            lastClickDecision = t
            insideSince = nil
        }
    }
}
