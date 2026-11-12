// EKFCursorFusion.swift — gaze + head + hand → cursor, via an extended Kalman filter.
//
// STATE  s = [x, y, vx, vy, bx, by]  (screen points, pt/s)
//   (x, y)   cursor position = the user's intended point of regard
//   (vx, vy) correction velocity driven by the AirPods head channel
//   (bx, by) gaze bias: the slowly varying error of the calibrated gaze estimate
//            (calibration drift, head-pose leakage, spatially varying residuals)
//
// MEASUREMENTS
//   gaze   z = (x + bx, y + by) + n,  n ~ N(0, σg²/confidence)          H = [I 0 I]
//   head   z = v_head + n,  v_head = HeadPointerGain(rotation rate)       H = [0 I 0]
//   hand   relative delta Δ (pinch-drag): deterministic control input
//
// THE KEY IDEA — control inputs move x and b in opposite directions.
// Head and hand corrections are what the user does when the cursor is *not* where
// they look, i.e. they are direct observations of the gaze bias. The process model
// therefore integrates the correction velocity as x += v·dt, b −= v·dt (and Δ as
// x += Δ, b −= Δ), which keeps the gaze-observable sum x + b unchanged. Consequences:
//   • gaze measurements never "fight" a head correction: after the user nudges the
//     cursor by Δ, the gaze innovation stays ≈ 0 and the cursor stays where it was put;
//   • the bias state becomes an online re-calibration learnt from the corrections,
//     so the next target is pre-corrected by the learnt bias;
//   • gaze innovations update the observable sum x + b through the correlated x/b
//     covariance; they do not directly observe velocity. Head/hand corrections keep
//     that sum fixed while moving x and b in opposite directions, so the gaze channel
//     can continue to smooth the fixation without undoing a deliberate correction.
// Across a saccade the bias is only partially transferable (spatially varying
// calibration error): b ← b·exp(−amplitude / biasCorrelationLength), with the
// covariance restored toward `biasPriorSigma²` accordingly (an AR(1) field model).
//
// SACCADE DETECTION (fast re-targeting vs. heavy smoothing during fixation)
//   (A) single-sample χ² gate on the gaze innovation: an outlier (NIS > saccadeGateNIS
//       and |y| > saccadeMinJumpPoints) is *not* applied but queued; when
//       `saccadeConfirmSamples` consecutive outliers within `saccadeConfirmWindow`
//       agree with each other (spread ≤ saccadeClusterSigmas·σg) the position is reset
//       to their mean. Single glitches (blink artefacts) therefore never move the cursor.
//   (B) windowed test for medium shifts (50–90 pt) that pass the single-sample gate:
//       the mean innovation of the last m ≤ saccadeMaxWindow applied samples is tested
//       against a stricter gate (saccadeWindowGateNIS, ≈4σ) with variance R/m; a hit
//       resets the position to the window mean. This bounds the lag on small saccades
//       that a running mean would otherwise follow very slowly.
//   A reset sets Var(x + b) = R/m and inflates the bias covariance as described.
//
// TIME HANDLING
//   The filter time is the newest measurement time. Gaze samples carry *capture*
//   timestamps and typically arrive 50–100 ms after head samples of the same time;
//   a gaze sample older than the filter time (but within `reorderWindow`) is applied
//   against the *current* gaze-observable sum x + b. Head/hand corrections preserve
//   that sum by construction, so comparing the late sample with a historical x and a
//   current b would invent a residual equal to the motion during the delivery lag.
//   This current-state OOSM approximation avoids that false correction; older samples
//   are dropped.
//   `cursor(at:)` extrapolates with the held head velocity (zero-order hold between
//   25 Hz AirPods samples, zeroed after `head.timeout`) without mutating the filter.
//
// ASSUMPTIONS / LIMITATIONS
//   • The user looks at the *target* while correcting with the head (not at the cursor).
//   • Head motion while gazing leaks into the webcam gaze estimate only partially
//     (the gaze mapper compensates head pose); any residual leak is absorbed by b
//     through the same correction loop.
//   • A delayed saccade is gated at delivery time by the current-state OOSM
//     approximation; very late jump samples can therefore reset slightly after
//     the physical eye movement.
//   • Single-threaded: call from one serial queue.
import Foundation

public struct EKFCursorFusionConfig: Codable, Equatable {
    /// Per-sample gaze noise std (pt) at confidence 1; R = σ²/confidence.
    public var gazeSigmaPoints: Double
    /// Innovation-based adaptive R (Mehra 1972 / Sage–Husa style): the effective gaze σ is
    /// max(gazeSigmaPoints, σ̂) where σ̂² is a robust running estimate from in-fixation
    /// innovations. Prevents false saccade resets when the real tracker is noisier than assumed.
    public var adaptiveGazeNoise: Bool = true
    /// The gaze bias b is only observable through relative corrections (head / hand). When
    /// none arrived within this window, gaze updates treat b as a Schmidt "consider" state
    /// (never corrected by gaze), so b cannot silently absorb real gaze motion. 0 disables.
    public var biasObservabilityWindow: Double = 2.0
    /// Position random-walk σ (pt/√s) used while no correction channel is active (gaze-only
    /// operation): the cursor must then follow small compensatory gaze shifts itself, whereas
    /// with head/hand correction a near-zero `positionProcessSigma` is optimal. 0 = unchanged.
    public var gazeOnlyPositionProcessSigma: Double = 20
    /// Set when the hand fine-adjust channel is enabled: hand clutches are intermittent, so
    /// gaze-only mode is then never entered (simulation: entering it cost ~0.2–0.3 BPS).
    public var handCorrectionEnabled: Bool = false
    /// EMA weight of the innovation-variance estimate per accepted sample.
    public var adaptiveNoiseAlpha: Double = 0.02
    /// Upper bound for σ̂ (pt).
    public var adaptiveNoiseMaxSigma: Double = 120
    /// Gaze samples below this confidence are ignored (blink).
    public var minGazeConfidence: Double
    /// Without usable gaze or a fresh head sample for this long the cursor mode becomes `.idle`.
    public var gazeTimeout: TimeSec
    /// Measurements older than the filter time by more than this are dropped.
    public var reorderWindow: TimeSec
    /// Random walk on the cursor position, pt/√s. Lets the cursor follow slow gaze drift.
    public var positionProcessSigma: Double
    /// Discrete white-noise acceleration of the correction velocity, pt/s² (head dynamics).
    public var headAccelerationSigma: Double
    /// Ornstein–Uhlenbeck decay time of the correction velocity (s).
    public var velocityDecayTau: TimeSec
    /// Random walk on the bias, pt/√s (attributes slow gaze changes partly to drift).
    public var biasProcessSigma: Double
    /// Prior std of the gaze bias (pt), restored after saccades.
    public var biasPriorSigma: Double
    /// Spatial correlation length (pt) of the bias field: keep = exp(−amplitude/L).
    public var biasCorrelationLength: Double
    /// χ²(2) gate for a single gaze innovation (outlier / saccade candidate).
    public var saccadeGateNIS: Double
    /// Stricter gate for the windowed medium-shift test (B).
    public var saccadeWindowGateNIS: Double
    /// Innovations shorter than this (pt) never trigger a reset.
    public var saccadeMinJumpPoints: Double
    /// Consecutive agreeing outliers needed to confirm a saccade (≥ 1).
    public var saccadeConfirmSamples: Int
    /// Outliers older than this (s) no longer count toward confirmation.
    public var saccadeConfirmWindow: TimeSec
    /// Outliers must agree within this many σg to be one saccade.
    public var saccadeClusterSigmas: Double
    /// Max window length for test (B); 0 or 1 disables it.
    public var saccadeMaxWindow: Int
    /// Gaze samples below this confidence may never trigger (or count toward) a
    /// saccade reset — blink-edge artefacts are low-confidence, large, and
    /// otherwise look exactly like a 2-sample saccade. They are simply rejected.
    public var saccadeMinConfidence: Double
    /// Mode is `.saccade` for this long after a reset.
    public var saccadeModeDuration: TimeSec
    public var head: HeadPointerConfig
    /// Online head→gaze leak compensation (see GazeHeadLeakCompensator).
    public var leak: GazeHeadLeakConfig
    /// Uncertainty (pt) added per hand relative-delta input.
    public var relativeDeltaSigma: Double
    /// Mode is `.drag` while relative deltas arrived within this window.
    public var dragModeHold: TimeSec
    /// The unclamped state may leave the screen by this many points.
    public var stateMargin: Double
    /// Position std (pt) at which the reported confidence reaches 0.
    public var confidenceScalePoints: Double
    /// Number of recent display/filter states retained for onset locking and OOSM lookup.
    /// At the default 512 entries and 60 Hz this covers about 8.5 seconds.
    public var historyCapacity: Int

    public init(gazeSigmaPoints: Double = FusionDefaultTuning.gazeSigmaPoints,
                minGazeConfidence: Double = 0.2,
                gazeTimeout: TimeSec = 0.5,
                reorderWindow: TimeSec = 0.2,
                positionProcessSigma: Double = FusionDefaultTuning.positionProcessSigma,
                headAccelerationSigma: Double = FusionDefaultTuning.headAccelerationSigma,
                velocityDecayTau: TimeSec = 1.0,
                biasProcessSigma: Double = FusionDefaultTuning.biasProcessSigma,
                biasPriorSigma: Double = FusionDefaultTuning.biasPriorSigma,
                biasCorrelationLength: Double = FusionDefaultTuning.biasCorrelationLength,
                saccadeGateNIS: Double = FusionDefaultTuning.saccadeGateNIS,
                saccadeWindowGateNIS: Double = FusionDefaultTuning.saccadeWindowGateNIS,
                saccadeMinJumpPoints: Double = FusionDefaultTuning.saccadeMinJumpPoints,
                saccadeConfirmSamples: Int = FusionDefaultTuning.saccadeConfirmSamples,
                saccadeConfirmWindow: TimeSec = 0.12,
                saccadeClusterSigmas: Double = FusionDefaultTuning.saccadeClusterSigmas,
                saccadeMaxWindow: Int = 8,
                saccadeMinConfidence: Double = 0.6,
                saccadeModeDuration: TimeSec = 0.15,
                head: HeadPointerConfig = HeadPointerConfig(),
                leak: GazeHeadLeakConfig = GazeHeadLeakConfig(),
                relativeDeltaSigma: Double = 1.0,
                dragModeHold: TimeSec = 0.15,
                stateMargin: Double = 100,
                confidenceScalePoints: Double = 120,
                historyCapacity: Int = 512) {
        self.gazeSigmaPoints = gazeSigmaPoints; self.minGazeConfidence = minGazeConfidence
        self.gazeTimeout = gazeTimeout; self.reorderWindow = reorderWindow
        self.positionProcessSigma = positionProcessSigma; self.headAccelerationSigma = headAccelerationSigma
        self.velocityDecayTau = velocityDecayTau; self.biasProcessSigma = biasProcessSigma
        self.biasPriorSigma = biasPriorSigma; self.biasCorrelationLength = biasCorrelationLength
        self.saccadeGateNIS = saccadeGateNIS; self.saccadeWindowGateNIS = saccadeWindowGateNIS
        self.saccadeMinJumpPoints = saccadeMinJumpPoints; self.saccadeConfirmSamples = saccadeConfirmSamples
        self.saccadeConfirmWindow = saccadeConfirmWindow; self.saccadeClusterSigmas = saccadeClusterSigmas
        self.saccadeMaxWindow = saccadeMaxWindow; self.saccadeModeDuration = saccadeModeDuration
        self.saccadeMinConfidence = saccadeMinConfidence
        self.head = head; self.leak = leak; self.relativeDeltaSigma = relativeDeltaSigma; self.dragModeHold = dragModeHold
        self.stateMargin = stateMargin; self.confidenceScalePoints = confidenceScalePoints
        self.historyCapacity = historyCapacity
    }
}

public final class EKFCursorFusion: CursorFusing {
    public struct Stats: Codable, Equatable {
        public var gazeAccepted = 0, gazeRejected = 0, gazeDropped = 0, gazeOutOfOrder = 0, gazeLowConfidence = 0
        public var resets = 0, headUpdates = 0, handDeltas = 0
        public init() {}
    }

    public var config: EKFCursorFusionConfig
    public var screen: ScreenRect
    public private(set) var stats = Stats()

    private var ekf: ExtendedKalmanFilter
    private var filterTime: TimeSec = 0
    private var hasFilterTime = false
    private var gazeInitialized = false
    private var lastGazeTime: TimeSec?
    private var lastHeadTime: TimeSec?
    private var lastHeadSample: HeadMotionSample?
    private var lastHandTime: TimeSec?
    private var lastResetTime: TimeSec?
    private var history: FusionCursorHistory
    private var leakComp: GazeHeadLeakCompensator
    private var innovationVar: Double = 0
    /// Gaze σ actually used (pt): configured σ, or the adaptive estimate if larger.
    public var effectiveGazeSigma: Double {
        guard config.adaptiveGazeNoise else { return config.gazeSigmaPoints }
        return min(max(config.gazeSigmaPoints, innovationVar.squareRoot()), max(config.adaptiveNoiseMaxSigma, config.gazeSigmaPoints))
    }
    /// Current head→gaze leak estimate (pt/rad).
    public var headLeak: Point2 { leakComp.clampedLeak }

    private struct GazeObs { var t: TimeSec; var y: Point2; var r: Double }
    private var pendingOutliers: [GazeObs] = []
    private var recentInliers: [GazeObs] = []

    private static let X = 0, Y = 1, VX = 2, VY = 3, BX = 4, BY = 5
    private static let Hgaze = Matrix([[1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1]])
    private static let Hvel = Matrix([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
    /// Velocity variance assigned when the velocity is (re)set to zero, so the next head
    /// sample is applied with gain ≈ 1.
    private static let freshVelocityVariance = 1e5

    public init(config: EKFCursorFusionConfig = EKFCursorFusionConfig(),
                screen: ScreenRect = ScreenRect(x: 0, y: 0, width: 1512, height: 982)) {
        self.config = config
        self.screen = screen
        history = FusionCursorHistory(capacity: config.historyCapacity)
        leakComp = GazeHeadLeakCompensator(config: config.leak, pointerGain: config.head.pointsPerRadian)
        innovationVar = config.gazeSigmaPoints * config.gazeSigmaPoints
        ekf = EKFCursorFusion.initialFilter(screen: screen, config: config)
    }

    private static func initialFilter(screen: ScreenRect, config: EKFCursorFusionConfig) -> ExtendedKalmanFilter {
        let c = screen.center
        let sb2 = config.biasPriorSigma * config.biasPriorSigma
        return ExtendedKalmanFilter(
            state: [c.x, c.y, 0, 0, 0, 0],
            covariance: .diagonal([screen.width * screen.width / 4, screen.height * screen.height / 4,
                                   freshVelocityVariance, freshVelocityVariance, sb2, sb2]))
    }

    // MARK: - Introspection

    /// Unclamped filter position.
    public var statePosition: Point2 { Point2(ekf.x[Self.X], ekf.x[Self.Y]) }
    public var stateVelocity: Point2 { Point2(ekf.x[Self.VX], ekf.x[Self.VY]) }
    /// Current gaze bias estimate (gaze − cursor).
    public var bias: Point2 { Point2(ekf.x[Self.BX], ekf.x[Self.BY]) }
    public var covariance: Matrix { ekf.P }
    /// Var(x + b) per axis — the uncertainty of the gaze-observable sum.
    public var sumVariance: Point2 {
        let P = ekf.P
        return Point2(P[Self.X, Self.X] + 2 * P[Self.X, Self.BX] + P[Self.BX, Self.BX],
                      P[Self.Y, Self.Y] + 2 * P[Self.Y, Self.BY] + P[Self.BY, Self.BY])
    }
    public var lastReset: TimeSec? { lastResetTime }
    public var isGazeInitialized: Bool { gazeInitialized }

    // MARK: - CursorFusing

    public func reset() {
        ekf = EKFCursorFusion.initialFilter(screen: screen, config: config)
        hasFilterTime = false; gazeInitialized = false
        lastGazeTime = nil; lastHeadTime = nil; lastHeadSample = nil; lastHandTime = nil; lastResetTime = nil
        pendingOutliers.removeAll(); recentInliers.removeAll()
        history.removeAll()
        leakComp = GazeHeadLeakCompensator(config: config.leak, pointerGain: config.head.pointsPerRadian)
        innovationVar = config.gazeSigmaPoints * config.gazeSigmaPoints
        stats = Stats()
    }

    public func ingest(gaze: GazeEstimate) {
        let cfg = config
        // Confidence zero is the shared blink/unusable sentinel, even when a
        // caller sets the configurable threshold to zero.
        guard gaze.confidence > 0, gaze.confidence >= cfg.minGazeConfidence else { stats.gazeLowConfidence += 1; return }
        let t = gaze.t
        if hasFilterTime && t < filterTime {
            if filterTime - t > cfg.reorderWindow { stats.gazeDropped += 1; return }
            stats.gazeOutOfOrder += 1
        } else {
            predict(to: t)
        }
        let conf = min(max(gaze.confidence, 0.1), 1.0)
        let sigmaG = effectiveGazeSigma
        let r = sigmaG * sigmaG / conf
        let useLeak = cfg.head.enabled && cfg.leak.enabled
        let z = useLeak ? leakComp.compensate(gaze.point, at: t) : gaze.point
        lastGazeTime = max(lastGazeTime ?? t, t)

        if !gazeInitialized {
            gazeInitialized = true
            resetPosition(sum: z, sumVariance: r, jump: 0, at: t)
            stats.gazeAccepted += 1
            return
        }

        // For a late sample use the current state. The x+b observable is invariant
        // under the relative head/hand corrections, while mixing historical x with
        // current b would create a synthetic innovation proportional to latency.
        let ref = statePosition
        let b = bias
        let y = z - (ref + b)
        let vs = sumVariance
        let nis1 = y.x * y.x / (vs.x + r) + y.y * y.y / (vs.y + r)
        let clusterRadius = cfg.saccadeClusterSigmas * sigmaG

        // (A) single-sample gate → outlier queue → confirmed saccade.
        if nis1 > cfg.saccadeGateNIS && y.length > cfg.saccadeMinJumpPoints {
            if gaze.confidence < cfg.saccadeMinConfidence { stats.gazeRejected += 1; return }
            pendingOutliers.removeAll { t - $0.t > cfg.saccadeConfirmWindow }
            pendingOutliers.append(GazeObs(t: t, y: y, r: r))
            if pendingOutliers.count >= max(cfg.saccadeConfirmSamples, 1) {
                let (mean, spread, rMean) = Self.summarize(pendingOutliers)
                if spread <= clusterRadius {
                    resetPosition(sum: statePosition + b + mean, sumVariance: rMean / Double(pendingOutliers.count),
                                  jump: mean.length, at: t)
                    stats.gazeAccepted += 1
                    return
                }
                pendingOutliers = [pendingOutliers[pendingOutliers.count - 1]]
            }
            stats.gazeRejected += 1
            return
        }
        pendingOutliers.removeAll()

        // Low-confidence samples (blink edges) with a clearly inconsistent innovation are
        // rejected outright (χ²₂ 99%), and never enter the windowed saccade test below.
        let lowConfidence = gaze.confidence < cfg.saccadeMinConfidence
        if lowConfidence, isGazeOnlyMode(at: t), nis1 > 9.21 { stats.gazeRejected += 1; return }

        // Normal update. The same current-state measurement function is used for
        // in-order and late samples; see the OOSM note above.
        let biasUnobservable = isGazeOnlyMode(at: t)
        ekf.update(z: [z.x, z.y], H: Self.Hgaze, R: .diagonal([r, r]),
                   consider: biasUnobservable ? [Self.BX, Self.BY] : [])
        clampState()
        stats.gazeAccepted += 1
        if useLeak && gaze.confidence >= cfg.saccadeMinConfidence { leakComp.learn(gaze.point, at: t) }
        if cfg.adaptiveGazeNoise && gaze.confidence >= cfg.saccadeMinConfidence {
            // Per-axis innovation power minus the predicted state part, scaled to confidence 1.
            let e2 = 0.5 * (max(y.x * y.x - vs.x, 0) + max(y.y * y.y - vs.y, 0)) * conf
            let capped = min(e2, 9 * innovationVar)   // robust: limit single-sample influence
            innovationVar += cfg.adaptiveNoiseAlpha * (capped - innovationVar)
        }

        // (B) windowed medium-shift test.
        let maxWindow = cfg.saccadeMaxWindow
        guard maxWindow >= 2, !lowConfidence else { return }
        recentInliers.append(GazeObs(t: t, y: y, r: r))
        if recentInliers.count > maxWindow { recentInliers.removeFirst(recentInliers.count - maxWindow) }
        let minM = max(cfg.saccadeConfirmSamples, 2)
        guard recentInliers.count >= minM else { return }
        let vsNow = sumVariance
        var m = recentInliers.count
        while m >= minM {
            let (mean, spread, rMean) = Self.summarize(Array(recentInliers.suffix(m)))
            if spread <= clusterRadius && mean.length > cfg.saccadeMinJumpPoints {
                let rm = rMean / Double(m)
                let nisM = mean.x * mean.x / (vsNow.x + rm) + mean.y * mean.y / (vsNow.y + rm)
                if nisM > cfg.saccadeWindowGateNIS {
                    resetPosition(sum: statePosition + bias + mean, sumVariance: rm, jump: mean.length, at: t)
                    return
                }
            }
            m -= 1
        }
    }

    public func ingest(head: HeadMotionSample) {
        let cfg = config
        guard cfg.head.enabled else { return }
        let prev = lastHeadSample
        let t = head.t
        let outOfOrder = hasFilterTime && t < filterTime
        if outOfOrder {
            if filterTime - t > cfg.reorderWindow { return }
            // An attitude derivative needs a newer sample as its reference. An
            // out-of-order sample would have a negative interval and, if retained,
            // would also corrupt the reference for the next in-order update.
            if cfg.head.source == .attitudeDerivative { return }
            // Slightly late head sample: apply at the current filter time (velocity is a hold).
        } else {
            predict(to: t)
        }
        let rates = HeadPointerGain.rates(from: head, previous: prev, config: cfg.head)
        // Keep the newest capture sample for attitude differentiation. The first
        // sample intentionally produces no derivative but still becomes the anchor.
        if !outOfOrder { lastHeadSample = head }
        guard let rates else { return }
        if cfg.leak.enabled && !outOfOrder { leakComp.ingestHead(yawRate: rates.yaw, pitchRate: rates.pitch, at: t) }
        let v = HeadPointerGain.velocity(yawRate: rates.yaw, pitchRate: rates.pitch, config: cfg.head)
        let r = cfg.head.velocitySigma * cfg.head.velocitySigma
        ekf.update(z: [v.x, v.y], H: Self.Hvel, R: .diagonal([r, r]))
        lastHeadTime = max(lastHeadTime ?? t, t)
        stats.headUpdates += 1
    }

    public func ingest(relativeDelta: Point2, at t: TimeSec) {
        if !(hasFilterTime && t < filterTime) { predict(to: t) }
        var x = ekf.x, P = ekf.P
        let s2 = config.relativeDeltaSigma * config.relativeDeltaSigma
        for axis in 0..<2 {
            let ix = Self.X + axis, ib = Self.BX + axis
            let d = axis == 0 ? relativeDelta.x : relativeDelta.y
            x[ix] += d; x[ib] -= d
            P[ix, ix] += s2; P[ib, ib] += s2; P[ix, ib] -= s2; P[ib, ix] -= s2
        }
        ekf.x = x; ekf.P = P
        clampState()
        lastHandTime = max(lastHandTime ?? t, t)
        stats.handDeltas += 1
    }

    public func cursor(at t: TimeSec) -> CursorState {
        guard hasFilterTime else {
            return CursorState(t: t, position: screen.center, velocity: .zero, mode: .idle, confidence: 0)
        }
        if t < filterTime, let h = history.lookup(at: t) {
            return CursorState(t: t, position: h.display, velocity: .zero, mode: mode(at: t), confidence: confidence(at: t))
        }
        var pos = statePosition
        var v = stateVelocity
        let dt = max(t - filterTime, 0)
        if dt > 0 && (v.x != 0 || v.y != 0) {
            let deadline = (lastHeadTime ?? -.infinity) + config.head.timeout
            let d = min(max(min(t, deadline) - filterTime, 0), 1.0)
            let tau = max(config.velocityDecayTau, 1e-3)
            let a = exp(-d / tau)
            pos = pos + v * (tau * (1 - a))
            v = t > deadline ? .zero : v * a
        }
        let display = screen.clamp(pos)
        history.append(FusionCursorHistory.Entry(t: t, state: pos, display: display))
        return CursorState(t: t, position: display, velocity: v, mode: mode(at: t), confidence: confidence(at: t))
    }

    public func position(at t: TimeSec) -> Point2? { history.lookup(at: t)?.display }

    // MARK: - Internals

    private func mode(at t: TimeSec) -> CursorMode {
        if !gazeInitialized && lastHeadTime == nil { return .idle }
        if let h = lastHandTime, t - h < config.dragModeHold { return .drag }
        if let r = lastResetTime, t - r < config.saccadeModeDuration { return .saccade }
        if let g = lastGazeTime, t - g > config.gazeTimeout { return .idle }
        // With no gaze stream, a fresh AirPods velocity still gives a useful
        // relative cursor. `.fixation` is the closest public mode for this
        // head-only state; it falls back to `.idle` once the velocity is stale.
        if !gazeInitialized {
            if let h = lastHeadTime, t - h <= config.head.timeout { return .fixation }
            return .idle
        }
        return .fixation
    }

    private func confidence(at t: TimeSec) -> Double {
        if !gazeInitialized {
            guard let h = lastHeadTime else { return 0 }
            let age = max(t - h, 0)
            let speedVariance = 0.5 * (ekf.P[Self.VX, Self.VX] + ekf.P[Self.VY, Self.VY])
            let speedSigma = max(speedVariance, 0).squareRoot()
            let quality = 1 / (1 + speedSigma / max(config.head.velocitySigma, 1))
            let freshness = age > config.head.timeout ? 0.2 : (1 - 0.5 * age / max(config.head.timeout, 1e-6))
            return min(max((0.25 + 0.45 * quality) * freshness, 0.05), 0.7)
        }
        let vs = sumVariance
        let sigma = (0.5 * (vs.x + vs.y)).squareRoot()
        var c = min(max(1 - sigma / max(config.confidenceScalePoints, 1e-6), 0.05), 1)
        if let g = lastGazeTime, t - g > config.gazeTimeout { c *= 0.5 }
        return c
    }

    /// Advances the filter to `t` (zero-order hold of the head velocity, zeroed when
    /// the head channel is stale).
    private func predict(to t: TimeSec) {
        guard hasFilterTime else { filterTime = t; hasFilterTime = true; return }
        let dt = t - filterTime
        guard dt > 0 else { return }
        let deadline = (lastHeadTime ?? -.infinity) + config.head.timeout
        if stateVelocity != .zero && t > deadline {
            let d1 = deadline - filterTime
            if d1 > 0 { predictStep(d1) }
            zeroVelocity()
            predictStep(t - max(filterTime, deadline))
        } else {
            predictStep(dt)
        }
        filterTime = t
        clampState()
    }

    /// No correction channel enabled/active: the gaze bias is unobservable.
    public func isGazeOnlyMode(at t: TimeSec) -> Bool {
        let cfg = config
        guard cfg.biasObservabilityWindow > 0, !cfg.handCorrectionEnabled else { return false }
        let lastCorrection = max(lastHeadTime ?? -.infinity, lastHandTime ?? -.infinity)
        return t - lastCorrection > cfg.biasObservabilityWindow
    }

    private func predictStep(_ dt: TimeSec) {
        let d = min(max(dt, 0), 1.0)
        guard d > 0 else { return }
        let cfg = config
        let tau = max(cfg.velocityDecayTau, 1e-3)
        let a = exp(-d / tau), c = tau * (1 - a)
        var F = Matrix.identity(6)
        F[Self.X, Self.VX] = c; F[Self.Y, Self.VY] = c
        F[Self.VX, Self.VX] = a; F[Self.VY, Self.VY] = a
        F[Self.BX, Self.VX] = -c; F[Self.BY, Self.VY] = -c
        let qa = cfg.headAccelerationSigma * cfg.headAccelerationSigma
        let gazeOnly = cfg.gazeOnlyPositionProcessSigma > 0 && isGazeOnlyMode(at: filterTime)
        let ps = gazeOnly ? max(cfg.positionProcessSigma, cfg.gazeOnlyPositionProcessSigma) : cfg.positionProcessSigma
        let qx = ps * ps * d
        // Unobservable bias: do not let its variance grow either (it would only dilute the
        // gaze gain through S = P_x + P_b + 2·P_xb + R).
        let qb = gazeOnly ? 0 : cfg.biasProcessSigma * cfg.biasProcessSigma * d
        var Q = Matrix(rows: 6, cols: 6)
        let g = [d * d / 2, d, -d * d / 2]
        for axis in 0..<2 {
            let idx = [Self.X + axis, Self.VX + axis, Self.BX + axis]
            for i in 0..<3 { for j in 0..<3 { Q[idx[i], idx[j]] += qa * g[i] * g[j] } }
            Q[idx[0], idx[0]] += qx
            Q[idx[2], idx[2]] += qb
        }
        ekf.predict(F: F, Q: Q)
    }

    private func zeroVelocity() {
        var x = ekf.x, P = ekf.P
        for iv in [Self.VX, Self.VY] {
            x[iv] = 0
            for j in 0..<6 { P[iv, j] = 0; P[j, iv] = 0 }
            P[iv, iv] = Self.freshVelocityVariance
        }
        ekf.x = x; ekf.P = P
    }

    /// Re-initializes the position so that x + b = `sum` with Var(x + b) = `sumVariance`,
    /// decaying the bias by exp(−jump / biasCorrelationLength) and restoring its
    /// covariance toward the prior.
    private func resetPosition(sum: Point2, sumVariance: Double, jump: Double, at t: TimeSec) {
        let cfg = config
        let keep = jump > 0 ? exp(-jump / max(cfg.biasCorrelationLength, 1e-6)) : 1.0
        let sb2 = cfg.biasPriorSigma * cfg.biasPriorSigma
        var x = ekf.x, P = ekf.P
        for axis in 0..<2 {
            let ix = Self.X + axis, ib = Self.BX + axis
            let pbb = keep * keep * P[ib, ib] + (1 - keep * keep) * sb2
            for j in 0..<6 { P[ix, j] = 0; P[j, ix] = 0; P[ib, j] = 0; P[j, ib] = 0 }
            P[ix, ix] = sumVariance + pbb; P[ib, ib] = pbb; P[ix, ib] = -pbb; P[ib, ix] = -pbb
            let bNew = x[ib] * keep
            x[ib] = bNew
            x[ix] = (axis == 0 ? sum.x : sum.y) - bNew
        }
        ekf.x = x; ekf.P = P
        clampState()
        lastResetTime = t
        stats.resets += 1
        leakComp.newFixation(at: t)
        pendingOutliers.removeAll(); recentInliers.removeAll()
    }

    private func clampState() {
        let m = config.stateMargin
        let r = ScreenRect(x: screen.origin.x - m, y: screen.origin.y - m, width: screen.width + 2 * m, height: screen.height + 2 * m)
        let p = r.clamp(statePosition)
        ekf.x[Self.X] = p.x; ekf.x[Self.Y] = p.y
    }

    private static func summarize(_ obs: [GazeObs]) -> (mean: Point2, spread: Double, rMean: Double) {
        var mean = Point2.zero, rMean = 0.0
        for o in obs { mean = mean + o.y; rMean += o.r }
        let n = Double(obs.count)
        mean = mean / n; rMean /= n
        var spread = 0.0
        for o in obs { spread = max(spread, (o.y - mean).length) }
        return (mean, spread, rMean)
    }
}
