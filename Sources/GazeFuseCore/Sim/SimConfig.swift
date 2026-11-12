// SimConfig.swift — every parameter of the Webgrid simulator, with the literature
// values the defaults are based on. All structs are Codable so a configuration can
// be saved next to benchmark results.
//
// Unit conventions: seconds, screen points, degrees of visual angle where noted.
// `pointsPerDegree` converts the two: on a 14" MacBook Pro (1512×982 pt, ≈49.7 pt/cm)
// viewed from ≈60 cm, 1° ≈ 1.05 cm ≈ 52 pt; 50 pt/° is used.
import Foundation

public enum SimPreset: String, Codable, CaseIterable {
    /// Fitts'-law ideal pointer through the click model and arbiter (sanity baseline).
    case mouseLike
    /// Gaze → One Euro (DirectGazeCursor), no correction channel, compensatory gazing.
    case gazeOnly
    /// Gaze → EKFCursorFusion, no correction channel, compensatory gazing.
    case gazeEKF
    /// Gaze + AirPods head correction → EKF.
    case gazeHeadEKF
    /// Gaze + hand pinch-drag fine adjust → EKF (no AirPods).
    case gazeHandEKF
    /// Gaze + head + hand fine adjust → EKF.
    case gazeHeadHandEKF
}

public enum SimNoiseLevel: String, Codable, CaseIterable { case optimistic, typical, pessimistic }
public enum SimClickSource: String, Codable { case pinch, jaw }
public enum SimCursorMode: String, Codable { case direct, ekf, ideal }

/// Oculomotor model of the simulated user.
public struct SimEyeConfig: Codable, Equatable {
    /// Saccade latency to a newly appearing target: mean ≈ 200 ms, CV ≈ 0.2
    /// (Carpenter 1988; Fischer & Weber 1993 report 180–250 ms for ordinary saccades).
    public var saccadeLatencyMean: Double = 0.20
    public var saccadeLatencyCV: Double = 0.2
    /// Main-sequence duration D ≈ 21 ms + 2.2 ms/° · amplitude (Carpenter 1988, Bahill 1975).
    public var saccadeDurationBase: Double = 0.021
    public var saccadeDurationPerDeg: Double = 0.0022
    /// Primary saccades undershoot by ≈10 % of amplitude (Becker 1989; Harris 1995).
    public var saccadeUndershootFraction: Double = 0.10
    /// Landing scatter (σ, degrees) beyond the systematic undershoot (van Opstal & van Gisbergen 1989).
    public var saccadeLandingSigmaDeg: Double = 0.3
    /// Corrective saccades are usually tighter than the primary landing; this is
    /// an explicit simulator assumption rather than a Vision accuracy claim.
    public var correctiveSaccadeSigmaFraction: Double = 0.5
    /// Corrective saccade after ≈150 ms when the residual error exceeds `correctiveThresholdDeg`
    /// (Becker & Fuchs 1969: 100–200 ms).
    public var correctiveSaccadeLatency: Double = 0.15
    public var correctiveThresholdDeg: Double = 0.5
    /// Fixational drift/tremor: σ ≈ 0.1–0.2°, modelled as an Ornstein–Uhlenbeck process (Rolfs 2009).
    public var fixationDriftSigmaDeg: Double = 0.15
    public var fixationDriftTau: Double = 0.3
    /// Microsaccades: ≈1–2 /s, 0.1–0.5° (Martinez-Conde, Macknik & Hubel 2004).
    public var microsaccadeRatePerSec: Double = 1.0
    public var microsaccadeAmplitudeDeg: Double = 0.3
    /// Uniform amplitude range as a fraction of `microsaccadeAmplitudeDeg`.
    public var microsaccadeAmplitudeMinFraction: Double = 0.5
    public var microsaccadeAmplitudeMaxFraction: Double = 1.0
    /// Spontaneous blink rate ≈ 15–20 /min (Bentivoglio et al. 1997), duration 100–400 ms (Stern 1984).
    public var blinkRatePerMin: Double = 15
    public var blinkDuration: Double = 0.15
    public init() {}
}

/// Webcam gaze estimator (Vision landmarks + personal calibration) error model.
public struct SimGazeSensorConfig: Codable, Equatable {
    public var frameRate: Double = 30
    /// Per-sample white noise σ (degrees). This is an explicit synthetic-camera
    /// assumption for stress testing; published WebGazer/iTracker papers report
    /// aggregate pixel errors under their own hardware and calibration protocols,
    /// not a transferable Vision frame-to-frame angular bound.
    public var noiseSigmaDeg: Double = 0.7
    /// Temporally correlated wobble (landmark jitter), OU σ and time constant.
    public var noiseCorrelatedSigmaDeg: Double = 0.2
    public var noiseCorrelatedTau: Double = 0.5
    /// RMS of the spatially varying calibration bias (degrees). The 1–2° default is
    /// an explicit short-calibration stress-test assumption; published webcam
    /// studies report task-specific pixel errors that do not transfer directly.
    public var biasRMSDeg: Double = 1.5
    /// Smoothness of the bias field (Gaussian bump length scale, points).
    public var biasFieldLengthScale: Double = 300
    /// Number of Gaussian bumps in the synthetic spatial bias field.
    public var biasFieldBumps: Int = 6
    /// Relative weight of the constant component before RMS normalisation.
    public var biasConstantFraction: Double = 0.7
    /// Slow temporal drift of the bias (OU σ in degrees, time constant s) — posture changes.
    public var biasDriftSigmaDeg: Double = 0.3
    public var biasDriftTau: Double = 30
    /// Fraction of head rotation (as screen displacement, rad·pointsPerRadian) that leaks
    /// into the gaze estimate because head-pose compensation is imperfect.
    public var headLeakage: Double = 0.3
    /// Head rotation → screen displacement conversion used for the leak (pt/rad), i.e. the
    /// geometric displacement of the viewed point at 60 cm: ≈ 60 cm·49.7 pt/cm ≈ 3000 pt/rad.
    public var headLeakPointsPerRadian: Double = 3000
    /// Capture → delivery latency (camera exposure + Vision + mapping), with jitter.
    public var latency: Double = 0.06
    public var latencyJitter: Double = 0.01
    /// Probability that a frame yields no estimate (face lost).
    public var dropoutProbability: Double = 0.01
    /// Eyelid artefact at blink edges: shift (degrees, downward) and reported confidence.
    public var blinkEdgeArtifactDeg: Double = 2.5
    public var blinkEdgeConfidence: Double = 0.4
    public var blinkEdgeWindow: Double = 0.04
    /// Reported confidence range for good samples.
    public var confidenceMin: Double = 0.7
    public init() {}
}

/// AirPods head channel: IMU error model + the user's visuomotor head controller.
public struct SimHeadConfig: Codable, Equatable {
    public var sampleRate: Double = 25
    public var latency: Double = 0.03
    /// Gyro white noise σ (rad/s) at 25 Hz. This is a synthetic AirPods-IMU
    /// assumption and should be replaced by device characterization when available.
    public var gyroNoiseSigma: Double = 0.008
    /// Slowly varying gyro bias (OU σ rad/s, time constant s).
    public var gyroBiasSigma: Double = 0.003
    public var gyroBiasTau: Double = 60
    /// Slow attitude integration drift used by the synthetic sensor (rad/s).
    /// This is a model assumption; it is not a measured AirPods specification.
    public var attitudeDriftRate: Double = 0.0003
    /// Involuntary head tremor/postural sway (OU σ rad/s, time constant s).
    public var tremorSigma: Double = 0.01
    public var tremorTau: Double = 0.5
    /// Visuomotor reaction time of the correction loop (150–250 ms).
    public var reactionTime: Double = 0.20
    /// Corrections are issued intermittently, every ≈100 ms (psychological refractory period).
    public var controlInterval: Double = 0.10
    /// Desired cursor velocity = loopGain · perceived error (1/s), capped at `maxCursorSpeed`.
    public var loopGain: Double = 5.0
    public var maxCursorSpeed: Double = 500
    /// Comfortable maximum head rotation rate (rad/s) and excursion (rad).
    public var maxRate: Double = 0.8
    public var maxExcursion: Double = 0.5
    /// Neuromuscular first-order lag (s).
    public var motorTau: Double = 0.08
    /// Signal-dependent motor noise (Harris & Wolpert 1998): σ = fraction·|rate| + floor.
    public var motorNoiseFraction: Double = 0.10
    public var motorNoiseFloor: Double = 0.005
    /// The user stops correcting once the perceived error is below this (points).
    public var stopDeadzone: Double = 7
    /// The user waits this long after a saccade lands before judging the cursor
    /// (knows the cursor needs a beat to follow the eyes).
    public var postSaccadeAssessDelay: Double = 0.15
    /// Jaw clench disturbance seen by the IMU: impulse magnitude (rad/s) and duration.
    public var clenchImpulseRate: Double = 0.15
    public var clenchImpulseDuration: Double = 0.10
    /// White noise on the unused user-acceleration axes (g), retained so the
    /// emitted HeadMotionSample resembles a complete CoreMotion sample.
    public var userAccelerationNoiseSigma: Double = 0.005
    /// Intermittent submovement controller (Meyer et al. 1988; Crossman & Goodeve 1983):
    /// each correction is a discrete minimum-jerk head rotation planned from the
    /// perceived error, lasting base + perBit·log2(err/cell + 1) seconds, aimed at
    /// `submovementGain` of the error, with signal-dependent endpoint noise
    /// (`motorNoiseFraction` of the amplitude). After it ends the user waits
    /// `reactionTime` to see the result before planning the next one.
    public var useSubmovements: Bool = true
    public var submovementBase: Double = 0.15
    public var submovementPerBit: Double = 0.07
    public var submovementGain: Double = 0.95
    /// Rate at which the user re-learns the head→cursor gain from each submovement (0…1).
    public var gainAdaptationRate: Double = 0.4
    public init() {}
}

/// Hand pinch-drag fine-adjust channel (relative deltas) and its controller.
public struct SimHandConfig: Codable, Equatable {
    public var frameRate: Double = 30
    /// Decision → clutch engaged (gesture formed and recognised).
    public var engageLatency: Double = 0.15
    public var reactionTime: Double = 0.20
    public var controlInterval: Double = 0.08
    public var loopGain: Double = 6.0
    public var maxSpeed: Double = 600
    public var motorNoiseFraction: Double = 0.08
    /// Per-frame landmark jitter after the app's hand gain (points).
    public var trackingNoise: Double = 2.0
    /// Engage when the perceived error has not improved for this long (head stalled) —
    /// or immediately after assessment when there is no head channel.
    public var engageAfterStall: Double = 0.35
    /// Fractional improvement required to reset the stall timer.
    public var stallImprovementFraction: Double = 0.85
    public var releaseLatency: Double = 0.10
    public var stopDeadzone: Double = 6
    /// Submovement model for the clutched hand (see SimHeadConfig).
    public var useSubmovements: Bool = true
    public var submovementBase: Double = 0.12
    public var submovementPerBit: Double = 0.06
    public var submovementGain: Double = 0.95
    /// "Take-off" selection (Potter et al. 1988): releasing the clutch inside the
    /// target is the click. The app implements this as pinch-hold → adjust → release.
    public var releaseClicks: Bool = true
    public init() {}
}

/// Click gesture model: pinch (hand classifier) or jaw clench (AirPods IMU detector).
public struct SimClickConfig: Codable, Equatable {
    /// Decision → physical gesture onset.
    public var motorDelay: Double = 0.08
    /// Pinch: classifier latency (log-normal mean / CV), hold duration, miss and false-positive rates.
    public var pinchDetectLatencyMean: Double = 0.08
    public var pinchDetectLatencyCV: Double = 0.3
    public var pinchHoldDuration: Double = 0.15
    /// Gesture settle time during which the simulated user keeps the gaze still
    /// after a click onset. This models the existing classifier/action path.
    public var postClickHold: Double = 0.10
    public var pinchMissProbability: Double = 0.05
    public var pinchFalsePositivesPerMin: Double = 1.0
    /// Jaw clench: slower detector (needs the IMU transient to develop).
    public var jawDetectLatencyMean: Double = 0.15
    public var jawDetectLatencyCV: Double = 0.3
    public var jawMissProbability: Double = 0.08
    public var jawFalsePositivesPerMin: Double = 1.0
    public var confidenceMin: Double = 0.7
    /// Without visible effect of a click attempt the user retries after this long.
    public var retryTimeout: Double = 0.5
    public init() {}
}

/// Perception and decision parameters.
public struct SimDecisionConfig: Codable, Equatable {
    /// Visual processing delay of the cursor position used for the click decision.
    public var perceptionDelay: Double = 0.10
    /// The perceived cursor must sit inside the target for this long before a click.
    public var clickDwell: Double = 0.10
    /// The user clicks only when the perceived cursor is inside the central
    /// fraction of the target cell (speed–accuracy strategy; 1 = anywhere in the cell).
    public var clickInnerFraction: Double = 0.6
    /// …and only once it moved less than this fraction of a cell during the dwell.
    public var settleFraction: Double = 0.25
    /// Patience: click anyway after this long on one target.
    public var patience: Double = 3.0
    /// Compensatory gazing (no correction channel): re-aim the eyes by the perceived
    /// offset after this reaction time, if the offset exceeds `compensatoryMinOffset`.
    public var compensatoryReaction: Double = 0.35
    public var compensatoryMinOffset: Double = 12
    public var compensatoryGain: Double = 0.8
    public var compensatoryScatterDeg: Double = 0.3
    /// Perceptual threshold: the user cannot see offsets smaller than this (points).
    public var perceptualThreshold: Double = 3
    public init() {}
}

/// Fitts'-law ideal pointer for the `mouseLike` sanity baseline:
/// MT = a + b·log2(D/W + 1) (MacKenzie 1992; a ≈ 0.1 s, b ≈ 0.12 s/bit for a mouse).
public struct SimMouseConfig: Codable, Equatable {
    public var reactionTime: Double = 0.2
    public var fittsA: Double = 0.1
    public var fittsB: Double = 0.12
    /// Endpoint scatter σ as a fraction of target width.
    public var endpointSigmaFraction: Double = 0.17
    public init() {}
}

public struct SimConfig: Codable, Equatable {
    public var preset: SimPreset
    public var noise: SimNoiseLevel
    public var webgrid: WebgridConfig
    public var tickHz: Double
    public var displayHz: Double
    public var pointsPerDegree: Double
    public var cursorMode: SimCursorMode
    public var headEnabled: Bool
    public var handAdjustEnabled: Bool
    public var compensatoryGaze: Bool
    public var clickSource: SimClickSource
    public var eye: SimEyeConfig
    public var gazeSensor: SimGazeSensorConfig
    public var head: SimHeadConfig
    public var hand: SimHandConfig
    public var click: SimClickConfig
    public var decision: SimDecisionConfig
    public var mouse: SimMouseConfig
    public var tuning: FusionTuningBundle

    public init(preset: SimPreset = .gazeHeadEKF, noise: SimNoiseLevel = .typical,
                webgrid: WebgridConfig = WebgridConfig(), tickHz: Double = 240, displayHz: Double = 60,
                pointsPerDegree: Double = 50, cursorMode: SimCursorMode = .ekf, headEnabled: Bool = true,
                handAdjustEnabled: Bool = false, compensatoryGaze: Bool = false, clickSource: SimClickSource = .pinch,
                eye: SimEyeConfig = SimEyeConfig(), gazeSensor: SimGazeSensorConfig = SimGazeSensorConfig(),
                head: SimHeadConfig = SimHeadConfig(), hand: SimHandConfig = SimHandConfig(),
                click: SimClickConfig = SimClickConfig(), decision: SimDecisionConfig = SimDecisionConfig(),
                mouse: SimMouseConfig = SimMouseConfig(), tuning: FusionTuningBundle = FusionTuningBundle()) {
        self.preset = preset; self.noise = noise; self.webgrid = webgrid; self.tickHz = tickHz; self.displayHz = displayHz
        self.pointsPerDegree = pointsPerDegree; self.cursorMode = cursorMode; self.headEnabled = headEnabled
        self.handAdjustEnabled = handAdjustEnabled; self.compensatoryGaze = compensatoryGaze; self.clickSource = clickSource
        self.eye = eye; self.gazeSensor = gazeSensor; self.head = head; self.hand = hand; self.click = click
        self.decision = decision; self.mouse = mouse; self.tuning = tuning
    }

    /// Builds the configuration for a preset at a noise level, using `tuning` for the
    /// real fusion/arbiter code path. The head channel of the fusion configs is switched
    /// on/off to match the preset.
    public static func make(preset: SimPreset, noise: SimNoiseLevel = .typical,
                            tuning: FusionTuningBundle = FusionTuningBundle(),
                            webgrid: WebgridConfig = WebgridConfig(),
                            clickSource: SimClickSource = .pinch) -> SimConfig {
        var c = SimConfig(preset: preset, noise: noise, webgrid: webgrid, tuning: tuning)
        c.clickSource = clickSource
        switch preset {
        case .mouseLike:
            c.cursorMode = .ideal; c.headEnabled = false; c.handAdjustEnabled = false; c.compensatoryGaze = false
        case .gazeOnly:
            c.cursorMode = .direct; c.headEnabled = false; c.handAdjustEnabled = false; c.compensatoryGaze = true
        case .gazeEKF:
            c.cursorMode = .ekf; c.headEnabled = false; c.handAdjustEnabled = false; c.compensatoryGaze = true
        case .gazeHeadEKF:
            c.cursorMode = .ekf; c.headEnabled = true; c.handAdjustEnabled = false; c.compensatoryGaze = false
        case .gazeHandEKF:
            c.cursorMode = .ekf; c.headEnabled = false; c.handAdjustEnabled = true; c.compensatoryGaze = false
        case .gazeHeadHandEKF:
            c.cursorMode = .ekf; c.headEnabled = true; c.handAdjustEnabled = true; c.compensatoryGaze = false
        }
        c.tuning.fusion.head.enabled = c.headEnabled
        c.tuning.direct.head.enabled = c.headEnabled
        c.applyNoise(noise)
        return c
    }

    /// Scales the sensor / gesture error parameters to a noise level.
    public mutating func applyNoise(_ level: SimNoiseLevel) {
        noise = level
        switch level {
        case .optimistic:
            gazeSensor.noiseSigmaDeg = 0.45; gazeSensor.noiseCorrelatedSigmaDeg = 0.1
            gazeSensor.biasRMSDeg = 0.9; gazeSensor.biasDriftSigmaDeg = 0.15; gazeSensor.headLeakage = 0.1
            gazeSensor.dropoutProbability = 0.005; gazeSensor.latency = 0.05
            head.gyroNoiseSigma = 0.005; head.tremorSigma = 0.006
            click.pinchDetectLatencyMean = 0.06; click.pinchMissProbability = 0.02; click.pinchFalsePositivesPerMin = 0.5
            click.jawDetectLatencyMean = 0.12; click.jawMissProbability = 0.04; click.jawFalsePositivesPerMin = 0.5
            hand.trackingNoise = 1.5
        case .typical:
            gazeSensor.noiseSigmaDeg = 0.7; gazeSensor.noiseCorrelatedSigmaDeg = 0.2
            gazeSensor.biasRMSDeg = 1.5; gazeSensor.biasDriftSigmaDeg = 0.3; gazeSensor.headLeakage = 0.3
            gazeSensor.dropoutProbability = 0.01; gazeSensor.latency = 0.06
            head.gyroNoiseSigma = 0.008; head.tremorSigma = 0.01
            click.pinchDetectLatencyMean = 0.08; click.pinchMissProbability = 0.05; click.pinchFalsePositivesPerMin = 1.0
            click.jawDetectLatencyMean = 0.15; click.jawMissProbability = 0.08; click.jawFalsePositivesPerMin = 1.0
            hand.trackingNoise = 2.0
        case .pessimistic:
            gazeSensor.noiseSigmaDeg = 1.1; gazeSensor.noiseCorrelatedSigmaDeg = 0.35
            gazeSensor.biasRMSDeg = 2.5; gazeSensor.biasDriftSigmaDeg = 0.5; gazeSensor.headLeakage = 0.5
            gazeSensor.dropoutProbability = 0.03; gazeSensor.latency = 0.09
            head.gyroNoiseSigma = 0.015; head.tremorSigma = 0.015
            click.pinchDetectLatencyMean = 0.12; click.pinchMissProbability = 0.10; click.pinchFalsePositivesPerMin = 2.5
            click.jawDetectLatencyMean = 0.2; click.jawMissProbability = 0.15; click.jawFalsePositivesPerMin = 2.5
            hand.trackingNoise = 3.0
        }
    }
}
