// HandSyntheticHand.swift — deterministic synthetic hand kinematics for the hand-gesture
// tests: a parametric 2-D hand (21 Vision joints) in normalized image space with
// scale / position / rotation / tilt, pinch aperture trajectories, finger curl, joint
// noise, confidence, outliers, occlusion, motion blur and dropouts — plus a scenario
// builder that records ground truth and an evaluator that scores classifier events.
import Foundation
@testable import GazeFuseCore

// MARK: - RNG (SplitMix64)

struct HandTestRNG: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func uniform(_ a: Double = 0, _ b: Double = 1) -> Double {
        a + (b - a) * Double(next() >> 11) / Double(1 << 53)
    }
    mutating func gaussian() -> Double {
        let u1 = max(uniform(), 1e-12), u2 = uniform()
        return (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
    }
    mutating func chance(_ p: Double) -> Bool { uniform() < p }
}

// MARK: - Kinematic model

/// Three joint angle increments (radians) of a finger chain.
struct HandTestFlex {
    var a: Double, b: Double, c: Double
    static func deg(_ a: Double, _ b: Double, _ c: Double) -> HandTestFlex {
        HandTestFlex(a: a * .pi / 180, b: b * .pi / 180, c: c * .pi / 180)
    }
    static func * (f: HandTestFlex, s: Double) -> HandTestFlex { HandTestFlex(a: f.a * s, b: f.b * s, c: f.c * s) }
    static func lerp(_ x: HandTestFlex, _ y: HandTestFlex, _ q: Double) -> HandTestFlex {
        HandTestFlex(a: x.a + (y.a - x.a) * q, b: x.b + (y.b - x.b) * q, c: x.c + (y.c - x.c) * q)
    }
}

/// Pose parameters. Curl: 0 = straight, 1 = fist. Pinch: 0 = open, 1 = thumb touching the
/// tip (`contactGap` apart). A pinch overrides that finger's curl and drives the thumb.
struct HandTestPose {
    var indexCurl = 0.0, middleCurl = 0.0, ringCurl = 0.0, littleCurl = 0.0
    var indexPinch = 0.0
    var middlePinch = 0.0
    /// Fist: thumb wrapped over the curled index finger (tip near the index DIP).
    var thumbWrapped = false
    /// Tip-to-tip distance (palm-length units) at full closure.
    var contactGap = 0.06
}

/// Local frame: palm-length units, wrist at the origin, +y distal, +x thumb side.
enum HandTestModel {
    static let wrist = Point2(0, 0)
    static let thumbCMC = Point2(0.35, 0.30)
    static let indexMCP = Point2(0.32, 0.98)
    static let middleMCP = Point2(0, 1.0)
    static let ringMCP = Point2(-0.26, 0.95)
    static let littleMCP = Point2(-0.40, 0.85)
    /// Centroid of wrist + 4 MCPs (what `HandGestureFeatures.palmCenter` estimates).
    static let palmCentroid = Point2((0.32 + 0 - 0.26 - 0.40) / 5, (0.98 + 1.0 + 0.95 + 0.85) / 5)

    struct Finger { let base: Point2; let angle: Double; let lengths: (Double, Double, Double) }
    static let index = Finger(base: indexMCP, angle: 8 * .pi / 180, lengths: (0.42, 0.26, 0.20))
    static let middle = Finger(base: middleMCP, angle: 0, lengths: (0.46, 0.29, 0.22))
    static let ring = Finger(base: ringMCP, angle: -6 * .pi / 180, lengths: (0.42, 0.27, 0.21))
    static let little = Finger(base: littleMCP, angle: -14 * .pi / 180, lengths: (0.33, 0.21, 0.17))
    static let thumbLengths = (0.40, 0.32, 0.27)
    static let thumbOpen = HandTestFlex.deg(65, -10, -10)        // a = base angle, b/c = increments
    static let thumbIPFlex = -15 * Double.pi / 180
    static let curlFlex = HandTestFlex.deg(80, 100, 60)
    static let indexPinchFlex = HandTestFlex.deg(30, 70, 35)
    static let middlePinchFlex = HandTestFlex.deg(35, 75, 35)

    @inline(__always) static func dir(_ a: Double) -> Point2 { Point2(sin(a), cos(a)) }

    /// Three joints of a chain: angles accumulate (`flex.a` is added to the base angle).
    static func chain(base: Point2, angle: Double, lengths: (Double, Double, Double), flex: HandTestFlex) -> (Point2, Point2, Point2) {
        var a = angle + flex.a
        let p1 = base + dir(a) * lengths.0
        a += flex.b
        let p2 = p1 + dir(a) * lengths.1
        a += flex.c
        let p3 = p2 + dir(a) * lengths.2
        return (p1, p2, p3)
    }

    static func fingerJoints(_ f: Finger, flex: HandTestFlex) -> (Point2, Point2, Point2) {
        chain(base: f.base, angle: f.angle, lengths: f.lengths, flex: flex)
    }

    /// Thumb chain from the CMC: `flex.a` is the absolute base angle, b/c increments.
    static func thumbJoints(_ flex: HandTestFlex) -> (Point2, Point2, Point2) {
        chain(base: thumbCMC, angle: 0, lengths: thumbLengths, flex: flex)
    }

    /// Planar 2-link IK (IP flexion fixed) putting the thumb tip on `target`.
    static func thumbIK(target: Point2) -> HandTestFlex {
        let l1 = thumbLengths.0
        let phi = thumbIPFlex
        let cx = thumbLengths.1 + thumbLengths.2 * cos(phi), cy = thumbLengths.2 * sin(phi)
        let l2 = (cx * cx + cy * cy).squareRoot()
        let gamma = atan2(cy, cx)
        var v = target - thumbCMC
        let dMax = l1 + l2 - 1e-3
        if v.length > dMax { v = v * (dMax / v.length) }
        let d = v.length
        let c2 = min(max((d * d - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1), 1)
        let theta2 = -acos(c2)                                   // flex towards the fingers
        let a = atan2(v.x, v.y) - atan2(l2 * sin(theta2), l1 + l2 * cos(theta2))
        return HandTestFlex(a: a, b: theta2 - gamma, c: phi)
    }

    /// All 21 joints in the local frame.
    static func joints(_ pose: HandTestPose) -> [HandJoint: Point2] {
        var j: [HandJoint: Point2] = [
            .wrist: wrist, .thumbCMC: thumbCMC,
            .indexMCP: indexMCP, .middleMCP: middleMCP, .ringMCP: ringMCP, .littleMCP: littleMCP,
        ]
        let qi = pose.indexPinch, qm = pose.middlePinch
        let indexFlex = HandTestFlex.lerp(curlFlex * pose.indexCurl, indexPinchFlex, qi)
        let middleFlex = HandTestFlex.lerp(curlFlex * pose.middleCurl, middlePinchFlex, qm)
        let (ip, id, it) = fingerJoints(index, flex: indexFlex)
        let (mp, md, mt) = fingerJoints(middle, flex: middleFlex)
        let (rp, rd, rt) = fingerJoints(ring, flex: curlFlex * pose.ringCurl)
        let (lp, ld, lt) = fingerJoints(little, flex: curlFlex * pose.littleCurl)
        j[.indexPIP] = ip; j[.indexDIP] = id; j[.indexTip] = it
        j[.middlePIP] = mp; j[.middleDIP] = md; j[.middleTip] = mt
        j[.ringPIP] = rp; j[.ringDIP] = rd; j[.ringTip] = rt
        j[.littlePIP] = lp; j[.littleDIP] = ld; j[.littleTip] = lt

        /// Where the thumb tip lands when "touching" `tip`: offset by `contactGap`
        /// perpendicular to the finger's distal phalanx (`dip` -> `tip`), on the thumb's side —
        /// the pads touch, the tip landmarks stay a fingertip-thickness apart.
        func target(_ tip: Point2, _ dip: Point2) -> Point2 {
            let axis = tip - dip
            var perp = Point2(-axis.y, axis.x) / max(axis.length, 1e-9)
            let toThumb = thumbCMC - tip
            if perp.x * toThumb.x + perp.y * toThumb.y < 0 { perp = perp * -1 }
            return tip + perp * pose.contactGap
        }
        var thumb = thumbOpen
        if qi > 0 || qm > 0 {
            let closed = qi >= qm ? fingerJoints(index, flex: indexPinchFlex) : fingerJoints(middle, flex: middlePinchFlex)
            thumb = HandTestFlex.lerp(thumbOpen, thumbIK(target: target(closed.2, closed.1)), max(qi, qm))
        } else if pose.thumbWrapped {
            // Thumb over the middle phalanx: tip lands on the DIP, offset towards the CMC.
            thumb = thumbIK(target: id + (thumbCMC - id) * (pose.contactGap / (thumbCMC - id).length))
        }
        let (tm, ti, tt) = thumbJoints(thumb)
        j[.thumbMP] = tm; j[.thumbIP] = ti; j[.thumbTip] = tt
        return j
    }
}

// MARK: - Placement in the image

struct HandTestTransform {
    /// Palm length in normalized image units.
    var scale = 0.16
    /// Image position of the palm centroid.
    var center = Point2(0.5, 0.45)
    /// In-plane rotation (radians, counter-clockwise).
    var rotation = 0.0
    /// Out-of-plane tilt: x is scaled by cos(yaw), y by cos(pitch) before rotating.
    var yaw = 0.0
    var pitch = 0.0

    func apply(_ p: Point2) -> Point2 {
        let q = p - HandTestModel.palmCentroid
        let x = q.x * cos(yaw), y = q.y * cos(pitch)
        let c = cos(rotation), s = sin(rotation)
        return center + Point2(x * c - y * s, x * s + y * c) * scale
    }
}

struct HandTestNoise {
    /// Gaussian position noise (image units) on palm joints; tips get `tipFactor` times more.
    var sigma = 0.004
    var tipFactor = 1.5
    /// Per-joint probability of a confident-but-wrong position jump of up to `outlierMagnitude`.
    var outlierProb = 0.01
    var outlierMagnitude = 0.06
    /// Motion blur: noise x4, confidence x0.7.
    var blur = false
    /// Joints reported with confidence 0.05–0.25 and large error.
    var occluded: Set<HandJoint> = []
}

enum HandTestSynth {
    static let tips: Set<HandJoint> = [.thumbTip, .indexTip, .middleTip, .ringTip, .littleTip]

    static func sample(pose: HandTestPose, transform: HandTestTransform, noise: HandTestNoise,
                       t: TimeSec, rng: inout HandTestRNG, chirality: String? = "right") -> HandSample {
        var joints: [HandJoint: HandPoint] = [:]
        joints.reserveCapacity(21)
        // Iterate in the enum's canonical order. Dictionary iteration order is not a
        // deterministic source of RNG consumption, so iterating the model dictionary
        // directly would make a seeded scenario vary across processes.
        let modelJoints = HandTestModel.joints(pose)
        for j in HandJoint.allCases {
            guard let local = modelJoints[j] else { continue }
            var p = transform.apply(local)
            let isTip = tips.contains(j)
            var sigma = noise.sigma * (isTip ? noise.tipFactor : 1)
            var conf = isTip ? 0.8 + 0.08 * rng.gaussian() : 0.9 + 0.05 * rng.gaussian()
            if noise.blur { sigma *= 4; conf *= 0.7 }
            if noise.occluded.contains(j) { sigma *= 6; conf = rng.uniform(0.05, 0.25) }
            p = p + Point2(rng.gaussian() * sigma, rng.gaussian() * sigma)
            if rng.chance(noise.outlierProb) {
                p = p + Point2(rng.uniform(-noise.outlierMagnitude, noise.outlierMagnitude),
                               rng.uniform(-noise.outlierMagnitude, noise.outlierMagnitude))
            }
            joints[j] = HandPoint(x: p.x, y: p.y, confidence: min(max(conf, 0), 1))
        }
        return HandSample(t: t, chirality: chirality, joints: joints)
    }
}

// MARK: - Scenario builder with ground truth

struct HandTestFrame {
    var t: TimeSec
    var sample: HandSample?
    /// Physical truth (independent of detector dropouts): fingers touching.
    var indexContact: Bool
    var middleContact: Bool
    /// Noise-free normalized tip-to-tip distances (palm-length units).
    var cleanIndex: Double
    var cleanMiddle: Double
    /// True image position of the palm centroid (nil when the hand is absent).
    var palm: Point2?
    var pose: HandTestPose
    /// Which builder produced the frame ("idle", "pinch", "wave", ...), for diagnostics.
    var segment: String
}

final class HandTestScenario {
    private(set) var frames: [HandTestFrame] = []
    var rng: HandTestRNG
    let fps: Double
    var transform = HandTestTransform()
    var pose = HandTestPose()
    var noise = HandTestNoise()
    /// Per-frame probability that the detector returns no hand.
    var dropProb = 0.0
    /// Normal benchmark scenarios model the detector's expected 1–2 frame gaps. Longer
    /// loss is covered explicitly by `absent` / `occluded` tests, so independent Bernoulli
    /// drops are capped at two consecutive frames instead of accidentally testing a
    /// different contract in every click-precision run.
    private var randomDropRun = 0
    var segment = "idle"
    /// Pinch parameter at or above which the fingers physically touch (ground truth).
    static let contactThreshold = 0.9

    init(seed: UInt64, fps: Double = 30) {
        rng = HandTestRNG(seed: seed)
        self.fps = fps
    }

    var count: Int { frames.count }
    var dt: Double { 1 / fps }

    /// Randomize scale / position / rotation / tilt within realistic laptop-camera ranges.
    func randomizePlacement() {
        transform.scale = rng.uniform(0.10, 0.22)
        transform.center = Point2(rng.uniform(0.25, 0.75), rng.uniform(0.3, 0.7))
        transform.rotation = rng.uniform(-40, 40) * .pi / 180
        transform.yaw = rng.uniform(0, 35) * .pi / 180
        transform.pitch = rng.uniform(0, 35) * .pi / 180
    }

    /// Append one frame from the current state. Ground truth comes from the pose even when
    /// the detector drops the frame (the fingers are still physically touching).
    func step(present: Bool = true) {
        let t = Double(frames.count) * dt
        let clean = HandTestModel.joints(pose)
        let ci = clean[.thumbTip]!.distance(to: clean[.indexTip]!)
        let cm = clean[.thumbTip]!.distance(to: clean[.middleTip]!)
        let randomDropped: Bool
        if present {
            randomDropped = randomDropRun < 2 && rng.chance(dropProb)
            randomDropRun = randomDropped ? randomDropRun + 1 : 0
        } else {
            randomDropped = false
            randomDropRun = 0
        }
        let dropped = !present || randomDropped
        let s = dropped ? nil : HandTestSynth.sample(pose: pose, transform: transform, noise: noise, t: t, rng: &rng)
        frames.append(HandTestFrame(t: t, sample: s,
                                    indexContact: pose.indexPinch >= Self.contactThreshold,
                                    middleContact: pose.middlePinch >= Self.contactThreshold,
                                    cleanIndex: ci, cleanMiddle: cm,
                                    palm: present ? transform.center : nil, pose: pose,
                                    segment: present ? segment : "absent"))
    }

    /// Hand visible, slowly drifting, fingers slightly relaxed.
    func idle(frames n: Int, drift: Double = 0.02) {
        segment = "idle"
        let v = Point2(rng.uniform(-drift, drift), rng.uniform(-drift, drift)) / Double(max(n, 1))
        for _ in 0..<n {
            transform.center = transform.center + v
            step()
        }
    }

    /// Close (`closeFrames`), hold (`holdFrames`, translating by `translation` per frame), open (`openFrames`).
    func pinch(_ button: MouseButton, closeFrames: Int = 4, holdFrames: Int = 6, openFrames: Int = 4, translation: Point2 = .zero) {
        segment = "pinch"
        func set(_ q: Double) {
            if button == .left { pose.indexPinch = q } else { pose.middlePinch = q }
        }
        for i in 0..<closeFrames { set(Double(i + 1) / Double(closeFrames)); step() }
        for _ in 0..<holdFrames { transform.center = transform.center + translation; step() }
        for i in 0..<openFrames { set(1 - Double(i + 1) / Double(openFrames)); step() }
        set(0)
    }

    /// Fingers wiggling without touching: index aperture oscillates down to `minAperture`,
    /// hand rotates, other fingers curl randomly.
    func wave(frames n: Int, minAperture: Double = 0.4) {
        segment = "wave"
        let period = rng.uniform(8, 20)
        let rot0 = transform.rotation
        let curl = rng.uniform(0, 0.4)
        for i in 0..<n {
            let phase = Double(i) / period * 2 * .pi
            pose.indexPinch = (1 - minAperture) * (0.5 - 0.5 * cos(phase))
            pose.middleCurl = curl; pose.ringCurl = curl * 1.5; pose.littleCurl = curl * 1.5
            transform.rotation = rot0 + 0.4 * sin(phase * 0.5)
            step()
        }
        pose.indexPinch = 0; pose.middleCurl = 0; pose.ringCurl = 0; pose.littleCurl = 0
        transform.rotation = rot0
    }

    /// Whole-hand translation at `speed` image units / s with motion blur.
    func fastSweep(frames n: Int, speed: Double) {
        segment = "sweep"
        let ang = rng.uniform(0, 2 * .pi)
        let v = Point2(cos(ang), sin(ang)) * (speed / fps)
        noise.blur = true
        for _ in 0..<n {
            transform.center = transform.center + v
            step()
        }
        noise.blur = false
    }

    /// Hand absent.
    func absent(frames n: Int) { for _ in 0..<n { step(present: false) } }

    /// Closed fist with the thumb wrapped over the fingers.
    func fist(frames n: Int) {
        segment = "fist"
        pose.indexCurl = 1; pose.middleCurl = 1; pose.ringCurl = 1; pose.littleCurl = 1; pose.thumbWrapped = true
        for _ in 0..<n { step() }
        pose.indexCurl = 0; pose.middleCurl = 0; pose.ringCurl = 0; pose.littleCurl = 0; pose.thumbWrapped = false
    }

    /// Some joints occluded (low confidence, wrong positions).
    func occluded(frames n: Int, joints: Set<HandJoint>) {
        segment = "occluded"
        noise.occluded = joints
        for _ in 0..<n { step() }
        noise.occluded = []
    }
}

// MARK: - Diagnostics

enum HandTestDiagnostics {
    /// Prints the frames leading up to each unmatched `.down` (raw features per frame).
    static func dumpFalsePositives(_ scenario: HandTestScenario, _ r: HandTestReport, label: String, context: Int = 5) {
        let contact = scenario.frames.map { $0.indexContact || $0.middleContact }
        for e in r.events where e.event.kind == .down {
            // Unmatched = no physical contact within a few frames.
            let lo = max(0, e.frame - 3), hi = min(contact.count - 1, e.frame + 3)
            if contact[lo...hi].contains(true) { continue }
            print("FP \(label) \(e.event.button) at frame \(e.frame) segment \(scenario.frames[e.frame].segment) pose \(scenario.frames[e.frame].pose) scale \(scenario.transform.scale)")
            for i in max(0, e.frame - context)...e.frame {
                let f = scenario.frames[i]
                guard let s = f.sample, let feat = HandGestureFeatures(sample: s) else { print("  \(i): no hand"); continue }
                let idx = feat.index.map { String(format: "idx d %.3f dip %.3f pip %.3f conf %.2f", $0.distance, $0.dipDistance ?? -1, $0.pipDistance ?? -1, $0.confidence) } ?? "idx nil"
                let mid = feat.middle.map { String(format: "mid d %.3f dip %.3f", $0.distance, $0.dipDistance ?? -1) } ?? "mid nil"
                print(String(format: "  %d [%@] %@ | %@ | curl idx %.2f mid %.2f | size %.3f", i, f.segment, idx, mid, feat.indexCurl ?? -1, feat.middleCurl ?? -1, feat.handSize))
            }
        }
    }
}

// MARK: - Evaluation

struct HandTestEvent {
    var frame: Int
    var event: InputEvent
}

struct HandTestReport: CustomStringConvertible {
    var episodes = 0
    var truePositives = 0
    var falsePositives = 0
    var falseNegatives = 0
    var wrongButton = 0
    /// Frames from the first physical-contact frame to `.down` (negative = detected while closing).
    var downLatencies: [Int] = []
    /// Frames from the first frame whose noise-free distance exceeds the off-threshold to `.up`.
    var upLatencies: [Int] = []
    var missingUps = 0
    var extraUps = 0
    var scrollEvents = 0
    var events: [HandTestEvent] = []
    /// False-positive downs by scenario segment.
    var falsePositiveSegments: [String: Int] = [:]
    /// Per frame: the classifier's `pinchMotionDelta`.
    var deltas: [Point2?] = []
    var isPinching: [Bool] = []

    var precision: Double { truePositives + falsePositives == 0 ? 1 : Double(truePositives) / Double(truePositives + falsePositives) }
    var recall: Double { episodes == 0 ? 1 : Double(truePositives) / Double(episodes) }
    var meanDownLatency: Double { downLatencies.isEmpty ? .nan : Double(downLatencies.reduce(0, +)) / Double(downLatencies.count) }
    var maxDownLatency: Int { downLatencies.max() ?? 0 }
    var meanUpLatency: Double { upLatencies.isEmpty ? .nan : Double(upLatencies.reduce(0, +)) / Double(upLatencies.count) }
    var maxUpLatency: Int { upLatencies.max() ?? 0 }

    var description: String {
        let fpText = falsePositiveSegments.isEmpty ? "" : " FP by segment \(falsePositiveSegments)"
        return String(format: "episodes %d  TP %d  FP %d  FN %d  wrongButton %d  precision %.3f  recall %.3f  downLatency mean %.2f max %d  upLatency mean %.2f max %d  missingUps %d extraUps %d  scroll %d",
                      episodes, truePositives, falsePositives, falseNegatives, wrongButton, precision, recall,
                      meanDownLatency, maxDownLatency, meanUpLatency, maxUpLatency, missingUps, extraUps, scrollEvents) + fpText
    }
}

enum HandTestEvaluator {
    /// Runs the classifier over the scenario and scores `.down` / `.up` events against the
    /// ground-truth contact episodes. A down matches an episode when it falls within
    /// [start - earlySlack, end + lateSlack]. The release truth of an episode is the first
    /// frame after it whose noise-free distance exceeds `offThreshold`; the matched down's
    /// `.up` must land in [end - 1, release + upSlack].
    static func evaluate(_ scenario: HandTestScenario, classifier: PinchGestureClassifier,
                         earlySlack: Int = 2, lateSlack: Int = 3, upSlack: Int = 3) -> HandTestReport {
        var r = HandTestReport()
        let offThreshold = classifier.config.pinchOffDistance
        for (i, f) in scenario.frames.enumerated() {
            let evs = classifier.process(f.sample, at: f.t)
            for e in evs { r.events.append(HandTestEvent(frame: i, event: e)) }
            r.deltas.append(classifier.pinchMotionDelta)
            r.isPinching.append(classifier.isPinching)
        }
        r.scrollEvents = r.events.filter { $0.event.kind == .scroll }.count

        for button in [MouseButton.left, .right] {
            let contact = scenario.frames.map { button == .left ? $0.indexContact : $0.middleContact }
            let clean = scenario.frames.map { button == .left ? $0.cleanIndex : $0.cleanMiddle }
            var episodes: [(start: Int, end: Int, release: Int)] = []
            var i = 0
            while i < contact.count {
                if contact[i] {
                    var j = i
                    while j + 1 < contact.count && contact[j + 1] { j += 1 }
                    var rel = j + 1
                    while rel < contact.count && clean[rel] < offThreshold && rel < j + 20 { rel += 1 }
                    episodes.append((i, j, rel))
                    i = j + 1
                } else { i += 1 }
            }
            r.episodes += episodes.count
            let downs = r.events.filter { $0.event.kind == .down && $0.event.button == button }
            let ups = r.events.filter { $0.event.kind == .up && $0.event.button == button }
            // The classifier may need both its off-threshold debounce frames and its
            // tolerated dropout window before emitting `.up`. Keep the caller's explicit
            // slack as a lower bound, while deriving the minimum from the active config so
            // the metric does not call a contractually valid release "missing".
            let releaseSlack = max(upSlack, classifier.config.minOffFrames + classifier.config.maxDropoutFrames)
            var usedDowns = Set<Int>()
            var usedUps = Set<Int>()
            for ep in episodes {
                guard let (k, d) = downs.enumerated().first(where: { !usedDowns.contains($0.offset)
                    && $0.element.frame >= ep.start - earlySlack && $0.element.frame <= ep.end + lateSlack })
                    .map({ ($0.offset, $0.element) }) else {
                    r.falseNegatives += 1
                    // A down with the other button during this episode = wrong button.
                    let other: MouseButton = button == .left ? .right : .left
                    if r.events.contains(where: { $0.event.kind == .down && $0.event.button == other
                        && $0.frame >= ep.start - earlySlack && $0.frame <= ep.end + lateSlack }) { r.wrongButton += 1 }
                    continue
                }
                usedDowns.insert(k)
                r.truePositives += 1
                r.downLatencies.append(d.frame - ep.start)
                // A deferred scroll-mode tap emits `.down` and `.up` during the same
                // release frame. Match that pair by timestamp while retaining the
                // strict frame ordering for ordinary held gestures. Keep this as a small
                // loop rather than a compound `first(where:)` closure: Swift's type
                // checker otherwise spends excessive time on the tuple-heavy expression.
                var matchingUp: (index: Int, event: HandTestEvent)?
                for (upIndex, candidate) in ups.enumerated() {
                    guard !usedUps.contains(upIndex), candidate.frame >= ep.end - 1,
                          candidate.frame <= ep.release + releaseSlack else { continue }
                    let afterDown = candidate.frame > d.frame
                        || (candidate.frame == d.frame && candidate.event.t > d.event.t)
                    if afterDown {
                        matchingUp = (upIndex, candidate)
                        break
                    }
                }
                if let (ku, u) = matchingUp {
                    usedUps.insert(ku)
                    r.upLatencies.append(u.frame - ep.release)
                } else {
                    r.missingUps += 1
                }
            }
            for (k, d) in downs.enumerated() where !usedDowns.contains(k) {
                r.falsePositives += 1
                r.falsePositiveSegments[scenario.frames[d.frame].segment, default: 0] += 1
            }
            r.extraUps += ups.count - usedUps.count
        }
        return r
    }
}
