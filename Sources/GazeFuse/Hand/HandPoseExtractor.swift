// HandPoseExtractor.swift — app-side wrapper around Vision's
// `VNDetectHumanHandPoseRequest`. Owns the request, picks the best hand per frame and
// converts its 21 joints into the hardware-free `HandSample` consumed by
// `PinchGestureClassifier`.
//
// Usage in the camera loop (one `VNImageRequestHandler` per frame):
//
//     let hand = HandPoseExtractor()
//     try handler.perform(face.requests + hand.requests)
//     let sample: HandSample? = hand.extract(t: t)    // nil when no hand
//
// Coordinates are Vision's normalized image space (origin bottom-left, +y up) in the
// un-mirrored camera image; chirality is whatever Vision reports for that image.
//
// Best-hand selection when several hands are detected (`maximumHandCount` > 1):
//   score = mean joint confidence
//         + preferredChiralityBonus      (chirality matches `preferredChirality`)
//         + consistencyBonus * (1 - d / consistencyRadius)+   (d = palm-centre distance to
//                                                              the hand picked last frame)
// so the tracked hand is kept unless another one is clearly better.
import Foundation
import Vision
import GazeFuseCore

public final class HandPoseExtractor {
    public struct Config: Codable, Equatable {
        /// Hands Vision looks for. 1 is cheapest; 2 lets `preferredChirality` pick between hands.
        public var maximumHandCount: Int = 1
        /// "left" / "right" — favoured when several hands are visible. nil = no preference.
        public var preferredChirality: String? = nil
        public var preferredChiralityBonus: Double = 0.25
        public var consistencyBonus: Double = 0.5
        /// Normalized image distance at which the temporal-consistency bonus falls to zero.
        public var consistencyRadius: Double = 0.25
        /// Hands whose mean joint confidence is below this are ignored (returns nil).
        public var minMeanConfidence: Double = 0.2
        /// Minimum confidence for a joint to contribute to the palm-center geometry.
        /// A candidate must have at least two such palm joints at distinct positions.
        public var minPalmJointConfidence: Double = 0.1
        public init() {}
    }

    public var config: Config {
        didSet { request.maximumHandCount = max(1, config.maximumHandCount) }
    }
    private let request: VNDetectHumanHandPoseRequest
    private var lastPalmCenter: Point2?

    /// Number of hands Vision returned on the last `extract` (debug / UI).
    public private(set) var lastObservationCount = 0

    public init(config: Config = Config()) {
        self.config = config
        request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = max(1, config.maximumHandCount)
    }

    public convenience init(maximumHandCount: Int, preferredChirality: String? = nil) {
        var c = Config()
        c.maximumHandCount = maximumHandCount
        c.preferredChirality = preferredChirality
        self.init(config: c)
    }

    /// Requests to pass to `VNImageRequestHandler.perform`, together with the face requests.
    public var requests: [VNRequest] { [request] }

    /// Converts the best hand of the last performed request. nil when no usable hand.
    public func extract(t: TimeSec) -> HandSample? {
        let observations = request.results ?? []
        lastObservationCount = observations.count
        var best: (score: Double, sample: HandSample, palm: Point2)?
        for obs in observations {
            guard let points = try? obs.recognizedPoints(.all), !points.isEmpty else { continue }
            var joints: [HandJoint: HandPoint] = [:]
            joints.reserveCapacity(HandJoint.allCases.count)
            var confSum = 0.0
            for (joint, name) in Self.jointNames {
                // Vision can omit a point, or return a non-finite location/confidence
                // under blur/occlusion. Keep the shared HandSample shape stable by
                // representing either case as an explicit zero-confidence point.
                guard let p = points[name] else {
                    joints[joint] = HandPoint(x: 0, y: 0, confidence: 0)
                    continue
                }
                let x = Double(p.location.x)
                let y = Double(p.location.y)
                let rawConfidence = Double(p.confidence)
                guard x.isFinite, y.isFinite, rawConfidence.isFinite else {
                    joints[joint] = HandPoint(x: 0, y: 0, confidence: 0)
                    continue
                }
                let c = min(max(rawConfidence, 0), 1)
                joints[joint] = HandPoint(x: x, y: y, confidence: c)
                confSum += c
            }
            // `joints` always has all 21 entries. Keeping the denominator fixed makes
            // mean confidence reflect detector coverage as well as point certainty.
            let meanConf = confSum / Double(HandJoint.allCases.count)
            guard meanConf >= config.minMeanConfidence else { continue }

            let chirality = Self.chiralityString(obs.chirality)
            var score = meanConf
            if let pref = config.preferredChirality, pref == chirality { score += config.preferredChiralityBonus }
            guard let palm = Self.palmCenter(joints, minConfidence: config.minPalmJointConfidence) else { continue }
            if let last = lastPalmCenter {
                let d = palm.distance(to: last)
                score += config.consistencyBonus * max(0, 1 - d / config.consistencyRadius)
            }
            if best == nil || score > best!.score {
                best = (score, HandSample(t: t, chirality: chirality, joints: joints), palm)
            }
        }
        lastPalmCenter = best?.palm
        return best?.sample
    }

    /// Forget the previously tracked hand (e.g. when tracking is paused).
    public func reset() {
        lastPalmCenter = nil
        lastObservationCount = 0
    }

    // MARK: - Helpers

    private static func chiralityString(_ c: VNChirality) -> String? {
        switch c {
        case .left: return "left"
        case .right: return "right"
        default: return nil
        }
    }

    private static func palmCenter(_ joints: [HandJoint: HandPoint], minConfidence: Double) -> Point2? {
        let palmJoints: [HandJoint] = [.wrist, .indexMCP, .middleMCP, .ringMCP, .littleMCP]
        let points = palmJoints.compactMap { joint -> Point2? in
            guard let p = joints[joint], p.confidence >= minConfidence,
                  p.x.isFinite, p.y.isFinite else { return nil }
            return p.point
        }
        // A single confident point (or a collapsed cluster) is not enough to
        // establish a hand identity or a meaningful palm center.
        guard points.count >= 2, let first = points.first,
              points.dropFirst().contains(where: { $0.distance(to: first) > 1e-4 }) else {
            return nil
        }
        let sum = points.reduce(into: Point2.zero) { partial, point in partial = partial + point }
        return sum / Double(points.count)
    }

    private static let jointNames: [(HandJoint, VNHumanHandPoseObservation.JointName)] = [
        (.wrist, .wrist),
        (.thumbCMC, .thumbCMC), (.thumbMP, .thumbMP), (.thumbIP, .thumbIP), (.thumbTip, .thumbTip),
        (.indexMCP, .indexMCP), (.indexPIP, .indexPIP), (.indexDIP, .indexDIP), (.indexTip, .indexTip),
        (.middleMCP, .middleMCP), (.middlePIP, .middlePIP), (.middleDIP, .middleDIP), (.middleTip, .middleTip),
        (.ringMCP, .ringMCP), (.ringPIP, .ringPIP), (.ringDIP, .ringDIP), (.ringTip, .ringTip),
        (.littleMCP, .littleMCP), (.littlePIP, .littlePIP), (.littleDIP, .littleDIP), (.littleTip, .littleTip),
    ]
}
