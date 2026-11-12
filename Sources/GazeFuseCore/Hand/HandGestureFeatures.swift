// HandGestureFeatures.swift — scale-invariant, hardware-free features computed from one
// `HandSample` (21 Vision joints). Used by `PinchGestureClassifier` and available to
// future gestures (open palm = clutch, fist, pointing, ...).
//
// Everything is normalized by `handSize`, a reference length that does not change when
// the fingers move:
//   handSize = max( |wrist - middleMCP| , |indexMCP - littleMCP| / palmWidthToLengthRatio )
// The two palm axes are roughly orthogonal, so when the hand tilts and one axis
// foreshortens, the other may remain useful. This reduces, but cannot eliminate,
// projection error. If the wrist is out of frame (hand close to the camera)
// the palm width alone is used, and vice versa.
//
// Finger extension is measured by the *curl ratio* |tip - MCP| / (sum of the three phalanx
// lengths): 1.0 for a straight finger, ~0.7 for a pinching index finger, ~0.4 for a fist.
// A 2-D projection cannot reliably distinguish all curled, overlapping or camera-facing
// fingers; see docs/agent-notes/hand.md for this limitation.
import Foundation

public enum HandFinger: Int, Codable, CaseIterable {
    case thumb, index, middle, ring, little
}

public enum HandFingerState: String, Codable {
    case extended, curled
    /// Not enough confident joints, or the curl ratio lies in the ambiguous band.
    case unknown
}

public struct HandGestureFeatures {
    public struct Config: Codable, Equatable {
        /// Joints below this Vision confidence are ignored (treated as missing).
        public var minJointConfidence: Double = 0.3
        /// Anthropometric ratio palm width (indexMCP–littleMCP) / palm length (wrist–middleMCP),
        /// used to put both hand-size estimates in the same unit.
        public var palmWidthToLengthRatio: Double = 0.72
        /// Curl ratio at or above which a finger counts as `.extended`.
        public var extendedCurlRatio: Double = 0.85
        /// Curl ratio at or below which a finger counts as `.curled`.
        public var curledCurlRatio: Double = 0.55
        /// For `isOpenPalm` the thumb tip must be at least this far (hand-size units) from the
        /// index and middle tips, i.e. the thumb is abducted and nothing is pinched.
        public var openPalmMinThumbDistance: Double = 0.6
        public init() {}
    }

    /// One thumb-tip ↔ finger-tip pair.
    public struct Pinch: Equatable {
        /// Tip-to-tip distance in hand-size units (≈ 1.0 open, ≈ 0.05–0.2 touching).
        public var distance: Double
        /// Midpoint of the two tips, normalized image units.
        public var point: Point2
        /// min(thumbTip confidence, fingerTip confidence, mean palm confidence).
        public var confidence: Double
        /// Thumb tip to the same finger's DIP / PIP joint (hand-size units; nil if not confident).
        /// In a real pinch the thumb touches the finger*tip*, so the tip is its nearest joint;
        /// in a fist the thumb wraps over the middle phalanx and the DIP/PIP is nearer.
        public var dipDistance: Double?
        public var pipDistance: Double?
        /// True when the fingertip is at least as near to the thumb tip as the DIP and PIP.
        public var tipIsNearest: Bool {
            distance <= (dipDistance ?? .infinity) && distance <= (pipDistance ?? .infinity)
        }
    }

    /// Reference length in normalized image units (see file comment).
    public let handSize: Double
    /// Mean of the confident palm joints (wrist + four MCPs), normalized image units.
    public let palmCenter: Point2
    /// Mean confidence of the joints that went into `handSize` / `palmCenter`.
    public let palmConfidence: Double
    /// Thumb–index pair; nil when either tip is not confident.
    public let index: Pinch?
    /// Thumb–middle pair; nil when either tip is not confident.
    public let middle: Pinch?

    public let thumbCurl: Double?
    public let indexCurl: Double?
    public let middleCurl: Double?
    public let ringCurl: Double?
    public let littleCurl: Double?
    public let thumbState: HandFingerState
    public let indexState: HandFingerState
    public let middleState: HandFingerState
    public let ringState: HandFingerState
    public let littleState: HandFingerState

    /// Four fingers extended, thumb not curled, thumb far from the index/middle tips.
    /// Intended as a "clutch / pause tracking" signal.
    public let isOpenPalm: Bool
    /// Index, middle, ring and little all curled.
    public let isFist: Bool

    public func curl(of finger: HandFinger) -> Double? {
        switch finger {
        case .thumb: return thumbCurl
        case .index: return indexCurl
        case .middle: return middleCurl
        case .ring: return ringCurl
        case .little: return littleCurl
        }
    }

    public func state(of finger: HandFinger) -> HandFingerState {
        switch finger {
        case .thumb: return thumbState
        case .index: return indexState
        case .middle: return middleState
        case .ring: return ringState
        case .little: return littleState
        }
    }

    /// Returns nil when the hand size cannot be estimated (palm joints missing / degenerate).
    public init?(sample: HandSample, config: Config = Config()) {
        let joints = sample.joints
        let minC = config.minJointConfidence
        @inline(__always) func pt(_ j: HandJoint) -> HandPoint? {
            guard let p = joints[j], p.confidence.isFinite, p.confidence >= minC,
                  p.x.isFinite, p.y.isFinite else { return nil }
            return p
        }

        // --- hand size & palm centre -------------------------------------------------
        let wrist = pt(.wrist), midMCP = pt(.middleMCP), idxMCP = pt(.indexMCP), litMCP = pt(.littleMCP)
        var size = 0.0
        if let w = wrist, let m = midMCP { size = max(size, w.point.distance(to: m.point)) }
        if let a = idxMCP, let b = litMCP { size = max(size, a.point.distance(to: b.point) / config.palmWidthToLengthRatio) }
        guard size > 1e-4 else { return nil }
        handSize = size

        var cx = 0.0, cy = 0.0, cc = 0.0, n = 0
        func accumulate(_ p: HandPoint?) {
            if let p = p { cx += p.x; cy += p.y; cc += p.confidence; n += 1 }
        }
        accumulate(wrist); accumulate(idxMCP); accumulate(midMCP); accumulate(pt(.ringMCP)); accumulate(litMCP)
        // size > 0 guarantees at least two confident palm joints, so n >= 2.
        palmCenter = Point2(cx / Double(n), cy / Double(n))
        let palmConfidence = cc / Double(n)
        self.palmConfidence = palmConfidence

        // --- pinch pairs ----------------------------------------------------------------
        let thumbTip = pt(.thumbTip)
        func pinch(_ tip: HandPoint?, _ dip: HandPoint?, _ pip: HandPoint?) -> Pinch? {
            guard let t = thumbTip, let f = tip else { return nil }
            return Pinch(distance: t.point.distance(to: f.point) / size,
                         point: (t.point + f.point) / 2,
                         confidence: min(t.confidence, f.confidence, palmConfidence),
                         dipDistance: dip.map { t.point.distance(to: $0.point) / size },
                         pipDistance: pip.map { t.point.distance(to: $0.point) / size })
        }
        index = pinch(pt(.indexTip), pt(.indexDIP), pt(.indexPIP))
        middle = pinch(pt(.middleTip), pt(.middleDIP), pt(.middlePIP))

        // --- finger curl ----------------------------------------------------------------
        func curl(_ a: HandJoint, _ b: HandJoint, _ c: HandJoint, _ d: HandJoint) -> Double? {
            guard let p0 = pt(a), let p1 = pt(b), let p2 = pt(c), let p3 = pt(d) else { return nil }
            let len = p0.point.distance(to: p1.point) + p1.point.distance(to: p2.point) + p2.point.distance(to: p3.point)
            guard len > 1e-6 else { return nil }
            return min(1, p0.point.distance(to: p3.point) / len)
        }
        func state(_ r: Double?) -> HandFingerState {
            guard let r = r else { return .unknown }
            if r >= config.extendedCurlRatio { return .extended }
            if r <= config.curledCurlRatio { return .curled }
            return .unknown
        }
        thumbCurl = curl(.thumbCMC, .thumbMP, .thumbIP, .thumbTip)
        indexCurl = curl(.indexMCP, .indexPIP, .indexDIP, .indexTip)
        middleCurl = curl(.middleMCP, .middlePIP, .middleDIP, .middleTip)
        ringCurl = curl(.ringMCP, .ringPIP, .ringDIP, .ringTip)
        littleCurl = curl(.littleMCP, .littlePIP, .littleDIP, .littleTip)
        thumbState = state(thumbCurl)
        indexState = state(indexCurl)
        middleState = state(middleCurl)
        ringState = state(ringCurl)
        littleState = state(littleCurl)

        let fingersExtended = indexState == .extended && middleState == .extended
            && ringState == .extended && littleState == .extended
        let thumbAway = (index?.distance ?? 0) >= config.openPalmMinThumbDistance
            && (middle?.distance ?? 0) >= config.openPalmMinThumbDistance
        isOpenPalm = fingersExtended && thumbState == .extended && thumbAway
        isFist = indexState == .curled && middleState == .curled && ringState == .curled && littleState == .curled
    }
}
