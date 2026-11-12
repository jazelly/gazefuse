// FaceGazeExtractor.swift
//
// Vision face landmarks -> the small, hardware-independent GazeSample contract.
// The camera pipeline owns the serial queue on which `prepare`, Vision's request,
// and `extract` are called.  No frame is retained after `extract` returns.

import AVFoundation
import CoreGraphics
import Foundation
import GazeFuseCore
import Vision

/// Converts a Vision face-landmark observation into a calibrated-gaze input.
///
/// Vision landmark points are relative to the face bounding box.  This extractor
/// converts them to un-mirrored image-normalised coordinates (bottom-left origin),
/// then expresses each pupil offset in eye-width units.  The same `VNRequest` is
/// exposed through `requests` so the camera loop can perform it together with its
/// other requests.
public final class FaceGazeExtractor {
    /// Runtime tunables.  The defaults are deliberately conservative for a
    /// 1280x720/1920x1080 full-range Y plane and a 30--60 Hz camera stream.
    public struct Config: Codable, Equatable {
        /// Refine Vision's pupil landmark using a tiny dark-pixel centroid around
        /// the landmark.  Disable this when the camera format has no Y plane.
        public var refinePupil: Bool
        /// Search radius as a fraction of the eye width.
        public var refinementRadiusFraction: Double
        /// Pixel stride used by the refinement scan.  2 keeps the scan below a
        /// couple of milliseconds at ordinary webcam resolutions.
        public var refinementStride: Int
        /// Minimum luminance contrast below the local ROI mean before a pixel is
        /// allowed to contribute to the dark-pupil centroid.
        public var refinementMinContrast: Double
        /// Minimum weighted dark-pixel mass before a refined centre is accepted.
        public var refinementMinimumWeight: Double
        /// Faces below this confidence are ignored by `extract`.
        public var minimumFaceConfidence: Double
        /// Refinement algorithm: Timm–Barth gradient centre (default) or the dark-pixel centroid.
        public var useTimmBarth: Bool = true
        /// Eye patch is resampled to this width before Timm–Barth (cost ∝ width⁴).
        public var timmBarthPatchWidth: Int = 40
        /// A refined centre farther than this (eye widths) from Vision's landmark is rejected.
        public var maximumRefinementShift: Double = 0.25
        /// Skip refinement on nearly closed eyes (openness = height / width).
        public var minimumOpennessForRefinement: Double = 0.12

        public init(refinePupil: Bool = true,
                    refinementRadiusFraction: Double = 0.34,
                    refinementStride: Int = 2,
                    refinementMinContrast: Double = 18,
                    refinementMinimumWeight: Double = 8,
                    minimumFaceConfidence: Double = 0.05) {
            self.refinePupil = refinePupil
            self.refinementRadiusFraction = max(0.05, refinementRadiusFraction)
            self.refinementStride = max(1, refinementStride)
            self.refinementMinContrast = max(0, refinementMinContrast)
            self.refinementMinimumWeight = max(0, refinementMinimumWeight)
            self.minimumFaceConfidence = min(max(0, minimumFaceConfidence), 1)
        }
    }

    /// The request can be passed directly to `VNImageRequestHandler.perform`.
    public let requests: [VNRequest]
    public let config: Config

    private final class ObservationStore {
        var face: VNFaceObservation?
    }

    private struct EyeMetrics {
        var contour: [CGPoint]
        var centre: CGPoint
        var pupil: CGPoint?
        var axisA: CGPoint
        var axisB: CGPoint
        var widthPixels: Double
        var heightPixels: Double
        var perpendicularX: Double
        var perpendicularY: Double
        var perpendicularMinimum: Double
        var openness: Double?
    }

    private let store: ObservationStore
    private var pendingPixelBuffer: CVPixelBuffer?
    private let landmarksRequest: VNDetectFaceLandmarksRequest

    public init(config: Config = Config()) {
        self.config = config
        let observationStore = ObservationStore()
        self.store = observationStore

        // `currentRevision` is the latest revision known to the linked Vision
        // framework.  Keeping the closure on a separate store avoids retaining
        // the extractor through the request and permits safe initialisation.
        let request = VNDetectFaceLandmarksRequest { request, error in
            guard error == nil,
                  let landmarksRequest = request as? VNDetectFaceLandmarksRequest else {
                observationStore.face = nil
                return
            }

            // If more than one face is present, use the largest face.  GazeFuse's
            // cursor is single-user; this is deterministic and avoids a tracking
            // identity cache that would add state to the camera hot path.
            observationStore.face = landmarksRequest.results?
                .max { lhs, rhs in
                    let l = lhs.boundingBox.width * lhs.boundingBox.height
                    let r = rhs.boundingBox.width * rhs.boundingBox.height
                    return l < r
                }
        }
        request.revision = VNDetectFaceLandmarksRequest.supportedRevisions.max()
            ?? VNDetectFaceLandmarksRequest.currentRevision
        self.landmarksRequest = request
        self.requests = [request]
    }

    /// Supply the current pixel buffer before performing `requests` when the
    /// optional pupil refinement is enabled.  The buffer is used only by the
    /// immediately following `extract` call and is released there.  Calling
    /// `prepare` again replaces the previous frame, so a dropped frame cannot
    /// accumulate retained buffers.
    public func prepare(pixelBuffer: CVPixelBuffer) {
        pendingPixelBuffer = pixelBuffer
    }

    /// Explicitly release a prepared frame when the camera pipeline abandons a
    /// request without calling `extract` (for example after cancellation).
    public func clearPreparedFrame() {
        pendingPixelBuffer = nil
    }

    /// Extract the latest Vision result.  `imageSize` is the pixel size used by
    /// the request (not the AppKit display size).  The camera image is assumed to
    /// be un-mirrored and `.up`, matching the shared image-space convention.
    public func extract(t: TimeSec, imageSize: CGSize) -> GazeSample? {
        defer { pendingPixelBuffer = nil }
        guard imageSize.width > 1, imageSize.height > 1,
              let face = store.face,
              let landmarks = face.landmarks else { return nil }

        let rawFaceConfidence = Double(face.confidence)
        guard rawFaceConfidence >= config.minimumFaceConfidence else { return nil }

        let box = face.boundingBox.standardized
        let left = makeEyeMetrics(landmarks.leftEye,
                                  pupilRegion: landmarks.leftPupil,
                                  faceBox: box,
                                  imageSize: imageSize)
        let right = makeEyeMetrics(landmarks.rightEye,
                                   pupilRegion: landmarks.rightPupil,
                                   faceBox: box,
                                   imageSize: imageSize)
        guard left != nil || right != nil else { return nil }

        var leftEye = left
        var rightEye = right

        // The scan is intentionally local and bounded.  It is a pragmatic
        // refinement of Vision's landmark, not a full Timm--Barth gradient
        // implementation; use that method in a future image-processing module if
        // a larger CPU budget is available.
        if config.refinePupil, let pixelBuffer = pendingPixelBuffer {
            func refine(_ eye: EyeMetrics?) -> EyeMetrics? {
                guard var e = eye, let pupil = e.pupil, (e.openness ?? 1) >= config.minimumOpennessForRefinement else { return eye }
                let refined = config.useTimmBarth
                    ? refinePupilTimmBarth(pupil, eye: e, pixelBuffer: pixelBuffer, imageSize: imageSize)
                    : refinePupil(pupil, eye: e, pixelBuffer: pixelBuffer, imageSize: imageSize)
                if let refined {
                    let dx = Double(refined.x - pupil.x) * imageSize.width, dy = Double(refined.y - pupil.y) * imageSize.height
                    if (dx * dx + dy * dy).squareRoot() <= config.maximumRefinementShift * e.widthPixels { e.pupil = refined }
                }
                return e
            }
            leftEye = refine(leftEye)
            rightEye = refine(rightEye)
        }

        let leftPupil = relativePupil(leftEye, imageSize: imageSize)
        let rightPupil = relativePupil(rightEye, imageSize: imageSize)
        let faceCentre = Point2(box.midX, box.midY)

        var confidence = rawFaceConfidence
        if let quality = face.faceCaptureQuality {
            confidence *= min(max(Double(quality), 0), 1)
        }
        if leftEye?.pupil == nil { confidence *= 0.72 }
        if rightEye?.pupil == nil { confidence *= 0.72 }
        confidence = min(max(confidence, 0), 1)

        let extras = makeExtras(left: leftEye, right: rightEye, faceBox: box, imageSize: imageSize)
        return GazeSample(
            t: t,
            leftPupil: leftPupil,
            rightPupil: rightPupil,
            leftEyeOpenness: leftEye?.openness,
            rightEyeOpenness: rightEye?.openness,
            headYaw: face.yaw?.doubleValue,
            headPitch: face.pitch?.doubleValue,
            headRoll: face.roll?.doubleValue,
            faceCenter: faceCentre,
            faceSize: max(0, box.width),
            confidence: confidence,
            extra: extras
        )
    }

    // MARK: - Landmark conversion

    private func makeEyeMetrics(_ contourRegion: VNFaceLandmarkRegion2D?,
                                pupilRegion: VNFaceLandmarkRegion2D?,
                                faceBox: CGRect,
                                imageSize: CGSize) -> EyeMetrics? {
        guard let contourRegion,
              let contourPoints = points(contourRegion, faceBox: faceBox),
              contourPoints.count >= 2 else { return nil }

        var sumX = 0.0
        var sumY = 0.0
        for point in contourPoints {
            sumX += point.x; sumY += point.y
        }
        let centre = CGPoint(x: sumX / CGFloat(contourPoints.count),
                             y: sumY / CGFloat(contourPoints.count))

        // The farthest pair approximates the eye corners.  Measuring the span
        // perpendicular to this axis makes openness invariant to head roll;
        // an axis-aligned bounding box would report a falsely open eye when the
        // face is rotated.
        var farthestSquared = 0.0
        var axisA = contourPoints[0]
        var axisB = contourPoints[1]
        for i in 0..<(contourPoints.count - 1) {
            for j in (i + 1)..<contourPoints.count {
                let dx = Double(contourPoints[j].x - contourPoints[i].x) * Double(imageSize.width)
                let dy = Double(contourPoints[j].y - contourPoints[i].y) * Double(imageSize.height)
                let d2 = dx * dx + dy * dy
                if d2 > farthestSquared {
                    farthestSquared = d2
                    axisA = contourPoints[i]
                    axisB = contourPoints[j]
                }
            }
        }
        let widthPixels = farthestSquared.squareRoot()
        guard widthPixels > 0.5 else { return nil }
        let axisDX = (Double(axisB.x) - Double(axisA.x)) * Double(imageSize.width) / widthPixels
        let axisDY = (Double(axisB.y) - Double(axisA.y)) * Double(imageSize.height) / widthPixels
        let perpendicularX = -axisDY
        let perpendicularY = axisDX
        let centreXPixels = Double(centre.x) * Double(imageSize.width)
        let centreYPixels = Double(centre.y) * Double(imageSize.height)
        var minimumPerpendicular = Double.greatestFiniteMagnitude
        var maximumPerpendicular = -Double.greatestFiniteMagnitude
        for point in contourPoints {
            let dx = Double(point.x) * Double(imageSize.width) - centreXPixels
            let dy = Double(point.y) * Double(imageSize.height) - centreYPixels
            let projection = dx * perpendicularX + dy * perpendicularY
            minimumPerpendicular = min(minimumPerpendicular, projection)
            maximumPerpendicular = max(maximumPerpendicular, projection)
        }
        let heightPixels = max(0, maximumPerpendicular - minimumPerpendicular)
        let openness: Double = heightPixels / widthPixels
        var pupil: CGPoint?
        if let pupilRegion,
           let pupilPoints = points(pupilRegion, faceBox: faceBox),
           !pupilPoints.isEmpty {
            let px = pupilPoints.reduce(0.0) { $0 + $1.x } / CGFloat(pupilPoints.count)
            let py = pupilPoints.reduce(0.0) { $0 + $1.y } / CGFloat(pupilPoints.count)
            pupil = CGPoint(x: px, y: py)
        }

        return EyeMetrics(contour: contourPoints,
                          centre: centre,
                          pupil: pupil,
                          axisA: axisA,
                          axisB: axisB,
                          widthPixels: widthPixels,
                          heightPixels: heightPixels,
                          perpendicularX: perpendicularX,
                          perpendicularY: perpendicularY,
                          perpendicularMinimum: minimumPerpendicular,
                          openness: openness)
    }

    private func points(_ region: VNFaceLandmarkRegion2D, faceBox: CGRect) -> [CGPoint]? {
        guard region.pointCount > 0 else { return nil }
        let pointer = region.normalizedPoints
        return (0..<min(region.pointCount, pointer.count)).map { index in
            let point = pointer[index]
            return CGPoint(x: faceBox.origin.x + point.x * faceBox.width,
                           y: faceBox.origin.y + point.y * faceBox.height)
        }
    }

    private func relativePupil(_ eye: EyeMetrics?, imageSize: CGSize) -> Point2? {
        guard let eye, let pupil = eye.pupil, eye.widthPixels > 0 else { return nil }
        let dx = (pupil.x - eye.centre.x) * imageSize.width / eye.widthPixels
        let dy = (pupil.y - eye.centre.y) * imageSize.height / eye.widthPixels
        // Vision's image coordinates already have +y up; no camera mirroring or
        // top-left conversion is applied here.
        return Point2(Double(dx), Double(dy))
    }

    /// Stable 24-element extras vector.  Missing measurements are represented by
    /// zero plus an explicit presence flag at indices 22 and 23.
    private func makeExtras(left: EyeMetrics?, right: EyeMetrics?, faceBox: CGRect, imageSize: CGSize) -> [Double] {
        var x = [Double](repeating: 0, count: 24)
        func cornerNormalised(_ eye: EyeMetrics?) -> Point2? {
            guard let eye, let pupil = eye.pupil, eye.widthPixels > 0 else { return nil }
            let axisX = (Double(eye.axisB.x) - Double(eye.axisA.x)) * Double(imageSize.width) / eye.widthPixels
            let axisY = (Double(eye.axisB.y) - Double(eye.axisA.y)) * Double(imageSize.height) / eye.widthPixels
            let pupilX = Double(pupil.x) * Double(imageSize.width)
            let pupilY = Double(pupil.y) * Double(imageSize.height)
            let axisOriginX = Double(eye.axisA.x) * Double(imageSize.width)
            let axisOriginY = Double(eye.axisA.y) * Double(imageSize.height)
            let along = ((pupilX - axisOriginX) * axisX + (pupilY - axisOriginY) * axisY) / eye.widthPixels
            let centreX = Double(eye.centre.x) * Double(imageSize.width)
            let centreY = Double(eye.centre.y) * Double(imageSize.height)
            let perpendicular = (pupilX - centreX) * eye.perpendicularX + (pupilY - centreY) * eye.perpendicularY
            let across = eye.heightPixels > 0
                ? (perpendicular - eye.perpendicularMinimum) / eye.heightPixels
                : 0.5
            return Point2(min(max(along, 0), 1), min(max(across, 0), 1))
        }
        if let p = cornerNormalised(left) { x[0] = p.x; x[1] = p.y }
        if let p = cornerNormalised(right) { x[2] = p.x; x[3] = p.y }
        if let eye = left {
            x[4] = Double(eye.centre.x); x[5] = Double(eye.centre.y)
            x[8] = eye.widthPixels / imageSize.width
            x[10] = eye.openness ?? 0
            if let pupil = eye.pupil {
                x[18] = (Double(pupil.x) - Double(eye.centre.x)) * imageSize.width / eye.widthPixels
                x[19] = (Double(pupil.y) - Double(eye.centre.y)) * imageSize.height / eye.widthPixels
            }
        }
        if let eye = right {
            x[6] = Double(eye.centre.x); x[7] = Double(eye.centre.y)
            x[9] = eye.widthPixels / imageSize.width
            x[11] = eye.openness ?? 0
            if let pupil = eye.pupil {
                x[20] = (Double(pupil.x) - Double(eye.centre.x)) * imageSize.width / eye.widthPixels
                x[21] = (Double(pupil.y) - Double(eye.centre.y)) * imageSize.height / eye.widthPixels
            }
        }
        if let l = left, let r = right {
            let dx = (Double(r.centre.x) - Double(l.centre.x)) * imageSize.width
            let dy = (Double(r.centre.y) - Double(l.centre.y)) * imageSize.height
            x[12] = (dx * dx + dy * dy).squareRoot() / (imageSize.width * imageSize.width + imageSize.height * imageSize.height).squareRoot()
            x[13] = Double(r.centre.x - l.centre.x)
            x[14] = Double(r.centre.y - l.centre.y)
        }
        x[15] = Double(faceBox.midX); x[16] = Double(faceBox.midY); x[17] = Double(faceBox.width)
        x[22] = left?.pupil == nil ? 0 : 1
        x[23] = right?.pupil == nil ? 0 : 1
        return x
    }

    // MARK: - Optional luminance refinement

    private func refinePupil(_ rough: CGPoint,
                             eye: EyeMetrics,
                             pixelBuffer: CVPixelBuffer,
                             imageSize: CGSize) -> CGPoint? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard width > 2, height > 2,
              CVPixelBufferGetPlaneCount(pixelBuffer) > 0,
              Int(imageSize.width.rounded()) == width,
              Int(imageSize.height.rounded()) == height,
              CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange else { return nil }

        let radiusX = max(2, eye.widthPixels * config.refinementRadiusFraction)
        let radiusY = max(2, radiusX * imageSize.height / imageSize.width)
        let centreX = rough.x * imageSize.width
        let centreYTop = (1 - rough.y) * imageSize.height
        let x0 = max(0, Int(centreX - radiusX))
        let x1 = min(width - 1, Int(centreX + radiusX))
        let y0 = max(0, Int(centreYTop - radiusY))
        let y1 = min(height - 1, Int(centreYTop + radiusY))
        guard x1 > x0, y1 > y0 else { return nil }

        let step = max(1, config.refinementStride)
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else { return nil }
        let stride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let pixels = baseAddress.assumingMemoryBound(to: UInt8.self)

        var sum = 0.0
        var count = 0.0
        var y = y0
        while y <= y1 {
            let row = pixels.advanced(by: y * stride)
            var x = x0
            while x <= x1 {
                let dx = (Double(x) - centreX) / radiusX
                let dy = (Double(y) - centreYTop) / radiusY
                if dx * dx + dy * dy <= 1,
                   isInsideEyeContour(x: x, y: y, eye: eye, imageSize: imageSize) {
                    sum += Double(row[x]); count += 1
                }
                x += step
            }
            y += step
        }
        guard count >= 4 else { return nil }
        let threshold = max(8, sum / count - config.refinementMinContrast)

        var weighted = 0.0
        var weightedX = 0.0
        var weightedY = 0.0
        y = y0
        while y <= y1 {
            let row = pixels.advanced(by: y * stride)
            var x = x0
            while x <= x1 {
                let dx = (Double(x) - centreX) / radiusX
                let dy = (Double(y) - centreYTop) / radiusY
                if dx * dx + dy * dy <= 1,
                   isInsideEyeContour(x: x, y: y, eye: eye, imageSize: imageSize) {
                    let weight = max(0, threshold - Double(row[x]))
                    weighted += weight
                    weightedX += weight * Double(x)
                    weightedY += weight * Double(y)
                }
                x += step
            }
            y += step
        }
        guard weighted >= config.refinementMinimumWeight else { return nil }
        return CGPoint(x: weightedX / weighted / imageSize.width,
                       y: 1 - weightedY / weighted / imageSize.height)
    }

    /// Timm–Barth gradient eye-centre refinement on the Y plane (see GazeFuseCore.PupilLocator).
    private func refinePupilTimmBarth(_ rough: CGPoint, eye: EyeMetrics, pixelBuffer: CVPixelBuffer,
                                      imageSize: CGSize) -> CGPoint? {
        let width = CVPixelBufferGetWidth(pixelBuffer), height = CVPixelBufferGetHeight(pixelBuffer)
        guard width > 2, height > 2, CVPixelBufferGetPlaneCount(pixelBuffer) > 0,
              Int(imageSize.width.rounded()) == width, Int(imageSize.height.rounded()) == height,
              CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
              eye.contour.count >= 3 else { return nil }
        // Contour bounding box in top-left pixel coordinates, padded.
        let xs = eye.contour.map { Double($0.x) * Double(width) }
        let ys = eye.contour.map { (1 - Double($0.y)) * Double(height) }
        let pad = 0.12 * eye.widthPixels
        let x0 = max(0, Int((xs.min()! - pad).rounded(.down))), x1 = min(width - 1, Int((xs.max()! + pad).rounded(.up)))
        let y0 = max(0, Int((ys.min()! - pad).rounded(.down))), y1 = min(height - 1, Int((ys.max()! + pad).rounded(.up)))
        let cw = x1 - x0 + 1, ch = y1 - y0 + 1
        guard cw >= 8, ch >= 4 else { return nil }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else { return nil }
        let stride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let px = base.assumingMemoryBound(to: UInt8.self)
        var crop = [Double](repeating: 0, count: cw * ch)
        for y in 0..<ch {
            let row = px.advanced(by: (y0 + y) * stride + x0)
            for x in 0..<cw { crop[y * cw + x] = Double(row[x]) }
        }
        let tw = max(12, min(config.timmBarthPatchWidth, cw))
        let th = max(6, Int((Double(ch) * Double(tw) / Double(cw)).rounded()))
        let patch = GrayImage(width: cw, height: ch, pixels: crop).resized(width: tw, height: th)
        let sx = Double(cw) / Double(tw), sy = Double(ch) / Double(th)
        var mask = [Bool](repeating: false, count: tw * th)
        for y in 0..<th { for x in 0..<tw {
            let srcX = Int(Double(x0) + (Double(x) + 0.5) * sx), srcY = Int(Double(y0) + (Double(y) + 0.5) * sy)
            mask[y * tw + x] = isInsideEyeContour(x: srcX, y: srcY, eye: eye, imageSize: imageSize)
        } }
        guard mask.filter({ $0 }).count >= 12,
              let r = PupilLocator.timmBarth(patch, mask: mask) else { return nil }
        let fx = Double(x0) + (r.center.x + 0.5) * sx, fy = Double(y0) + (r.center.y + 0.5) * sy
        return CGPoint(x: fx / Double(width), y: 1 - fy / Double(height))
    }

    /// Ray-casting test in normalized Vision image space.  Limiting the dark
    /// centroid to the detected eye polygon keeps eyelids/eyebrows outside the
    /// contour from winning the refinement scan.
    private func isInsideEyeContour(x: Int, y: Int, eye: EyeMetrics, imageSize: CGSize) -> Bool {
        guard eye.contour.count >= 3 else { return true }
        let pointX = Double(x) / Double(imageSize.width)
        let pointY = 1 - Double(y) / Double(imageSize.height)
        var inside = false
        var j = eye.contour.count - 1
        for i in 0..<eye.contour.count {
            let xi = Double(eye.contour[i].x)
            let yi = Double(eye.contour[i].y)
            let xj = Double(eye.contour[j].x)
            let yj = Double(eye.contour[j].y)
            let crosses = (yi > pointY) != (yj > pointY)
            if crosses {
                let intersection = (xj - xi) * (pointY - yi) / (yj - yi) + xi
                if pointX < intersection { inside.toggle() }
            }
            j = i
        }
        return inside
    }
}
