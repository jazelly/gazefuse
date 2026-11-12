// AppearanceGazeEstimator.swift
//
// Optional appearance-based (CNN) gaze: a Core ML model you convert locally with
// scripts/convert_gaze_model.py (GazeFuse ships no weights — see the licence note there).
// Contract: image input 448×448 RGB face crop → output "gaze" = [yaw, pitch] radians.
// The yaw/pitch go into GazeSample.appearanceGaze; the ridge calibration decides (by
// leave-one-target-out CV) whether they improve the personal mapping.

import CoreML
import CoreVideo
import Foundation
import GazeFuseCore
import Vision

final class AppearanceGazeEstimator {
    static var supportDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("GazeFuse", isDirectory: true)
    }
    /// Looked up in order: GazeAppearance.mlmodelc, GazeAppearance.mlpackage (compiled on first use).
    static var defaultCandidates: [URL] {
        [supportDirectory.appendingPathComponent("GazeAppearance.mlmodelc"),
         supportDirectory.appendingPathComponent("GazeAppearance.mlpackage")]
    }

    private let request: VNCoreMLRequest
    let modelDescription: String
    /// Square crop side relative to the larger face-box side (L2CS-style face crops are loose).
    var cropScale: Double = 1.15
    private(set) var lastLatencyMs: Double = 0

    /// Returns nil when no model is installed or it does not match the contract.
    static func loadDefault(log: ((String) -> Void)? = nil) -> AppearanceGazeEstimator? {
        for url in defaultCandidates where FileManager.default.fileExists(atPath: url.path) {
            do { return try AppearanceGazeEstimator(url: url) }
            catch { log?("Appearance gaze model at \(url.lastPathComponent) not usable: \(error.localizedDescription)") }
        }
        return nil
    }

    init(url: URL) throws {
        var compiled = url
        if url.pathExtension == "mlpackage" || url.pathExtension == "mlmodel" {
            let cached = url.deletingPathExtension().appendingPathExtension("mlmodelc")
            let srcDate = (try? FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date) ?? .distantPast
            let dstDate = (try? FileManager.default.attributesOfItem(atPath: cached.path)[.modificationDate] as? Date) ?? .distantPast
            if dstDate < srcDate {
                let tmp = try MLModel.compileModel(at: url)
                try? FileManager.default.removeItem(at: cached)
                do { try FileManager.default.moveItem(at: tmp, to: cached); compiled = cached } catch { compiled = tmp }
            } else {
                compiled = cached
            }
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        let ml = try MLModel(contentsOf: compiled, configuration: cfg)
        guard ml.modelDescription.outputDescriptionsByName["gaze"] != nil,
              let input = ml.modelDescription.inputDescriptionsByName.values.first, input.type == .image else {
            throw NSError(domain: "GazeFuse", code: 2, userInfo: [NSLocalizedDescriptionKey: "model must take an image and output 'gaze'"])
        }
        modelDescription = ml.modelDescription.metadata[.description] as? String ?? compiled.lastPathComponent
        request = VNCoreMLRequest(model: try VNCoreMLModel(for: ml))
        request.imageCropAndScaleOption = .scaleFill
    }

    /// `faceBox`: Vision-normalized (bottom-left origin) face bounding box in `pixelBuffer`.
    func estimate(pixelBuffer: CVPixelBuffer, faceBox: CGRect) -> Point2? {
        let w = Double(CVPixelBufferGetWidth(pixelBuffer)), h = Double(CVPixelBufferGetHeight(pixelBuffer))
        let side = max(faceBox.width * w, faceBox.height * h) * cropScale
        guard side >= 48 else { return nil }
        var roi = CGRect(x: faceBox.midX - side / w / 2, y: faceBox.midY - side / h / 2, width: side / w, height: side / h)
        roi = roi.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        guard roi.width > 0.02, roi.height > 0.02 else { return nil }
        request.regionOfInterest = roi
        let t0 = CFAbsoluteTimeGetCurrent()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        guard (try? handler.perform([request])) != nil,
              let obs = request.results?.first as? VNCoreMLFeatureValueObservation,
              let arr = obs.featureValue.multiArrayValue, arr.count >= 2 else { return nil }
        lastLatencyMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        let yaw = arr[0].doubleValue, pitch = arr[1].doubleValue
        guard yaw.isFinite, pitch.isFinite else { return nil }
        return Point2(yaw, pitch)
    }
}
