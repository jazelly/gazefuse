// VideoSelfTest.swift — `GazeFuse --gaze-video FILE [--appearance-model FILE]`
//
// Runs the real camera-side gaze pipeline (Vision landmarks, Timm–Barth refinement, optional
// Core ML appearance model) on a video file instead of the camera, prints diagnostics, exits.
// No camera / Accessibility permission needed; nothing is posted to the system.

import AVFoundation
import Foundation
import GazeFuseCore
import Vision

enum VideoSelfTest {
    static func run(path: String, modelPath: String?) -> Int32 {
        let url = URL(fileURLWithPath: path)
        let asset = AVURLAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first, let reader = try? AVAssetReader(asset: asset) else {
            print("cannot open \(path)"); return 1
        }
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange])
        reader.add(out)
        reader.startReading()
        var refined = FaceGazeExtractor()
        let plain = FaceGazeExtractor(config: .init(refinePupil: false))
        if let w = ProcessInfo.processInfo.environment["GF_MIN_EYE_PX"].flatMap(Double.init) {
            var c = FaceGazeExtractor.Config(); c.minimumEyeWidthForRefinement = w
            refined = FaceGazeExtractor(config: c)
        }
        if let dir = ProcessInfo.processInfo.environment["GF_DUMP_DIR"] {
            var n = 0
            refined.debugPatchSink = { img, mask, tb, lm in
                n += 1
                guard n % 40 == 1 else { return }
                // PGM ×4 upscale: mask dimmed outside, TB = white dot, landmark = black dot.
                let k = 4, W = img.width * k, H = img.height * k
                var bytes = [UInt8](repeating: 0, count: W * H)
                for y in 0..<H { for x in 0..<W {
                    let sx = x / k, sy = y / k
                    var v = img[sx, sy]
                    if !mask[sy * img.width + sx] { v *= 0.6 }
                    bytes[y * W + x] = UInt8(max(0, min(255, v)))
                } }
                func dot(_ p: Point2, _ val: UInt8) {
                    let cx = Int((p.x + 0.5) * Double(k)), cy = Int((p.y + 0.5) * Double(k))
                    for y in (cy - 2)...(cy + 2) { for x in (cx - 2)...(cx + 2) where x >= 0 && y >= 0 && x < W && y < H { bytes[y * W + x] = val } }
                }
                dot(tb, 255); dot(lm, 0)
                var data = Data("P5\n\(W) \(H)\n255\n".utf8); data.append(contentsOf: bytes)
                try? data.write(to: URL(fileURLWithPath: dir).appendingPathComponent("eye_\(n).pgm"))
            }
        }
        var appearance: AppearanceGazeEstimator?
        if let modelPath { appearance = try? AppearanceGazeEstimator(url: URL(fileURLWithPath: modelPath)) }
        else { appearance = AppearanceGazeEstimator.loadDefault() }
        print("appearance model: \(appearance?.modelDescription ?? "none")")
        var frames = 0, faces = 0
        var visionMs = 0.0, refineMs = 0.0, appMs = 0.0, appN = 0
        var prevR: GazeSample?, prevP: GazeSample?, prevA: Point2?
        var jitterR: [Double] = [], jitterP: [Double] = [], jitterA: [Double] = []
        var yaws: [Double] = [], pitches: [Double] = []
        func pupilAvg(_ s: GazeSample) -> Point2? {
            guard let l = s.leftPupil, let r = s.rightPupil else { return nil }
            return Point2((l.x + r.x) / 2, (l.y + r.y) / 2)
        }
        var size = CGSize.zero
        var eyeWidths: [Double] = []
        while let sb = out.copyNextSampleBuffer(), let pb = CMSampleBufferGetImageBuffer(sb) {
            frames += 1
            size = CGSize(width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))
            let t = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sb))
            let t0 = CFAbsoluteTimeGetCurrent()
            let h = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up, options: [:])
            try? h.perform(refined.requests + plain.requests)
            let t1 = CFAbsoluteTimeGetCurrent()
            refined.prepare(pixelBuffer: pb)
            let r = refined.extract(t: t, imageSize: size)
            let t2 = CFAbsoluteTimeGetCurrent()
            let p = plain.extract(t: t, imageSize: size)
            visionMs += (t1 - t0) * 1000; refineMs += (t2 - t1) * 1000
            guard let r, let p else { prevR = nil; prevP = nil; prevA = nil; continue }
            faces += 1
            if let x = r.extra, x.count > 9 { eyeWidths.append(x[8] * size.width); eyeWidths.append(x[9] * size.width) }
            if let a = pupilAvg(r), let pr = prevR.flatMap(pupilAvg) { jitterR.append((a - pr).length) }
            if let a = pupilAvg(p), let pp = prevP.flatMap(pupilAvg) { jitterP.append((a - pp).length) }
            prevR = r; prevP = p
            if let model = appearance, let box = refined.lastFaceBox, let g = model.estimate(pixelBuffer: pb, faceBox: box) {
                appMs += model.lastLatencyMs; appN += 1
                yaws.append(g.x); pitches.append(g.y)
                if let pa = prevA { jitterA.append((g - pa).length) }
                prevA = g
            } else { prevA = nil }
        }
        func med(_ v: [Double]) -> Double { v.isEmpty ? .nan : v.sorted()[v.count / 2] }
        func meanSD(_ v: [Double]) -> String {
            guard !v.isEmpty else { return "-" }
            let m = v.reduce(0, +) / Double(v.count)
            let sd = (v.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(v.count)).squareRoot()
            return String(format: "%.1f° ± %.1f°", m * 180 / .pi, sd * 180 / .pi)
        }
        print(String(format: "video %@ %.0fx%.0f: frames %d, face found %d (%.1f%%)", url.lastPathComponent, size.width, size.height, frames, faces, 100 * Double(faces) / Double(max(frames, 1))))
        print(String(format: "eye width: median %.1f px (p10 %.1f, p90 %.1f)", med(eyeWidths),
                     eyeWidths.isEmpty ? .nan : eyeWidths.sorted()[eyeWidths.count / 10], eyeWidths.isEmpty ? .nan : eyeWidths.sorted()[eyeWidths.count * 9 / 10]))
        print(String(format: "Vision (face landmarks ×2 requests) %.2f ms/frame, extract+Timm–Barth %.2f ms/frame", visionMs / Double(max(frames, 1)), refineMs / Double(max(frames, 1))))
        print(String(format: "Timm–Barth refinement: accepted %d / %d eyes (%.1f%%), mean shift from landmark %.3f eye-widths",
                     refined.refinementAccepted, refined.refinementAttempts,
                     100 * Double(refined.refinementAccepted) / Double(max(refined.refinementAttempts, 1)),
                     refined.refinementShiftSum / Double(max(refined.refinementAccepted, 1))))
        print(String(format: "median frame-to-frame pupil-offset change: landmark %.4f, refined %.4f eye-widths (lower = less jitter; includes real eye motion)", med(jitterP), med(jitterR)))
        if appN > 0 {
            print(String(format: "appearance model: %d frames, %.2f ms/frame, yaw %@, pitch %@, median frame-to-frame change %.2f°",
                         appN, appMs / Double(appN), meanSD(yaws), meanSD(pitches), med(jitterA) * 180 / .pi))
        }
        return 0
    }
}
