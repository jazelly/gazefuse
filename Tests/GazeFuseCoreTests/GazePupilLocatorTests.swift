import XCTest
@testable import GazeFuseCore

/// Synthetic webcam eye patches: skin, almond eye opening, sclera, textured iris, pupil,
/// eyelash band on the upper lid, specular glint, partial lid occlusion, blur and noise.
struct SyntheticEyePatch {
    var image: GrayImage
    var mask: [Bool]
    var truth: Point2      // iris centre, patch pixels
    var eyeWidth: Double   // pixels

    static func make(rng: inout GazeTestRNG, eyeWidthPx: Double, noise: Double, glint: Bool, lashes: Bool) -> SyntheticEyePatch {
        let W = Int((eyeWidthPx * 1.3).rounded()), H = Int((eyeWidthPx * 0.75).rounded())
        let a = eyeWidthPx / 2, ex = Double(W) / 2, ey = Double(H) / 2
        let b = eyeWidthPx * (0.17 + 0.06 * rng.uniform())      // opening half-height
        let ri = eyeWidthPx * (0.20 + 0.03 * rng.uniform())    // iris radius
        let cx = ex + (rng.uniform() * 2 - 1) * 0.45 * a
        let cy = ey + (rng.uniform() * 2 - 1) * 0.35 * b
        let irisI = 60 + 60 * rng.uniform(), pupilI = 20 + 10 * rng.uniform()
        let sclera = 185 + 30 * rng.uniform(), skin = 130 + 50 * rng.uniform()
        let gx = cx - 0.35 * ri, gy = cy - 0.35 * ri
        func upper(_ x: Double) -> Double { let u = (x - ex) / a; return ey - b * max(0, 1 - u * u) }
        func lower(_ x: Double) -> Double { let u = (x - ex) / a; return ey + 0.8 * b * max(0, 1 - u * u) }
        let ss = 3
        var img = GrayImage(width: W, height: H)
        var mask = [Bool](repeating: false, count: W * H)
        for y in 0..<H { for x in 0..<W {
            var acc = 0.0
            for j in 0..<ss { for i in 0..<ss {
                let fx = Double(x) + (Double(i) + 0.5) / Double(ss), fy = Double(y) + (Double(j) + 0.5) / Double(ss)
                var v = skin
                let inOpening = abs(fx - ex) < a && fy > upper(fx) && fy < lower(fx)
                if inOpening {
                    v = sclera
                    let d = ((fx - cx) * (fx - cx) + (fy - cy) * (fy - cy)).squareRoot()
                    if d < ri { v = irisI + 12 * sin(atan2(fy - cy, fx - cx) * 9) * (d / ri) }
                    if d < 0.45 * ri { v = pupilI }
                    if glint && ((fx - gx) * (fx - gx) + (fy - gy) * (fy - gy)) < 1.5 * 1.5 * (eyeWidthPx / 48) { v = 245 }
                } else if lashes && abs(fx - ex) < a * 1.05 && fy <= upper(fx) && fy > upper(fx) - eyeWidthPx * 0.07 {
                    v = 45
                }
                acc += v
            } }
            img[x, y] = acc / Double(ss * ss)
            let fx = Double(x) + 0.5, fy = Double(y) + 0.5
            mask[y * W + x] = abs(fx - ex) < a && fy > upper(fx) - 1 && fy < lower(fx) + 1
        } }
        img = img.gaussian3()
        for i in 0..<img.pixels.count { img.pixels[i] = min(255, max(0, img.pixels[i] + noise * rng.normal())) }
        // Pixel centres are at integer coordinates in GrayImage space → shift truth by -0.5.
        return SyntheticEyePatch(image: img, mask: mask, truth: Point2(cx - 0.5, cy - 0.5), eyeWidth: eyeWidthPx)
    }
}

final class GazePupilLocatorTests: XCTestCase {
    struct Stats: CustomStringConvertible {
        var errs: [Double] = []
        var fails = 0
        mutating func add(_ p: Point2?, _ truth: Point2, _ width: Double) {
            guard let p else { fails += 1; return }
            errs.append((p - truth).length / width)
        }
        var mean: Double { errs.isEmpty ? .nan : errs.reduce(0, +) / Double(errs.count) }
        var p95: Double { let s = errs.sorted(); return s.isEmpty ? .nan : s[min(s.count - 1, Int(0.95 * Double(s.count)))] }
        var description: String { String(format: "mean %.4f p95 %.4f eye-widths (fail %d)", mean, p95, fails) }
    }

    /// Error of each method in eye-width units (the unit of GazeSample pupil offsets).
    func runCondition(_ name: String, eyeWidth: Double, noise: Double, glint: Bool, lashes: Bool,
                      landmarkSigma: Double, n: Int = 300) -> (landmark: Stats, dark: Stats, tb: Stats, hybrid: Stats) {
        var rng = GazeTestRNG(state: 0x7B_2011 &+ UInt64(eyeWidth * 10))
        var lm = Stats(), dk = Stats(), tb = Stats(), hy = Stats()
        for _ in 0..<n {
            let e = SyntheticEyePatch.make(rng: &rng, eyeWidthPx: eyeWidth, noise: noise, glint: glint, lashes: lashes)
            // "Landmark" estimate: truth + isotropic Gaussian error (assumed Vision pupil-landmark error).
            let l = Point2(e.truth.x + landmarkSigma * e.eyeWidth * rng.normal(), e.truth.y + landmarkSigma * e.eyeWidth * rng.normal())
            lm.add(l, e.truth, e.eyeWidth)
            dk.add(PupilLocator.darkCentroid(e.image, mask: e.mask), e.truth, e.eyeWidth)
            let r = PupilLocator.timmBarth(e.image, mask: e.mask)
            tb.add(r?.center, e.truth, e.eyeWidth)
            // Hybrid used by the app: trust TB when it is near the landmark prior, else keep the landmark.
            var h = l
            if let r, (r.center - l).length < 0.25 * e.eyeWidth { h = r.center }
            hy.add(h, e.truth, e.eyeWidth)
        }
        print("pupil[\(name)] landmark(σ=\(landmarkSigma)): \(lm)")
        print("pupil[\(name)] dark-centroid: \(dk)")
        print("pupil[\(name)] Timm–Barth: \(tb)")
        print("pupil[\(name)] hybrid (TB gated by landmark): \(hy)")
        return (lm, dk, tb, hy)
    }

    func testTimmBarthBeatsDarkCentroidAndLandmarkPrior() {
        let clean = runCondition("48px clean", eyeWidth: 48, noise: 3, glint: false, lashes: false, landmarkSigma: 0.05)
        let real = runCondition("48px glint+lashes+noise", eyeWidth: 48, noise: 7, glint: true, lashes: true, landmarkSigma: 0.05)
        let small = runCondition("32px glint+lashes+noise", eyeWidth: 32, noise: 7, glint: true, lashes: true, landmarkSigma: 0.05, n: 200)
        XCTAssertLessThan(clean.tb.mean, 0.03)
        XCTAssertLessThan(real.tb.mean, real.dark.mean, "TB must beat the dark-centroid baseline under lashes/glint")
        XCTAssertLessThan(real.hybrid.mean, real.landmark.mean * 0.7)
        XCTAssertLessThan(small.hybrid.mean, small.landmark.mean)
    }

    func testTimmBarthRuntimeBudget() {
        var rng = GazeTestRNG(state: 99)
        let e = SyntheticEyePatch.make(rng: &rng, eyeWidthPx: 40, noise: 5, glint: true, lashes: true)
        let t0 = Date()
        for _ in 0..<20 { _ = PupilLocator.timmBarth(e.image, mask: e.mask) }
        let ms = Date().timeIntervalSince(t0) / 20 * 1000
        print(String(format: "pupil Timm–Barth %dx%d patch: %.2f ms/eye (debug build)", e.image.width, e.image.height, ms))
        XCTAssertLessThan(ms, 30)
    }
}
