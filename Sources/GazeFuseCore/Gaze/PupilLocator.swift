// PupilLocator.swift
//
// Image-based pupil/iris centre refinement on a small grayscale eye patch.
//
// * Timm & Barth (2011), "Accurate eye centre localisation by means of gradients":
//   the centre c maximises  w(c) · mean_i max(0, d_i(c) · g_i)²,  where g_i are normalised
//   image gradients (above a dynamic magnitude threshold), d_i the normalised displacement
//   from c to pixel i, and w(c) the inverted (smoothed) intensity at c (dark-centre prior).
// * A dark-pixel centroid baseline (the method the app used before), for comparison.
//
// Pure Swift, no Vision/CoreImage, so it is unit-tested on synthetic eye images. The app
// crops the eye from the camera's Y plane, resamples it to ~`targetWidth` pixels and calls
// `locate`. Coordinates are patch pixels, origin top-left, +y down.
import Foundation

public struct GrayImage: Equatable {
    public let width: Int
    public let height: Int
    public var pixels: [Double]

    public init(width: Int, height: Int, pixels: [Double]) {
        precondition(width > 0 && height > 0 && pixels.count == width * height)
        self.width = width; self.height = height; self.pixels = pixels
    }
    public init(width: Int, height: Int, fill: Double = 0) {
        self.init(width: width, height: height, pixels: [Double](repeating: fill, count: width * height))
    }
    @inline(__always) public subscript(x: Int, y: Int) -> Double {
        get { pixels[y * width + x] }
        set { pixels[y * width + x] = newValue }
    }
    /// Bilinear sample with edge clamping.
    public func sample(_ x: Double, _ y: Double) -> Double {
        let cx = min(max(x, 0), Double(width - 1)), cy = min(max(y, 0), Double(height - 1))
        let x0 = Int(cx), y0 = Int(cy)
        let x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1)
        let fx = cx - Double(x0), fy = cy - Double(y0)
        let a = self[x0, y0] * (1 - fx) + self[x1, y0] * fx
        let b = self[x0, y1] * (1 - fx) + self[x1, y1] * fx
        return a * (1 - fy) + b * fy
    }
    /// Area-correct-ish resample (bilinear on a pre-blurred image) to a new size.
    public func resized(width w: Int, height h: Int) -> GrayImage {
        var out = GrayImage(width: w, height: h)
        let sx = Double(width) / Double(w), sy = Double(height) / Double(h)
        let src = (sx > 1.5 || sy > 1.5) ? boxBlurred(radius: Int((max(sx, sy) / 2).rounded())) : self
        for y in 0..<h { for x in 0..<w {
            out[x, y] = src.sample((Double(x) + 0.5) * sx - 0.5, (Double(y) + 0.5) * sy - 0.5)
        } }
        return out
    }
    public func boxBlurred(radius r: Int) -> GrayImage {
        guard r > 0 else { return self }
        var tmp = GrayImage(width: width, height: height), out = tmp
        for y in 0..<height { for x in 0..<width {
            var s = 0.0, n = 0.0
            for k in max(0, x - r)...min(width - 1, x + r) { s += self[k, y]; n += 1 }
            tmp[x, y] = s / n
        } }
        for y in 0..<height { for x in 0..<width {
            var s = 0.0, n = 0.0
            for k in max(0, y - r)...min(height - 1, y + r) { s += tmp[x, k]; n += 1 }
            out[x, y] = s / n
        } }
        return out
    }
    /// 3×3 binomial (≈ Gaussian σ≈0.85) blur.
    public func gaussian3() -> GrayImage {
        var out = GrayImage(width: width, height: height)
        let k = [1.0, 2.0, 1.0]
        for y in 0..<height { for x in 0..<width {
            var s = 0.0, n = 0.0
            for j in -1...1 { for i in -1...1 {
                let xx = x + i, yy = y + j
                guard xx >= 0, yy >= 0, xx < width, yy < height else { continue }
                let w = k[i + 1] * k[j + 1]; s += w * self[xx, yy]; n += w
            } }
            out[x, y] = s / n
        } }
        return out
    }
}

public struct PupilLocatorResult: Equatable {
    /// Centre in patch pixels (top-left origin).
    public var center: Point2
    /// Peak objective relative to the patch mean objective (≥ 1; higher = more distinct).
    public var distinctness: Double
}

public enum PupilLocator {
    public struct Config: Equatable {
        /// Gradient magnitude threshold = mean + k · std (Timm & Barth use 0.3).
        public var gradientThresholdStd: Double = 0.3
        /// Inverted-intensity weight w(c) = (maxI - I_blur(c)) ^ weightPower.
        public var weightPower: Double = 1
        /// Candidate centres are restricted to this fraction of the patch, centred (eyelid/brow margins).
        public var searchFraction: Double = 0.85
        /// Optional mask: candidate centres and gradients only inside this ellipse-ish region.
        public var useMask: Bool = true
        /// Sub-pixel refinement: objective-weighted centroid over a (2r+1)² window around the peak.
        public var subpixelRadius: Int = 1
        /// Blur radius (box) applied for the dark-centre weight.
        public var weightBlurRadius: Int = 2
        public init() {}
    }

    /// Timm–Barth gradient centre. `mask` (same size, true = inside eye opening) is optional.
    public static func timmBarth(_ image: GrayImage, mask: [Bool]? = nil, config: Config = Config()) -> PupilLocatorResult? {
        let w = image.width, h = image.height
        guard w >= 8, h >= 6 else { return nil }
        let img = image.gaussian3()
        // Gradients (central differences).
        var gx = [Double](repeating: 0, count: w * h), gy = gx, mag = gx
        for y in 1..<(h - 1) { for x in 1..<(w - 1) {
            let i = y * w + x
            gx[i] = (img[x + 1, y] - img[x - 1, y]) / 2
            gy[i] = (img[x, y + 1] - img[x, y - 1]) / 2
            mag[i] = (gx[i] * gx[i] + gy[i] * gy[i]).squareRoot()
        } }
        let inside: (Int) -> Bool = { i in config.useMask ? (mask?[i] ?? true) : true }
        var sum = 0.0, sum2 = 0.0, n = 0.0
        for i in 0..<(w * h) where inside(i) && mag[i] > 0 { sum += mag[i]; sum2 += mag[i] * mag[i]; n += 1 }
        guard n > 8 else { return nil }
        let mean = sum / n, std = max(0, sum2 / n - mean * mean).squareRoot()
        let thr = mean + config.gradientThresholdStd * std
        var px: [Double] = [], py: [Double] = [], ux: [Double] = [], uy: [Double] = []
        for y in 1..<(h - 1) { for x in 1..<(w - 1) {
            let i = y * w + x
            guard mag[i] > thr, inside(i) || nearMask(mask, x, y, w, h) else { continue }
            px.append(Double(x)); py.append(Double(y)); ux.append(gx[i] / mag[i]); uy.append(gy[i] / mag[i])
        } }
        guard px.count >= 6 else { return nil }
        let blurred = img.boxBlurred(radius: config.weightBlurRadius)
        let maxI = blurred.pixels.max() ?? 255
        let mx = Int(Double(w) * (1 - config.searchFraction) / 2), my = Int(Double(h) * (1 - config.searchFraction) / 2)
        var obj = [Double](repeating: 0, count: w * h)
        var best = -1.0, bx = 0, by = 0, objSum = 0.0, objN = 0.0
        for cy in my..<(h - my) { for cx in mx..<(w - mx) {
            let ci = cy * w + cx
            guard inside(ci) else { continue }
            let fx = Double(cx), fy = Double(cy)
            var acc = 0.0
            for k in 0..<px.count {
                let dx = px[k] - fx, dy = py[k] - fy
                let d2 = dx * dx + dy * dy
                guard d2 > 0 else { continue }
                let dot = (dx * ux[k] + dy * uy[k]) / d2.squareRoot()
                if dot > 0 { acc += dot * dot }
            }
            let wgt = pow(max(maxI - blurred[cx, cy], 0), config.weightPower)
            let v = wgt * acc / Double(px.count)
            obj[ci] = v; objSum += v; objN += 1
            if v > best { best = v; bx = cx; by = cy }
        } }
        guard best > 0, objN > 0 else { return nil }
        // Sub-pixel: objective-weighted centroid around the peak (values above 0.9·peak).
        var sx = 0.0, sy = 0.0, sw = 0.0
        let r = config.subpixelRadius
        for y in max(0, by - r)...min(h - 1, by + r) { for x in max(0, bx - r)...min(w - 1, bx + r) {
            let v = obj[y * w + x] - 0.9 * best
            if v > 0 { sx += v * Double(x); sy += v * Double(y); sw += v }
        } }
        let c = sw > 0 ? Point2(sx / sw, sy / sw) : Point2(Double(bx), Double(by))
        return PupilLocatorResult(center: c, distinctness: best / (objSum / objN))
    }

    private static func nearMask(_ mask: [Bool]?, _ x: Int, _ y: Int, _ w: Int, _ h: Int) -> Bool {
        guard let mask else { return true }
        for j in -1...1 { for i in -1...1 {
            let xx = x + i, yy = y + j
            if xx >= 0, yy >= 0, xx < w, yy < h, mask[yy * w + xx] { return true }
        } }
        return false
    }

    /// Baseline: centroid of pixels darker than (local mean − contrast) inside the mask,
    /// weighted by darkness. This is what `FaceGazeExtractor.refinePupil` does.
    public static func darkCentroid(_ image: GrayImage, mask: [Bool]? = nil, minContrast: Double = 18) -> Point2? {
        var s = 0.0, n = 0.0
        for i in 0..<(image.width * image.height) where mask?[i] ?? true { s += image.pixels[i]; n += 1 }
        guard n > 4 else { return nil }
        let thr = s / n - minContrast
        var sx = 0.0, sy = 0.0, sw = 0.0
        for y in 0..<image.height { for x in 0..<image.width {
            let i = y * image.width + x
            guard mask?[i] ?? true else { continue }
            let d = thr - image.pixels[i]
            if d > 0 { sx += d * Double(x); sy += d * Double(y); sw += d }
        } }
        guard sw > 8 else { return nil }
        return Point2(sx / sw, sy / sw)
    }
}
