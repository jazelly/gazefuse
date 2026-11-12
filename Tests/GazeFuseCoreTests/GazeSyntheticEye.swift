import Foundation
import GazeFuseCore

/// Deterministic geometric fixture: rotate a spherical pupil relative to the head,
/// project it into the eye contour, then add independent landmark noise.
struct GazeSyntheticEye {
    let screen = ScreenRect(x: -180, y: 70, width: 1440, height: 900)
    var rng = GazeTestRNG(state: 0xEA7E_2026)

    mutating func sample(target: Point2, t: Double, phase: Double,
                         headAmplitude: Double = 0.055, noise: Double = 0.0007,
                         blink: Bool = false, outlier: Bool = false) -> GazeSample {
        // 0.22 mm per point, 650 mm viewing distance; no fitted inverse is used
        // by the implementation. Head motion is independent of screen target.
        let yaw = headAmplitude * sin(phase * 1.71)
        let pitch = headAmplitude * cos(phase * 1.13)
        let roll = 0.35 * headAmplitude * sin(phase * 0.79)
        let tx = 9 * sin(phase * 0.53)
        let ty = 7 * cos(phase * 0.67)
        let distance = 650 + 12 * sin(phase * 0.41)
        let angleX = atan(((target.x - screen.center.x) * 0.22 - tx) / distance)
        let angleY = atan(((target.y - screen.center.y) * 0.22 - ty) / distance)
        let px = 0.42 * sin(angleX - yaw)
        let py = -0.42 * sin(angleY + pitch)
        let rotatedX = px * cos(roll) - py * sin(roll)
        let rotatedY = px * sin(roll) + py * cos(roll)
        func point(_ nx: Double, _ ny: Double) -> Point2 {
            Point2(rotatedX + nx, rotatedY + ny)
        }
        let left = point(noise * rng.normal(), noise * rng.normal())
        let right = point(noise * rng.normal(), noise * rng.normal())
        return GazeSample(
            t: t,
            leftPupil: outlier ? Point2(0.38, -0.3) : left,
            rightPupil: outlier ? Point2(-0.32, 0.3) : right,
            leftEyeOpenness: blink ? 0.025 : 0.26,
            rightEyeOpenness: blink ? 0.025 : 0.25,
            headYaw: yaw, headPitch: pitch, headRoll: roll,
            faceCenter: Point2(0.5 + tx / 500, 0.5 - ty / 500),
            faceSize: 0.30 * 650 / distance, confidence: 0.98
        )
    }

    /// A 30 Hz fixation with 267 ms target-acquisition latency and deterministic
    /// blinks, gross pupil outliers and missing-eye frames after acquisition.
    mutating func fixation(target: Point2, previous: Point2, index: Int,
                           contaminated: Bool) -> [GazeSample] {
        (0..<45).map { frame in
            let t = Double(index) * 2 + Double(frame) / 30
            var s = sample(target: frame < 8 ? previous : target, t: t,
                           phase: Double(index) * 1.37 + Double(frame) * 0.19,
                           blink: contaminated && frame % 13 == 0,
                           outlier: contaminated && frame % 17 == 0)
            if contaminated && frame % 19 == 0 { s.leftPupil = nil }
            return s
        }
    }
}

struct GazeTestRNG {
    var state: UInt64
    mutating func uniform() -> Double {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return Double((z ^ (z >> 31)) >> 11) / 9_007_199_254_740_992
    }
    mutating func normal() -> Double {
        sqrt(-2 * log(max(uniform(), 1e-12))) * cos(2 * .pi * uniform())
    }
}
