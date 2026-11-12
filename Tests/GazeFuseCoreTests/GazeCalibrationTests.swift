import XCTest
@testable import GazeFuseCore

final class GazeCalibrationTests: XCTestCase {
    private func train(config: RidgeGazeCalibrator.Config = .init(), contaminated: Bool = false,
                       landmarkNoise: Double? = nil, appearanceSigma: Double? = nil)
        throws -> (RidgeGazeCalibrator, CalibrationReport) {
        var eye = GazeSyntheticEye()
        eye.landmarkNoise = landmarkNoise; eye.appearanceSigma = appearanceSigma
        let mapper = RidgeGazeCalibrator(config: config)
        let targets = (0..<16).map { i in
            Point2(eye.screen.origin.x + 110 + Double(i % 4) * 405,
                   eye.screen.origin.y + 80 + Double(i / 4) * 245)
        }
        for (index, target) in targets.enumerated() {
            mapper.addCalibrationSamples(
                eye.fixation(target: target, previous: targets[(index + 15) % 16],
                             index: index, contaminated: contaminated), target: target)
        }
        return (mapper, try mapper.fit())
    }

    private func heldOutErrors(_ mapper: RidgeGazeCalibrator, headAmplitude: Double = 0.055,
                               bias: Point2 = .zero, landmarkNoise: Double? = nil,
                               appearanceSigma: Double? = nil) throws -> [Double] {
        var eye = GazeSyntheticEye(rng: GazeTestRNG(state: 0xC0FFEE))
        eye.landmarkNoise = landmarkNoise; eye.appearanceSigma = appearanceSigma
        return try (0..<240).map { i in
            let target = Point2(eye.screen.origin.x + 70 + 1300 * eye.rng.uniform(),
                                eye.screen.origin.y + 60 + 780 * eye.rng.uniform())
            let sample = eye.sample(target: target + bias, t: 100 + Double(i) / 30,
                                    phase: 50 + Double(i) * 0.233,
                                    headAmplitude: headAmplitude)
            return try XCTUnwrap(mapper.estimate(sample)).point.distance(to: target)
        }
    }

    private func mean(_ errors: [Double]) -> Double { errors.reduce(0, +) / Double(errors.count) }
    private func p95(_ errors: [Double]) -> Double { errors.sorted()[Int(Double(errors.count - 1) * 0.95)] }

    func testGeometricCalibrationAndRobustContaminatedFixations() throws {
        let (clean, cleanReport) = try train()
        let (robust, robustReport) = try train(contaminated: true)
        let cleanErrors = try heldOutErrors(clean)
        let robustErrors = try heldOutErrors(robust)
        print(String(format: "GAZE_METRIC clean_mean=%.3f clean_p95=%.3f clean_cv=%.3f contaminated_mean=%.3f contaminated_p95=%.3f contaminated_cv=%.3f accepted=%d",
                     mean(cleanErrors), p95(cleanErrors), cleanReport.meanErrorPoints,
                     mean(robustErrors), p95(robustErrors), robustReport.meanErrorPoints,
                     robustReport.sampleCount))
        XCTAssertLessThan(mean(cleanErrors), 30)
        XCTAssertLessThan(mean(robustErrors), 30)
        XCTAssertLessThan(p95(robustErrors), 60)
        XCTAssertEqual(robustReport.pointCount, 16)
        XCTAssertGreaterThan(robustReport.sampleCount, 100)
        XCTAssertLessThan(robustReport.sampleCount, 16 * 36)
        XCTAssertTrue(robustReport.meanErrorPoints.isFinite)
    }

    /// Webcam-realistic per-frame noise. Landmark σ 0.05 eye-widths is the assumed Vision
    /// pupil-landmark error; 0.011 is the Timm–Barth error measured on synthetic eye patches
    /// (GazePupilLocatorTests); appearance σ 2.5° per frame is a typical calibrated CNN.
    func testPupilRefinementAndAppearanceFusionCutCalibrationError() throws {
        var rows: [(String, Double, Double, Double)] = []
        func run(_ name: String, landmark: Double, appearance: Double?) throws -> Double {
            let (m, report) = try train(landmarkNoise: landmark, appearanceSigma: appearance)
            let e = try heldOutErrors(m, landmarkNoise: landmark, appearanceSigma: appearance)
            rows.append((name, mean(e), p95(e), report.meanErrorPoints))
            print(String(format: "GAZE_FUSION %@: held-out mean %.1f pt p95 %.1f pt (CV %.1f) — %@",
                         name, mean(e), p95(e), report.meanErrorPoints, report.notes))
            return mean(e)
        }
        let landmark = try run("landmark σ0.05", landmark: 0.05, appearance: nil)
        let tb = try run("Timm–Barth σ0.011", landmark: 0.011, appearance: nil)
        let app = try run("landmark σ0.05 + appearance 2.5°", landmark: 0.05, appearance: 2.5 * .pi / 180)
        let both = try run("Timm–Barth σ0.011 + appearance 2.5°", landmark: 0.011, appearance: 2.5 * .pi / 180)
        let appOnlyNoisy = try run("landmark σ0.05 + appearance 5°", landmark: 0.05, appearance: 5 * .pi / 180)
        XCTAssertLessThan(tb, landmark * 0.5)
        XCTAssertLessThan(app, landmark)
        // With a precise pupil the appearance head is only kept when CV says it helps.
        XCTAssertLessThan(both, tb * 1.05)
        XCTAssertLessThan(appOnlyNoisy, landmark * 1.02, "a noisy appearance head must not make things worse")
    }

    func testBlinkAndEyeDropoutConfidence() throws {
        let (mapper, _) = try train()
        var eye = GazeSyntheticEye()
        let good = eye.sample(target: eye.screen.center, t: 100, phase: 0.5)
        let confidence = try XCTUnwrap(mapper.estimate(good)).confidence
        var oneEye = good
        oneEye.leftPupil = nil
        let degraded = try XCTUnwrap(mapper.estimate(oneEye))
        XCTAssertGreaterThan(degraded.confidence, 0)
        XCTAssertLessThan(degraded.confidence, confidence)
        var noEyes = good
        noEyes.leftPupil = nil
        noEyes.rightPupil = nil
        XCTAssertNil(mapper.estimate(noEyes))
        var blink = good
        blink.leftEyeOpenness = 0.02
        blink.rightEyeOpenness = 0.02
        XCTAssertNil(mapper.estimate(blink))
        mapper.addCorrection(sample: blink, truePoint: eye.screen.center)
        XCTAssertTrue(mapper.isCalibrated, "A blink correction must not invalidate calibration")
        var lowConfidence = good
        lowConfidence.confidence = 0
        let priorPoint = mapper.estimate(good)?.point
        mapper.addCorrection(sample: lowConfidence, truePoint: Point2(4000, 4000))
        XCTAssertEqual(mapper.estimate(good)?.point, priorPoint)
        XCTAssertNil(RidgeGazeCalibrator().estimate(good))
    }

    func testOnlyTargetOnsetFramesCannotProduceCalibration() {
        let mapper = RidgeGazeCalibrator()
        var eye = GazeSyntheticEye()
        for i in 0..<5 {
            let target = Point2(100 + Double(i % 3) * 400, 100 + Double(i / 3) * 500)
            let samples = (0..<7).map { frame in
                eye.sample(target: target, t: Double(i) * 2 + Double(frame) / 30,
                           phase: Double(i) + Double(frame) * 0.1)
            }
            mapper.addCalibrationSamples(samples, target: target)
        }
        XCTAssertThrowsError(try mapper.fit())
        XCTAssertFalse(mapper.isCalibrated)
    }

    func testHeadPoseFeaturesGeneralizeToLargerUnseenMotion() throws {
        let (compensated, _) = try train()
        var config = RidgeGazeCalibrator.Config()
        config.headPoseFeatures = false
        let (uncompensated, _) = try train(config: config)
        let on = try heldOutErrors(compensated, headAmplitude: 0.09)
        let off = try heldOutErrors(uncompensated, headAmplitude: 0.09)
        print(String(format: "GAZE_METRIC head_on_mean=%.3f head_off_mean=%.3f", mean(on), mean(off)))
        XCTAssertLessThan(mean(on), 30)
        XCTAssertLessThan(mean(on), mean(off) * 0.4)
    }

    func testTargetGroupedCrossValidationSelectsBestLambda() throws {
        let candidates = [1e-5, 0.01, 100.0]
        var config = RidgeGazeCalibrator.Config()
        config.lambdaCandidates = candidates
        let (selected, report) = try train(config: config)
        var scores: [Double] = []
        for lambda in candidates {
            config.lambdaCandidates = [lambda]
            let (_, oneReport) = try train(config: config)
            scores.append(oneReport.meanErrorPoints)
        }
        let best = try XCTUnwrap(scores.enumerated().min { $0.element < $1.element })
        XCTAssertEqual(try XCTUnwrap(selected.model).lambda, candidates[best.offset])
        XCTAssertEqual(report.meanErrorPoints, best.element, accuracy: 1e-7)
        XCTAssertLessThan(best.element, scores.last! * 0.25)
        print("GAZE_METRIC cv_lambdas=\(candidates) cv_errors=\(scores) selected=\(candidates[best.offset])")
    }

    func testCodableModelPreservesPredictionsAndFeatureConfiguration() throws {
        // Deliberately load with default runtime Config, so a saved model must
        // carry its own choice of feature basis.
        var config = RidgeGazeCalibrator.Config()
        config.headPoseFeatures = false
        let (original, _) = try train(config: config)
        var eye = GazeSyntheticEye()
        let sample = eye.sample(target: eye.screen.center, t: 40, phase: 2)
        original.addCorrection(sample: sample, truePoint: eye.screen.center + Point2(20, -10))
        let model = try XCTUnwrap(original.model)
        let data = try JSONEncoder().encode(model)
        let decoded = try JSONDecoder().decode(RidgeGazeCalibrator.Model.self, from: data)
        let restored = try RidgeGazeCalibrator(model: decoded)
        XCTAssertEqual(try XCTUnwrap(original.estimate(sample)).point,
                       try XCTUnwrap(restored.estimate(sample)).point)
        XCTAssertEqual(model, decoded)
        let savedPoint = try XCTUnwrap(restored.estimate(sample)).point
        restored.addCorrection(sample: sample, truePoint: savedPoint + Point2(4, 2))
        let continued = try XCTUnwrap(restored.estimate(sample)).point
        XCTAssertEqual(continued.x, savedPoint.x + 2, accuracy: 1e-8)
        XCTAssertEqual(continued.y, savedPoint.y + 1, accuracy: 1e-8)
        let configRoundTrip = try JSONDecoder().decode(RidgeGazeCalibrator.Config.self,
                                                       from: JSONEncoder().encode(config))
        XCTAssertEqual(config, configRoundTrip)
        let start = ProcessInfo.processInfo.systemUptime
        var sum = 0.0
        for _ in 0..<10_000 { sum += restored.estimate(sample)?.point.x ?? 0 }
        let microseconds = (ProcessInfo.processInfo.systemUptime - start) * 100
        XCTAssertTrue(sum.isFinite)
        print(String(format: "GAZE_METRIC estimate_debug_microseconds=%.3f model_json_bytes=%d", microseconds, data.count))
    }

    func testOnlineCorrectionsReduceNewSystematicBias() throws {
        let (mapper, _) = try train()
        let bias = Point2(75, -48)
        let before = try heldOutErrors(mapper, bias: bias)
        var eye = GazeSyntheticEye(rng: GazeTestRNG(state: 0xB1A5))
        for i in 0..<32 {
            let target = Point2(eye.screen.origin.x + 180 + 1080 * eye.rng.uniform(),
                                eye.screen.origin.y + 120 + 660 * eye.rng.uniform())
            mapper.addCorrection(sample: eye.sample(target: target + bias, t: 300 + Double(i),
                                                     phase: 80 + Double(i) * 0.57), truePoint: target)
        }
        let after = try heldOutErrors(mapper, bias: bias)
        print(String(format: "GAZE_METRIC correction_before=%.3f correction_after=%.3f", mean(before), mean(after)))
        XCTAssertLessThan(mean(after), mean(before) * 0.4)
    }
}
