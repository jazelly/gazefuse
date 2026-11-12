// RidgeGazeCalibrator.swift
//
// A small, hardware-free gaze mapper.  Vision supplies normalized eye/face
// landmarks; this type learns the user's camera-to-display mapping from those
// landmarks with weighted ridge regression.  The implementation deliberately
// keeps the model explicit and Codable so a calibration can be inspected and
// replayed without any runtime dependencies.

import Foundation

/// Errors reported by `RidgeGazeCalibrator.fit()` and model persistence.
public enum RidgeGazeError: Error, LocalizedError, Equatable {
    case noCalibrationData
    case insufficientUsableSamples
    case invalidModel
    case noFittedModel

    public var errorDescription: String? {
        switch self {
        case .noCalibrationData: return "No calibration targets were supplied."
        case .insufficientUsableSamples: return "Calibration did not contain enough usable eye samples."
        case .invalidModel: return "The stored gaze model is invalid."
        case .noFittedModel: return "The gaze calibrator has not been fitted."
        }
    }
}

/// Maps Vision eye/face features to a global top-left-origin screen point.
///
/// The feature vector contains both binocular geometry and the detector's head
/// pose.  A small set of second-order terms captures the most common webcam
/// non-linearities while ridge regularisation keeps the fit stable for a
/// 5--16 point calibration.  Samples are collected on the caller's serial
/// camera queue; this class does not create threads or locks.
public final class RidgeGazeCalibrator: GazeMapper {
    // MARK: Configuration

    public struct Config: Codable, Equatable {
        /// Seconds at the beginning of each target presentation to ignore. This
        /// removes saccade/latency frames before a fixation settles.
        public var onsetExclusionSeconds: Double
        /// Eye-contour openness below this value is treated as a blink.
        public var blinkOpennessThreshold: Double
        /// Frames below this detector confidence are not used for fitting.
        public var minimumConfidence: Double
        /// Robust per-target outlier cutoff in median absolute deviations.
        public var outlierMADs: Double
        /// Bounded retained frames for each target.
        public var maxSamplesPerTarget: Int
        /// Consecutive accepted frames combined by a component-wise median.
        public var aggregationBinSize: Int
        /// Maximum number of extractor `extra` values included in the model.
        public var maxExtraFeatures: Int
        /// Ridge candidates evaluated by leave-one-target-out validation.
        public var lambdaCandidates: [Double]
        /// Ridge value used if the candidate list is empty.
        public var defaultLambda: Double
        /// Include Vision head yaw/pitch/roll features.
        public var headPoseFeatures: Bool
        /// Include selected quadratic/cross terms.
        public var polynomialFeatures: Bool
        /// Bounded recent click corrections used for online bias refinement.
        public var maxCorrections: Int
        /// Also fit a second head that adds appearance-model yaw/pitch (+ quadratic terms) when
        /// enough calibration samples carry `appearanceGaze`; it is kept only if its
        /// leave-one-target-out error beats the landmark-only model.
        public var appearanceFeatures: Bool = true
        /// Minimum fraction of usable rows that must carry appearance gaze to fit that head.
        public var minimumAppearanceFraction: Double = 0.6

        public init(
            onsetExclusionSeconds: Double = 0.30,
            blinkOpennessThreshold: Double = 0.16,
            minimumConfidence: Double = 0.10,
            outlierMADs: Double = 5.0,
            maxSamplesPerTarget: Int = 240,
            aggregationBinSize: Int = 3,
            maxExtraFeatures: Int = 12,
            lambdaCandidates: [Double] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
            defaultLambda: Double = 1e-2,
            headPoseFeatures: Bool = true,
            polynomialFeatures: Bool = true,
            maxCorrections: Int = 32
        ) {
            self.onsetExclusionSeconds = max(0, onsetExclusionSeconds)
            self.blinkOpennessThreshold = max(0, blinkOpennessThreshold)
            self.minimumConfidence = min(max(0, minimumConfidence), 1)
            self.outlierMADs = max(0.5, outlierMADs)
            self.maxSamplesPerTarget = max(8, maxSamplesPerTarget)
            self.aggregationBinSize = max(1, aggregationBinSize)
            self.maxExtraFeatures = max(0, maxExtraFeatures)
            self.lambdaCandidates = lambdaCandidates.filter { $0.isFinite && $0 > 0 }
            self.defaultLambda = (defaultLambda.isFinite && defaultLambda > 0) ? defaultLambda : 1e-2
            self.headPoseFeatures = headPoseFeatures
            self.polynomialFeatures = polynomialFeatures
            self.maxCorrections = max(0, maxCorrections)
        }

    }

    /// Serializable fitted weights.  JSON encoding this value is the portable
    /// save/load format for a calibration.
    public struct Model: Codable, Equatable {
        public var featureCount: Int
        public var extraFeatureCount: Int
        public var weightsX: [Double]
        public var weightsY: [Double]
        public var lambda: Double
        public var headPoseFeatures: Bool
        public var polynomialFeatures: Bool
        /// Median residual from recent click corrections, in screen points.
        public var correctionBias: Point2
        /// Optional appearance head (landmark features + 5 appearance terms). Used for samples
        /// that carry `appearanceGaze`; others fall back to the landmark-only weights.
        public var appearanceWeightsX: [Double]?
        public var appearanceWeightsY: [Double]?
        public var appearanceLambda: Double?

        public init(
            featureCount: Int,
            extraFeatureCount: Int,
            weightsX: [Double],
            weightsY: [Double],
            lambda: Double,
            headPoseFeatures: Bool = true,
            polynomialFeatures: Bool = true,
            correctionBias: Point2 = .zero
        ) {
            self.featureCount = featureCount
            self.extraFeatureCount = extraFeatureCount
            self.weightsX = weightsX
            self.weightsY = weightsY
            self.lambda = lambda
            self.headPoseFeatures = headPoseFeatures
            self.polynomialFeatures = polynomialFeatures
            self.correctionBias = correctionBias
        }
    }

    // MARK: Public state and construction

    public let config: Config
    public private(set) var model: Model?
    public private(set) var lastReport: CalibrationReport?

    public var isCalibrated: Bool { model != nil }

    public init(config: Config = Config()) {
        self.config = config
        self.model = nil
        self.lastReport = nil
    }

    /// Construct a mapper from a previously decoded model.
    public init(model: Model, config: Config = Config()) throws {
        guard RidgeGazeCalibrator.isValid(model) else { throw RidgeGazeError.invalidModel }
        self.config = config
        self.model = model
        self.lastReport = nil
        if config.maxCorrections > 0, model.correctionBias != .zero {
            corrections = [model.correctionBias]
        }
    }

    public func reset() {
        groups.removeAll(keepingCapacity: false)
        corrections.removeAll(keepingCapacity: false)
        model = nil
        lastReport = nil
    }

    // MARK: Codable model convenience

    public func encodedModel(using encoder: JSONEncoder = JSONEncoder()) throws -> Data {
        guard let model else { throw RidgeGazeError.noFittedModel }
        return try encoder.encode(model)
    }

    public static func loadModel(from data: Data, config: Config = Config(), using decoder: JSONDecoder = JSONDecoder()) throws -> RidgeGazeCalibrator {
        let model = try decoder.decode(Model.self, from: data)
        return try RidgeGazeCalibrator(model: model, config: config)
    }

    // MARK: Calibration collection

    private struct CalibrationGroup {
        var target: Point2
        var samples: [GazeSample]
    }

    private var groups: [CalibrationGroup] = []
    private var corrections: [Point2] = []

    public func addCalibrationSamples(_ samples: [GazeSample], target: Point2) {
        guard !samples.isEmpty, target.x.isFinite, target.y.isFinite else { return }
        if let index = groups.firstIndex(where: { $0.target == target }) {
            groups[index].samples.append(contentsOf: samples)
            if groups[index].samples.count > config.maxSamplesPerTarget {
                groups[index].samples.removeFirst(groups[index].samples.count - config.maxSamplesPerTarget)
            }
        } else {
            groups.append(CalibrationGroup(target: target, samples: Array(samples.suffix(config.maxSamplesPerTarget))))
        }
        // A new target invalidates fitted weights, but leaves the recorded groups
        // intact for the caller to call fit() after the whole plan.
        model = nil
        lastReport = nil
    }

    /// Incorporate a click target as an implicit calibration correction. The
    /// residual is kept in a bounded list and its component-wise median is added
    /// to future predictions. This is intentionally a bias update rather than a
    /// second online optimizer: it is allocation-light, stable after one click,
    /// and cannot grow without bound.
    public func addCorrection(sample: GazeSample, truePoint: Point2) {
        guard truePoint.x.isFinite, truePoint.y.isFinite,
              sampleConfidence(sample) >= config.minimumConfidence else { return }
        guard let fitted = model, let predicted = predict(sample, model: fitted, includeCorrection: false) else { return }
        let delta = Point2(truePoint.x - predicted.x, truePoint.y - predicted.y)
        guard delta.x.isFinite, delta.y.isFinite else { return }
        if config.maxCorrections > 0 {
            corrections.append(delta)
            if corrections.count > config.maxCorrections {
                corrections.removeFirst(corrections.count - config.maxCorrections)
            }
        }
        var updated = fitted
        updated.correctionBias = correctionMedian()
        model = updated
    }

    // MARK: Fit and estimate

    @discardableResult
    public func fit() throws -> CalibrationReport {
        guard !groups.isEmpty else { throw RidgeGazeError.noCalibrationData }

        let extraCount = min(config.maxExtraFeatures, groups.flatMap { $0.samples }.compactMap { $0.extra?.count }.max() ?? 0)
        func buildRows(appearance: Bool) -> (rows: [TrainingRow], usable: [Int], accepted: Int) {
            var rows: [TrainingRow] = []
            var usableCounts = Array(repeating: 0, count: groups.count)
            var accepted = 0
            for (groupIndex, group) in groups.enumerated() {
                let candidateSamples = usableSamples(in: group.samples)
                guard !candidateSamples.isEmpty else { continue }
                let candidates: [TrainingRow] = candidateSamples.compactMap { sample in
                    guard let raw = rawFeatures(sample, extraCount: extraCount, headPose: config.headPoseFeatures, appearance: appearance), raw.hasFinite else { return nil }
                    return TrainingRow(time: sample.t, features: makeFeatures(raw.values, extraCount: extraCount, headPose: config.headPoseFeatures, polynomial: config.polynomialFeatures), target: group.target, groupIndex: groupIndex, confidence: sampleConfidence(sample))
                }
                guard !candidates.isEmpty else { continue }
                let robust = robustRows(candidates)
                usableCounts[groupIndex] = robust.count
                accepted += robust.count
                rows.append(contentsOf: medianBins(robust, binSize: config.aggregationBinSize))
            }
            return (rows, usableCounts, accepted)
        }
        let base = buildRows(appearance: false)
        let rows = base.rows, usableCounts = base.usable, acceptedRawSampleCount = base.accepted

        guard rows.count >= 3, usableCounts.filter({ $0 > 0 }).count >= 3 else {
            throw RidgeGazeError.insufficientUsableSamples
        }
        let featureCount = rows[0].features.count
        let selectedLambda = chooseLambda(rows: rows, featureCount: featureCount)
        guard let solved = solve(rows: rows, featureCount: featureCount, lambda: selectedLambda) else {
            throw RidgeGazeError.insufficientUsableSamples
        }
        let priorBias = model?.correctionBias ?? correctionMedian()
        var fitted = Model(featureCount: featureCount, extraFeatureCount: extraCount,
                           weightsX: solved.x, weightsY: solved.y, lambda: selectedLambda,
                           headPoseFeatures: config.headPoseFeatures,
                           polynomialFeatures: config.polynomialFeatures,
                           correctionBias: priorBias)
        var errors = crossValidatedErrors(rows: rows, featureCount: featureCount, lambda: selectedLambda)
        guard !errors.isEmpty else { throw RidgeGazeError.insufficientUsableSamples }
        var appearanceNote = "appearance off"
        if config.appearanceFeatures {
            let app = buildRows(appearance: true)
            if app.rows.count >= 3, app.usable.filter({ $0 > 0 }).count >= 3,
               Double(app.rows.count) >= config.minimumAppearanceFraction * Double(rows.count) {
                let fc = app.rows[0].features.count
                let lam = chooseLambda(rows: app.rows, featureCount: fc)
                if let sol = solve(rows: app.rows, featureCount: fc, lambda: lam) {
                    let appErrors = crossValidatedErrors(rows: app.rows, featureCount: fc, lambda: lam)
                    let baseMean = errors.reduce(0, +) / Double(errors.count)
                    let appMean = appErrors.isEmpty ? .infinity : appErrors.reduce(0, +) / Double(appErrors.count)
                    appearanceNote = String(format: "appearance head CV %.1f pt vs landmark %.1f pt", appMean, baseMean)
                    if appMean < baseMean {
                        fitted.appearanceWeightsX = sol.x; fitted.appearanceWeightsY = sol.y; fitted.appearanceLambda = lam
                        errors = appErrors
                        appearanceNote += " (used)"
                    } else {
                        appearanceNote += " (not better, dropped)"
                    }
                }
            } else {
                appearanceNote = "appearance: too few samples"
            }
        }
        let mean = errors.reduce(0, +) / Double(errors.count)
        let p95 = percentile(errors, 0.95)
        let report = CalibrationReport(pointCount: usableCounts.filter({ $0 > 0 }).count,
                                       sampleCount: acceptedRawSampleCount,
                                       meanErrorPoints: mean,
                                       p95ErrorPoints: p95,
                                       notes: String(format: "ridge lambda %.6g; head pose %@; %d target groups; %@",
                                                     selectedLambda, config.headPoseFeatures ? "on" : "off", usableCounts.filter({ $0 > 0 }).count, appearanceNote))
        model = fitted
        lastReport = report
        return report
    }

    public func estimate(_ sample: GazeSample) -> GazeEstimate? {
        guard let fitted = model else { return nil }
        guard let point = predict(sample, model: fitted, includeCorrection: true) else { return nil }
        let confidence = estimateConfidence(sample)
        guard confidence > 0 else { return nil }
        return GazeEstimate(t: sample.t, point: point, confidence: confidence)
    }

    // MARK: Feature engineering

    private struct RawFeatures {
        var values: [Double]
        var hasFinite: Bool { values.allSatisfy { $0.isFinite } }
    }

    private struct TrainingRow {
        var time: TimeSec
        var features: [Double]
        var target: Point2
        var groupIndex: Int
        var confidence: Double
    }

    private func rawFeatures(_ sample: GazeSample, extraCount: Int, headPose: Bool, appearance: Bool = false) -> RawFeatures? {
        var appearanceTerms: [Double] = []
        if appearance {
            guard let a = sample.appearanceGaze, a.x.isFinite, a.y.isFinite else { return nil }
            appearanceTerms = [a.x, a.y, a.x * a.x, a.y * a.y, a.x * a.y]
        }
        let leftOpen = finite(sample.leftEyeOpenness) ?? 1
        let rightOpen = finite(sample.rightEyeOpenness) ?? 1
        let leftUsable = leftOpen >= config.blinkOpennessThreshold
        let rightUsable = rightOpen >= config.blinkOpennessThreshold
        let left = leftUsable ? sample.leftPupil : nil
        let right = rightUsable ? sample.rightPupil : nil
        guard left != nil || right != nil else { return nil }

        let lp = left ?? right ?? .zero
        let rp = right ?? left ?? .zero
        let avg = Point2((lp.x + rp.x) * 0.5, (lp.y + rp.y) * 0.5)
        let diff = Point2(lp.x - rp.x, lp.y - rp.y)
        let center = Point2(finite(sample.faceCenter.x) ?? 0, finite(sample.faceCenter.y) ?? 0)
        let size = finite(sample.faceSize) ?? 0
        let yaw = headPose ? (finite(sample.headYaw) ?? 0) : 0
        let pitch = headPose ? (finite(sample.headPitch) ?? 0) : 0
        let roll = headPose ? (finite(sample.headRoll) ?? 0) : 0
        var values: [Double] = [
            lp.x, lp.y, rp.x, rp.y,
            avg.x, avg.y, diff.x, diff.y,
            center.x, center.y, size,
            yaw, pitch, roll,
            leftUsable ? leftOpen : 0, rightUsable ? rightOpen : 0,
            left != nil ? 1 : 0, right != nil ? 1 : 0,
            min(max(finite(sample.confidence) ?? 0, 0), 1)
        ]
        if extraCount > 0 {
            let extras = sample.extra ?? []
            for index in 0..<extraCount {
                values.append(index < extras.count ? (finite(extras[index]) ?? 0) : 0)
            }
        }
        values.append(contentsOf: appearanceTerms)
        return RawFeatures(values: values)
    }

    private func makeFeatures(_ raw: [Double], extraCount: Int, headPose: Bool, polynomial: Bool) -> [Double] {
        // raw layout is documented by rawFeatures: [eyes, binocular, face,
        // head, openness, presence, detector confidence, extras].
        var result: [Double] = [1]
        result.append(contentsOf: raw)
        guard polynomial else { return result }

        // Quadratic terms for pupil geometry, face geometry and head pose.
        let squareIndices = Array(0..<14)
        for index in squareIndices { result.append(raw[index] * raw[index]) }

        // Cross terms chosen for the dominant webcam nuisance factors. Keeping
        // this list small avoids an unnecessary high-dimensional polynomial.
        let pairs = [(4, 11), (4, 12), (5, 11), (5, 12),
                     (8, 11), (9, 12), (10, 11), (10, 12),
                     (6, 13), (7, 13)]
        if headPose {
            for (a, b) in pairs { result.append(raw[a] * raw[b]) }
        }
        return result
    }

    // MARK: Robust sample preparation

    private func usableSamples(in samples: [GazeSample]) -> [GazeSample] {
        let ordered = samples.sorted { $0.t < $1.t }
        guard let first = ordered.first else { return [] }
        let postOnset = ordered.filter { $0.t >= first.t + config.onsetExclusionSeconds && sampleConfidence($0) >= config.minimumConfidence && hasUsableEye($0) }
        return postOnset
    }

    private func hasUsableEye(_ sample: GazeSample) -> Bool {
        let lo = finite(sample.leftEyeOpenness) ?? 1
        let ro = finite(sample.rightEyeOpenness) ?? 1
        let left = lo >= config.blinkOpennessThreshold && sample.leftPupil != nil
        let right = ro >= config.blinkOpennessThreshold && sample.rightPupil != nil
        return left || right
    }

    private func robustRows(_ candidates: [TrainingRow]) -> [TrainingRow] {
        guard candidates.count >= 4 else { return candidates }
        let count = candidates[0].features.count
        var medians = [Double](repeating: 0, count: count)
        var scales = [Double](repeating: 0, count: count)
        for i in 0..<count {
            let column = candidates.map { $0.features[i] }
            medians[i] = median(column)
            let deviations = column.map { abs($0 - medians[i]) }
            scales[i] = max(1.4826 * median(deviations), 1e-5)
        }
        var distances: [Double] = []
        distances.reserveCapacity(candidates.count)
        for row in candidates {
            var sum = 0.0
            for i in 0..<count {
                let z = (row.features[i] - medians[i]) / scales[i]
                sum += min(z * z, 400)
            }
            distances.append(sum.squareRoot())
        }
        let center = median(distances)
        let spread = max(1.4826 * median(distances.map { abs($0 - center) }), 1e-5)
        let limit = center + config.outlierMADs * spread
        let kept = candidates.enumerated().filter { distances[$0.offset] <= limit }.map { $0.element }
        // Keep at least a small representative set for very short clips.
        if kept.count >= 2 { return kept }
        return candidates.enumerated().sorted { distances[$0.offset] < distances[$1.offset] }.prefix(min(2, candidates.count)).map { $0.element }
    }

    /// Replace short chronological runs with component-wise medians.  This
    /// reduces landmark noise while retaining several windows of each target,
    /// so head-pose changes within a fixation remain visible to the regressor.
    private func medianBins(_ rows: [TrainingRow], binSize: Int) -> [TrainingRow] {
        guard rows.count > 1, binSize > 1 else { return rows }
        let ordered = rows.sorted { $0.time < $1.time }
        var result: [TrainingRow] = []
        result.reserveCapacity((ordered.count + binSize - 1) / binSize)
        var start = 0
        while start < ordered.count {
            let end = min(start + binSize, ordered.count)
            let bin = Array(ordered[start..<end])
            let featureCount = bin[0].features.count
            var featureMedians = [Double](repeating: 0, count: featureCount)
            for i in 0..<featureCount { featureMedians[i] = median(bin.map { $0.features[i] }) }
            result.append(TrainingRow(time: median(bin.map { $0.time }), features: featureMedians,
                                      target: bin[0].target, groupIndex: bin[0].groupIndex,
                                      confidence: median(bin.map { $0.confidence })))
            start = end
        }
        return result
    }

    private func sampleConfidence(_ sample: GazeSample) -> Double {
        let base = min(max(finite(sample.confidence) ?? 0, 0), 1)
        let lo = finite(sample.leftEyeOpenness) ?? 1
        let ro = finite(sample.rightEyeOpenness) ?? 1
        let eyeFactor: Double
        if hasUsableEye(sample) {
            eyeFactor = (sample.leftPupil != nil && lo >= config.blinkOpennessThreshold && sample.rightPupil != nil && ro >= config.blinkOpennessThreshold) ? 1 : 0.62
        } else {
            eyeFactor = 0
        }
        return base * eyeFactor
    }

    private func estimateConfidence(_ sample: GazeSample) -> Double {
        let base = sampleConfidence(sample)
        guard base >= config.minimumConfidence else { return 0 }
        return min(max(base, 0), 1)
    }

    // MARK: Regression and validation

    private struct SolvedWeights { var x: [Double]; var y: [Double] }

    private func solve(rows: [TrainingRow], featureCount: Int, lambda: Double) -> SolvedWeights? {
        guard !rows.isEmpty, featureCount > 0 else { return nil }
        var normal = Matrix(rows: featureCount, cols: featureCount)
        var rhs = Matrix(rows: featureCount, cols: 2)
        for row in rows {
            let weight = max(row.confidence, 1e-5)
            for i in 0..<featureCount {
                let fi = row.features[i]
                rhs[i, 0] += weight * fi * row.target.x
                rhs[i, 1] += weight * fi * row.target.y
                for j in 0..<featureCount { normal[i, j] += weight * fi * row.features[j] }
            }
        }
        let ridge = max(lambda.isFinite ? lambda : config.defaultLambda, 1e-9)
        // Do not penalise the intercept; tiny diagonal jitter handles an exact
        // duplicate feature column without changing the fitted bias.
        normal[0, 0] += 1e-9
        for i in 1..<featureCount { normal[i, i] += ridge }
        guard let solved = normal.choleskySolve(rhs) else { return nil }
        guard solved.rows == featureCount, solved.cols == 2 else { return nil }
        return SolvedWeights(x: solved.data.enumerated().filter { $0.offset % 2 == 0 }.map { $0.element },
                             y: solved.data.enumerated().filter { $0.offset % 2 == 1 }.map { $0.element })
    }

    private func chooseLambda(rows: [TrainingRow], featureCount: Int) -> Double {
        let candidates = config.lambdaCandidates.isEmpty ? [config.defaultLambda] : config.lambdaCandidates
        let groups = Set(rows.map { $0.groupIndex })
        guard groups.count > 1 else { return config.defaultLambda }
        var best = config.defaultLambda
        var bestError = Double.greatestFiniteMagnitude
        for candidate in candidates {
            let errors = crossValidatedErrors(rows: rows, featureCount: featureCount, lambda: candidate)
            guard !errors.isEmpty else { continue }
            let score = errors.reduce(0, +) / Double(errors.count)
            if score < bestError { bestError = score; best = candidate }
        }
        return best
    }

    private func crossValidatedErrors(rows: [TrainingRow], featureCount: Int, lambda: Double) -> [Double] {
        let groups = Array(Set(rows.map { $0.groupIndex })).sorted()
        var errors: [Double] = []
        errors.reserveCapacity(groups.count)
        for heldOut in groups {
            let trainRows = rows.filter { $0.groupIndex != heldOut }
            let testRows = rows.filter { $0.groupIndex == heldOut }
            guard let solved = solve(rows: trainRows, featureCount: featureCount, lambda: lambda), !testRows.isEmpty else { return [] }
            var points: [Point2] = []
            points.reserveCapacity(testRows.count)
            for row in testRows {
                let x = dot(solved.x, row.features)
                let y = dot(solved.y, row.features)
                if x.isFinite && y.isFinite { points.append(Point2(x, y)) }
            }
            guard !points.isEmpty else { return [] }
            errors.append(Self.medianPoint(points).distance(to: testRows[0].target))
        }
        return errors
    }

    private func predict(_ sample: GazeSample, model: Model, includeCorrection: Bool) -> Point2? {
        guard RidgeGazeCalibrator.isValid(model) else { return nil }
        if let wx = model.appearanceWeightsX, let wy = model.appearanceWeightsY, wx.count == model.featureCount + 5,
           wy.count == wx.count, sample.appearanceGaze != nil,
           let raw = rawFeatures(sample, extraCount: model.extraFeatureCount, headPose: model.headPoseFeatures, appearance: true), raw.hasFinite {
            let features = makeFeatures(raw.values, extraCount: model.extraFeatureCount, headPose: model.headPoseFeatures, polynomial: model.polynomialFeatures)
            if features.count == wx.count {
                var point = Point2(dot(wx, features), dot(wy, features))
                if includeCorrection { point = point + model.correctionBias }
                if point.x.isFinite, point.y.isFinite { return point }
            }
        }
        guard let raw = rawFeatures(sample, extraCount: model.extraFeatureCount, headPose: model.headPoseFeatures), raw.hasFinite else { return nil }
        let features = makeFeatures(raw.values, extraCount: model.extraFeatureCount, headPose: model.headPoseFeatures, polynomial: model.polynomialFeatures)
        guard features.count == model.featureCount else { return nil }
        var point = Point2(dot(model.weightsX, features), dot(model.weightsY, features))
        if includeCorrection { point = point + model.correctionBias }
        guard point.x.isFinite, point.y.isFinite else { return nil }
        return point
    }

    // MARK: Online corrections and helpers

    private func correctionMedian() -> Point2 {
        guard !corrections.isEmpty else { return model?.correctionBias ?? .zero }
        return Point2(median(corrections.map { $0.x }), median(corrections.map { $0.y }))
    }

    private static func isValid(_ model: Model) -> Bool {
        model.featureCount >= 20 && model.extraFeatureCount >= 0 &&
        model.extraFeatureCount <= model.featureCount &&
        model.featureCount - model.extraFeatureCount == 20 + (model.polynomialFeatures ? 14 + (model.headPoseFeatures ? 10 : 0) : 0) &&
        model.weightsX.count == model.featureCount && model.weightsY.count == model.featureCount &&
        model.weightsX.allSatisfy { $0.isFinite } && model.weightsY.allSatisfy { $0.isFinite } &&
        model.lambda.isFinite && model.lambda > 0 && model.correctionBias.x.isFinite && model.correctionBias.y.isFinite
    }

    private func finite(_ value: Double?) -> Double? {
        guard let value, value.isFinite else { return nil }
        return value
    }

    private func dot(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count else { return .nan }
        var result = 0.0
        for i in 0..<a.count { result += a[i] * b[i] }
        return result
    }

    private func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) * 0.5 : sorted[middle]
    }

    private static func medianPoint(_ values: [Point2]) -> Point2 {
        Point2(values.map { $0.x }.sorted().middleValue, values.map { $0.y }.sorted().middleValue)
    }

    private func percentile(_ values: [Double], _ quantile: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let position = min(max(quantile, 0), 1) * Double(sorted.count - 1)
        let lower = Int(position.rounded(.down))
        let upper = Int(position.rounded(.up))
        if lower == upper { return sorted[lower] }
        let fraction = position - Double(lower)
        return sorted[lower] * (1 - fraction) + sorted[upper] * fraction
    }
}

private extension Array where Element == Double {
    var middleValue: Double {
        guard !isEmpty else { return 0 }
        let i = count / 2
        return count.isMultiple(of: 2) ? (self[i - 1] + self[i]) * 0.5 : self[i]
    }
}

