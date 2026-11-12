import XCTest
@testable import GazeFuseCore

/// Small deterministic generator used by the filter tests.  Keeping the noise
/// source local makes the tests reproducible without introducing a dependency.
private struct FilterTestRNG {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func nextUnit() -> Double {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        z ^= z >> 31
        return Double(z >> 11) * (1.0 / 9007199254740992.0)
    }

    mutating func normal() -> Double {
        // Box–Muller. The lower bound avoids log(0) while retaining a fixed stream.
        let u1 = max(nextUnit(), 1e-12)
        let u2 = nextUnit()
        return (-2.0 * log(u1)).squareRoot() * cos(2.0 * Double.pi * u2)
    }
}

final class FilterTests: XCTestCase {
    func testOneEuroReducesRestJitter() {
        var rng = FilterTestRNG(seed: 0xF17E_0001)
        var filter = OneEuroFilter(config: OneEuroConfig(minCutoff: 1.0, beta: 0.007, dCutoff: 1.0))
        let truth = 240.0
        let dt = 1.0 / 60.0
        var rawSquared = 0.0
        var filteredSquared = 0.0
        var count = 0

        for i in 0..<600 {
            let sample = truth + 3.0 * rng.normal()
            let estimate = filter.filter(sample, at: Double(i) * dt)
            if i >= 60 {
                rawSquared += (sample - truth) * (sample - truth)
                filteredSquared += (estimate - truth) * (estimate - truth)
                count += 1
            }
        }

        let rawRMS = (rawSquared / Double(count)).squareRoot()
        let filteredRMS = (filteredSquared / Double(count)).squareRoot()
        print("Filter metric oneEuroRest rawRMS=\(rawRMS) filteredRMS=\(filteredRMS)")
        XCTAssertLessThan(filteredRMS, rawRMS * 0.55, "resting gaze should be strongly smoothed")
    }

    func testOneEuroTracksFastMoveWithSmallLag() {
        var rng = FilterTestRNG(seed: 0xF17E_0002)
        var filter = OneEuroFilter(config: OneEuroConfig(minCutoff: 1.0, beta: 0.02, dCutoff: 1.0))
        let dt = 1.0 / 60.0
        var absoluteError = 0.0
        var count = 0

        for i in 0..<120 {
            let t = Double(i) * dt
            let truth = 100.0 + 500.0 * t
            let estimate = filter.filter(truth + 0.5 * rng.normal(), at: t)
            if i >= 30 {
                absoluteError += abs(estimate - truth)
                count += 1
            }
        }

        let meanAbsoluteError = absoluteError / Double(count)
        print("Filter metric oneEuroFast meanAbsoluteError=\(meanAbsoluteError)")
        XCTAssertLessThan(meanAbsoluteError, 18.0, "speed-adaptive cutoff should limit ramp lag")
    }

    func testOneEuro2DUsesIsotropicSpeed() {
        var filter = OneEuroFilter2D(config: OneEuroConfig(minCutoff: 1.0, beta: 0.01, dCutoff: 1.0))
        let dt = 1.0 / 60.0
        var previous = Point2.zero
        for i in 0..<30 {
            previous = Point2(Double(i) * 2.0, Double(i) * 2.0)
            _ = filter.filter(previous, at: Double(i) * dt)
        }
        let diagonal = filter.value!
        var axisFilter = OneEuroFilter2D(config: OneEuroConfig(minCutoff: 1.0, beta: 0.01, dCutoff: 1.0))
        for i in 0..<30 { _ = axisFilter.filter(Point2(Double(i) * 2.0, 0), at: Double(i) * dt) }
        let axis = axisFilter.value!
        XCTAssertGreaterThan(diagonal.x, 40)
        XCTAssertGreaterThan(diagonal.y, 40)
        // The two axes of an isotropic filter receive the same cutoff.
        XCTAssertEqual(diagonal.x - diagonal.y, 0, accuracy: 1e-12)
        XCTAssertGreaterThan(axis.x, 40)
    }

    func testExtendedKalmanFilterLinearConverges() {
        var rng = FilterTestRNG(seed: 0xE1F1)
        var filter = ExtendedKalmanFilter(
            state: [0, 0],
            covariance: .diagonal([100, 25])
        )
        let dt = 0.1
        let F = Matrix([[1, dt], [0, 1]])
        let q = Matrix([[0.0025, 0], [0, 0.01]])
        let H = Matrix([[1, 0]])
        let R = Matrix([[0.25]])
        var positionSquaredError = 0.0
        var velocitySquaredError = 0.0

        for i in 0..<120 {
            filter.predict(F: F, Q: q)
            let t = Double(i + 1) * dt
            let truth = 5.0 + 2.0 * t
            _ = filter.update(z: [truth + 0.5 * rng.normal()], H: H, R: R)
            positionSquaredError += (filter.x[0] - truth) * (filter.x[0] - truth)
            velocitySquaredError += (filter.x[1] - 2.0) * (filter.x[1] - 2.0)
        }

        let positionRMSE = (positionSquaredError / 120.0).squareRoot()
        let velocityRMSE = (velocitySquaredError / 120.0).squareRoot()
        print("Filter metric linearEKF positionRMSE=\(positionRMSE) velocityRMSE=\(velocityRMSE) finalPosition=\(filter.x[0]) finalVelocity=\(filter.x[1])")
        XCTAssertEqual(filter.x[0], 5.0 + 2.0 * 120.0 * dt, accuracy: 0.45)
        XCTAssertEqual(filter.x[1], 2.0, accuracy: 0.25)
        XCTAssertEqual(filter.P[0, 1], filter.P[1, 0], accuracy: 1e-12)
        XCTAssertGreaterThan(filter.P[0, 0], 0)
        XCTAssertGreaterThan(filter.P[1, 1], 0)
    }

    func testExtendedKalmanFilterNonlinearRangeBearing() {
        var rng = FilterTestRNG(seed: 0xE1F2)
        var filter = ExtendedKalmanFilter(
            state: [4.5, 3.5],
            covariance: .diagonal([16, 16])
        )
        let truth = Point2(10, 8)
        let range = truth.length
        let bearing = atan2(truth.y, truth.x)
        let R = Matrix.diagonal([0.15 * 0.15, 0.01 * 0.01])

        for _ in 0..<45 {
            let h = { (state: [Double]) -> [Double] in
                [((state[0] * state[0] + state[1] * state[1]).squareRoot()), atan2(state[1], state[0])]
            }
            var z = [range + 0.15 * rng.normal(), bearing + 0.01 * rng.normal()]
            // This trajectory stays away from the ±π seam; no angle unwrap is needed.
            let result = filter.update(
                z: z,
                h: h,
                H: { state in
                    let px = state[0], py = state[1]
                    let r2 = max(px * px + py * py, 1e-12)
                    let r = r2.squareRoot()
                    return Matrix([[px / r, py / r], [-py / r2, px / r2]])
                },
                R: { _ in R }
            )
            XCTAssertTrue(result.applied)
            z.removeAll(keepingCapacity: true)
        }

        let positionError = Point2(filter.x[0], filter.x[1]).distance(to: truth)
        print("Filter metric rangeBearing positionError=\(positionError) final=\(filter.x[0]),\(filter.x[1])")
        XCTAssertEqual(filter.x[0], truth.x, accuracy: 0.35)
        XCTAssertEqual(filter.x[1], truth.y, accuracy: 0.35)
    }

    func testExtendedKalmanFilterClosureModelPrediction() {
        var filter = ExtendedKalmanFilter(state: [0], covariance: Matrix([[1]]))
        filter.predict(
            f: { state in [state[0] + 2] },
            F: { _ in Matrix([[1]]) },
            Q: { _ in Matrix([[0.25]]) }
        )
        XCTAssertEqual(filter.x[0], 2, accuracy: 1e-12)
        XCTAssertEqual(filter.P[0, 0], 1.25, accuracy: 1e-12)
    }

    func testNISHasChiSquareOneDistribution() {
        var rng = FilterTestRNG(seed: 0xE1F3)
        let H = Matrix([[1]])
        let R = Matrix([[4]])
        let expectedVariance = 5.0
        var sum = 0.0
        var inside95 = 0
        let sampleCount = 4000

        for _ in 0..<sampleCount {
            let filter = ExtendedKalmanFilter(state: [0], covariance: Matrix([[1]]))
            let z = sqrt(expectedVariance) * rng.normal()
            let result = filter.innovation(z: [z], H: H, R: R)!
            sum += result.nis
            if result.nis <= ChiSquareGate.quantile(dof: 1, p: 0.95) { inside95 += 1 }
        }

        let mean = sum / Double(sampleCount)
        let fraction = Double(inside95) / Double(sampleCount)
        print("Filter metric NIS mean=\(mean) fraction95=\(fraction)")
        XCTAssertEqual(mean, 1.0, accuracy: 0.08)
        XCTAssertEqual(fraction, 0.95, accuracy: 0.025)
    }

    func testNISGateRejectsOutlierAndLeavesStateUntouched() {
        var filter = ExtendedKalmanFilter(state: [0], covariance: Matrix([[1]]))
        let H = Matrix([[1]])
        let R = Matrix([[0.04]])
        let accepted = filter.update(z: [0.2], H: H, R: R, gate: ChiSquareGate.quantile(dof: 1, p: 0.99))
        XCTAssertTrue(accepted.applied)
        let stateBeforeOutlier = filter.x
        let covarianceBeforeOutlier = filter.P
        let rejected = filter.update(z: [100], H: H, R: R, gate: ChiSquareGate.quantile(dof: 1, p: 0.99))
        XCTAssertFalse(rejected.applied)
        XCTAssertGreaterThan(rejected.nis, 6.635)
        XCTAssertEqual(filter.x, stateBeforeOutlier)
        XCTAssertEqual(filter.P, covarianceBeforeOutlier)
    }
}
