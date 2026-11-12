// ConstantVelocityKF.swift — 2-D constant-velocity Kalman filter helper.
//
// State [x, y, vx, vy]; transition x += v·dt; process noise = discrete white-noise
// acceleration (DWNA, Bar-Shalom §6.3.2): per axis Q = σa² [[dt⁴/4, dt³/2],[dt³/2, dt²]].
// Measurements are positions with isotropic noise σ. This is a thin wrapper over
// `ExtendedKalmanFilter` and doubles as its linear test case.
import Foundation

public struct ConstantVelocityKF {
    public private(set) var ekf: ExtendedKalmanFilter
    /// Process noise: standard deviation of the (piecewise constant) acceleration, units/s².
    public var accelerationSigma: Double

    public init(position: Point2, velocity: Point2 = .zero, positionSigma: Double, velocitySigma: Double, accelerationSigma: Double) {
        ekf = ExtendedKalmanFilter(
            state: [position.x, position.y, velocity.x, velocity.y],
            covariance: .diagonal([positionSigma * positionSigma, positionSigma * positionSigma,
                                   velocitySigma * velocitySigma, velocitySigma * velocitySigma]))
        self.accelerationSigma = accelerationSigma
    }

    public var position: Point2 { Point2(ekf.x[0], ekf.x[1]) }
    public var velocity: Point2 { Point2(ekf.x[2], ekf.x[3]) }
    public var covariance: Matrix { ekf.P }

    public mutating func predict(dt: TimeSec) {
        guard dt > 0 else { return }
        var F = Matrix.identity(4)
        F[0, 2] = dt; F[1, 3] = dt
        let q = accelerationSigma * accelerationSigma
        let d2 = dt * dt, d3 = d2 * dt, d4 = d3 * dt
        var Q = Matrix(rows: 4, cols: 4)
        for a in 0..<2 {
            Q[a, a] = q * d4 / 4; Q[a, a + 2] = q * d3 / 2; Q[a + 2, a] = q * d3 / 2; Q[a + 2, a + 2] = q * d2
        }
        ekf.predict(F: F, Q: Q)
    }

    private static let H: Matrix = Matrix([[1, 0, 0, 0], [0, 1, 0, 0]])

    /// Position measurement with noise std `sigma`. Optional χ²(2) gate.
    @discardableResult
    public mutating func update(position z: Point2, sigma: Double, gate: Double? = nil) -> EKFUpdateResult {
        ekf.update(z: [z.x, z.y], H: Self.H, R: .diagonal([sigma * sigma, sigma * sigma]), gate: gate)
    }
}
