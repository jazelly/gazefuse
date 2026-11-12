// ExtendedKalmanFilter.swift — a small, generic EKF over `Matrix`.
//
// State x (n), covariance P (n×n). The caller supplies the model per step:
//   predict:  x ← f(x) (or F·x when f is nil),  P ← F P Fᵀ + Q
//   update:   y = z − h(x) (or z − H·x),  S = H P Hᵀ + R,  K = P Hᵀ S⁻¹,
//             x ← x + K y,  P ← (I − K H) P (I − K H)ᵀ + K R Kᵀ   (Joseph form)
// The Joseph form keeps P symmetric positive semi-definite under round-off, which
// matters for long-running real-time filters. `update` returns the normalized
// innovation squared NIS = yᵀ S⁻¹ y; for a consistent filter NIS ~ χ²(m). Passing a
// `gate` (a χ² quantile) rejects measurements whose NIS exceeds it (outliers) —
// the result reports `applied == false` and the state is untouched.
//
// References: Bar-Shalom, Li & Kirubarajan, "Estimation with Applications to
// Tracking and Navigation" (2001), §5.2 (EKF), §5.4 (NIS consistency), §11.2 (gating).
import Foundation

/// Result of an EKF measurement update.
public struct EKFUpdateResult {
    /// Innovation y = z − h(x⁻).
    public var innovation: [Double]
    /// Innovation covariance S = H P⁻ Hᵀ + R.
    public var innovationCovariance: Matrix
    /// Normalized innovation squared yᵀ S⁻¹ y (∞ if S was singular).
    public var nis: Double
    /// False if the measurement was rejected by the gate (or S was singular).
    public var applied: Bool
}

public struct ExtendedKalmanFilter {
    public var x: [Double]
    public var P: Matrix

    public init(state: [Double], covariance: Matrix) {
        precondition(covariance.rows == state.count && covariance.cols == state.count, "P must be n×n")
        x = state; P = covariance
    }

    public var stateCount: Int { x.count }

    /// Time update. `F` is the Jacobian of `f` (or the transition matrix itself for a
    /// linear model, in which case `f` may be nil).
    public mutating func predict(F: Matrix, Q: Matrix, f: (([Double]) -> [Double])? = nil) {
        if let f = f { x = f(x) } else { x = (F * Matrix(column: x)).column }
        P = F * P * F.transposed + Q
        symmetrize()
    }

    /// Closure-model form of the time update. `F` and `Q` are evaluated at the
    /// prior state, as required by the usual first-order EKF linearisation. This
    /// keeps state-dependent dynamics on the same Joseph/NIS implementation path
    /// as the matrix form above.
    public mutating func predict(f: ([Double]) -> [Double],
                                F: ([Double]) -> Matrix,
                                Q: ([Double]) -> Matrix) {
        let prior = x
        let transition = F(prior)
        let process = Q(prior)
        x = f(prior)
        P = transition * P * transition.transposed + process
        symmetrize()
    }

    /// Innovation, its covariance and the NIS for measurement `z` *without* applying it.
    /// Returns nil if S is singular.
    public func innovation(z: [Double], H: Matrix, R: Matrix,
                           h: (([Double]) -> [Double])? = nil) -> (y: [Double], S: Matrix, Sinv: Matrix, nis: Double)? {
        let hx = h?(x) ?? (H * Matrix(column: x)).column
        precondition(hx.count == z.count && H.rows == z.count && H.cols == x.count, "measurement dimension mismatch")
        var y = z
        for i in 0..<y.count { y[i] -= hx[i] }
        let S = H * P * H.transposed + R
        guard let Sinv = S.inverse else { return nil }
        var nis = 0.0
        for i in 0..<y.count { for j in 0..<y.count { nis += y[i] * Sinv[i, j] * y[j] } }
        return (y, S, Sinv, nis)
    }

    /// Measurement update (Joseph form). If `gate` is given and NIS > gate the
    /// measurement is rejected and the state is left unchanged.
    @discardableResult
    public mutating func update(z: [Double], H: Matrix, R: Matrix,
                                h: (([Double]) -> [Double])? = nil, gate: Double? = nil,
                                consider: [Int] = []) -> EKFUpdateResult {
        guard let (y, S, Sinv, nis) = innovation(z: z, H: H, R: R, h: h) else {
            return EKFUpdateResult(innovation: [], innovationCovariance: Matrix(rows: z.count, cols: z.count),
                                   nis: .infinity, applied: false)
        }
        if let g = gate, nis > g {
            return EKFUpdateResult(innovation: y, innovationCovariance: S, nis: nis, applied: false)
        }
        var K = P * H.transposed * Sinv
        // Schmidt-Kalman "consider" states: kept in the covariance but never corrected by this
        // measurement (the Joseph-form update below stays valid for any gain).
        for i in consider where i >= 0 && i < x.count { for j in 0..<K.cols { K[i, j] = 0 } }
        let Ky = (K * Matrix(column: y)).column
        for i in 0..<x.count { x[i] += Ky[i] }
        let IKH = Matrix.identity(x.count) - K * H
        P = IKH * P * IKH.transposed + K * R * K.transposed
        symmetrize()
        return EKFUpdateResult(innovation: y, innovationCovariance: S, nis: nis, applied: true)
    }

    /// Closure-model form of the measurement update. `h`, `H`, and `R` are all
    /// evaluated at the same prior state, so nonlinear sensors can provide their
    /// Jacobian and state-dependent covariance without mutating the filter first.
    @discardableResult
    public mutating func update(z: [Double],
                                h: @escaping ([Double]) -> [Double],
                                H: ([Double]) -> Matrix,
                                R: ([Double]) -> Matrix,
                                gate: Double? = nil) -> EKFUpdateResult {
        let prior = x
        let measurementJacobian = H(prior)
        let covariance = R(prior)
        return update(z: z, H: measurementJacobian, R: covariance, h: h, gate: gate)
    }

    /// P ← (P + Pᵀ)/2, removing round-off asymmetry.
    public mutating func symmetrize() {
        let n = P.rows
        for i in 0..<n {
            for j in (i + 1)..<max(n, i + 1) {
                let v = 0.5 * (P[i, j] + P[j, i]); P[i, j] = v; P[j, i] = v
            }
        }
    }
}

/// χ² quantiles commonly used as EKF gates (Bar-Shalom §11.2).
public enum ChiSquareGate {
    /// Upper quantile of χ²(dof) at probability p ∈ {0.95, 0.99, 0.999}, dof ∈ 1...4.
    public static func quantile(dof: Int, p: Double) -> Double {
        let table: [Int: [Double: Double]] = [
            1: [0.95: 3.841, 0.99: 6.635, 0.999: 10.828],
            2: [0.95: 5.991, 0.99: 9.210, 0.999: 13.816],
            3: [0.95: 7.815, 0.99: 11.345, 0.999: 16.266],
            4: [0.95: 9.488, 0.99: 13.277, 0.999: 18.467],
        ]
        precondition(table[dof]?[p] != nil, "unsupported dof/p")
        return table[dof]![p]!
    }
}
