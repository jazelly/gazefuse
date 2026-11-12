// Matrix.swift — tiny dense row-major Double matrix (no dependencies).
// Shared by the EKF (fusion) and ridge regression (gaze calibration).
import Foundation

public struct Matrix: Equatable, CustomStringConvertible {
    public let rows: Int
    public let cols: Int
    public var data: [Double]

    public init(rows: Int, cols: Int, repeating v: Double = 0) {
        self.rows = rows; self.cols = cols; data = Array(repeating: v, count: rows * cols)
    }
    public init(_ rowsArray: [[Double]]) {
        rows = rowsArray.count; cols = rowsArray.first?.count ?? 0
        data = rowsArray.flatMap { $0 }
        precondition(data.count == rows * cols, "ragged matrix")
    }
    public init(column: [Double]) { rows = column.count; cols = 1; data = column }
    public static func identity(_ n: Int) -> Matrix {
        var m = Matrix(rows: n, cols: n); for i in 0..<n { m[i, i] = 1 }; return m
    }
    public static func diagonal(_ d: [Double]) -> Matrix {
        var m = Matrix(rows: d.count, cols: d.count); for (i, v) in d.enumerated() { m[i, i] = v }; return m
    }

    @inline(__always) public subscript(r: Int, c: Int) -> Double {
        get { data[r * cols + c] }
        set { data[r * cols + c] = newValue }
    }
    public var transposed: Matrix {
        var t = Matrix(rows: cols, cols: rows)
        for r in 0..<rows { for c in 0..<cols { t[c, r] = self[r, c] } }
        return t
    }
    public var column: [Double] { precondition(cols == 1); return data }

    public static func + (a: Matrix, b: Matrix) -> Matrix {
        precondition(a.rows == b.rows && a.cols == b.cols)
        var m = a; for i in 0..<m.data.count { m.data[i] += b.data[i] }; return m
    }
    public static func - (a: Matrix, b: Matrix) -> Matrix {
        precondition(a.rows == b.rows && a.cols == b.cols)
        var m = a; for i in 0..<m.data.count { m.data[i] -= b.data[i] }; return m
    }
    public static func * (a: Matrix, s: Double) -> Matrix { var m = a; for i in 0..<m.data.count { m.data[i] *= s }; return m }
    public static func * (a: Matrix, b: Matrix) -> Matrix {
        precondition(a.cols == b.rows, "dim mismatch \(a.rows)x\(a.cols) * \(b.rows)x\(b.cols)")
        var m = Matrix(rows: a.rows, cols: b.cols)
        for i in 0..<a.rows {
            for k in 0..<a.cols {
                let av = a[i, k]; if av == 0 { continue }
                for j in 0..<b.cols { m.data[i * m.cols + j] += av * b.data[k * b.cols + j] }
            }
        }
        return m
    }

    /// Inverse via Gauss–Jordan with partial pivoting. nil if singular.
    public var inverse: Matrix? {
        precondition(rows == cols)
        let n = rows
        var a = self, inv = Matrix.identity(n)
        for col in 0..<n {
            var piv = col; var best = abs(a[col, col])
            for r in (col + 1)..<max(n, col + 1) where abs(a[r, col]) > best { best = abs(a[r, col]); piv = r }
            if best < 1e-14 { return nil }
            if piv != col {
                for c in 0..<n { a.data.swapAt(col * n + c, piv * n + c); inv.data.swapAt(col * n + c, piv * n + c) }
            }
            let d = a[col, col]
            for c in 0..<n { a[col, c] /= d; inv[col, c] /= d }
            for r in 0..<n where r != col {
                let f = a[r, col]; if f == 0 { continue }
                for c in 0..<n { a[r, c] -= f * a[col, c]; inv[r, c] -= f * inv[col, c] }
            }
        }
        return inv
    }

    /// Solve A x = B for symmetric positive-definite A (Cholesky). nil if not SPD.
    public func choleskySolve(_ b: Matrix) -> Matrix? {
        precondition(rows == cols && b.rows == rows)
        let n = rows
        var L = Matrix(rows: n, cols: n)
        for i in 0..<n {
            for j in 0...i {
                var s = self[i, j]
                for k in 0..<j { s -= L[i, k] * L[j, k] }
                if i == j { if s <= 0 { return nil }; L[i, i] = s.squareRoot() } else { L[i, j] = s / L[j, j] }
            }
        }
        var x = b
        for c in 0..<b.cols {
            var y = [Double](repeating: 0, count: n)
            for i in 0..<n { var s = b[i, c]; for k in 0..<i { s -= L[i, k] * y[k] }; y[i] = s / L[i, i] }
            for i in stride(from: n - 1, through: 0, by: -1) {
                var s = y[i]; for k in (i + 1)..<max(n, i + 1) { s -= L[k, i] * x[k, c] }; x[i, c] = s / L[i, i]
            }
        }
        return x
    }

    public var description: String {
        (0..<rows).map { r in (0..<cols).map { String(format: "%9.4f", self[r, $0]) }.joined(separator: " ") }.joined(separator: "\n")
    }
}
