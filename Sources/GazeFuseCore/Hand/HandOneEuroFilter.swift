// HandOneEuroFilter.swift — One-Euro filter (Casiez, Roussel & Vogel, CHI 2012)
// used by the hand-gesture module to smooth pinch distances and the pinch point.
//
// A One-Euro filter is a first-order low-pass whose cutoff frequency rises with the
// speed of the signal:   cutoff = minCutoff + beta * |smoothed derivative|.
// At rest the low cutoff removes jitter; during fast motion the cutoff opens up so the
// output follows the input with little lag. This is exactly what a pinch classifier
// wants: a steady distance while the fingers are still, and a quick response when they
// close. The derivative itself is low-passed with a fixed `dCutoff`.
//
// alpha(cutoff, dt) = 1 / (1 + tau / dt),  tau = 1 / (2 pi cutoff)
//
// The filter is a value type so a classifier can embed several without allocation.
import Foundation

/// Scalar One-Euro filter. `filter(_:dt:)` must be called with the elapsed time since the
/// previous call (seconds); the first call after `init`/`reset` returns its input unchanged.
public struct HandOneEuroFilter: Codable, Equatable {
    /// Cutoff frequency at rest (Hz). Lower = smoother but laggier at low speed.
    public var minCutoff: Double
    /// Speed coefficient: how much the cutoff rises per unit of signal speed (signal units / s).
    public var beta: Double
    /// Cutoff (Hz) used to smooth the derivative estimate.
    public var dCutoff: Double

    /// The last filtered value (nil before the first sample).
    public private(set) var value: Double?
    private var dValue: Double = 0

    public init(minCutoff: Double = 1, beta: Double = 0, dCutoff: Double = 1) {
        self.minCutoff = minCutoff; self.beta = beta; self.dCutoff = dCutoff
    }

    @inline(__always)
    static func alpha(cutoff: Double, dt: Double) -> Double {
        let tau = 1 / (2 * Double.pi * cutoff)
        return 1 / (1 + tau / dt)
    }

    /// Feed one sample. `dt` is the time since the previous sample (seconds, > 0).
    @discardableResult
    public mutating func filter(_ x: Double, dt: Double) -> Double {
        guard let prev = value, dt > 0 else { value = x; dValue = 0; return x }
        let dx = (x - prev) / dt
        dValue += Self.alpha(cutoff: dCutoff, dt: dt) * (dx - dValue)
        let cutoff = minCutoff + beta * abs(dValue)
        let v = prev + Self.alpha(cutoff: cutoff, dt: dt) * (x - prev)
        value = v
        return v
    }

    public mutating func reset() { value = nil; dValue = 0 }
}

/// Two-dimensional One-Euro filter for points. Both axes share one cutoff derived from the
/// speed *magnitude*, so diagonal motion is not smoothed differently from axis-aligned motion.
public struct HandOneEuroFilter2D: Codable, Equatable {
    public var minCutoff: Double
    public var beta: Double
    public var dCutoff: Double
    public private(set) var value: Point2?
    private var dValue: Point2 = .zero

    public init(minCutoff: Double = 1, beta: Double = 0, dCutoff: Double = 1) {
        self.minCutoff = minCutoff; self.beta = beta; self.dCutoff = dCutoff
    }

    @discardableResult
    public mutating func filter(_ p: Point2, dt: Double) -> Point2 {
        guard let prev = value, dt > 0 else { value = p; dValue = .zero; return p }
        let d = (p - prev) / dt
        dValue = dValue + (d - dValue) * HandOneEuroFilter.alpha(cutoff: dCutoff, dt: dt)
        let cutoff = minCutoff + beta * dValue.length
        let v = prev + (p - prev) * HandOneEuroFilter.alpha(cutoff: cutoff, dt: dt)
        value = v
        return v
    }

    public mutating func reset() { value = nil; dValue = .zero }
}
