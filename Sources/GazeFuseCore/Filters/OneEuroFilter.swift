// OneEuroFilter.swift — the "1€ filter" (Casiez, Roussel & Vogel, CHI 2012).
//
// A first-order low-pass whose cutoff frequency rises with the signal's speed:
//     cutoff = minCutoff + beta * |dx/dt|
// so slow signals (resting cursor) get heavy smoothing (little jitter) while fast
// signals (saccades, quick head moves) get a high cutoff (little lag). The
// derivative itself is low-passed with a fixed `dCutoff` before being used.
//
// Both a scalar and an isotropic 2-D variant are provided. The 2-D variant uses the
// *magnitude* of the 2-D derivative to drive the cutoff of both axes, so diagonal
// motion is treated the same as axis-aligned motion (the reference implementation
// filters axes independently, which makes the lag direction-dependent).
//
// Reference: G. Casiez, N. Roussel, D. Vogel. "1€ Filter: A Simple Speed-based
// Low-pass Filter for Noisy Input in Interactive Systems", CHI 2012.
import Foundation

/// Tunables of a One Euro filter. Units of `beta` are 1/(signal unit) so that
/// `beta * speed` is a frequency in Hz; for screen points (speeds of 100–3000 pt/s)
/// values around 0.001–0.02 are sensible.
public struct OneEuroConfig: Codable, Equatable {
    /// Minimum cutoff frequency in Hz. Lower = less jitter at rest but more lag.
    public var minCutoff: Double
    /// Speed coefficient. Higher = cutoff rises faster with speed = less lag when moving.
    public var beta: Double
    /// Cutoff (Hz) of the low-pass applied to the derivative estimate.
    public var dCutoff: Double

    public init(minCutoff: Double = 1.0, beta: Double = 0.007, dCutoff: Double = 1.0) {
        self.minCutoff = minCutoff; self.beta = beta; self.dCutoff = dCutoff
    }

    /// Smoothing factor of a first-order low-pass with the given cutoff and sample interval.
    @inline(__always) static func alpha(cutoff: Double, dt: TimeSec) -> Double {
        let tau = 1.0 / (2.0 * Double.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / dt)
    }
}

/// Scalar One Euro filter. Value type; call `filter(_:at:)` with monotonic timestamps.
public struct OneEuroFilter {
    public var config: OneEuroConfig
    private var xHat: Double?
    private var dxHat: Double = 0
    private var tPrev: TimeSec = 0

    public init(config: OneEuroConfig = OneEuroConfig()) { self.config = config }

    /// Last filtered value (nil before the first sample).
    public var value: Double? { xHat }

    public mutating func reset() { xHat = nil; dxHat = 0; tPrev = 0 }

    /// Filters sample `x` taken at time `t`. Non-increasing timestamps are treated as
    /// a repeated sample (the previous estimate is returned unchanged).
    @discardableResult
    public mutating func filter(_ x: Double, at t: TimeSec) -> Double {
        guard let prev = xHat else { xHat = x; dxHat = 0; tPrev = t; return x }
        let dt = t - tPrev
        guard dt > 0 else { return prev }
        tPrev = t
        let dx = (x - prev) / dt
        let aD = OneEuroConfig.alpha(cutoff: config.dCutoff, dt: dt)
        dxHat += aD * (dx - dxHat)
        let cutoff = config.minCutoff + config.beta * abs(dxHat)
        let a = OneEuroConfig.alpha(cutoff: cutoff, dt: dt)
        let out = prev + a * (x - prev)
        xHat = out
        return out
    }
}

/// Isotropic 2-D One Euro filter (shared, speed-magnitude-driven cutoff for both axes).
public struct OneEuroFilter2D {
    public var config: OneEuroConfig
    private var pHat: Point2?
    private var dHat: Point2 = .zero
    private var tPrev: TimeSec = 0

    public init(config: OneEuroConfig = OneEuroConfig()) { self.config = config }

    public var value: Point2? { pHat }

    public mutating func reset() { pHat = nil; dHat = .zero; tPrev = 0 }

    @discardableResult
    public mutating func filter(_ p: Point2, at t: TimeSec) -> Point2 {
        guard let prev = pHat else { pHat = p; dHat = .zero; tPrev = t; return p }
        let dt = t - tPrev
        guard dt > 0 else { return prev }
        tPrev = t
        let d = (p - prev) / dt
        let aD = OneEuroConfig.alpha(cutoff: config.dCutoff, dt: dt)
        dHat = dHat + (d - dHat) * aD
        let cutoff = config.minCutoff + config.beta * dHat.length
        let a = OneEuroConfig.alpha(cutoff: cutoff, dt: dt)
        let out = prev + (p - prev) * a
        pHat = out
        return out
    }
}
