// SimRNG.swift — deterministic SplitMix64 generator for the simulator and tests.
import Foundation

public final class SimRNG {
    private var state: UInt64
    private var spareGaussian: Double?

    public init(seed: UInt64) { state = seed &+ 0x9E37_79B9_7F4A_7C15 }

    public func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    public func uniform() -> Double { Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0) }
    public func uniform(_ a: Double, _ b: Double) -> Double { a + (b - a) * uniform() }
    public func int(_ n: Int) -> Int { Int(next() % UInt64(max(n, 1))) }
    public func bernoulli(_ p: Double) -> Bool { uniform() < p }
    public func exponential(rate: Double) -> Double { rate <= 0 ? .infinity : -log(1 - uniform()) / rate }

    /// Standard normal (Marsaglia polar method, cached pair).
    public func gaussian() -> Double {
        if let s = spareGaussian { spareGaussian = nil; return s }
        var u = 0.0, v = 0.0, s = 0.0
        repeat {
            u = uniform(-1, 1); v = uniform(-1, 1); s = u * u + v * v
        } while s >= 1 || s == 0
        let m = (-2 * log(s) / s).squareRoot()
        spareGaussian = v * m
        return u * m
    }
    public func gaussian(mean: Double, sd: Double) -> Double { mean + sd * gaussian() }
    public func gaussian2D(sd: Double) -> Point2 { Point2(gaussian() * sd, gaussian() * sd) }

    /// Log-normal with the given mean and coefficient of variation (sd/mean).
    public func lognormal(mean: Double, cv: Double) -> Double {
        guard mean > 0, cv > 0 else { return mean }
        let s2 = log(1 + cv * cv)
        let mu = log(mean) - s2 / 2
        return exp(mu + s2.squareRoot() * gaussian())
    }

    /// Unit vector in a uniformly random direction.
    public func direction() -> Point2 { let a = uniform(0, 2 * .pi); return Point2(cos(a), sin(a)) }

    /// Derive an independent generator (for sub-streams).
    public func fork() -> SimRNG { SimRNG(seed: next()) }
}
