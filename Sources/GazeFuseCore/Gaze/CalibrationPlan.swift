// CalibrationPlan.swift
//
// Hardware-free target geometry used by the calibration UI and by deterministic
// tests.  Coordinates are in GazeFuse's global screen space: points, origin at
// the top-left of the main display, with +y down.

import Foundation

/// A deterministic set of targets for gaze calibration.
///
/// The four standard layouts deliberately use a small number of points so that a
/// first calibration can finish quickly.  `targets` is shuffled with the supplied
/// seed, while the geometry itself remains fixed and reproducible.  `validationTargets`
/// is generated from a separate lattice and does not intentionally reuse calibration
/// targets, allowing the UI to report an independent post-fit error.
public struct CalibrationPlan: Codable, Equatable {
    public enum Layout: Int, Codable, CaseIterable, Equatable {
        case five = 5
        case nine = 9
        case thirteen = 13
        case sixteen = 16

        public var pointCount: Int { rawValue }
    }

    /// Tunables are Codable so an application can save the exact calibration
    /// protocol used for a session.
    public struct Config: Codable, Equatable {
        /// Fraction of width/height kept clear around the screen edge (0...0.45).
        public var marginFraction: Double
        /// Stable order randomisation seed.  The same seed gives the same order.
        public var seed: UInt64
        /// Whether to shuffle the generated layout.  Geometry is never changed.
        public var randomizeOrder: Bool
        /// Time for which a calibration target is shown.
        public var targetDuration: TimeSec
        /// Number of points in the independent validation pass.
        public var validationPointCount: Int
        /// Time for which a validation target is shown.
        public var validationDuration: TimeSec
        /// Initial acquisition interval ignored by the UI's validation median.
        /// The mapper also applies its own onset rejection during fitting.
        public var settleDuration: TimeSec
        /// Time for which the measured validation summary remains visible.
        public var summaryDuration: TimeSec
        /// Assumptions used only to label the screen-point to visual-angle
        /// estimate in the UI report.
        public var assumedDisplayWidthMetres: Double
        public var assumedViewingDistanceMetres: Double

        public init(marginFraction: Double = 0.08,
                    seed: UInt64 = 0x475A_4643_414C_4942,
                    randomizeOrder: Bool = true,
                    targetDuration: TimeSec = 1.5,
                    validationPointCount: Int = 5,
                    validationDuration: TimeSec = 0.8,
                    settleDuration: TimeSec = 0.3,
                    summaryDuration: TimeSec = 2.0,
                    assumedDisplayWidthMetres: Double = 0.344,
                    assumedViewingDistanceMetres: Double = 0.60) {
            self.marginFraction = min(max(marginFraction, 0), 0.45)
            self.seed = seed
            self.randomizeOrder = randomizeOrder
            self.targetDuration = max(targetDuration, 0.1)
            self.validationPointCount = min(max(validationPointCount, 1), 16)
            self.validationDuration = max(validationDuration, 0.1)
            self.settleDuration = min(max(settleDuration, 0), max(self.validationDuration - 0.05, 0))
            self.summaryDuration = max(summaryDuration, 0.1)
            self.assumedDisplayWidthMetres = max(assumedDisplayWidthMetres, 0.01)
            self.assumedViewingDistanceMetres = max(assumedViewingDistanceMetres, 0.05)
        }
    }

    public let screen: ScreenRect
    public let layout: Layout
    public let config: Config
    public let targets: [Point2]
    public let validationTargets: [Point2]

    public init(screen: ScreenRect, layout: Layout = .nine, config: Config = Config()) {
        self.screen = screen
        self.layout = layout
        self.config = config

        let calibration = Self.geometry(layout: layout, screen: screen, margin: config.marginFraction)
        self.targets = config.randomizeOrder ? Self.shuffled(calibration, seed: config.seed) : calibration

        let validation = Self.validationGeometry(screen: screen,
                                                 margin: config.marginFraction,
                                                 excluding: calibration,
                                                 count: config.validationPointCount)
        self.validationTargets = config.randomizeOrder
            ? Self.shuffled(validation, seed: config.seed ^ 0x9E37_79B9_7F4A_7C15)
            : validation
    }

    private static func geometry(layout: Layout, screen: ScreenRect, margin: Double) -> [Point2] {
        func point(_ u: Double, _ v: Double) -> Point2 {
            let x = screen.origin.x + screen.width * (margin + u * (1 - 2 * margin))
            let y = screen.origin.y + screen.height * (margin + v * (1 - 2 * margin))
            return Point2(x, y)
        }

        switch layout {
        case .five:
            return [point(0, 0), point(1, 0), point(0.5, 0.5), point(0, 1), point(1, 1)]
        case .nine:
            return [0, 0.5, 1].flatMap { v in [0, 0.5, 1].map { u in point(u, v) } }
        case .thirteen:
            // A 3x3 scaffold plus quarter points on each edge.  This samples
            // the long edges without making the center disproportionately dense.
            let scaffold = [0, 0.5, 1].flatMap { v in [0, 0.5, 1].map { u in point(u, v) } }
            let edgeQuarter = [point(0.25, 0), point(0.75, 0), point(0.25, 1), point(0.75, 1)]
            return scaffold + edgeQuarter
        case .sixteen:
            let grid: [Double] = [0, 1.0 / 3.0, 2.0 / 3.0, 1]
            return grid.flatMap { v in grid.map { u in point(u, v) } }
        }
    }

    private static func validationGeometry(screen: ScreenRect,
                                           margin: Double,
                                           excluding: [Point2],
                                           count: Int) -> [Point2] {
        func point(_ u: Double, _ v: Double) -> Point2 {
            let x = screen.origin.x + screen.width * (margin + u * (1 - 2 * margin))
            let y = screen.origin.y + screen.height * (margin + v * (1 - 2 * margin))
            return Point2(x, y)
        }

        // An offset 7x7 lattice gives validation points in the interior and near
        // the edges while leaving enough candidates for a caller that requests
        // more than the default five validation points.
        let levels: [Double] = [0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875]
        var candidates: [Point2] = []
        for v in levels {
            for u in levels {
                let p = point(u, v)
                if !excluding.contains(where: { $0.distance(to: p) < 1e-6 }) {
                    candidates.append(p)
                }
            }
        }
        // Keep the order spatially balanced: corners first, then the center-ish
        // points.  The caller may shuffle this deterministic list.
        let center = screen.center
        candidates.sort { a, b in
            let da = a.distance(to: center), db = b.distance(to: center)
            if abs(da - db) > 1e-9 { return da > db }
            if abs(a.y - b.y) > 1e-9 { return a.y < b.y }
            return a.x < b.x
        }
        return Array(candidates.prefix(max(1, count)))
    }

    private static func shuffled(_ values: [Point2], seed: UInt64) -> [Point2] {
        var result = values
        var state = seed
        // SplitMix64 is tiny, deterministic, and does not depend on Foundation's
        // process-randomised collection hashing.
        func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        guard result.count > 1 else { return result }
        for i in stride(from: result.count - 1, through: 1, by: -1) {
            let j = Int(next() % UInt64(i + 1))
            if i != j { result.swapAt(i, j) }
        }
        return result
    }
}

