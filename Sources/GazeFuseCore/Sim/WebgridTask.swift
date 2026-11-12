// WebgridTask.swift — the Webgrid task: an N×N grid on a square canvas, one lit
// target at a time; a click inside the target is correct (and a new random target
// lights up), inside any other cell is incorrect (the target stays), outside the
// canvas does nothing.
import Foundation

public struct WebgridConfig: Codable, Equatable {
    /// Cells per side (Neuralink's default is 30; 35 is also common).
    public var gridSize: Int
    /// Side of the square canvas in points.
    public var canvasSize: Double
    /// Top-left of the canvas; nil = centred on `screen`.
    public var canvasOrigin: Point2?
    public var sessionSeconds: Double
    public var clickTypes: Int
    public var screen: ScreenRect

    public init(gridSize: Int = 30, canvasSize: Double = 900, canvasOrigin: Point2? = nil,
                sessionSeconds: Double = 60, clickTypes: Int = 1,
                screen: ScreenRect = ScreenRect(x: 0, y: 0, width: 1512, height: 982)) {
        self.gridSize = gridSize; self.canvasSize = canvasSize; self.canvasOrigin = canvasOrigin
        self.sessionSeconds = sessionSeconds; self.clickTypes = clickTypes; self.screen = screen
    }

    public var cellSize: Double { canvasSize / Double(gridSize) }
    public var canvas: ScreenRect {
        let o = canvasOrigin ?? Point2(screen.origin.x + (screen.width - canvasSize) / 2,
                                      screen.origin.y + (screen.height - canvasSize) / 2)
        return ScreenRect(x: o.x, y: o.y, width: canvasSize, height: canvasSize)
    }
}

public final class WebgridTask {
    public enum ClickOutcome: String, Codable { case correct, incorrect, outside }

    public let config: WebgridConfig
    private let rng: SimRNG
    public private(set) var targetRow = 0
    public private(set) var targetCol = 0
    public private(set) var correct = 0
    public private(set) var incorrect = 0
    public private(set) var outsideClicks = 0
    public private(set) var targetOnsetTime: TimeSec = 0
    /// Acquisition time of every correct target (onset → correct click).
    public private(set) var targetTimes: [Double] = []

    public init(config: WebgridConfig, rng: SimRNG, startTime: TimeSec = 0) {
        self.config = config; self.rng = rng
        targetRow = rng.int(config.gridSize); targetCol = rng.int(config.gridSize)
        targetOnsetTime = startTime
    }

    public var targetRect: ScreenRect {
        let c = config.canvas, s = config.cellSize
        return ScreenRect(x: c.origin.x + Double(targetCol) * s, y: c.origin.y + Double(targetRow) * s, width: s, height: s)
    }
    public var targetCenter: Point2 { targetRect.center }

    /// (row, col) of the cell containing `p`, nil outside the canvas.
    public func cell(at p: Point2) -> (row: Int, col: Int)? {
        let c = config.canvas
        guard c.contains(p) else { return nil }
        let s = config.cellSize
        return (min(Int((p.y - c.origin.y) / s), config.gridSize - 1), min(Int((p.x - c.origin.x) / s), config.gridSize - 1))
    }

    @discardableResult
    public func click(at p: Point2, t: TimeSec) -> ClickOutcome {
        guard let cell = cell(at: p) else { outsideClicks += 1; return .outside }
        if cell.row == targetRow && cell.col == targetCol {
            correct += 1
            targetTimes.append(t - targetOnsetTime)
            newTarget(at: t)
            return .correct
        }
        incorrect += 1
        return .incorrect
    }

    private func newTarget(at t: TimeSec) {
        let n = config.gridSize
        var r = targetRow, c = targetCol
        while r == targetRow && c == targetCol { r = rng.int(n); c = rng.int(n) }
        targetRow = r; targetCol = c
        targetOnsetTime = t
    }

    public func score(elapsed: Double) -> WebgridScore {
        WebgridScore(gridSize: config.gridSize, clickTypes: config.clickTypes, correct: correct, incorrect: incorrect, elapsedSeconds: elapsed)
    }
}
