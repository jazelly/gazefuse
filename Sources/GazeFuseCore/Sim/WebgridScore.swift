// WebgridScore.swift — Neuralink Webgrid scoring.
//
// The Webgrid frontend computes bits per second as
//     bps = max(log2(t·t − 1) · f / 60, 0)
// with t = cells per side and f = net targets per minute (correct − incorrect)
// (see github.com/ofou/webgrid_eval, `Math.max(Math.log2(t*t-1)*f/60,0)`). When
// several click types are in play (left / right targets) the alphabet is
// t·t·clickTypes symbols, i.e. N = cells × clickTypes in log2(N − 1).
import Foundation

public struct WebgridScore: Codable, Equatable {
    /// Cells per side.
    public var gridSize: Int
    public var clickTypes: Int
    public var correct: Int
    public var incorrect: Int
    public var elapsedSeconds: Double

    public init(gridSize: Int, clickTypes: Int = 1, correct: Int, incorrect: Int, elapsedSeconds: Double) {
        self.gridSize = gridSize; self.clickTypes = clickTypes
        self.correct = correct; self.incorrect = incorrect; self.elapsedSeconds = elapsedSeconds
    }

    /// Information per correct target: log2(N·clickTypes − 1).
    public var bitsPerTarget: Double {
        Self.bitsPerTarget(gridSize: gridSize, clickTypes: clickTypes)
    }
    /// NTPM = (correct − incorrect) per minute.
    public var netTargetsPerMinute: Double {
        elapsedSeconds > 0 ? Double(correct - incorrect) / (elapsedSeconds / 60) : 0
    }
    public var bitsPerSecond: Double { max(bitsPerTarget * netTargetsPerMinute / 60, 0) }
    public var accuracy: Double {
        let total = correct + incorrect
        return total > 0 ? Double(correct) / Double(total) : 0
    }

    public static func bitsPerTarget(gridSize: Int, clickTypes: Int = 1) -> Double {
        let n = Double(gridSize * gridSize * max(clickTypes, 1))
        return n > 1 ? log2(n - 1) : 0
    }

    public static func bitsPerSecond(gridSize: Int, clickTypes: Int = 1, correct: Int, incorrect: Int, seconds: Double) -> Double {
        WebgridScore(gridSize: gridSize, clickTypes: clickTypes, correct: correct, incorrect: incorrect, elapsedSeconds: seconds).bitsPerSecond
    }
}
