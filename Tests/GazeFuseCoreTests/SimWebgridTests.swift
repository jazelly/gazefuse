import XCTest
@testable import GazeFuseCore

final class SimWebgridTests: XCTestCase {
    func testWebgridScoreUsesNetTargetsAndClickTypes() {
        let score = WebgridScore(gridSize: 30, clickTypes: 2, correct: 12, incorrect: 2, elapsedSeconds: 60)
        let expected = log2(Double(30 * 30 * 2 - 1)) * 10 / 60
        XCTAssertEqual(score.bitsPerTarget, log2(1799), accuracy: 1e-12)
        XCTAssertEqual(score.netTargetsPerMinute, 10, accuracy: 1e-12)
        XCTAssertEqual(score.bitsPerSecond, expected, accuracy: 1e-12)
        XCTAssertEqual(WebgridScore.bitsPerSecond(gridSize: 1, correct: 4, incorrect: 10, seconds: 60), 0)
    }

    func testWebgridTaskCorrectIncorrectAndOutsideClicks() {
        let config = WebgridConfig(gridSize: 3, canvasSize: 90, sessionSeconds: 2)
        let task = WebgridTask(config: config, rng: SimRNG(seed: 17))
        let first = task.targetCenter
        XCTAssertEqual(task.click(at: first, t: 0.25), .correct)
        XCTAssertEqual(task.correct, 1)
        var wrong = config.canvas.center
        for row in 0..<config.gridSize {
            for col in 0..<config.gridSize {
                let p = Point2(config.canvas.origin.x + (Double(col) + 0.5) * config.cellSize,
                               config.canvas.origin.y + (Double(row) + 0.5) * config.cellSize)
                if !task.targetRect.contains(p) { wrong = p; break }
            }
            if !task.targetRect.contains(wrong) { break }
        }
        XCTAssertEqual(task.click(at: wrong, t: 0.4), .incorrect)
        XCTAssertEqual(task.click(at: Point2(-1, -1), t: 0.5), .outside)
        XCTAssertEqual(task.incorrect, 1)
        XCTAssertEqual(task.outsideClicks, 1)
    }

    func testSimulatorIsSeededAndReturnsBoundedMetrics() {
        var grid = WebgridConfig(gridSize: 6, canvasSize: 360, sessionSeconds: 1.5)
        grid.screen = ScreenRect(x: 0, y: 0, width: 600, height: 450)
        let config = SimConfig.make(preset: .gazeHeadEKF, noise: .typical, webgrid: grid)
        let a = WebgridSimulator(config: config, seed: 42).run()
        let b = WebgridSimulator(config: config, seed: 42).run()
        XCTAssertEqual(a, b)
        XCTAssertGreaterThanOrEqual(a.accuracy, 0)
        XCTAssertLessThanOrEqual(a.accuracy, 1)
        XCTAssertTrue(a.bps.isFinite)
        XCTAssertTrue(a.ntpm.isFinite)
        XCTAssertTrue(a.meanClickError.isFinite)
        XCTAssertTrue(a.targetTimes.allSatisfy { $0 >= 0 && $0 <= grid.sessionSeconds })
        XCTAssertLessThanOrEqual(a.correct + a.incorrect, a.clickAttempts + a.falsePositiveClicks)
    }

    func testMouseLikeBaselineRecoversFromAnOffTargetEndpoint() {
        var grid = WebgridConfig(gridSize: 10, canvasSize: 500, sessionSeconds: 5)
        grid.screen = ScreenRect(x: 0, y: 0, width: 800, height: 600)
        let config = SimConfig.make(preset: .mouseLike, noise: .typical, webgrid: grid)
        let result = WebgridSimulator(config: config, seed: 42).run()
        XCTAssertGreaterThanOrEqual(result.correct, 4)
        XCTAssertEqual(result.accuracy, 1, accuracy: 1e-12)
        XCTAssertGreaterThan(result.bps, 0)
    }

    func testConfigsRoundTripForEveryPresetAndNoiseLevel() throws {
        for preset in SimPreset.allCases {
            for noise in SimNoiseLevel.allCases {
                let config = SimConfig.make(preset: preset, noise: noise)
                let data = try JSONEncoder().encode(config)
                let decoded = try JSONDecoder().decode(SimConfig.self, from: data)
                XCTAssertEqual(decoded, config)
            }
        }
    }
}
