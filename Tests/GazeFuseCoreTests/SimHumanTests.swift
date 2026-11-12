import XCTest
@testable import GazeFuseCore

final class SimHumanTests: XCTestCase {
    func testHeadControllerReducesKnownCursorResidual() {
        var grid = WebgridConfig(gridSize: 4, canvasSize: 400, sessionSeconds: 2)
        grid.screen = ScreenRect(x: 0, y: 0, width: 600, height: 500)
        var config = SimConfig.make(preset: .gazeHeadEKF, noise: .optimistic, webgrid: grid)
        config.eye.blinkRatePerMin = 0
        config.head.postSaccadeAssessDelay = 0
        let start = grid.screen.center
        let gain = config.tuning.fusion.head
        let human = SimHuman(config: config, rng: SimRNG(seed: 5), headGain: gain, startGaze: start)
        let target = Point2(start.x + 160, start.y)
        human.newTarget(center: target, rect: ScreenRect(x: target.x - 50, y: target.y - 50, width: 100, height: 100), at: 0)

        // Feed a deliberately stale/incorrect displayed cursor. Once the eye has
        // landed, the closed-loop head controller should command a non-zero rate.
        var cursor = start
        for i in 0..<600 {
            let t = Double(i) / 240
            human.perceive(cursor: cursor, at: t)
            _ = human.step(t: t, dt: 1 / 240)
            let rate = human.headTrueRate(at: t)
            let velocity = HeadPointerGain.velocity(yawRate: rate.x, pitchRate: rate.y, config: gain)
            cursor = cursor + velocity * (1 / 240)
        }
        let initialError = start.distance(to: target)
        let residual = cursor.distance(to: target)
        XCTAssertLessThan(residual, initialError * 0.4,
                          "head correction should reduce the perceived cursor error")
        XCTAssertTrue(human.headRate.length.isFinite)
    }
}
