import XCTest
@testable import GazeFuseCore
final class ZZDebugTrace: XCTestCase {
    func testSweep() {
        for p in [SimPreset.gazeHeadEKF, .gazeHandEKF, .gazeHeadHandEKF] {
            for n in SimNoiseLevel.allCases {
                let cfg = SimConfig.make(preset: p, noise: n)
                let s = SimSummary.run(config: cfg, seeds: [1, 2, 3, 4, 5, 6])
                let r = s.results[0]
                print(String(format: "%@ %@ BPS %.2f±%.2f NTPM %.1f acc %.2f t/target %.2f patience %d clickErr %.1f", p.rawValue, n.rawValue, s.bpsMean, s.bpsSD, s.ntpmMean, s.accuracyMean, s.timePerTargetMean, r.patienceClicks, r.meanClickError))
            }
        }
    }
}
