import XCTest
@testable import GazeFuseCore

final class GazePlanTests: XCTestCase {
    func testLayoutsHaveRequestedGeometryAndIndependentValidation() throws {
        // An above-left secondary display catches assumptions that the origin is zero.
        let screen = ScreenRect(x: -1600, y: -900, width: 1600, height: 900)
        let margin = 0.1
        for layout in CalibrationPlan.Layout.allCases {
            let plan = CalibrationPlan(screen: screen, layout: layout,
                                       config: .init(marginFraction: margin, seed: 1234))
            XCTAssertEqual(plan.targets.count, layout.rawValue)
            XCTAssertEqual(Set(plan.targets).count, layout.rawValue)
            XCTAssertEqual(plan.validationTargets.count, 5)
            for p in plan.targets + plan.validationTargets {
                XCTAssertTrue(screen.contains(p))
                XCTAssertGreaterThanOrEqual(p.x, screen.origin.x + screen.width * margin - 1e-8)
                XCTAssertLessThanOrEqual(p.x, screen.origin.x + screen.width * (1 - margin) + 1e-8)
                XCTAssertGreaterThanOrEqual(p.y, screen.origin.y + screen.height * margin - 1e-8)
                XCTAssertLessThanOrEqual(p.y, screen.origin.y + screen.height * (1 - margin) + 1e-8)
            }
            XCTAssertTrue(Set(plan.targets).isDisjoint(with: Set(plan.validationTargets)))
            let same = CalibrationPlan(screen: screen, layout: layout,
                                       config: .init(marginFraction: margin, seed: 1234))
            let another = CalibrationPlan(screen: screen, layout: layout,
                                          config: .init(marginFraction: margin, seed: 5678))
            XCTAssertEqual(plan.targets, same.targets)
            XCTAssertNotEqual(plan.targets, another.targets)
            XCTAssertEqual(Set(plan.targets), Set(another.targets))
            let decoded = try JSONDecoder().decode(CalibrationPlan.self,
                                                   from: JSONEncoder().encode(plan))
            XCTAssertEqual(plan, decoded)
        }
    }
}
