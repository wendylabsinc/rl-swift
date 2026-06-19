import Testing
@testable import SwiftRL

@Suite struct ConstraintsTests {
    @Test func validatesConstraintWeights() {
        #expect(throws: SwiftRLError.invalidWeight(-1)) {
            _ = try ConstraintSignal(name: "force", value: 1, limit: 2, relation: .lessThanOrEqual, weight: -1)
        }
    }

    @Test func evaluatesLessThanAndGreaterThanConstraints() throws {
        let forceSafe = try ConstraintSignal(name: "force", value: 2, limit: 3, relation: .lessThanOrEqual, weight: 10)
        let forceViolation = try ConstraintSignal(name: "force", value: 5, limit: 3, relation: .lessThanOrEqual, weight: 10)
        let batterySafe = try ConstraintSignal(name: "battery", value: 0.7, limit: 0.2, relation: .greaterThanOrEqual, weight: 4)
        let batteryViolation = try ConstraintSignal(name: "battery", value: 0.1, limit: 0.2, relation: .greaterThanOrEqual, weight: 4)

        #expect(forceSafe.isSatisfied)
        #expect(forceSafe.violationAmount == 0)
        #expect(forceSafe.cost == 0)
        #expect(!forceViolation.isSatisfied)
        #expect(forceViolation.violationAmount == 2)
        #expect(forceViolation.cost == 20)
        #expect(batterySafe.isSatisfied)
        #expect(batterySafe.violationAmount == 0)
        #expect(!batteryViolation.isSatisfied)
        #expect(batteryViolation.violationAmount == 0.1)
        #expect(batteryViolation.cost == 0.4)
    }

    @Test func summarizesConstraintReports() throws {
        let safe = ConstraintReport([])
        let report = ConstraintReport([
            try ConstraintSignal(name: "force", value: 5, limit: 3, relation: .lessThanOrEqual, weight: 2),
            try ConstraintSignal(name: "battery", value: 0.7, limit: 0.2, relation: .greaterThanOrEqual),
        ])

        #expect(safe.isSatisfied)
        #expect(safe.totalCost == 0)
        #expect(!report.isSatisfied)
        #expect(report.totalCost == 4)
        #expect(report.violations.map(\.name) == ["force"])
        #expect(report.cost(named: "force") == 4)
        #expect(report.cost(named: "missing") == nil)
    }
}
