import Testing
@testable import SwiftRL

@Suite struct SafetyTests {
    @Test func validatesSafetyEnvelope() throws {
        let space = try ContinuousBoxSpace(lowerBounds: [-1, -1], upperBounds: [1, 1])

        #expect(throws: SwiftRLError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try RobotSafetyEnvelope(commandSpace: space, maximumDelta: [0.1])
        }
        #expect(throws: SwiftRLError.invalidBounds(index: 0, lower: 0, upper: -0.1)) {
            _ = try RobotSafetyEnvelope(commandSpace: space, maximumDelta: [-0.1, 0.2])
        }
    }

    @Test func allowsSafeActions() throws {
        let space = try ContinuousBoxSpace(lowerBounds: [-1, -1], upperBounds: [1, 1])
        let unrestricted = try RobotSafetyEnvelope(commandSpace: space)
        let rateLimited = try RobotSafetyEnvelope(commandSpace: space, maximumDelta: [0.2, 0.2])
        let previous = RobotAction(commands: [0, 0], mode: .velocity)

        #expect(unrestricted.allows(RobotAction(commands: [0.5, -0.5], mode: .velocity)))
        #expect(!unrestricted.allows(RobotAction(commands: [2, 0], mode: .velocity)))
        #expect(rateLimited.allows(RobotAction(commands: [0.1, -0.1], mode: .velocity), previousAction: previous))
        #expect(!rateLimited.allows(RobotAction(commands: [0.3, 0], mode: .velocity), previousAction: previous))
        #expect(!rateLimited.allows(RobotAction(commands: [0.1, 0], mode: .velocity), previousAction: RobotAction(commands: [0], mode: .velocity)))
    }

    @Test func shieldsActionsWithBoundsAndRateLimits() throws {
        let space = try ContinuousBoxSpace(lowerBounds: [-1, -1], upperBounds: [1, 1])
        let unrestricted = try RobotSafetyEnvelope(commandSpace: space)
        let rateLimited = try RobotSafetyEnvelope(commandSpace: space, maximumDelta: [0.2, 0.3])
        let previous = RobotAction(commands: [0.8, -0.8], mode: .torque)

        #expect(try unrestricted.shield(RobotAction(commands: [2, -2], mode: .torque)) == RobotAction(commands: [1, -1], mode: .torque))
        #expect(try rateLimited.shield(RobotAction(commands: [2, -2], mode: .torque), previousAction: previous) == RobotAction(commands: [1, -1], mode: .torque))
        #expect(throws: SwiftRLError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try rateLimited.shield(RobotAction(commands: [0, 0], mode: .torque), previousAction: RobotAction(commands: [0], mode: .torque))
        }
    }

    @Test func reportsSafetyInterventions() throws {
        let space = try ContinuousBoxSpace(lowerBounds: [-1, -1], upperBounds: [1, 1])
        let envelope = try RobotSafetyEnvelope(commandSpace: space, maximumDelta: [0.2, 0.2])
        let requested = RobotAction(commands: [2, -0.9], mode: .velocity)
        let previous = RobotAction(commands: [0.5, -0.5], mode: .velocity)

        let assessment = try envelope.assess(requested, previousAction: previous)

        #expect(assessment.requestedAction == requested)
        #expect(assessment.safeAction == RobotAction(commands: [0.7, -0.7], mode: .velocity))
        #expect(assessment.didIntervene)
        #expect(assessment.interventions == [
            .commandClipped(index: 0, requested: 2, applied: 1),
            .rateLimited(index: 0, requested: 1, applied: 0.7),
            .rateLimited(index: 1, requested: -0.9, applied: -0.7),
        ])

        let noIntervention = RobotSafetyAssessment(
            requestedAction: requested,
            safeAction: requested,
            interventions: []
        )
        #expect(!noIntervention.didIntervene)
    }
}
