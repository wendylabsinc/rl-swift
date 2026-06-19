import Testing
@testable import SwiftRL

@Suite struct RobotTypesTests {
    @Test func robotActionClipsCommands() throws {
        let action = RobotAction(commands: [-2, 3], mode: .torque)
        let space = try ContinuousBoxSpace(lowerBounds: [-1, -1], upperBounds: [1, 1])

        #expect(try action.clipped(to: space) == RobotAction(commands: [-1, 1], mode: .torque))
        #expect(RobotControlMode.velocity.rawValue == "velocity")
    }

    @Test func robotObservationValidatesAndFlattensFeatures() throws {
        #expect(throws: SwiftRLError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try RobotObservation(jointPositions: [0, 1], jointVelocities: [0])
        }

        let observation = try RobotObservation(
            jointPositions: [1, 2],
            jointVelocities: [0.1, 0.2],
            endEffectorPose: [3, 4, 5],
            sensorReadings: ["force": 9],
            stepIndex: 7
        )

        #expect(observation.stepIndex == 7)
        #expect(observation.flattenedFeatures(sensorOrder: ["force", "missing"]) == [1, 2, 0.1, 0.2, 3, 4, 5, 9, 0])
    }

    @Test func rewardBreakdownTotalsNamedContributions() {
        let breakdown = RewardBreakdown([
            RewardComponent(name: "reach", value: 2, weight: 3),
            RewardComponent(name: "energy", value: -4, weight: 0.5),
        ])

        #expect(breakdown.components.count == 2)
        #expect(breakdown.total == 4)
        #expect(breakdown.contribution(named: "reach") == 6)
        #expect(breakdown.contribution(named: "missing") == nil)
    }
}
