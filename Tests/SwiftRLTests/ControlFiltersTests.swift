import Testing
@testable import SwiftRL

@Suite struct ControlFiltersTests {
    @Test func validatesSmoothingAlpha() {
        #expect(throws: SwiftRLError.invalidProbability(-0.1)) {
            _ = try ActionSmoother(alpha: -0.1)
        }
        #expect(throws: SwiftRLError.invalidProbability(1.1)) {
            _ = try ActionSmoother(alpha: 1.1)
        }
    }

    @Test func smoothsAndResetsActions() throws {
        var smoother = try ActionSmoother(alpha: 0.25)
        let first = RobotAction(commands: [0, 4], mode: .position)
        let second = RobotAction(commands: [4, 0], mode: .position)

        #expect(smoother.currentCommands == nil)
        #expect(try smoother.smooth(first) == first)
        #expect(smoother.currentCommands == [0, 4])
        #expect(try smoother.smooth(second) == RobotAction(commands: [1, 3], mode: .position))
        #expect(smoother.currentCommands == [1, 3])
        #expect(throws: SwiftRLError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try smoother.smooth(RobotAction(commands: [1], mode: .position))
        }

        smoother.reset()
        #expect(smoother.currentCommands == nil)
        #expect(try smoother.smooth(RobotAction(commands: [2], mode: .velocity)) == RobotAction(commands: [2], mode: .velocity))
    }
}
