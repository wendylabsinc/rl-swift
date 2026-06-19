import Testing
@testable import RLSwift

@Suite struct ControlFiltersTests {
    @Test func validatesSmoothingAlpha() {
        #expect(throws: RLSwiftError.invalidProbability(-0.1)) {
            _ = try ActionSmoother(alpha: -0.1)
        }
        #expect(throws: RLSwiftError.invalidProbability(1.1)) {
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
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try smoother.smooth(RobotAction(commands: [1], mode: .position))
        }

        smoother.reset()
        #expect(smoother.currentCommands == nil)
        #expect(try smoother.smooth(RobotAction(commands: [2], mode: .velocity)) == RobotAction(commands: [2], mode: .velocity))
    }
}
