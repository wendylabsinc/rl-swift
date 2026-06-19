import Testing
@testable import RLSwift

@Suite struct PolicyTests {
    @Test func validatesPolicyInputs() {
        #expect(throws: RLSwiftError.invalidProbability(-0.1)) {
            _ = try EpsilonGreedyPolicy(actions: [0], epsilon: -0.1)
        }
        #expect(throws: RLSwiftError.invalidTemperature(0)) {
            _ = try SoftmaxPolicy(actions: [0], temperature: 0)
        }
        #expect(throws: RLSwiftError.emptyActionSpace) {
            _ = try EpsilonGreedyPolicy<Int>(actions: [], epsilon: 0)
        }
        #expect(throws: RLSwiftError.emptyActionSpace) {
            _ = try SoftmaxPolicy<Int>(actions: [], temperature: 1)
        }
    }

    @Test func epsilonGreedyExploitsHighestValue() throws {
        var policy = try EpsilonGreedyPolicy(actions: ["a", "b"], epsilon: 0, seed: 1)

        #expect(policy.selectAction(values: ["a": 1, "b": 3]) == "b")
        #expect(policy.selectAction(values: [:]) == "a")
    }

    @Test func epsilonGreedyExploresWithinActionSpace() throws {
        var policy = try EpsilonGreedyPolicy(actions: ["a", "b"], epsilon: 1, seed: 2)
        let action = policy.selectAction(values: ["a": 100, "b": -100])

        #expect(["a", "b"].contains(action))
    }

    @Test func softmaxSelectsWithinActionSpace() throws {
        var policy = try SoftmaxPolicy(actions: ["a", "b"], temperature: 0.5, seed: 3)
        let action = policy.selectAction(values: ["a": 0, "b": 5])

        #expect(action == "b")
    }

    @Test func softmaxCanReturnFirstAction() throws {
        var policy = try SoftmaxPolicy(actions: ["a", "b"], temperature: 0.5, seed: 3)

        #expect(policy.selectAction(values: ["a": 5, "b": 0]) == "a")
    }

    @Test func softmaxDefaultsMissingValuesToZero() throws {
        var policy = try SoftmaxPolicy(actions: ["a", "b"], temperature: 0.5, seed: 3)

        #expect(policy.selectAction(values: ["a": 5]) == "a")
    }

    @Test func softmaxHandlesSingleActionSpace() throws {
        var policy = try SoftmaxPolicy(actions: ["only"], temperature: 1, seed: 3)

        #expect(policy.selectAction(values: [:]) == "only")
    }
}
