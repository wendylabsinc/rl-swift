import Testing
@testable import RLSwift

@Suite struct QLearningTests {
    @Test func validatesAgentInputs() {
        #expect(throws: RLSwiftError.emptyActionSpace) {
            _ = try TabularQAgent<Int, Int>(actions: [], learningRate: 0.5, discount: 0.9, epsilon: 0)
        }
        #expect(throws: RLSwiftError.invalidProbability(1.1)) {
            _ = try TabularQAgent<Int, Int>(actions: [0], learningRate: 1.1, discount: 0.9, epsilon: 0)
        }
        #expect(throws: RLSwiftError.invalidProbability(-0.1)) {
            _ = try TabularQAgent<Int, Int>(actions: [0], learningRate: 0.5, discount: -0.1, epsilon: 0)
        }
        #expect(throws: RLSwiftError.invalidProbability(1.1)) {
            _ = try TabularQAgent<Int, Int>(actions: [0], learningRate: 0.5, discount: 0.9, epsilon: 1.1)
        }
    }

    @Test func updatesFromNonterminalTransition() throws {
        var agent = try TabularQAgent<String, String>(
            actions: ["left", "right"],
            learningRate: 0.5,
            discount: 0.9,
            epsilon: 0
        )
        agent.setQValue(4, for: "next", action: "right")

        agent.observe(Transition(observation: "start", action: "left", reward: 2, nextObservation: "next", isTerminal: false))

        #expect(agent.qValue(for: "start", action: "left") == 2.8)
        #expect(agent.bestValue(for: "next") == 4)
        #expect(try agent.action(for: "next") == "right")
        #expect(try agent.action(for: "unseen") == "left")
    }

    @Test func updatesFromTerminalTransitionWithoutBootstrap() throws {
        var agent = try TabularQAgent<Int, Int>(actions: [0, 1], learningRate: 1, discount: 1, epsilon: 0)

        agent.observe(Transition(observation: 0, action: 1, reward: 7, nextObservation: 1, isTerminal: true))

        #expect(agent.qValue(for: 0, action: 1) == 7)
        #expect(agent.qValue(for: 100, action: 1) == 0)
        #expect(agent.qValue(for: 0, action: 0) == 0)
        #expect(agent.bestValue(for: 99) == 0)
    }
}
