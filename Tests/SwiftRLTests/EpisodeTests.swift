import Testing
@testable import SwiftRL

@Suite struct EpisodeTests {
    @Test func recordsTransitionsAndReturns() {
        var episode = Episode<Int, String>()
        let first = Transition(observation: 0, action: "right", reward: 1, nextObservation: 1, isTerminal: false)
        let second = Transition(observation: 1, action: "right", reward: 2, nextObservation: 2, isTerminal: true)

        episode.append(first)
        episode.append(second)

        #expect(episode.count == 2)
        #expect(episode.transitions == [first, second])
        #expect(episode.totalReward == 3)
        #expect(episode.discountedReturn(gamma: 0.5) == 2)

        episode.removeAll(keepingCapacity: true)
        #expect(episode.count == 0)
        #expect(episode.totalReward == 0)
        #expect(episode.discountedReturn(gamma: 0.9) == 0)
    }

    @Test func stepResultCarriesInfo() {
        let result = StepResult(observation: "state", reward: 1.5, isTerminal: true, info: ["phase": "done"])

        #expect(result == StepResult(observation: "state", reward: 1.5, isTerminal: true, info: ["phase": "done"]))
        #expect(result.info["phase"] == "done")
    }
}
