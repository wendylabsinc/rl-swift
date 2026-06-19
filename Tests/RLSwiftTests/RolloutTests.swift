import Testing
@testable import RLSwift

@Suite struct RolloutTests {
    @Test func computesDiscountedReturnsForEveryStep() {
        var episode = Episode<Int, Int>()

        #expect(episode.discountedReturns(gamma: 0.9) == [])

        episode.append(Transition(observation: 0, action: 0, reward: 1, nextObservation: 1, isTerminal: false))
        episode.append(Transition(observation: 1, action: 0, reward: 2, nextObservation: 2, isTerminal: true))

        #expect(episode.discountedReturns(gamma: 0.5) == [2, 2])
    }

    @Test func computesNStepReturns() throws {
        var episode = Episode<Int, Int>()
        episode.append(Transition(observation: 0, action: 0, reward: 1, nextObservation: 1, isTerminal: false))
        episode.append(Transition(observation: 1, action: 0, reward: 2, nextObservation: 2, isTerminal: false))
        episode.append(Transition(observation: 2, action: 0, reward: 4, nextObservation: 3, isTerminal: true))

        #expect(throws: RLSwiftError.invalidHorizon(0)) {
            _ = try episode.nStepReturns(gamma: 0.5, horizon: 0)
        }
        #expect(try episode.nStepReturns(gamma: 0.5, horizon: 2, bootstrap: 10) == [4.5, 4, 4])
        #expect(try episode.nStepReturns(gamma: 0.5, horizon: 1, bootstrap: 10) == [6, 7, 4])
    }
}
