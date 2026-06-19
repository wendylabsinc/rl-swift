import Testing
@testable import SwiftRL

@Suite struct TerminationTests {
    @Test func terminationClassifiesEpisodeEndings() {
        let continuing = StepTermination.continuing
        let terminated = StepTermination.terminated(reason: "goal")
        let truncated = StepTermination.truncated(reason: "time_limit")
        let interrupted = StepTermination.interrupted(reason: "e_stop")

        #expect(!continuing.endsEpisode)
        #expect(terminated.endsEpisode)
        #expect(truncated.endsEpisode)
        #expect(interrupted.endsEpisode)
        #expect(!continuing.isTruncated)
        #expect(!terminated.isTruncated)
        #expect(truncated.isTruncated)
        #expect(!interrupted.isTruncated)
        #expect(!continuing.isInterrupted)
        #expect(!terminated.isInterrupted)
        #expect(!truncated.isInterrupted)
        #expect(interrupted.isInterrupted)
        #expect(continuing.reason == nil)
        #expect(terminated.reason == "goal")
        #expect(truncated.reason == "time_limit")
        #expect(interrupted.reason == "e_stop")
    }

    @Test func stepResultAndTransitionResolveTermination() {
        let continuing = StepResult(observation: 0, reward: 1, isTerminal: false)
        let terminal = StepResult(observation: 1, reward: 2, isTerminal: true)
        let truncated = StepResult(observation: 2, reward: 3, isTerminal: false, termination: .truncated(reason: "timeout"))
        let interrupted = Transition(observation: 0, action: 1, reward: -1, nextObservation: 1, isTerminal: false, termination: .interrupted(reason: "guard"))

        #expect(!continuing.isTerminal)
        #expect(continuing.termination == .continuing)
        #expect(terminal.isTerminal)
        #expect(terminal.termination == .terminated(reason: "terminal"))
        #expect(truncated.isTerminal)
        #expect(truncated.termination == .truncated(reason: "timeout"))
        #expect(interrupted.isTerminal)
        #expect(interrupted.termination == .interrupted(reason: "guard"))
    }
}
