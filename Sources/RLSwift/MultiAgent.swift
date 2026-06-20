/// Batched result returned by a multi-agent environment.
public struct MultiAgentStepResult<Observation: Sendable>: Sendable {
    /// Observation by agent identifier.
    public let observations: [String: Observation]

    /// Reward by agent identifier.
    public let rewards: [String: Double]

    /// Termination by agent identifier.
    public let terminations: [String: StepTermination]

    /// Diagnostic metadata by agent identifier.
    public let info: [String: [String: String]]

    /// Creates a multi-agent step result.
    public init(
        observations: [String: Observation],
        rewards: [String: Double],
        terminations: [String: StepTermination],
        info: [String: [String: String]] = [:]
    ) {
        self.observations = observations
        self.rewards = rewards
        self.terminations = terminations
        self.info = info
    }

    /// Whether every active agent has ended its episode.
    public var allAgentsDone: Bool {
        !terminations.isEmpty && terminations.values.allSatisfy(\.endsEpisode)
    }
}

extension MultiAgentStepResult: Equatable where Observation: Equatable {}

/// A stateful environment with multiple simultaneously acting agents.
public protocol MultiAgentEnvironment: Sendable {
    /// Observation type exposed to each agent.
    associatedtype Observation: Sendable

    /// Action type accepted from each agent.
    associatedtype Action: Sendable

    /// Stable agent identifiers.
    var agentIDs: [String] { get }

    /// Resets all agents.
    mutating func reset() -> [String: Observation]

    /// Applies one action per active agent.
    mutating func step(_ actions: [String: Action]) throws -> MultiAgentStepResult<Observation>
}

/// Actions for the built-in two-agent matrix game.
public enum MatrixGameAction: String, CaseIterable, Sendable, Equatable, Codable, Hashable {
    /// Cooperate with the other agent.
    case cooperate

    /// Defect against the other agent.
    case defect
}

/// Observation for the built-in matrix game.
public struct MatrixGameObservation: Sendable, Equatable, Codable, Hashable {
    /// Current round index.
    public let round: Int

    /// Creates a matrix-game observation.
    public init(round: Int) {
        self.round = round
    }
}

/// A deterministic two-agent matrix game for self-play and multi-agent smoke tests.
public struct MatrixGameEnvironment: MultiAgentEnvironment {
    /// Stable agent identifiers.
    public let agentIDs: [String]

    private let maxRounds: Int
    private var round: Int

    /// Creates a two-agent matrix game.
    public init(agentIDs: [String] = ["agent-0", "agent-1"], maxRounds: Int = 4) throws {
        guard agentIDs.count == 2 else {
            throw RLSwiftError.dimensionMismatch(expected: 2, actual: agentIDs.count)
        }
        guard Set(agentIDs).count == agentIDs.count else {
            throw RLSwiftError.duplicateIdentifier("agentID")
        }
        guard maxRounds > 0 else {
            throw RLSwiftError.invalidHorizon(maxRounds)
        }
        self.agentIDs = agentIDs
        self.maxRounds = maxRounds
        round = 0
    }

    /// Resets the game round.
    public mutating func reset() -> [String: MatrixGameObservation] {
        round = 0
        return observations()
    }

    /// Applies one cooperate/defect action per agent.
    public mutating func step(_ actions: [String: MatrixGameAction]) throws -> MultiAgentStepResult<MatrixGameObservation> {
        guard Set(actions.keys) == Set(agentIDs) else {
            throw RLSwiftError.dimensionMismatch(expected: agentIDs.count, actual: actions.count)
        }
        let first = actions[agentIDs[0]]!
        let second = actions[agentIDs[1]]!
        round += 1

        let rewards: [String: Double]
        if first == .cooperate && second == .cooperate {
            rewards = [agentIDs[0]: 2, agentIDs[1]: 2]
        } else if first == .defect && second == .defect {
            rewards = [agentIDs[0]: 0, agentIDs[1]: 0]
        } else if first == .defect {
            rewards = [agentIDs[0]: 3, agentIDs[1]: -1]
        } else {
            rewards = [agentIDs[0]: -1, agentIDs[1]: 3]
        }

        let termination: StepTermination = round >= maxRounds ? .truncated(reason: "max_rounds") : .continuing
        return MultiAgentStepResult(
            observations: observations(),
            rewards: rewards,
            terminations: [agentIDs[0]: termination, agentIDs[1]: termination]
        )
    }

    private func observations() -> [String: MatrixGameObservation] {
        [agentIDs[0]: MatrixGameObservation(round: round), agentIDs[1]: MatrixGameObservation(round: round)]
    }
}
