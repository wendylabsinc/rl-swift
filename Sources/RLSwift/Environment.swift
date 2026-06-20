/// The reason an environment step did or did not end an episode.
public enum StepTermination: Sendable, Equatable, Codable {
    /// The episode is still running.
    case continuing

    /// The task reached a natural terminal condition.
    case terminated(reason: String)

    /// The episode stopped because of an external limit such as a time horizon.
    case truncated(reason: String)

    /// The episode stopped because execution was interrupted, commonly by a safety system.
    case interrupted(reason: String)

    /// Whether the episode has ended for any reason.
    public var endsEpisode: Bool {
        switch self {
        case .continuing:
            return false
        case .terminated, .truncated, .interrupted:
            return true
        }
    }

    /// Whether the episode ended because of an external truncation.
    public var isTruncated: Bool {
        switch self {
        case .truncated:
            return true
        case .continuing, .terminated, .interrupted:
            return false
        }
    }

    /// Whether the episode ended because a safety or supervisory system interrupted execution.
    public var isInterrupted: Bool {
        switch self {
        case .interrupted:
            return true
        case .continuing, .terminated, .truncated:
            return false
        }
    }

    /// The human-readable reason associated with ending states.
    public var reason: String? {
        switch self {
        case .continuing:
            return nil
        case let .terminated(reason), let .truncated(reason), let .interrupted(reason):
            return reason
        }
    }
}

/// A single response returned by an environment after an action is applied.
public struct StepResult<Observation: Sendable>: Sendable {
    /// The observation visible to the agent after the step.
    public let observation: Observation

    /// The scalar reward emitted for the transition.
    public let reward: Double

    /// Whether the transition ended the episode.
    public let isTerminal: Bool

    /// The termination semantics for the step.
    public let termination: StepTermination

    /// Optional diagnostic metadata supplied by the environment.
    public let info: [String: String]

    /// Creates a step result from its environment-facing fields.
    public init(
        observation: Observation,
        reward: Double,
        isTerminal: Bool,
        info: [String: String] = [:],
        termination: StepTermination? = nil
    ) {
        self.observation = observation
        self.reward = reward
        let resolvedTermination = termination ?? (isTerminal ? .terminated(reason: "terminal") : .continuing)
        self.isTerminal = resolvedTermination.endsEpisode
        self.termination = resolvedTermination
        self.info = info
    }
}

extension StepResult: Equatable where Observation: Equatable {}

/// A stateful process that maps actions to observations, rewards, and terminal signals.
public protocol Environment: Sendable {
    /// The observation type exposed to agents.
    associatedtype Observation: Sendable

    /// The action type accepted by the environment.
    associatedtype Action: Sendable

    /// Resets the environment and returns the initial observation.
    mutating func reset() -> Observation

    /// Applies an action and returns the resulting transition data.
    mutating func step(_ action: Action) throws -> StepResult<Observation>
}

/// A policy-bearing learner that can choose actions and update from transitions.
public protocol Agent: Sendable {
    /// The observation type consumed by the agent.
    associatedtype Observation: Sendable

    /// The action type emitted by the agent.
    associatedtype Action: Sendable

    /// Chooses an action for the current observation.
    mutating func action(for observation: Observation) throws -> Action

    /// Updates internal state from a transition.
    mutating func observe(_ transition: Transition<Observation, Action>)
}
