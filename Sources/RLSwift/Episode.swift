/// A transition collected from one environment step.
public struct Transition<Observation: Sendable, Action: Sendable>: Sendable {
    /// The observation before the action was applied.
    public let observation: Observation

    /// The action selected by the agent.
    public let action: Action

    /// The scalar reward emitted by the environment.
    public let reward: Double

    /// The observation after the action was applied.
    public let nextObservation: Observation

    /// Whether this transition ended the episode.
    public let isTerminal: Bool

    /// The termination semantics for this transition.
    public let termination: StepTermination

    /// Creates a transition from its agent-environment fields.
    public init(
        observation: Observation,
        action: Action,
        reward: Double,
        nextObservation: Observation,
        isTerminal: Bool,
        termination: StepTermination? = nil
    ) {
        self.observation = observation
        self.action = action
        self.reward = reward
        self.nextObservation = nextObservation
        let resolvedTermination = termination ?? (isTerminal ? .terminated(reason: "terminal") : .continuing)
        self.isTerminal = resolvedTermination.endsEpisode
        self.termination = resolvedTermination
    }
}

extension Transition: Equatable where Observation: Equatable, Action: Equatable {}

/// An ordered collection of transitions produced during a single episode.
public struct Episode<Observation: Sendable, Action: Sendable>: Sendable {
    private var storage: [Transition<Observation, Action>]

    /// Creates an empty episode.
    public init() {
        storage = []
    }

    /// The transitions in chronological order.
    public var transitions: [Transition<Observation, Action>] {
        storage
    }

    /// The number of recorded transitions.
    public var count: Int {
        storage.count
    }

    /// The undiscounted sum of all rewards in the episode.
    public var totalReward: Double {
        storage.reduce(0) { $0 + $1.reward }
    }

    /// Appends a transition to the episode.
    public mutating func append(_ transition: Transition<Observation, Action>) {
        storage.append(transition)
    }

    /// Removes all transitions while keeping the episode ready for reuse.
    public mutating func removeAll(keepingCapacity keepCapacity: Bool = false) {
        storage.removeAll(keepingCapacity: keepCapacity)
    }

    /// Computes the discounted return from the start of the episode.
    public func discountedReturn(gamma: Double) -> Double {
        var multiplier = 1.0
        var result = 0.0
        for transition in storage {
            result += multiplier * transition.reward
            multiplier *= gamma
        }
        return result
    }
}

extension Episode: Equatable where Observation: Equatable, Action: Equatable {}
