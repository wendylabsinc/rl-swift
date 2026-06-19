/// A tabular Q-learning agent for small discrete state-action spaces.
public struct TabularQAgent<State: Hashable & Sendable, Action: Hashable & Sendable>: Agent {
    private let actions: [Action]
    private let learningRate: Double
    private let discount: Double
    private var policy: EpsilonGreedyPolicy<Action>
    private var table: [State: [Action: Double]]

    /// Creates a Q-learning agent with epsilon-greedy exploration.
    public init(
        actions: [Action],
        learningRate: Double,
        discount: Double,
        epsilon: Double,
        seed: UInt64 = 0
    ) throws {
        _ = try DiscreteActionSpace(actions)
        guard (0...1).contains(learningRate) else {
            throw RLSwiftError.invalidProbability(learningRate)
        }
        guard (0...1).contains(discount) else {
            throw RLSwiftError.invalidProbability(discount)
        }

        self.actions = actions
        self.learningRate = learningRate
        self.discount = discount
        policy = try EpsilonGreedyPolicy(actions: actions, epsilon: epsilon, seed: seed)
        table = [:]
    }

    /// Returns the current Q-value for a state-action pair.
    public func qValue(for state: State, action: Action) -> Double {
        guard let value = table[state]?[action] else {
            return 0
        }
        return value
    }

    /// Sets a Q-value, which is useful for bootstrapping tests or warm starts.
    public mutating func setQValue(_ value: Double, for state: State, action: Action) {
        table[state, default: [:]][action] = value
    }

    /// Chooses an action according to the current epsilon-greedy policy.
    public mutating func action(for observation: State) throws -> Action {
        let values: [Action: Double]
        if let knownValues = table[observation] {
            values = knownValues
        } else {
            values = [:]
        }
        return policy.selectAction(values: values)
    }

    /// Applies the Q-learning update for a transition.
    public mutating func observe(_ transition: Transition<State, Action>) {
        let current = qValue(for: transition.observation, action: transition.action)
        let bootstrap = transition.isTerminal ? 0 : bestValue(for: transition.nextObservation)
        let target = transition.reward + discount * bootstrap
        setQValue(current + learningRate * (target - current), for: transition.observation, action: transition.action)
    }

    /// Returns the highest Q-value currently known for a state.
    public func bestValue(for state: State) -> Double {
        var best = qValue(for: state, action: actions[0])
        for action in actions.dropFirst() {
            best = max(best, qValue(for: state, action: action))
        }
        return best
    }
}
