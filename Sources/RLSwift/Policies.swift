import Foundation

/// An epsilon-greedy policy over a discrete action space.
public struct EpsilonGreedyPolicy<Action: Hashable & Sendable>: Sendable {
    private let actionSpace: DiscreteActionSpace<Action>
    private let epsilon: Double
    private var generator: SeededGenerator

    /// Creates an epsilon-greedy policy.
    public init(actions: [Action], epsilon: Double, seed: UInt64 = 0) throws {
        guard (0...1).contains(epsilon) else {
            throw RLSwiftError.invalidProbability(epsilon)
        }
        actionSpace = try DiscreteActionSpace(actions)
        self.epsilon = epsilon
        generator = SeededGenerator(seed: seed)
    }

    /// Selects an action from action-value estimates, defaulting missing values to zero.
    public mutating func selectAction(values: [Action: Double]) -> Action {
        let draw = Double(generator.next()) / Double(UInt64.max)
        if draw < epsilon {
            return actionSpace.randomAction(using: &generator)
        }

        var bestAction = actionSpace.actions[0]
        var bestValue = values[bestAction] ?? 0
        for action in actionSpace.actions.dropFirst() {
            let candidateValue = values[action] ?? 0
            if candidateValue > bestValue {
                bestAction = action
                bestValue = candidateValue
            }
        }
        return bestAction
    }
}

/// A softmax policy over a discrete action space.
public struct SoftmaxPolicy<Action: Hashable & Sendable>: Sendable {
    private let actionSpace: DiscreteActionSpace<Action>
    private let temperature: Double
    private var generator: SeededGenerator

    /// Creates a softmax policy.
    public init(actions: [Action], temperature: Double, seed: UInt64 = 0) throws {
        guard temperature > 0 else {
            throw RLSwiftError.invalidTemperature(temperature)
        }
        actionSpace = try DiscreteActionSpace(actions)
        self.temperature = temperature
        generator = SeededGenerator(seed: seed)
    }

    /// Selects an action by sampling from exponentiated action values.
    public mutating func selectAction(values: [Action: Double]) -> Action {
        var maximum = values[actionSpace.actions[0]] ?? 0
        for action in actionSpace.actions.dropFirst() {
            maximum = max(maximum, values[action] ?? 0)
        }
        let weights = actionSpace.actions.map { exp(((values[$0] ?? 0) - maximum) / temperature) }
        let total = weights.reduce(0, +)
        var threshold = (Double(generator.next()) / Double(UInt64.max)) * total

        for index in weights.indices.dropLast() {
            threshold -= weights[index]
            if threshold <= 0 {
                return actionSpace.actions[index]
            }
        }
        return actionSpace.actions[actionSpace.actions.count - 1]
    }
}
