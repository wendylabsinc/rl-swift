import Foundation

/// Policy/value prediction paired with the next recurrent state.
public struct RecurrentPolicyValuePrediction<State: Sendable>: Sendable {
    /// Policy logits, probabilities, and scalar value estimate.
    public let policyValuePrediction: PPOPolicyValuePrediction

    /// Recurrent state after consuming the input observation.
    public let nextState: State
}

extension RecurrentPolicyValuePrediction: Equatable where State: Equatable {}

/// Policy/value model that carries recurrent state between observations.
public protocol RecurrentPolicyValueModel: Sendable {
    /// Model-specific recurrent state.
    associatedtype State: Sendable

    /// Returns the initial recurrent state for one sequence.
    func initialState() throws -> State

    /// Predicts an action distribution and value estimate from an observation and previous state.
    func prediction(for observation: [Double], state: State) throws -> RecurrentPolicyValuePrediction<State>
}

/// Hidden state used by ``MinGRUCell``.
public struct MinGRUState: Sendable, Equatable, Codable {
    /// Hidden activations.
    public let hidden: [Double]

    /// Creates a recurrent state.
    public init(hidden: [Double]) throws {
        guard !hidden.isEmpty else {
            throw RLSwiftError.invalidCapacity(0)
        }
        self.hidden = hidden
    }

    /// Creates a zero state for a hidden size.
    public static func zeros(hiddenDimension: Int) throws -> MinGRUState {
        guard hiddenDimension > 0 else {
            throw RLSwiftError.invalidCapacity(hiddenDimension)
        }
        return try MinGRUState(hidden: Array(repeating: 0.0, count: hiddenDimension))
    }
}

/// Minimal GRU reference cell with input-only gates and hidden-state interpolation.
public struct MinGRUCell: Sendable, Equatable, Codable {
    /// Number of input features.
    public let inputDimension: Int

    /// Number of hidden units.
    public let hiddenDimension: Int

    /// Flattened update-gate weights in input-major order.
    public let updateWeights: [Double]

    /// Update-gate bias per hidden unit.
    public let updateBiases: [Double]

    /// Flattened candidate weights in input-major order.
    public let candidateWeights: [Double]

    /// Candidate bias per hidden unit.
    public let candidateBiases: [Double]

    /// Creates a minimal GRU cell from explicit weights.
    public init(
        inputDimension: Int,
        hiddenDimension: Int,
        updateWeights: [Double],
        updateBiases: [Double],
        candidateWeights: [Double],
        candidateBiases: [Double]
    ) throws {
        guard inputDimension > 0 else {
            throw RLSwiftError.dimensionMismatch(expected: 1, actual: inputDimension)
        }
        guard hiddenDimension > 0 else {
            throw RLSwiftError.invalidCapacity(hiddenDimension)
        }
        let matrixCount = inputDimension * hiddenDimension
        guard updateWeights.count == matrixCount else {
            throw RLSwiftError.dimensionMismatch(expected: matrixCount, actual: updateWeights.count)
        }
        guard candidateWeights.count == matrixCount else {
            throw RLSwiftError.dimensionMismatch(expected: matrixCount, actual: candidateWeights.count)
        }
        guard updateBiases.count == hiddenDimension else {
            throw RLSwiftError.dimensionMismatch(expected: hiddenDimension, actual: updateBiases.count)
        }
        guard candidateBiases.count == hiddenDimension else {
            throw RLSwiftError.dimensionMismatch(expected: hiddenDimension, actual: candidateBiases.count)
        }
        self.inputDimension = inputDimension
        self.hiddenDimension = hiddenDimension
        self.updateWeights = updateWeights
        self.updateBiases = updateBiases
        self.candidateWeights = candidateWeights
        self.candidateBiases = candidateBiases
    }

    /// Returns the zero initial state for this cell.
    public func initialState() throws -> MinGRUState {
        try MinGRUState.zeros(hiddenDimension: hiddenDimension)
    }

    /// Advances the cell by one observation.
    public func step(input: [Double], state: MinGRUState) throws -> MinGRUState {
        guard input.count == inputDimension else {
            throw RLSwiftError.dimensionMismatch(expected: inputDimension, actual: input.count)
        }
        guard state.hidden.count == hiddenDimension else {
            throw RLSwiftError.dimensionMismatch(expected: hiddenDimension, actual: state.hidden.count)
        }
        var nextHidden = Array(repeating: 0.0, count: hiddenDimension)
        for hiddenIndex in 0..<hiddenDimension {
            let update = Self.sigmoid(projection(weights: updateWeights, biases: updateBiases, input: input, hiddenIndex: hiddenIndex))
            let candidate = tanh(projection(weights: candidateWeights, biases: candidateBiases, input: input, hiddenIndex: hiddenIndex))
            nextHidden[hiddenIndex] = (1 - update) * state.hidden[hiddenIndex] + update * candidate
        }
        return try MinGRUState(hidden: nextHidden)
    }

    private func projection(weights: [Double], biases: [Double], input: [Double], hiddenIndex: Int) -> Double {
        var value = biases[hiddenIndex]
        for inputIndex in 0..<inputDimension {
            value += input[inputIndex] * weights[inputIndex * hiddenDimension + hiddenIndex]
        }
        return value
    }

    private static func sigmoid(_ value: Double) -> Double {
        1 / (1 + exp(-value))
    }
}

/// Recurrent discrete actor-critic model backed by a ``MinGRUCell``.
public struct MinGRUDiscreteActorCriticModel: RecurrentPolicyValueModel {
    /// Recurrent feature extractor.
    public let cell: MinGRUCell

    /// Number of discrete actions.
    public let actionCount: Int

    /// Flattened actor weights in hidden-major order.
    public let actorWeights: [Double]

    /// Actor bias per action.
    public let actorBiases: [Double]

    /// Value head weights per hidden unit.
    public let valueWeights: [Double]

    /// Scalar value-head bias.
    public let valueBias: Double

    /// Creates a recurrent actor-critic model from explicit weights.
    public init(
        cell: MinGRUCell,
        actionCount: Int,
        actorWeights: [Double],
        actorBiases: [Double],
        valueWeights: [Double],
        valueBias: Double
    ) throws {
        guard actionCount > 0 else {
            throw RLSwiftError.emptyActionSpace
        }
        let actorWeightCount = cell.hiddenDimension * actionCount
        guard actorWeights.count == actorWeightCount else {
            throw RLSwiftError.dimensionMismatch(expected: actorWeightCount, actual: actorWeights.count)
        }
        guard actorBiases.count == actionCount else {
            throw RLSwiftError.dimensionMismatch(expected: actionCount, actual: actorBiases.count)
        }
        guard valueWeights.count == cell.hiddenDimension else {
            throw RLSwiftError.dimensionMismatch(expected: cell.hiddenDimension, actual: valueWeights.count)
        }
        self.cell = cell
        self.actionCount = actionCount
        self.actorWeights = actorWeights
        self.actorBiases = actorBiases
        self.valueWeights = valueWeights
        self.valueBias = valueBias
    }

    /// Returns the zero recurrent state.
    public func initialState() throws -> MinGRUState {
        try cell.initialState()
    }

    /// Predicts action logits and a value estimate for one recurrent step.
    public func prediction(
        for observation: [Double],
        state: MinGRUState
    ) throws -> RecurrentPolicyValuePrediction<MinGRUState> {
        let nextState = try cell.step(input: observation, state: state)
        var logits = actorBiases
        for hiddenIndex in 0..<cell.hiddenDimension {
            for actionIndex in 0..<actionCount {
                logits[actionIndex] += nextState.hidden[hiddenIndex] * actorWeights[hiddenIndex * actionCount + actionIndex]
            }
        }
        var value = valueBias
        for hiddenIndex in 0..<cell.hiddenDimension {
            value += nextState.hidden[hiddenIndex] * valueWeights[hiddenIndex]
        }
        return RecurrentPolicyValuePrediction(
            policyValuePrediction: try PPOPolicyValuePrediction(logits: logits, valueEstimate: value),
            nextState: nextState
        )
    }
}
