import Foundation

/// One supervised PPO optimizer sample for a discrete actor-critic model.
public struct PPOTrainingSample: Sendable, Equatable, Codable {
    /// Flattened observation features consumed by the model.
    public let observation: [Double]

    /// Zero-based action index sampled during rollout collection.
    public let actionIndex: Int

    /// Action sampled during rollout collection.
    public let action: PPOAction

    /// Log probability assigned to the action by the behavior policy.
    public let oldLogProbability: Double

    /// Advantage target for the policy-gradient update.
    public let advantage: Double

    /// Return target for value-function learning.
    public let returnEstimate: Double

    /// Value estimate recorded during rollout collection.
    public let oldValueEstimate: Double?

    /// Creates a PPO optimizer sample.
    public init(
        observation: [Double],
        actionIndex: Int,
        oldLogProbability: Double,
        advantage: Double,
        returnEstimate: Double,
        oldValueEstimate: Double? = nil
    ) throws {
        try self.init(
            observation: observation,
            action: .discrete(actionIndex),
            oldLogProbability: oldLogProbability,
            advantage: advantage,
            returnEstimate: returnEstimate,
            oldValueEstimate: oldValueEstimate
        )
    }

    /// Creates a PPO optimizer sample for any supported PPO action shape.
    public init(
        observation: [Double],
        action: PPOAction,
        oldLogProbability: Double,
        advantage: Double,
        returnEstimate: Double,
        oldValueEstimate: Double? = nil
    ) throws {
        guard !observation.isEmpty else {
            throw RLSwiftError.dimensionMismatch(expected: 1, actual: 0)
        }
        let resolvedActionIndex: Int
        switch action {
        case let .discrete(index):
            guard index >= 0 else {
                throw RLSwiftError.invalidCapacity(index)
            }
            resolvedActionIndex = index
        case let .multiDiscrete(indices):
            guard !indices.isEmpty else {
                throw RLSwiftError.invalidSampleCount(0)
            }
            for index in indices {
                guard index >= 0 else {
                    throw RLSwiftError.invalidCapacity(index)
                }
            }
            resolvedActionIndex = -1
        case let .continuous(values):
            guard !values.isEmpty else {
                throw RLSwiftError.invalidSampleCount(0)
            }
            resolvedActionIndex = -1
        }
        self.observation = observation
        self.actionIndex = resolvedActionIndex
        self.action = action
        self.oldLogProbability = oldLogProbability
        self.advantage = advantage
        self.returnEstimate = returnEstimate
        self.oldValueEstimate = oldValueEstimate
    }
}

/// Actor logits and scalar value predicted by a policy/value model.
public struct PPOPolicyValuePrediction: Sendable, Equatable, Codable {
    /// Unnormalized action scores.
    public let logits: [Double]

    /// Softmax action probabilities derived from ``logits``.
    public let probabilities: [Double]

    /// Scalar value estimate for the input observation.
    public let valueEstimate: Double

    /// Creates a policy/value prediction and computes stable softmax probabilities.
    public init(logits: [Double], valueEstimate: Double) throws {
        guard !logits.isEmpty else {
            throw RLSwiftError.emptyActionSpace
        }
        self.logits = logits
        self.probabilities = Self.softmax(logits)
        self.valueEstimate = valueEstimate
    }

    /// Returns the log probability for a zero-based action index.
    public func logProbability(actionIndex: Int) throws -> Double {
        guard actionIndex >= 0, actionIndex < probabilities.count else {
            throw RLSwiftError.dimensionMismatch(expected: probabilities.count, actual: actionIndex + 1)
        }
        return log(max(probabilities[actionIndex], 1e-12))
    }

    /// Categorical distribution represented by this policy head.
    public var actionDistribution: PPOActionDistribution {
        .categorical(logits: logits)
    }

    /// Returns the log probability for a PPO action.
    public func logProbability(of action: PPOAction) throws -> Double {
        try actionDistribution.logProbability(of: action)
    }

    /// Shannon entropy of the categorical action distribution.
    public var entropy: Double {
        probabilities.reduce(0.0) { partial, probability in
            partial - probability * log(max(probability, 1e-12))
        }
    }

    private static func softmax(_ logits: [Double]) -> [Double] {
        let maximum = logits.max()!
        let exponentials = logits.map { exp($0 - maximum) }
        let total = exponentials.reduce(0, +)
        return exponentials.map { $0 / total }
    }
}

/// A trainable discrete actor-critic policy/value model used by PPO trainers.
public protocol PPOPolicyValueModel: Sendable {
    /// Predicts action logits and a scalar value for one flattened observation.
    func prediction(for observation: [Double]) throws -> PPOPolicyValuePrediction

    /// Applies one PPO optimizer update to a minibatch.
    mutating func update(
        minibatch: [PPOTrainingSample],
        configuration: PPOConfiguration
    ) throws -> PPOObjectiveBreakdown
}

/// One optimizer step emitted by ``NeuralPPOTrainer``.
public struct PPOOptimizerStepSummary: Sendable, Equatable, Codable {
    /// Zero-based epoch index.
    public let epoch: Int

    /// Zero-based minibatch index within the epoch.
    public let minibatchIndex: Int

    /// Number of samples used by the optimizer step.
    public let sampleCount: Int

    /// PPO objective measured before the parameter update.
    public let objective: PPOObjectiveBreakdown
}

/// Summary returned after a PPO training pass.
public struct PPOTrainingSummary: Sendable, Equatable, Codable {
    /// Total samples seen in each optimization epoch.
    public let sampleCount: Int

    /// Number of epochs completed.
    public let epochCount: Int

    /// Per-minibatch objective measurements.
    public let optimizerSteps: [PPOOptimizerStepSummary]

    /// Objective from the final optimizer step.
    public var finalObjective: PPOObjectiveBreakdown? {
        optimizerSteps.last?.objective
    }
}

/// Full PPO minibatch loop for trainable policy/value models.
public struct NeuralPPOTrainer<Model: PPOPolicyValueModel>: Sendable {
    /// Model being optimized.
    public private(set) var model: Model

    /// PPO hyperparameters.
    public let configuration: PPOConfiguration

    /// Creates a trainer around a mutable policy/value model.
    public init(model: Model, configuration: PPOConfiguration) {
        self.model = model
        self.configuration = configuration
    }

    /// Runs the configured number of PPO epochs over deterministic minibatches.
    public mutating func update(samples: [PPOTrainingSample]) throws -> PPOTrainingSummary {
        guard !samples.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        var steps: [PPOOptimizerStepSummary] = []
        for epoch in 0..<configuration.epochs {
            let epochConfiguration = try configuration.scheduled(epoch: epoch)
            var minibatchIndex = 0
            var start = 0
            while start < samples.count {
                let end = min(start + configuration.minibatchSize, samples.count)
                let minibatch = Array(samples[start..<end])
                let objective = try model.update(minibatch: minibatch, configuration: epochConfiguration)
                steps.append(PPOOptimizerStepSummary(
                    epoch: epoch,
                    minibatchIndex: minibatchIndex,
                    sampleCount: minibatch.count,
                    objective: objective
                ))
                minibatchIndex += 1
                start = end
            }
        }
        return PPOTrainingSummary(
            sampleCount: samples.count,
            epochCount: configuration.epochs,
            optimizerSteps: steps
        )
    }

    /// Samples prioritized trajectory segments and runs the standard PPO update over their samples.
    public mutating func update(
        segments: [PPOTrajectorySegment],
        sampledSegmentCount: Int? = nil,
        seed: UInt64 = 0
    ) throws -> PPOTrainingSummary {
        let sampler = try PPOTrajectorySegmentSampler(
            segments: segments,
            priorityAlpha: configuration.trajectoryPriorityAlpha
        )
        let selected = try sampler.sample(count: sampledSegmentCount ?? segments.count, seed: seed)
        return try update(samples: selected.flatMap(\.samples))
    }
}

/// A compact one-hidden-layer actor-critic model for discrete PPO training.
public struct DenseDiscreteActorCriticModel: PPOPolicyValueModel {
    /// Number of observation features.
    public let observationDimension: Int

    /// Number of hidden tanh units.
    public let hiddenUnitCount: Int

    /// Number of discrete actions.
    public let actionCount: Int

    private var inputWeights: [Double]
    private var hiddenBiases: [Double]
    private var actorWeights: [Double]
    private var actorBiases: [Double]
    private var valueWeights: [Double]
    private var valueBias: Double

    /// Creates a deterministic dense actor-critic model.
    public init(
        observationDimension: Int,
        hiddenUnitCount: Int,
        actionCount: Int,
        seed: UInt64 = 0,
        weightScale: Double = 0.05
    ) throws {
        guard observationDimension > 0 else {
            throw RLSwiftError.dimensionMismatch(expected: 1, actual: observationDimension)
        }
        guard hiddenUnitCount > 0 else {
            throw RLSwiftError.invalidCapacity(hiddenUnitCount)
        }
        guard actionCount > 0 else {
            throw RLSwiftError.emptyActionSpace
        }
        guard weightScale >= 0 else {
            throw RLSwiftError.invalidWeight(weightScale)
        }
        self.observationDimension = observationDimension
        self.hiddenUnitCount = hiddenUnitCount
        self.actionCount = actionCount

        var generator = SeededGenerator(seed: seed)
        func weights(_ count: Int) -> [Double] {
            (0..<count).map { _ in
                let unit = Double(generator.next()) / Double(UInt64.max)
                return (unit * 2 - 1) * weightScale
            }
        }
        inputWeights = weights(observationDimension * hiddenUnitCount)
        hiddenBiases = weights(hiddenUnitCount)
        actorWeights = weights(hiddenUnitCount * actionCount)
        actorBiases = weights(actionCount)
        valueWeights = weights(hiddenUnitCount)
        valueBias = weights(1)[0]
    }

    /// Predicts action logits and value for one observation.
    public func prediction(for observation: [Double]) throws -> PPOPolicyValuePrediction {
        let forward = try forward(observation)
        return try PPOPolicyValuePrediction(logits: forward.logits, valueEstimate: forward.value)
    }

    /// Applies one stochastic-gradient PPO update.
    public mutating func update(
        minibatch: [PPOTrainingSample],
        configuration: PPOConfiguration
    ) throws -> PPOObjectiveBreakdown {
        guard !minibatch.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }

        var inputWeightGradients = Array(repeating: 0.0, count: inputWeights.count)
        var hiddenBiasGradients = Array(repeating: 0.0, count: hiddenBiases.count)
        var actorWeightGradients = Array(repeating: 0.0, count: actorWeights.count)
        var actorBiasGradients = Array(repeating: 0.0, count: actorBiases.count)
        var valueWeightGradients = Array(repeating: 0.0, count: valueWeights.count)
        var valueBiasGradient = 0.0
        var objectiveSamples: [PPOClippedObjectiveSample] = []
        objectiveSamples.reserveCapacity(minibatch.count)

        for sample in minibatch {
            guard sample.actionIndex >= 0, sample.actionIndex < actionCount else {
                throw RLSwiftError.dimensionMismatch(expected: actionCount, actual: sample.actionIndex + 1)
            }
            let forward = try forward(sample.observation)
            let prediction = try PPOPolicyValuePrediction(logits: forward.logits, valueEstimate: forward.value)
            let newLogProbability = try prediction.logProbability(actionIndex: sample.actionIndex)
            objectiveSamples.append(PPOClippedObjectiveSample(
                oldLogProbability: sample.oldLogProbability,
                newLogProbability: newLogProbability,
                advantage: sample.advantage,
                returnEstimate: sample.returnEstimate,
                valueEstimate: forward.value,
                entropy: prediction.entropy,
                oldValueEstimate: sample.oldValueEstimate
            ))

            var logitGradients = Array(repeating: 0.0, count: actionCount)
            let ratio = exp(newLogProbability - sample.oldLogProbability)
            let clippedRatio = min(max(ratio, 1 - configuration.clipRange), 1 + configuration.clipRange)
            let unclippedObjective = ratio * sample.advantage
            let clippedObjective = clippedRatio * sample.advantage
            if unclippedObjective <= clippedObjective {
                let logProbabilityGradient = -sample.advantage * ratio
                for action in 0..<actionCount {
                    let indicator = action == sample.actionIndex ? 1.0 : 0.0
                    logitGradients[action] += logProbabilityGradient * (indicator - forward.probabilities[action])
                }
            }

            for action in 0..<actionCount {
                let probability = forward.probabilities[action]
                let entropyGradient = -probability * (log(max(probability, 1e-12)) + prediction.entropy)
                logitGradients[action] -= configuration.entropyCoefficient * entropyGradient
            }

            let valueGradient = configuration.valueLossCoefficient * (forward.value - sample.returnEstimate)
            var hiddenGradients = Array(repeating: 0.0, count: hiddenUnitCount)
            for hidden in 0..<hiddenUnitCount {
                for action in 0..<actionCount {
                    let index = hidden * actionCount + action
                    actorWeightGradients[index] += forward.hidden[hidden] * logitGradients[action]
                    hiddenGradients[hidden] += actorWeights[index] * logitGradients[action]
                }
                valueWeightGradients[hidden] += forward.hidden[hidden] * valueGradient
                hiddenGradients[hidden] += valueWeights[hidden] * valueGradient
            }
            for action in 0..<actionCount {
                actorBiasGradients[action] += logitGradients[action]
            }
            valueBiasGradient += valueGradient

            for hidden in 0..<hiddenUnitCount {
                let hiddenPreActivationGradient = hiddenGradients[hidden] * (1 - forward.hidden[hidden] * forward.hidden[hidden])
                hiddenBiasGradients[hidden] += hiddenPreActivationGradient
                for input in 0..<observationDimension {
                    let index = input * hiddenUnitCount + hidden
                    inputWeightGradients[index] += sample.observation[input] * hiddenPreActivationGradient
                }
            }
        }

        let objective = try PPOClippedObjective.evaluate(samples: objectiveSamples, configuration: configuration)
        var gradientScale = 1.0
        if let maximumGradientNorm = configuration.maximumGradientNorm {
            let squaredNorm =
                inputWeightGradients.reduce(0.0) { $0 + $1 * $1 }
                + hiddenBiasGradients.reduce(0.0) { $0 + $1 * $1 }
                + actorWeightGradients.reduce(0.0) { $0 + $1 * $1 }
                + actorBiasGradients.reduce(0.0) { $0 + $1 * $1 }
                + valueWeightGradients.reduce(0.0) { $0 + $1 * $1 }
                + valueBiasGradient * valueBiasGradient
            let norm = sqrt(squaredNorm)
            gradientScale = norm > maximumGradientNorm ? maximumGradientNorm / norm : 1
        }
        let scale = configuration.learningRate * gradientScale / Double(minibatch.count)
        for index in inputWeights.indices {
            inputWeights[index] -= scale * inputWeightGradients[index]
        }
        for index in hiddenBiases.indices {
            hiddenBiases[index] -= scale * hiddenBiasGradients[index]
        }
        for index in actorWeights.indices {
            actorWeights[index] -= scale * actorWeightGradients[index]
        }
        for index in actorBiases.indices {
            actorBiases[index] -= scale * actorBiasGradients[index]
        }
        for index in valueWeights.indices {
            valueWeights[index] -= scale * valueWeightGradients[index]
        }
        valueBias -= scale * valueBiasGradient
        return objective
    }

    private struct ForwardPass {
        var hidden: [Double]
        var logits: [Double]
        var probabilities: [Double]
        var value: Double
    }

    private func forward(_ observation: [Double]) throws -> ForwardPass {
        guard observation.count == observationDimension else {
            throw RLSwiftError.dimensionMismatch(expected: observationDimension, actual: observation.count)
        }

        var hidden = Array(repeating: 0.0, count: hiddenUnitCount)
        for hiddenIndex in 0..<hiddenUnitCount {
            var activation = hiddenBiases[hiddenIndex]
            for input in 0..<observationDimension {
                activation += observation[input] * inputWeights[input * hiddenUnitCount + hiddenIndex]
            }
            hidden[hiddenIndex] = tanh(activation)
        }

        var logits = actorBiases
        for action in 0..<actionCount {
            for hiddenIndex in 0..<hiddenUnitCount {
                logits[action] += hidden[hiddenIndex] * actorWeights[hiddenIndex * actionCount + action]
            }
        }

        var value = valueBias
        for hiddenIndex in 0..<hiddenUnitCount {
            value += hidden[hiddenIndex] * valueWeights[hiddenIndex]
        }
        let prediction = try PPOPolicyValuePrediction(logits: logits, valueEstimate: value)
        return ForwardPass(hidden: hidden, logits: logits, probabilities: prediction.probabilities, value: value)
    }
}
