import Foundation

/// Hyperparameters for clipped proximal policy optimization.
public struct PPOConfiguration: Sendable, Equatable, Codable {
    /// Discount applied to future rewards.
    public let discount: Double

    /// Lambda used by generalized advantage estimation.
    public let gaeLambda: Double

    /// Symmetric clipping range around probability ratio `1`.
    public let clipRange: Double

    /// Weight applied to the value-function loss term.
    public let valueLossCoefficient: Double

    /// Weight applied to the policy entropy bonus.
    public let entropyCoefficient: Double

    /// Optimizer learning rate used by backend trainers.
    public let learningRate: Double

    /// Number of optimization epochs per collected rollout.
    public let epochs: Int

    /// Minibatch size used by backend trainers.
    public let minibatchSize: Int

    /// Creates a PPO configuration and validates the ranges used by stable training loops.
    public init(
        discount: Double = 0.99,
        gaeLambda: Double = 0.95,
        clipRange: Double = 0.2,
        valueLossCoefficient: Double = 0.5,
        entropyCoefficient: Double = 0.01,
        learningRate: Double = 3e-4,
        epochs: Int = 4,
        minibatchSize: Int = 256
    ) throws {
        guard (0...1).contains(discount) else {
            throw RLSwiftError.invalidProbability(discount)
        }
        guard (0...1).contains(gaeLambda) else {
            throw RLSwiftError.invalidProbability(gaeLambda)
        }
        guard clipRange >= 0 else {
            throw RLSwiftError.invalidWeight(clipRange)
        }
        guard valueLossCoefficient >= 0 else {
            throw RLSwiftError.invalidWeight(valueLossCoefficient)
        }
        guard entropyCoefficient >= 0 else {
            throw RLSwiftError.invalidWeight(entropyCoefficient)
        }
        guard learningRate > 0 else {
            throw RLSwiftError.invalidWeight(learningRate)
        }
        guard epochs > 0 else {
            throw RLSwiftError.invalidHorizon(epochs)
        }
        guard minibatchSize > 0 else {
            throw RLSwiftError.invalidCapacity(minibatchSize)
        }
        self.discount = discount
        self.gaeLambda = gaeLambda
        self.clipRange = clipRange
        self.valueLossCoefficient = valueLossCoefficient
        self.entropyCoefficient = entropyCoefficient
        self.learningRate = learningRate
        self.epochs = epochs
        self.minibatchSize = minibatchSize
    }
}

/// One timestep of policy/value data needed to compute PPO advantages.
public struct PPOTrajectoryStep: Sendable, Equatable, Codable {
    /// Reward observed after the action.
    public let reward: Double

    /// Value estimate for the observation before the action.
    public let valueEstimate: Double

    /// Log probability assigned to the sampled action by the behavior policy.
    public let logProbability: Double

    /// Policy entropy for the action distribution at this timestep.
    public let entropy: Double

    /// Termination semantics for the transition.
    public let termination: StepTermination

    /// Creates one PPO trajectory step.
    public init(
        reward: Double,
        valueEstimate: Double,
        logProbability: Double,
        entropy: Double,
        termination: StepTermination = .continuing
    ) {
        self.reward = reward
        self.valueEstimate = valueEstimate
        self.logProbability = logProbability
        self.entropy = entropy
        self.termination = termination
    }
}

/// Advantage and return targets produced from a PPO rollout.
public struct PPOAdvantageBatch: Sendable, Equatable, Codable {
    /// Advantage estimates in timestep order.
    public let advantages: [Double]

    /// Value-function return targets in timestep order.
    public let returns: [Double]

    /// Creates a batch of advantage and return targets.
    public init(advantages: [Double], returns: [Double]) throws {
        guard advantages.count == returns.count else {
            throw RLSwiftError.dimensionMismatch(expected: advantages.count, actual: returns.count)
        }
        self.advantages = advantages
        self.returns = returns
    }
}

/// Utilities for generalized advantage estimation.
public enum PPOAdvantageEstimator {
    /// Computes generalized advantage estimates and value targets for one rollout.
    public static func generalizedAdvantageEstimate(
        steps: [PPOTrajectoryStep],
        lastValue: Double,
        configuration: PPOConfiguration
    ) throws -> PPOAdvantageBatch {
        guard !steps.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        var advantages = Array(repeating: 0.0, count: steps.count)
        var returns = Array(repeating: 0.0, count: steps.count)
        var runningAdvantage = 0.0

        for index in steps.indices.reversed() {
            let nextValue = index == steps.indices.last ? lastValue : steps[index + 1].valueEstimate
            let nonTerminal = steps[index].termination.endsEpisode ? 0.0 : 1.0
            let delta = steps[index].reward + configuration.discount * nextValue * nonTerminal - steps[index].valueEstimate
            runningAdvantage = delta + configuration.discount * configuration.gaeLambda * nonTerminal * runningAdvantage
            advantages[index] = runningAdvantage
            returns[index] = runningAdvantage + steps[index].valueEstimate
        }

        return try PPOAdvantageBatch(advantages: advantages, returns: returns)
    }
}

/// One sample used by the clipped PPO objective.
public struct PPOClippedObjectiveSample: Sendable, Equatable, Codable {
    /// Log probability recorded under the behavior policy.
    public let oldLogProbability: Double

    /// Log probability produced by the current policy.
    public let newLogProbability: Double

    /// Advantage target for this sample.
    public let advantage: Double

    /// Return target for value learning.
    public let returnEstimate: Double

    /// Current value estimate.
    public let valueEstimate: Double

    /// Entropy of the current action distribution.
    public let entropy: Double

    /// Creates one clipped-objective sample.
    public init(
        oldLogProbability: Double,
        newLogProbability: Double,
        advantage: Double,
        returnEstimate: Double,
        valueEstimate: Double,
        entropy: Double
    ) {
        self.oldLogProbability = oldLogProbability
        self.newLogProbability = newLogProbability
        self.advantage = advantage
        self.returnEstimate = returnEstimate
        self.valueEstimate = valueEstimate
        self.entropy = entropy
    }

    /// Probability ratio between current and behavior policies.
    public var probabilityRatio: Double {
        exp(newLogProbability - oldLogProbability)
    }

    /// Probability ratio clipped to the PPO trust region.
    public func clippedRatio(clipRange: Double) -> Double {
        min(max(probabilityRatio, 1 - clipRange), 1 + clipRange)
    }
}

/// Scalar loss breakdown for a PPO minibatch.
public struct PPOObjectiveBreakdown: Sendable, Equatable, Codable {
    /// Policy-gradient loss after PPO clipping.
    public let policyLoss: Double

    /// Mean squared value-function loss scaled by `0.5`.
    public let valueLoss: Double

    /// Mean entropy bonus.
    public let entropyBonus: Double

    /// Combined scalar loss after value and entropy coefficients are applied.
    public let totalLoss: Double

    /// Mean old-policy minus new-policy log probability.
    public let meanApproximateKL: Double

    /// Fraction of samples whose probability ratio was clipped.
    public let clippedFraction: Double
}

/// Pure Swift PPO objective math shared by backend-specific learners.
public enum PPOClippedObjective {
    /// Evaluates the clipped PPO loss for a minibatch.
    public static func evaluate(
        samples: [PPOClippedObjectiveSample],
        configuration: PPOConfiguration
    ) throws -> PPOObjectiveBreakdown {
        guard !samples.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }

        var policyTotal = 0.0
        var valueTotal = 0.0
        var entropyTotal = 0.0
        var klTotal = 0.0
        var clippedCount = 0

        for sample in samples {
            let ratio = sample.probabilityRatio
            let clippedRatio = sample.clippedRatio(clipRange: configuration.clipRange)
            policyTotal += min(ratio * sample.advantage, clippedRatio * sample.advantage)
            let valueError = sample.returnEstimate - sample.valueEstimate
            valueTotal += 0.5 * valueError * valueError
            entropyTotal += sample.entropy
            klTotal += sample.oldLogProbability - sample.newLogProbability
            if abs(ratio - 1) > configuration.clipRange {
                clippedCount += 1
            }
        }

        let count = Double(samples.count)
        let policyLoss = -policyTotal / count
        let valueLoss = valueTotal / count
        let entropyBonus = entropyTotal / count
        return PPOObjectiveBreakdown(
            policyLoss: policyLoss,
            valueLoss: valueLoss,
            entropyBonus: entropyBonus,
            totalLoss: policyLoss
                + configuration.valueLossCoefficient * valueLoss
                - configuration.entropyCoefficient * entropyBonus,
            meanApproximateKL: klTotal / count,
            clippedFraction: Double(clippedCount) / count
        )
    }
}
