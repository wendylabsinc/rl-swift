import Foundation

/// Linear multiplier schedule for PPO coefficients across optimization epochs.
public struct PPOAnnealingSchedule: Sendable, Equatable, Codable {
    /// Coefficient multiplier used on the final optimization epoch.
    public let finalMultiplier: Double

    /// A schedule that keeps the coefficient unchanged.
    public static let constant = try! PPOAnnealingSchedule(finalMultiplier: 1)

    /// Creates a linear schedule from multiplier `1` to `finalMultiplier`.
    public init(finalMultiplier: Double = 1) throws {
        guard finalMultiplier > 0 else {
            throw RLSwiftError.invalidWeight(finalMultiplier)
        }
        self.finalMultiplier = finalMultiplier
    }

    /// Returns the multiplier for a zero-based epoch index.
    public func multiplier(epoch: Int, totalEpochs: Int) throws -> Double {
        guard totalEpochs > 0 else {
            throw RLSwiftError.invalidHorizon(totalEpochs)
        }
        guard epoch >= 0 else {
            throw RLSwiftError.invalidSampleCount(epoch)
        }
        guard epoch < totalEpochs else {
            throw RLSwiftError.dimensionMismatch(expected: totalEpochs, actual: epoch + 1)
        }
        if totalEpochs == 1 {
            return finalMultiplier
        }
        let progress = Double(epoch) / Double(totalEpochs - 1)
        return 1 + (finalMultiplier - 1) * progress
    }
}

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

    /// Optional symmetric clipping range around the previous value estimate.
    public let valueClipRange: Double?

    /// Optimizer learning rate used by backend trainers.
    public let learningRate: Double

    /// Linear schedule applied to ``learningRate`` across PPO epochs.
    public let learningRateSchedule: PPOAnnealingSchedule

    /// Linear schedule applied to ``entropyCoefficient`` across PPO epochs.
    public let entropyCoefficientSchedule: PPOAnnealingSchedule

    /// Optional global gradient-norm limit used by trainable models.
    public let maximumGradientNorm: Double?

    /// Priority exponent used when sampling trajectory segments.
    public let trajectoryPriorityAlpha: Double

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
        valueClipRange: Double? = nil,
        learningRate: Double = 3e-4,
        learningRateSchedule: PPOAnnealingSchedule = .constant,
        entropyCoefficientSchedule: PPOAnnealingSchedule = .constant,
        maximumGradientNorm: Double? = nil,
        trajectoryPriorityAlpha: Double = 0,
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
        if let valueClipRange {
            guard valueClipRange >= 0 else {
                throw RLSwiftError.invalidWeight(valueClipRange)
            }
        }
        guard learningRate > 0 else {
            throw RLSwiftError.invalidWeight(learningRate)
        }
        if let maximumGradientNorm {
            guard maximumGradientNorm > 0 else {
                throw RLSwiftError.invalidWeight(maximumGradientNorm)
            }
        }
        guard trajectoryPriorityAlpha >= 0 else {
            throw RLSwiftError.invalidWeight(trajectoryPriorityAlpha)
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
        self.valueClipRange = valueClipRange
        self.learningRate = learningRate
        self.learningRateSchedule = learningRateSchedule
        self.entropyCoefficientSchedule = entropyCoefficientSchedule
        self.maximumGradientNorm = maximumGradientNorm
        self.trajectoryPriorityAlpha = trajectoryPriorityAlpha
        self.epochs = epochs
        self.minibatchSize = minibatchSize
    }

    /// Scheduled learning rate for a zero-based optimization epoch.
    public func learningRate(epoch: Int) throws -> Double {
        learningRate * (try learningRateSchedule.multiplier(epoch: epoch, totalEpochs: epochs))
    }

    /// Scheduled entropy coefficient for a zero-based optimization epoch.
    public func entropyCoefficient(epoch: Int) throws -> Double {
        entropyCoefficient * (try entropyCoefficientSchedule.multiplier(epoch: epoch, totalEpochs: epochs))
    }

    /// Returns a copy with epoch-specific scheduled coefficients and constant schedules.
    public func scheduled(epoch: Int) throws -> PPOConfiguration {
        try PPOConfiguration(
            discount: discount,
            gaeLambda: gaeLambda,
            clipRange: clipRange,
            valueLossCoefficient: valueLossCoefficient,
            entropyCoefficient: entropyCoefficient(epoch: epoch),
            valueClipRange: valueClipRange,
            learningRate: learningRate(epoch: epoch),
            learningRateSchedule: .constant,
            entropyCoefficientSchedule: .constant,
            maximumGradientNorm: maximumGradientNorm,
            trajectoryPriorityAlpha: trajectoryPriorityAlpha,
            epochs: epochs,
            minibatchSize: minibatchSize
        )
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

    /// Value estimate recorded under the behavior policy before the update.
    public let oldValueEstimate: Double?

    /// Entropy of the current action distribution.
    public let entropy: Double

    /// Creates one clipped-objective sample.
    public init(
        oldLogProbability: Double,
        newLogProbability: Double,
        advantage: Double,
        returnEstimate: Double,
        valueEstimate: Double,
        entropy: Double,
        oldValueEstimate: Double? = nil
    ) {
        self.oldLogProbability = oldLogProbability
        self.newLogProbability = newLogProbability
        self.advantage = advantage
        self.returnEstimate = returnEstimate
        self.valueEstimate = valueEstimate
        self.oldValueEstimate = oldValueEstimate
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

    /// Value estimate clipped around ``oldValueEstimate`` when value clipping is configured.
    public func clippedValueEstimate(valueClipRange: Double) -> Double? {
        guard let oldValueEstimate else {
            return nil
        }
        let delta = min(max(valueEstimate - oldValueEstimate, -valueClipRange), valueClipRange)
        return oldValueEstimate + delta
    }

    /// PPO value loss for this sample, including optional value clipping.
    public func valueLoss(valueClipRange: Double?) -> Double {
        let valueError = returnEstimate - valueEstimate
        let unclippedLoss = valueError * valueError
        guard let valueClipRange,
              let clippedValueEstimate = clippedValueEstimate(valueClipRange: valueClipRange) else {
            return 0.5 * unclippedLoss
        }
        let clippedError = returnEstimate - clippedValueEstimate
        return 0.5 * max(unclippedLoss, clippedError * clippedError)
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

    /// Creates a scalar PPO objective breakdown.
    public init(
        policyLoss: Double,
        valueLoss: Double,
        entropyBonus: Double,
        totalLoss: Double,
        meanApproximateKL: Double,
        clippedFraction: Double
    ) {
        self.policyLoss = policyLoss
        self.valueLoss = valueLoss
        self.entropyBonus = entropyBonus
        self.totalLoss = totalLoss
        self.meanApproximateKL = meanApproximateKL
        self.clippedFraction = clippedFraction
    }
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
            valueTotal += sample.valueLoss(valueClipRange: configuration.valueClipRange)
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
