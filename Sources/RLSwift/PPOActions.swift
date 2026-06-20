import Foundation

/// Action value used by PPO policies across discrete, multidiscrete, and continuous spaces.
public enum PPOAction: Sendable, Equatable, Codable {
    /// One categorical action index.
    case discrete(Int)

    /// One categorical action index per action dimension.
    case multiDiscrete([Int])

    /// One continuous action value per action dimension.
    case continuous([Double])
}

/// Policy action distribution used to score PPO rollout actions.
public enum PPOActionDistribution: Sendable, Equatable, Codable {
    /// One categorical distribution represented by unnormalized logits.
    case categorical(logits: [Double])

    /// Independent categorical distributions represented by one logit array per action dimension.
    case multiCategorical(logitsByDimension: [[Double]])

    /// Independent normal distributions represented by means and log standard deviations.
    case diagonalGaussian(mean: [Double], logStandardDeviation: [Double])

    /// Returns the action log probability under this distribution.
    public func logProbability(of action: PPOAction) throws -> Double {
        switch (self, action) {
        case let (.categorical(logits), .discrete(index)):
            let probabilities = try Self.softmax(logits)
            guard index >= 0, index < probabilities.count else {
                throw RLSwiftError.dimensionMismatch(expected: probabilities.count, actual: index + 1)
            }
            return log(max(probabilities[index], 1e-12))
        case let (.multiCategorical(logitsByDimension), .multiDiscrete(indices)):
            guard logitsByDimension.count == indices.count else {
                throw RLSwiftError.dimensionMismatch(expected: logitsByDimension.count, actual: indices.count)
            }
            var total = 0.0
            for dimension in logitsByDimension.indices {
                let probabilities = try Self.softmax(logitsByDimension[dimension])
                let index = indices[dimension]
                guard index >= 0, index < probabilities.count else {
                    throw RLSwiftError.dimensionMismatch(expected: probabilities.count, actual: index + 1)
                }
                total += log(max(probabilities[index], 1e-12))
            }
            return total
        case let (.diagonalGaussian(mean, logStandardDeviation), .continuous(values)):
            try Self.validateGaussian(mean: mean, logStandardDeviation: logStandardDeviation, values: values)
            let logTwoPi = log(2 * Double.pi)
            var total = 0.0
            for index in mean.indices {
                let standardDeviation = exp(logStandardDeviation[index])
                let normalized = (values[index] - mean[index]) / standardDeviation
                total += -0.5 * normalized * normalized - logStandardDeviation[index] - 0.5 * logTwoPi
            }
            return total
        default:
            throw RLSwiftError.dimensionMismatch(expected: 1, actual: 0)
        }
    }

    /// Returns the Shannon entropy of this distribution.
    public func entropy() throws -> Double {
        switch self {
        case let .categorical(logits):
            return try Self.categoricalEntropy(logits)
        case let .multiCategorical(logitsByDimension):
            guard !logitsByDimension.isEmpty else {
                throw RLSwiftError.invalidSampleCount(0)
            }
            var total = 0.0
            for logits in logitsByDimension {
                total += try Self.categoricalEntropy(logits)
            }
            return total
        case let .diagonalGaussian(mean, logStandardDeviation):
            try Self.validateGaussian(mean: mean, logStandardDeviation: logStandardDeviation, values: mean)
            let constant = 0.5 * (1 + log(2 * Double.pi))
            return logStandardDeviation.reduce(0.0) { $0 + $1 + constant }
        }
    }

    private static func categoricalEntropy(_ logits: [Double]) throws -> Double {
        try softmax(logits).reduce(0.0) { partial, probability in
            partial - probability * log(max(probability, 1e-12))
        }
    }

    private static func softmax(_ logits: [Double]) throws -> [Double] {
        guard !logits.isEmpty else {
            throw RLSwiftError.emptyActionSpace
        }
        let maximum = logits.max()!
        let exponentials = logits.map { exp($0 - maximum) }
        let total = exponentials.reduce(0, +)
        return exponentials.map { $0 / total }
    }

    private static func validateGaussian(
        mean: [Double],
        logStandardDeviation: [Double],
        values: [Double]
    ) throws {
        guard !mean.isEmpty else {
            throw RLSwiftError.dimensionMismatch(expected: 1, actual: 0)
        }
        guard mean.count == logStandardDeviation.count else {
            throw RLSwiftError.dimensionMismatch(expected: mean.count, actual: logStandardDeviation.count)
        }
        guard mean.count == values.count else {
            throw RLSwiftError.dimensionMismatch(expected: mean.count, actual: values.count)
        }
    }
}

/// Norm statistics produced by PPO gradient clipping.
public struct PPOGradientClipSummary: Sendable, Equatable, Codable {
    /// Norm before clipping.
    public let originalNorm: Double

    /// Norm after clipping.
    public let clippedNorm: Double

    /// Multiplicative scale applied to every gradient element.
    public let scale: Double
}

/// Gradient vector after optional PPO norm clipping.
public struct PPOClippedGradientVector: Sendable, Equatable, Codable {
    /// Scaled gradient values.
    public let values: [Double]

    /// Norm and scale summary.
    public let summary: PPOGradientClipSummary
}

/// Utility for global-norm PPO gradient clipping.
public enum PPOGradientClipper {
    /// Returns a clipped copy of `gradients`.
    public static func clipped(_ gradients: [Double], maximumNorm: Double) throws -> PPOClippedGradientVector {
        guard maximumNorm > 0 else {
            throw RLSwiftError.invalidWeight(maximumNorm)
        }
        let originalNorm = sqrt(gradients.reduce(0.0) { $0 + $1 * $1 })
        let scale = originalNorm > maximumNorm ? maximumNorm / originalNorm : 1
        let values = gradients.map { $0 * scale }
        let summary = PPOGradientClipSummary(
            originalNorm: originalNorm,
            clippedNorm: originalNorm * scale,
            scale: scale
        )
        return PPOClippedGradientVector(values: values, summary: summary)
    }
}

/// Contiguous trajectory segment used by prioritized PPO replay.
public struct PPOTrajectorySegment: Sendable, Equatable, Codable {
    /// Stable segment identifier.
    public let id: String

    /// Samples in timestep order.
    public let samples: [PPOTrainingSample]

    /// Positive sampling priority.
    public let priority: Double

    /// Creates one prioritized trajectory segment.
    public init(id: String, samples: [PPOTrainingSample], priority: Double) throws {
        guard !id.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "ppo.segmentID")
        }
        guard !samples.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        guard priority > 0 else {
            throw RLSwiftError.invalidPriority(priority)
        }
        self.id = id
        self.samples = samples
        self.priority = priority
    }
}

/// Deterministic weighted sampler for PPO trajectory segments.
public struct PPOTrajectorySegmentSampler: Sendable, Equatable, Codable {
    /// Segments available for replay.
    public let segments: [PPOTrajectorySegment]

    /// Priority exponent. A value of `0` samples uniformly.
    public let priorityAlpha: Double

    /// Creates a trajectory segment sampler.
    public init(segments: [PPOTrajectorySegment], priorityAlpha: Double) throws {
        guard !segments.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        guard priorityAlpha >= 0 else {
            throw RLSwiftError.invalidWeight(priorityAlpha)
        }
        self.segments = segments
        self.priorityAlpha = priorityAlpha
    }

    /// Samples segments with replacement using deterministic weighted sampling.
    public func sample(count: Int, seed: UInt64) throws -> [PPOTrajectorySegment] {
        guard count > 0 else {
            throw RLSwiftError.invalidSampleCount(count)
        }
        let weights = segments.map { pow($0.priority, priorityAlpha) }
        var runningTotal = 0.0
        let cumulativeWeights = weights.map { weight in
            runningTotal += weight
            return runningTotal
        }
        var generator = SeededGenerator(seed: seed)
        return (0..<count).map { _ in
            let unit = Double(generator.next() >> 11) / 9_007_199_254_740_992.0
            let threshold = unit * runningTotal
            let index = cumulativeWeights.firstIndex { threshold < $0 }!
            return segments[index]
        }
    }
}
