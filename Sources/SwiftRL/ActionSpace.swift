import Foundation

/// Errors thrown by SwiftRL when an algorithm or data structure receives invalid input.
public enum SwiftRLError: Error, Equatable, Sendable {
    /// Indicates that an action-dependent component was created without any available actions.
    case emptyActionSpace

    /// Indicates that a probability value was outside the closed `0...1` range.
    case invalidProbability(Double)

    /// Indicates that a softmax temperature was not strictly positive.
    case invalidTemperature(Double)

    /// Indicates that a bounded storage container was created with a non-positive capacity.
    case invalidCapacity(Int)

    /// Indicates that a sample request used a negative count.
    case invalidSampleCount(Int)

    /// Indicates that a lower/upper bound pair is invalid at an index.
    case invalidBounds(index: Int, lower: Double, upper: Double)

    /// Indicates that a vector had a different dimension than required.
    case dimensionMismatch(expected: Int, actual: Int)

    /// Indicates that an n-step or rollout horizon was not positive.
    case invalidHorizon(Int)

    /// Indicates that a duration field was negative.
    case invalidDuration(name: String, value: Double)

    /// Indicates that a weighting value was negative.
    case invalidWeight(Double)

    /// Indicates that a replay priority was not strictly positive.
    case invalidPriority(Double)
}

/// A deterministic random number generator suitable for reproducible reinforcement learning tests.
public struct SeededGenerator: RandomNumberGenerator, Sendable {
    private var state: UInt64

    /// Creates a generator from a fixed seed.
    public init(seed: UInt64) {
        state = seed
    }

    /// Returns the next 64 bits from a SplitMix64 sequence.
    public mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        return value ^ (value >> 31)
    }
}

/// A finite set of available actions for discrete-control environments and policies.
public struct DiscreteActionSpace<Action: Hashable & Sendable>: Sendable {
    /// The ordered actions available to the agent.
    public let actions: [Action]

    /// Creates an action space, preserving the provided action order for tie-breaking.
    public init(_ actions: [Action]) throws {
        guard !actions.isEmpty else {
            throw SwiftRLError.emptyActionSpace
        }
        self.actions = actions
    }

    /// Returns `true` when the action belongs to the space.
    public func contains(_ action: Action) -> Bool {
        actions.contains(action)
    }

    /// Selects a reproducible random action from the space.
    public func randomAction(using generator: inout SeededGenerator) -> Action {
        actions[Int(generator.next() % UInt64(actions.count))]
    }
}

/// A bounded continuous action or observation space.
public struct ContinuousBoxSpace: Sendable, Equatable {
    /// The inclusive lower bounds for each dimension.
    public let lowerBounds: [Double]

    /// The inclusive upper bounds for each dimension.
    public let upperBounds: [Double]

    /// Creates a continuous box and validates every bound pair.
    public init(lowerBounds: [Double], upperBounds: [Double]) throws {
        guard lowerBounds.count == upperBounds.count else {
            throw SwiftRLError.dimensionMismatch(expected: lowerBounds.count, actual: upperBounds.count)
        }
        for index in lowerBounds.indices {
            guard lowerBounds[index] <= upperBounds[index] else {
                throw SwiftRLError.invalidBounds(index: index, lower: lowerBounds[index], upper: upperBounds[index])
            }
        }
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds
    }

    /// The number of dimensions in the space.
    public var dimension: Int {
        lowerBounds.count
    }

    /// Returns whether a vector lies inside all bounds.
    public func contains(_ values: [Double]) -> Bool {
        guard values.count == dimension else {
            return false
        }
        for index in values.indices where values[index] < lowerBounds[index] || values[index] > upperBounds[index] {
            return false
        }
        return true
    }

    /// Clips a vector into the box.
    public func clamp(_ values: [Double]) throws -> [Double] {
        try validateDimension(of: values)
        return values.indices.map { min(max(values[$0], lowerBounds[$0]), upperBounds[$0]) }
    }

    /// Converts a normalized `[-1, 1]` vector into the box's physical units.
    public func scaleFromUnit(_ values: [Double]) throws -> [Double] {
        try validateDimension(of: values)
        return values.indices.map { index in
            let midpoint = (lowerBounds[index] + upperBounds[index]) / 2
            let halfRange = (upperBounds[index] - lowerBounds[index]) / 2
            return midpoint + values[index] * halfRange
        }
    }

    /// Converts a vector from physical units into normalized `[-1, 1]` coordinates.
    public func normalizeToUnit(_ values: [Double]) throws -> [Double] {
        try validateDimension(of: values)
        return values.indices.map { index in
            let halfRange = (upperBounds[index] - lowerBounds[index]) / 2
            guard halfRange > 0 else {
                return 0
            }
            let midpoint = (lowerBounds[index] + upperBounds[index]) / 2
            return (values[index] - midpoint) / halfRange
        }
    }

    private func validateDimension(of values: [Double]) throws {
        guard values.count == dimension else {
            throw SwiftRLError.dimensionMismatch(expected: dimension, actual: values.count)
        }
    }
}
