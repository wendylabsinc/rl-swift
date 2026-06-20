/// A finite set of available actions for discrete-control environments and policies.
public struct DiscreteActionSpace<Action: Hashable & Sendable>: Sendable {
    /// The ordered actions available to the agent.
    public let actions: [Action]

    /// Creates an action space, preserving the provided action order for tie-breaking.
    public init(_ actions: [Action]) throws {
        guard !actions.isEmpty else {
            throw RLSwiftError.emptyActionSpace
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
public struct ContinuousBoxSpace: Sendable, Equatable, Codable {
    /// The inclusive lower bounds for each dimension.
    public let lowerBounds: [Double]

    /// The inclusive upper bounds for each dimension.
    public let upperBounds: [Double]

    /// Creates a continuous box and validates every bound pair.
    public init(lowerBounds: [Double], upperBounds: [Double]) throws {
        guard lowerBounds.count == upperBounds.count else {
            throw RLSwiftError.dimensionMismatch(expected: lowerBounds.count, actual: upperBounds.count)
        }
        for index in lowerBounds.indices {
            guard lowerBounds[index] <= upperBounds[index] else {
                throw RLSwiftError.invalidBounds(index: index, lower: lowerBounds[index], upper: upperBounds[index])
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
        for index in 0..<dimension where values[index] < lowerBounds[index] || values[index] > upperBounds[index] {
            return false
        }
        return true
    }

    /// Clips a vector into the box.
    public func clamp(_ values: [Double]) throws -> [Double] {
        try validateDimension(of: values)
        var scratch = VectorScratch(count: dimension)
        for index in 0..<dimension {
            scratch.set(min(max(values[index], lowerBounds[index]), upperBounds[index]), at: index)
        }
        return scratch.finish()
    }

    /// Converts a normalized `[-1, 1]` vector into the box's physical units.
    public func scaleFromUnit(_ values: [Double]) throws -> [Double] {
        try validateDimension(of: values)
        var scratch = VectorScratch(count: dimension)
        for index in 0..<dimension {
            let midpoint = (lowerBounds[index] + upperBounds[index]) / 2
            let halfRange = (upperBounds[index] - lowerBounds[index]) / 2
            scratch.set(midpoint + values[index] * halfRange, at: index)
        }
        return scratch.finish()
    }

    /// Converts a vector from physical units into normalized `[-1, 1]` coordinates.
    public func normalizeToUnit(_ values: [Double]) throws -> [Double] {
        try validateDimension(of: values)
        var scratch = VectorScratch(count: dimension)
        for index in 0..<dimension {
            let halfRange = (upperBounds[index] - lowerBounds[index]) / 2
            if halfRange > 0 {
                let midpoint = (lowerBounds[index] + upperBounds[index]) / 2
                scratch.set((values[index] - midpoint) / halfRange, at: index)
            } else {
                scratch.set(0, at: index)
            }
        }
        return scratch.finish()
    }

    private func validateDimension(of values: [Double]) throws {
        guard values.count == dimension else {
            throw RLSwiftError.dimensionMismatch(expected: dimension, actual: values.count)
        }
    }
}
