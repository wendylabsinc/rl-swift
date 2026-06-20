/// The direction used to evaluate a scalar safety or performance constraint.
public enum ConstraintRelation: String, Sendable, Equatable, Codable {
    /// The value must be less than or equal to the limit.
    case lessThanOrEqual

    /// The value must be greater than or equal to the limit.
    case greaterThanOrEqual
}

/// A scalar constraint signal for constrained or safety-aware reinforcement learning.
public struct ConstraintSignal: Sendable, Equatable, Codable {
    /// A stable name for the constraint.
    public let name: String

    /// The measured scalar value.
    public let value: Double

    /// The constraint limit.
    public let limit: Double

    /// The relation that determines satisfaction.
    public let relation: ConstraintRelation

    /// The multiplier applied to violation amount when computing cost.
    public let weight: Double

    /// Creates a constraint signal.
    public init(
        name: String,
        value: Double,
        limit: Double,
        relation: ConstraintRelation,
        weight: Double = 1
    ) throws {
        guard weight >= 0 else {
            throw RLSwiftError.invalidWeight(weight)
        }
        self.name = name
        self.value = value
        self.limit = limit
        self.relation = relation
        self.weight = weight
    }

    /// Whether the constraint is currently satisfied.
    public var isSatisfied: Bool {
        switch relation {
        case .lessThanOrEqual:
            return value <= limit
        case .greaterThanOrEqual:
            return value >= limit
        }
    }

    /// The non-negative amount by which the constraint is violated.
    public var violationAmount: Double {
        switch relation {
        case .lessThanOrEqual:
            return max(0, value - limit)
        case .greaterThanOrEqual:
            return max(0, limit - value)
        }
    }

    /// The weighted cost contribution for this constraint.
    public var cost: Double {
        violationAmount * weight
    }
}

/// A collection of constraint signals produced by a robot step or trajectory.
public struct ConstraintReport: Sendable, Equatable, Codable {
    /// The constraints included in the report.
    public let signals: [ConstraintSignal]

    /// Creates a report from constraint signals.
    public init(_ signals: [ConstraintSignal]) {
        self.signals = signals
    }

    /// Whether every constraint is currently satisfied.
    public var isSatisfied: Bool {
        signals.allSatisfy(\.isSatisfied)
    }

    /// The total weighted constraint violation cost.
    public var totalCost: Double {
        signals.reduce(0) { $0 + $1.cost }
    }

    /// The constraints that are currently violated.
    public var violations: [ConstraintSignal] {
        signals.filter { !$0.isSatisfied }
    }

    /// Returns the weighted cost for a named constraint, or `nil` when absent.
    public func cost(named name: String) -> Double? {
        signals.first { $0.name == name }?.cost
    }
}
