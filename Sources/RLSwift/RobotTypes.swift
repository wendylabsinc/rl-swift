/// The control interpretation for a robot command vector.
public enum RobotControlMode: String, Sendable, Equatable, Codable {
    /// Interprets command values as target joint positions.
    case position

    /// Interprets command values as target joint velocities.
    case velocity

    /// Interprets command values as target joint torques or efforts.
    case torque
}

/// A robot action represented by a continuous command vector.
public struct RobotAction: Sendable, Equatable, Codable {
    /// The command values in the units implied by `mode`.
    public let commands: [Double]

    /// The control mode used by the command vector.
    public let mode: RobotControlMode

    /// Creates a robot action.
    public init(commands: [Double], mode: RobotControlMode) {
        self.commands = commands
        self.mode = mode
    }

    /// Returns a copy of the action clipped into a continuous command space.
    public func clipped(to space: ContinuousBoxSpace) throws -> RobotAction {
        RobotAction(commands: try space.clamp(commands), mode: mode)
    }
}

/// A compact robot observation containing proprioception, task pose, and named sensors.
public struct RobotObservation: Sendable, Equatable, Codable {
    /// Joint positions in a deterministic robot-specific order.
    public let jointPositions: [Double]

    /// Joint velocities in the same order as `jointPositions`.
    public let jointVelocities: [Double]

    /// End-effector pose or task-space state vector.
    public let endEffectorPose: [Double]

    /// Additional named scalar sensor values.
    public let sensorReadings: [String: Double]

    /// The environment step index for this observation.
    public let stepIndex: Int

    /// Creates a robot observation and validates joint vector dimensions.
    public init(
        jointPositions: [Double],
        jointVelocities: [Double],
        endEffectorPose: [Double] = [],
        sensorReadings: [String: Double] = [:],
        stepIndex: Int = 0
    ) throws {
        guard jointPositions.count == jointVelocities.count else {
            throw RLSwiftError.dimensionMismatch(expected: jointPositions.count, actual: jointVelocities.count)
        }
        self.jointPositions = jointPositions
        self.jointVelocities = jointVelocities
        self.endEffectorPose = endEffectorPose
        self.sensorReadings = sensorReadings
        self.stepIndex = stepIndex
    }

    /// Flattens the observation into a deterministic feature vector.
    public func flattenedFeatures(sensorOrder: [String]) -> [Double] {
        jointPositions
            + jointVelocities
            + endEffectorPose
            + sensorOrder.map { sensorReadings[$0] ?? 0 }
    }
}

/// A scalar reward term that can be weighted and combined with other terms.
public struct RewardComponent: Sendable, Equatable, Codable {
    /// A stable name for the reward component.
    public let name: String

    /// The raw component value before weighting.
    public let value: Double

    /// The multiplier applied to `value`.
    public let weight: Double

    /// Creates a reward component.
    public init(name: String, value: Double, weight: Double = 1) {
        self.name = name
        self.value = value
        self.weight = weight
    }

    /// The weighted contribution to total reward.
    public var contribution: Double {
        value * weight
    }
}

/// A deterministic reward accumulator for shaped robot rewards.
public struct RewardBreakdown: Sendable, Equatable, Codable {
    /// The component terms used to build the total reward.
    public let components: [RewardComponent]

    /// Creates a reward breakdown.
    public init(_ components: [RewardComponent]) {
        self.components = components
    }

    /// The weighted total reward.
    public var total: Double {
        components.reduce(0) { $0 + $1.contribution }
    }

    /// Returns the contribution for a named component, or `nil` when absent.
    public func contribution(named name: String) -> Double? {
        components.first { $0.name == name }?.contribution
    }
}
