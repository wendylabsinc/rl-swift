/// Timing metadata for a single robot or autonomous-system control step.
public struct ControlTiming: Sendable, Equatable, Codable {
    /// The zero-based control step index.
    public let stepIndex: Int

    /// The requested or observed control-period duration in seconds.
    public let deltaTime: Double

    /// The age of the observation data in seconds when the action was chosen.
    public let sensorAge: Double

    /// The delay between action selection and command application in seconds.
    public let actionLatency: Double

    /// Creates validated timing metadata.
    public init(
        stepIndex: Int,
        deltaTime: Double,
        sensorAge: Double = 0,
        actionLatency: Double = 0
    ) throws {
        try Self.validateNonNegative(deltaTime, name: "deltaTime")
        try Self.validateNonNegative(sensorAge, name: "sensorAge")
        try Self.validateNonNegative(actionLatency, name: "actionLatency")
        self.stepIndex = stepIndex
        self.deltaTime = deltaTime
        self.sensorAge = sensorAge
        self.actionLatency = actionLatency
    }

    /// The combined observation and actuation delay.
    public var closedLoopLatency: Double {
        sensorAge + actionLatency
    }

    /// Returns whether the closed-loop latency exceeds a deadline.
    public func missesDeadline(maximumLatency: Double) throws -> Bool {
        try Self.validateNonNegative(maximumLatency, name: "maximumLatency")
        return closedLoopLatency > maximumLatency
    }

    private static func validateNonNegative(_ value: Double, name: String) throws {
        guard value >= 0 else {
            throw RLSwiftError.invalidDuration(name: name, value: value)
        }
    }
}
