/// The current state of a hardware emergency-stop circuit or software interlock.
public struct EmergencyStopState: Sendable, Equatable, Codable {
    /// Whether the stop is currently engaged.
    public let isEngaged: Bool

    /// Human-readable reason for the stop.
    public let reason: String?

    /// Creates an emergency-stop state.
    public init(isEngaged: Bool, reason: String? = nil) {
        self.isEngaged = isEngaged
        self.reason = reason
    }
}

/// A hardware-supervisor intervention that can stop or modify policy output.
public enum SafetySupervisorIntervention: Sendable, Equatable, Codable {
    /// Emergency stop prevented command application.
    case emergencyStop(reason: String?)

    /// Sensor data was older than the configured maximum.
    case staleSensor(age: Double, maximumAge: Double)

    /// Closed-loop latency exceeded the configured maximum.
    case deadlineMissed(latency: Double, maximumLatency: Double)

    /// The action was clipped or rate-limited by ``RobotSafetyEnvelope``.
    case safetyEnvelope([SafetyIntervention])
}

/// Input data evaluated by a hardware-facing safety supervisor.
public struct SafetySupervisorInput: Sendable, Equatable, Codable {
    /// Action requested by the learned policy or planner.
    public let requestedAction: RobotAction

    /// Previously applied action, if available for rate limiting.
    public let previousAction: RobotAction?

    /// Timing metadata for the current control step.
    public let timing: ControlTiming

    /// Emergency-stop state sampled before command application.
    public let emergencyStop: EmergencyStopState

    /// Creates safety-supervisor input.
    public init(
        requestedAction: RobotAction,
        previousAction: RobotAction? = nil,
        timing: ControlTiming,
        emergencyStop: EmergencyStopState = EmergencyStopState(isEngaged: false)
    ) {
        self.requestedAction = requestedAction
        self.previousAction = previousAction
        self.timing = timing
        self.emergencyStop = emergencyStop
    }
}

/// A hardware-supervisor decision for one control step.
public struct SafetySupervisorDecision: Sendable, Equatable, Codable {
    /// Action requested by the policy.
    public let requestedAction: RobotAction

    /// Action that may be sent to actuators, or `nil` when execution must halt.
    public let commandedAction: RobotAction?

    /// Whether execution should halt before command application.
    public let shouldHalt: Bool

    /// Termination semantics to record in the episode or dataset.
    public let termination: StepTermination

    /// All supervisor interventions observed for this decision.
    public let interventions: [SafetySupervisorIntervention]

    /// Detailed envelope assessment when an action was evaluated.
    public let safetyAssessment: RobotSafetyAssessment?

    /// Creates a safety-supervisor decision.
    public init(
        requestedAction: RobotAction,
        commandedAction: RobotAction?,
        shouldHalt: Bool,
        termination: StepTermination,
        interventions: [SafetySupervisorIntervention],
        safetyAssessment: RobotSafetyAssessment?
    ) {
        self.requestedAction = requestedAction
        self.commandedAction = commandedAction
        self.shouldHalt = shouldHalt
        self.termination = termination
        self.interventions = interventions
        self.safetyAssessment = safetyAssessment
    }
}

/// A deterministic safety supervisor for robot deployments outside the learned policy.
public struct HardwareSafetySupervisor: Sendable, Equatable, Codable {
    /// Envelope used to clip and rate-limit commands.
    public let envelope: RobotSafetyEnvelope

    /// Optional command sent when the supervisor halts execution.
    public let failsafeAction: RobotAction?

    /// Maximum permitted sensor age in seconds.
    public let maximumSensorAge: Double?

    /// Maximum permitted closed-loop latency in seconds.
    public let maximumClosedLoopLatency: Double?

    /// Creates a hardware-facing safety supervisor.
    public init(
        envelope: RobotSafetyEnvelope,
        failsafeAction: RobotAction? = nil,
        maximumSensorAge: Double? = nil,
        maximumClosedLoopLatency: Double? = nil
    ) throws {
        if let maximumSensorAge {
            try Self.validateNonNegative(maximumSensorAge, name: "maximumSensorAge")
        }
        if let maximumClosedLoopLatency {
            try Self.validateNonNegative(maximumClosedLoopLatency, name: "maximumClosedLoopLatency")
        }
        if let failsafeAction, !envelope.commandSpace.contains(failsafeAction.commands) {
            throw RLSwiftError.dimensionMismatch(
                expected: envelope.commandSpace.dimension,
                actual: failsafeAction.commands.count
            )
        }
        self.envelope = envelope
        self.failsafeAction = failsafeAction
        self.maximumSensorAge = maximumSensorAge
        self.maximumClosedLoopLatency = maximumClosedLoopLatency
    }

    /// Evaluates a control step and returns either a safe command or a halt decision.
    public func assess(_ input: SafetySupervisorInput) throws -> SafetySupervisorDecision {
        var interventions: [SafetySupervisorIntervention] = []
        if input.emergencyStop.isEngaged {
            interventions.append(.emergencyStop(reason: input.emergencyStop.reason))
        }
        if let maximumSensorAge, input.timing.sensorAge > maximumSensorAge {
            interventions.append(.staleSensor(age: input.timing.sensorAge, maximumAge: maximumSensorAge))
        }
        if let maximumClosedLoopLatency, input.timing.closedLoopLatency > maximumClosedLoopLatency {
            interventions.append(
                .deadlineMissed(
                    latency: input.timing.closedLoopLatency,
                    maximumLatency: maximumClosedLoopLatency
                )
            )
        }
        if !interventions.isEmpty {
            return SafetySupervisorDecision(
                requestedAction: input.requestedAction,
                commandedAction: failsafeAction,
                shouldHalt: true,
                termination: .interrupted(reason: "hardware-safety-supervisor"),
                interventions: interventions,
                safetyAssessment: nil
            )
        }
        let assessment = try envelope.assess(input.requestedAction, previousAction: input.previousAction)
        if assessment.didIntervene {
            interventions.append(.safetyEnvelope(assessment.interventions))
        }
        return SafetySupervisorDecision(
            requestedAction: input.requestedAction,
            commandedAction: assessment.safeAction,
            shouldHalt: false,
            termination: .continuing,
            interventions: interventions,
            safetyAssessment: assessment
        )
    }

    private static func validateNonNegative(_ value: Double, name: String) throws {
        guard value >= 0 else {
            throw RLSwiftError.invalidDuration(name: name, value: value)
        }
    }
}
