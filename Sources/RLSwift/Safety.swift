/// A single safety-layer modification made to a robot command.
public enum SafetyIntervention: Sendable, Equatable, Codable {
    /// A command was clipped to the absolute command bounds.
    case commandClipped(index: Int, requested: Double, applied: Double)

    /// A command was clipped to the maximum allowed change from the previous command.
    case rateLimited(index: Int, requested: Double, applied: Double)
}

/// The result of passing an action through a safety envelope.
public struct RobotSafetyAssessment: Sendable, Equatable, Codable {
    /// The action requested by a policy or caller.
    public let requestedAction: RobotAction

    /// The action after all safety interventions.
    public let safeAction: RobotAction

    /// The interventions applied while producing `safeAction`.
    public let interventions: [SafetyIntervention]

    /// Creates a safety assessment.
    public init(
        requestedAction: RobotAction,
        safeAction: RobotAction,
        interventions: [SafetyIntervention]
    ) {
        self.requestedAction = requestedAction
        self.safeAction = safeAction
        self.interventions = interventions
    }

    /// Whether the safety layer changed the requested action.
    public var didIntervene: Bool {
        !interventions.isEmpty
    }
}

/// A safety envelope that clips robot commands and rate-limits changes between actions.
public struct RobotSafetyEnvelope: Sendable, Equatable, Codable {
    /// The absolute command bounds enforced for every action.
    public let commandSpace: ContinuousBoxSpace

    /// The optional maximum absolute per-dimension command delta.
    public let maximumDelta: [Double]?

    /// Creates a safety envelope.
    public init(commandSpace: ContinuousBoxSpace, maximumDelta: [Double]? = nil) throws {
        if let maximumDelta {
            guard maximumDelta.count == commandSpace.dimension else {
                throw RLSwiftError.dimensionMismatch(expected: commandSpace.dimension, actual: maximumDelta.count)
            }
            for index in maximumDelta.indices {
                guard maximumDelta[index] >= 0 else {
                    throw RLSwiftError.invalidBounds(index: index, lower: 0, upper: maximumDelta[index])
                }
            }
        }
        self.commandSpace = commandSpace
        self.maximumDelta = maximumDelta
    }

    /// Returns whether an action already satisfies the command envelope.
    public func allows(_ action: RobotAction, previousAction: RobotAction? = nil) -> Bool {
        guard commandSpace.contains(action.commands) else {
            return false
        }
        guard let previousAction, let maximumDelta else {
            return true
        }
        guard previousAction.commands.count == action.commands.count else {
            return false
        }
        for index in action.commands.indices where abs(action.commands[index] - previousAction.commands[index]) > maximumDelta[index] {
            return false
        }
        return true
    }

    /// Clips an action into the envelope and applies optional rate limiting.
    public func shield(_ action: RobotAction, previousAction: RobotAction? = nil) throws -> RobotAction {
        try assess(action, previousAction: previousAction).safeAction
    }

    /// Returns the safety assessment for an action without discarding intervention details.
    public func assess(_ action: RobotAction, previousAction: RobotAction? = nil) throws -> RobotSafetyAssessment {
        var interventions: [SafetyIntervention] = []
        let boundsClamped = try commandSpace.clamp(action.commands)
        var commands = boundsClamped
        for index in commands.indices where commands[index] != action.commands[index] {
            interventions.append(.commandClipped(index: index, requested: action.commands[index], applied: commands[index]))
        }

        if let previousAction, let maximumDelta {
            guard previousAction.commands.count == commands.count else {
                throw RLSwiftError.dimensionMismatch(expected: commands.count, actual: previousAction.commands.count)
            }
            for index in commands.indices {
                let lower = previousAction.commands[index] - maximumDelta[index]
                let upper = previousAction.commands[index] + maximumDelta[index]
                let requested = commands[index]
                commands[index] = min(max(commands[index], lower), upper)
                if commands[index] != requested {
                    interventions.append(.rateLimited(index: index, requested: requested, applied: commands[index]))
                }
            }
            let finalClamped = try commandSpace.clamp(commands)
            for index in finalClamped.indices where finalClamped[index] != commands[index] {
                interventions.append(.commandClipped(index: index, requested: commands[index], applied: finalClamped[index]))
            }
            commands = finalClamped
        }
        let safeAction = RobotAction(commands: commands, mode: action.mode)
        return RobotSafetyAssessment(requestedAction: action, safeAction: safeAction, interventions: interventions)
    }
}
