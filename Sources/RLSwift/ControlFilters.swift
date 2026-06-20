/// A stateful low-pass filter for robot command vectors.
public struct ActionSmoother: Sendable, Equatable {
    /// The update factor in the closed range `0...1`.
    public let alpha: Double

    private var previousCommands: [Double]?

    /// Creates a smoother.
    public init(alpha: Double) throws {
        guard (0...1).contains(alpha) else {
            throw RLSwiftError.invalidProbability(alpha)
        }
        self.alpha = alpha
        previousCommands = nil
    }

    /// The most recent smoothed command vector.
    public var currentCommands: [Double]? {
        previousCommands
    }

    /// Smooths an action against the previous output.
    public mutating func smooth(_ action: RobotAction) throws -> RobotAction {
        guard let previousCommands else {
            self.previousCommands = action.commands
            return action
        }
        guard previousCommands.count == action.commands.count else {
            throw RLSwiftError.dimensionMismatch(expected: previousCommands.count, actual: action.commands.count)
        }
        var scratch = VectorScratch(count: action.commands.count)
        for index in 0..<action.commands.count {
            scratch.set(previousCommands[index] + alpha * (action.commands[index] - previousCommands[index]), at: index)
        }
        let commands = scratch.finish()
        self.previousCommands = commands
        return RobotAction(commands: commands, mode: action.mode)
    }

    /// Clears the smoothing history.
    public mutating func reset() {
        previousCommands = nil
    }
}
