/// Built-in benchmark environment identifiers.
public enum BuiltInEnvironmentID: String, CaseIterable, Sendable, Equatable, Codable {
    /// A small discrete navigation environment.
    case lineWorld

    /// A one-state binary bandit.
    case binaryBandit

    /// A two-agent matrix game.
    case matrixGame
}

/// Metadata for a built-in benchmark environment.
public struct EnvironmentCatalogEntry: Sendable, Equatable, Codable {
    /// Stable environment identifier.
    public let id: BuiltInEnvironmentID

    /// Human-readable display name.
    public let displayName: String

    /// Observation-space summary.
    public let observationSpace: String

    /// Action-space summary.
    public let actionSpace: String

    /// Default maximum episode length.
    public let defaultMaxSteps: Int

    /// Whether the environment exposes multiple agents.
    public let supportsMultiAgent: Bool

    /// Searchable environment tags.
    public let tags: [String]

    /// Creates a catalog entry.
    public init(
        id: BuiltInEnvironmentID,
        displayName: String,
        observationSpace: String,
        actionSpace: String,
        defaultMaxSteps: Int,
        supportsMultiAgent: Bool,
        tags: [String]
    ) throws {
        guard !displayName.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "environment.displayName")
        }
        guard !observationSpace.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "environment.observationSpace")
        }
        guard !actionSpace.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "environment.actionSpace")
        }
        guard defaultMaxSteps > 0 else {
            throw RLSwiftError.invalidHorizon(defaultMaxSteps)
        }
        self.id = id
        self.displayName = displayName
        self.observationSpace = observationSpace
        self.actionSpace = actionSpace
        self.defaultMaxSteps = defaultMaxSteps
        self.supportsMultiAgent = supportsMultiAgent
        self.tags = tags
    }
}

/// Catalog for the built-in RLSwift benchmark environments.
public enum BuiltInEnvironmentCatalog {
    /// Returns all built-in environment entries.
    public static func allEntries() throws -> [EnvironmentCatalogEntry] {
        [
            try EnvironmentCatalogEntry(
                id: .lineWorld,
                displayName: "LineWorld",
                observationSpace: "position, goal, step",
                actionSpace: "left | right",
                defaultMaxSteps: 32,
                supportsMultiAgent: false,
                tags: ["discrete", "navigation", "q-learning", "ppo-smoke"]
            ),
            try EnvironmentCatalogEntry(
                id: .binaryBandit,
                displayName: "BinaryBandit",
                observationSpace: "pull count",
                actionSpace: "optionA | optionB",
                defaultMaxSteps: 8,
                supportsMultiAgent: false,
                tags: ["bandit", "discrete", "exploration"]
            ),
            try EnvironmentCatalogEntry(
                id: .matrixGame,
                displayName: "MatrixGame",
                observationSpace: "round index per agent",
                actionSpace: "cooperate | defect",
                defaultMaxSteps: 4,
                supportsMultiAgent: true,
                tags: ["multi-agent", "game", "self-play"]
            ),
        ]
    }

    /// Returns a catalog entry by identifier.
    public static func entry(for id: BuiltInEnvironmentID) throws -> EnvironmentCatalogEntry {
        switch id {
        case .lineWorld:
            return try allEntries()[0]
        case .binaryBandit:
            return try allEntries()[1]
        case .matrixGame:
            return try allEntries()[2]
        }
    }
}

/// Discrete actions for ``LineWorldEnvironment``.
public enum LineWorldAction: String, CaseIterable, Sendable, Equatable, Codable, Hashable {
    /// Move one cell left.
    case left

    /// Move one cell right.
    case right
}

/// Observation returned by ``LineWorldEnvironment``.
public struct LineWorldObservation: Sendable, Equatable, Codable, Hashable {
    /// Current position.
    public let position: Int

    /// Goal position.
    public let goal: Int

    /// Current episode step.
    public let stepIndex: Int

    /// Creates a line-world observation.
    public init(position: Int, goal: Int, stepIndex: Int) {
        self.position = position
        self.goal = goal
        self.stepIndex = stepIndex
    }
}

/// A deterministic one-dimensional navigation benchmark.
public struct LineWorldEnvironment: Environment {
    private let length: Int
    private let maxSteps: Int
    private var position: Int
    private var stepIndex: Int

    /// Creates a line-world environment.
    public init(length: Int = 5, maxSteps: Int = 32) throws {
        guard length >= 2 else {
            throw RLSwiftError.invalidCapacity(length)
        }
        guard maxSteps > 0 else {
            throw RLSwiftError.invalidHorizon(maxSteps)
        }
        self.length = length
        self.maxSteps = maxSteps
        position = 0
        stepIndex = 0
    }

    /// Current observation without mutating the environment.
    public var observation: LineWorldObservation {
        LineWorldObservation(position: position, goal: length - 1, stepIndex: stepIndex)
    }

    /// Resets the agent to the start of the line.
    public mutating func reset() -> LineWorldObservation {
        position = 0
        stepIndex = 0
        return observation
    }

    /// Applies a left/right action.
    public mutating func step(_ action: LineWorldAction) throws -> StepResult<LineWorldObservation> {
        stepIndex += 1
        switch action {
        case .left:
            position = max(0, position - 1)
        case .right:
            position = min(length - 1, position + 1)
        }

        if position == length - 1 {
            return StepResult(
                observation: observation,
                reward: 1,
                isTerminal: true,
                termination: .terminated(reason: "goal")
            )
        }
        if stepIndex >= maxSteps {
            return StepResult(
                observation: observation,
                reward: -0.01,
                isTerminal: true,
                termination: .truncated(reason: "max_steps")
            )
        }
        return StepResult(observation: observation, reward: -0.01, isTerminal: false)
    }
}

/// Discrete actions for ``BinaryBanditEnvironment``.
public enum BinaryBanditAction: String, CaseIterable, Sendable, Equatable, Codable, Hashable {
    /// First arm.
    case optionA

    /// Second arm.
    case optionB
}

/// Observation returned by ``BinaryBanditEnvironment``.
public struct BinaryBanditObservation: Sendable, Equatable, Codable, Hashable {
    /// Number of pulls already made.
    public let pullCount: Int

    /// Creates a binary-bandit observation.
    public init(pullCount: Int) {
        self.pullCount = pullCount
    }
}

/// A deterministic binary bandit benchmark.
public struct BinaryBanditEnvironment: Environment {
    private let rewardingAction: BinaryBanditAction
    private let maxPulls: Int
    private var pullCount: Int

    /// Creates a binary bandit.
    public init(rewardingAction: BinaryBanditAction = .optionB, maxPulls: Int = 8) throws {
        guard maxPulls > 0 else {
            throw RLSwiftError.invalidHorizon(maxPulls)
        }
        self.rewardingAction = rewardingAction
        self.maxPulls = maxPulls
        pullCount = 0
    }

    /// Current observation without mutating the environment.
    public var observation: BinaryBanditObservation {
        BinaryBanditObservation(pullCount: pullCount)
    }

    /// Resets the pull counter.
    public mutating func reset() -> BinaryBanditObservation {
        pullCount = 0
        return observation
    }

    /// Pulls one bandit arm.
    public mutating func step(_ action: BinaryBanditAction) throws -> StepResult<BinaryBanditObservation> {
        pullCount += 1
        let reward = action == rewardingAction ? 1.0 : 0.0
        let termination: StepTermination = pullCount >= maxPulls ? .truncated(reason: "max_pulls") : .continuing
        return StepResult(observation: observation, reward: reward, isTerminal: termination.endsEpisode, termination: termination)
    }
}
