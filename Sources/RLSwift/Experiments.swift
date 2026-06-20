/// Configuration for a reproducible RLSwift experiment run.
public struct ExperimentConfiguration: Sendable, Equatable, Codable {
    /// Stable experiment identifier.
    public let experimentID: String

    /// Built-in environment used by the experiment.
    public let environmentID: BuiltInEnvironmentID

    /// Random seed used by the run.
    public let seed: UInt64

    /// Number of episodes to collect or evaluate.
    public let episodeCount: Int

    /// Maximum number of steps per episode.
    public let maxEpisodeSteps: Int

    /// PPO configuration used by trainable policies.
    public let ppoConfiguration: PPOConfiguration

    /// Optional vectorized rollout profile.
    public let vectorizationProfile: VectorizationProfile?

    /// Directory or URI where checkpoints should be written.
    public let checkpointDirectory: String

    /// Creates an experiment configuration.
    public init(
        experimentID: String,
        environmentID: BuiltInEnvironmentID,
        seed: UInt64,
        episodeCount: Int,
        maxEpisodeSteps: Int,
        ppoConfiguration: PPOConfiguration,
        vectorizationProfile: VectorizationProfile? = nil,
        checkpointDirectory: String
    ) throws {
        guard !experimentID.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "experimentID")
        }
        guard episodeCount > 0 else {
            throw RLSwiftError.invalidHorizon(episodeCount)
        }
        guard maxEpisodeSteps > 0 else {
            throw RLSwiftError.invalidHorizon(maxEpisodeSteps)
        }
        guard !checkpointDirectory.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "checkpointDirectory")
        }
        self.experimentID = experimentID
        self.environmentID = environmentID
        self.seed = seed
        self.episodeCount = episodeCount
        self.maxEpisodeSteps = maxEpisodeSteps
        self.ppoConfiguration = ppoConfiguration
        self.vectorizationProfile = vectorizationProfile
        self.checkpointDirectory = checkpointDirectory
    }
}

/// Checkpoint record that ties a policy artifact back to its experiment configuration.
public struct ExperimentCheckpointRecord: Sendable, Equatable, Codable {
    /// Configuration that produced the checkpoint.
    public let configuration: ExperimentConfiguration

    /// Saved policy manifest.
    public let manifest: PolicyCheckpointManifest

    /// Optional evaluation summary recorded near checkpoint time.
    public let evaluationSummary: EvaluationSummary?

    /// Creates an experiment checkpoint record.
    public init(
        configuration: ExperimentConfiguration,
        manifest: PolicyCheckpointManifest,
        evaluationSummary: EvaluationSummary? = nil
    ) {
        self.configuration = configuration
        self.manifest = manifest
        self.evaluationSummary = evaluationSummary
    }
}

/// Aggregate metrics from policy evaluation episodes.
public struct EvaluationSummary: Sendable, Equatable, Codable {
    /// Number of evaluated episodes.
    public let episodeCount: Int

    /// Mean undiscounted return.
    public let meanReturn: Double

    /// Fraction of episodes that reached a task success condition.
    public let successRate: Double

    /// Mean number of steps per episode.
    public let meanEpisodeLength: Double

    /// Creates an evaluation summary.
    public init(
        episodeCount: Int,
        meanReturn: Double,
        successRate: Double,
        meanEpisodeLength: Double
    ) throws {
        guard episodeCount > 0 else {
            throw RLSwiftError.invalidSampleCount(episodeCount)
        }
        guard (0...1).contains(successRate) else {
            throw RLSwiftError.invalidProbability(successRate)
        }
        guard meanEpisodeLength >= 0 else {
            throw RLSwiftError.invalidDuration(name: "meanEpisodeLength", value: meanEpisodeLength)
        }
        self.episodeCount = episodeCount
        self.meanReturn = meanReturn
        self.successRate = successRate
        self.meanEpisodeLength = meanEpisodeLength
    }
}

/// Built-in evaluation helpers used by the CLI and smoke tests.
public enum ExperimentEvaluator {
    /// Evaluates the deterministic right-moving policy in ``LineWorldEnvironment``.
    public static func evaluateLineWorldRightPolicy(
        episodes: Int,
        length: Int = 5,
        maxSteps: Int = 16
    ) throws -> EvaluationSummary {
        guard episodes > 0 else {
            throw RLSwiftError.invalidSampleCount(episodes)
        }
        var totalReturn = 0.0
        var totalSteps = 0
        var successes = 0
        for _ in 0..<episodes {
            var environment = try LineWorldEnvironment(length: length, maxSteps: maxSteps)
            _ = environment.reset()
            var episodeReturn = 0.0
            for stepIndex in 1...maxSteps {
                let result = try environment.step(.right)
                episodeReturn += result.reward
                if result.isTerminal {
                    totalSteps += stepIndex
                    if result.termination.reason == "goal" {
                        successes += 1
                    }
                    break
                }
            }
            totalReturn += episodeReturn
        }
        return try EvaluationSummary(
            episodeCount: episodes,
            meanReturn: totalReturn / Double(episodes),
            successRate: Double(successes) / Double(episodes),
            meanEpisodeLength: Double(totalSteps) / Double(episodes)
        )
    }
}
