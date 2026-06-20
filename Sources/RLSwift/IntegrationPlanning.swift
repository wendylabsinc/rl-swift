public import Foundation

/// Common robot integration surfaces that can connect to RLSwift policies.
public enum RobotIntegrationKind: String, Sendable, Equatable, Codable {
    /// ROS 2 graph integration.
    case ros2

    /// Simulator integration.
    case simulator

    /// WendyOS app or device integration.
    case wendyOS
}

/// A dependency-light adapter configuration for external robot runtimes.
public struct RobotIntegrationAdapterConfiguration: Sendable, Equatable, Codable {
    /// Runtime family.
    public let kind: RobotIntegrationKind

    /// Endpoint, namespace, or device address.
    public let endpoint: String

    /// Observation topic, stream, or channel name.
    public let observationChannel: String

    /// Action topic, stream, or channel name.
    public let actionChannel: String

    /// Additional runtime metadata.
    public let metadata: [String: String]

    /// Creates an adapter configuration.
    public init(
        kind: RobotIntegrationKind,
        endpoint: String,
        observationChannel: String,
        actionChannel: String,
        metadata: [String: String] = [:]
    ) throws {
        guard !endpoint.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "adapter.endpoint")
        }
        guard !observationChannel.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "adapter.observationChannel")
        }
        guard !actionChannel.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "adapter.actionChannel")
        }
        self.kind = kind
        self.endpoint = endpoint
        self.observationChannel = observationChannel
        self.actionChannel = actionChannel
        self.metadata = metadata
    }

    /// Creates a ROS 2 adapter configuration.
    public static func ros2(
        namespace: String,
        observationTopic: String,
        actionTopic: String,
        metadata: [String: String] = [:]
    ) throws -> RobotIntegrationAdapterConfiguration {
        try RobotIntegrationAdapterConfiguration(
            kind: .ros2,
            endpoint: namespace,
            observationChannel: observationTopic,
            actionChannel: actionTopic,
            metadata: metadata
        )
    }

    /// Creates a simulator adapter configuration.
    public static func simulator(
        endpoint: String,
        observationStream: String,
        actionStream: String,
        metadata: [String: String] = [:]
    ) throws -> RobotIntegrationAdapterConfiguration {
        try RobotIntegrationAdapterConfiguration(
            kind: .simulator,
            endpoint: endpoint,
            observationChannel: observationStream,
            actionChannel: actionStream,
            metadata: metadata
        )
    }

    /// Creates a MuJoCo simulator adapter configuration.
    public static func mujoco(
        modelPath: String,
        observationStream: String = "qpos,qvel,sensors",
        actionStream: String = "ctrl",
        metadata: [String: String] = [:]
    ) throws -> RobotIntegrationAdapterConfiguration {
        guard !modelPath.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "mujoco.modelPath")
        }
        var enrichedMetadata = metadata
        enrichedMetadata["simulator"] = "mujoco"
        enrichedMetadata["modelPath"] = modelPath
        return try RobotIntegrationAdapterConfiguration(
            kind: .simulator,
            endpoint: modelPath,
            observationChannel: observationStream,
            actionChannel: actionStream,
            metadata: enrichedMetadata
        )
    }

    /// Creates an Isaac Sim bridge adapter configuration.
    public static func isaacSim(
        endpoint: String,
        robotPath: String = "/World/Robot",
        observationStream: String = "observation",
        actionStream: String = "action",
        metadata: [String: String] = [:]
    ) throws -> RobotIntegrationAdapterConfiguration {
        guard !robotPath.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "isaacSim.robotPath")
        }
        var enrichedMetadata = metadata
        enrichedMetadata["simulator"] = "isaac-sim"
        enrichedMetadata["robotPath"] = robotPath
        return try RobotIntegrationAdapterConfiguration(
            kind: .simulator,
            endpoint: endpoint,
            observationChannel: observationStream,
            actionChannel: actionStream,
            metadata: enrichedMetadata
        )
    }

    /// Creates a WendyOS adapter configuration.
    public static func wendyOS(
        device: String,
        observationStream: String,
        actionStream: String,
        metadata: [String: String] = [:]
    ) throws -> RobotIntegrationAdapterConfiguration {
        try RobotIntegrationAdapterConfiguration(
            kind: .wendyOS,
            endpoint: device,
            observationChannel: observationStream,
            actionChannel: actionStream,
            metadata: metadata
        )
    }
}

/// Batched result returned by a vectorized environment runner.
public struct VectorizedStepResult<Observation: Sendable>: Sendable {
    /// Per-environment observations.
    public let observations: [Observation]

    /// Per-environment rewards.
    public let rewards: [Double]

    /// Per-environment terminal flags.
    public let terminalFlags: [Bool]

    /// Per-environment termination states.
    public let terminations: [StepTermination]

    /// Creates a vectorized step result.
    public init(observations: [Observation], rewards: [Double], terminalFlags: [Bool], terminations: [StepTermination]) {
        self.observations = observations
        self.rewards = rewards
        self.terminalFlags = terminalFlags
        self.terminations = terminations
    }
}

extension VectorizedStepResult: Equatable where Observation: Equatable {}

/// A deterministic sequential vectorized environment runner.
public struct VectorizedEnvironmentRunner<WrappedEnvironment: Environment>: Sendable {
    private var environments: [WrappedEnvironment]

    /// Creates a vectorized runner from environment instances.
    public init(_ environments: [WrappedEnvironment]) throws {
        guard !environments.isEmpty else {
            throw RLSwiftError.invalidCapacity(0)
        }
        self.environments = environments
    }

    /// Number of environments managed by the runner.
    public var count: Int {
        environments.count
    }

    /// Resets every environment and returns initial observations.
    public mutating func resetAll() -> [WrappedEnvironment.Observation] {
        environments.indices.map { environments[$0].reset() }
    }

    /// Steps every environment with one action per environment.
    public mutating func step(_ actions: [WrappedEnvironment.Action]) throws -> VectorizedStepResult<WrappedEnvironment.Observation> {
        guard actions.count == environments.count else {
            throw RLSwiftError.dimensionMismatch(expected: environments.count, actual: actions.count)
        }
        var observations: [WrappedEnvironment.Observation] = []
        var rewards: [Double] = []
        var terminalFlags: [Bool] = []
        var terminations: [StepTermination] = []
        for index in environments.indices {
            let result = try environments[index].step(actions[index])
            observations.append(result.observation)
            rewards.append(result.reward)
            terminalFlags.append(result.isTerminal)
            terminations.append(result.termination)
        }
        return VectorizedStepResult(
            observations: observations,
            rewards: rewards,
            terminalFlags: terminalFlags,
            terminations: terminations
        )
    }
}

/// Assignment metadata for distributed rollout collection.
public struct RolloutShardAssignment: Sendable, Equatable, Codable {
    /// Zero-based shard index.
    public let shardIndex: Int

    /// Total number of shards.
    public let shardCount: Int

    /// Seed mixed into deterministic environment assignment.
    public let seed: UInt64

    /// Creates a rollout shard assignment.
    public init(shardIndex: Int, shardCount: Int, seed: UInt64 = 0) throws {
        guard shardCount > 0 else {
            throw RLSwiftError.invalidCapacity(shardCount)
        }
        guard shardIndex >= 0, shardIndex < shardCount else {
            throw RLSwiftError.dimensionMismatch(expected: shardCount, actual: shardIndex)
        }
        self.shardIndex = shardIndex
        self.shardCount = shardCount
        self.seed = seed
    }

    /// Returns whether a stable environment ID belongs to this shard.
    public func owns(environmentID: String) -> Bool {
        Int(stableShardHash(environmentID, seed: seed) % UInt64(shardCount)) == shardIndex
    }
}

/// A model export request for ONNX-based deployment.
public struct ONNXExportDescriptor: Sendable, Equatable, Codable {
    /// Exported model name.
    public let modelName: String

    /// ONNX opset version.
    public let opsetVersion: Int

    /// Input tensor names.
    public let inputNames: [String]

    /// Output tensor names.
    public let outputNames: [String]

    /// Dynamic axes by tensor name.
    public let dynamicAxes: [String: [Int]]

    /// Additional export metadata.
    public let metadata: [String: String]

    /// Creates an ONNX export descriptor.
    public init(
        modelName: String,
        opsetVersion: Int,
        inputNames: [String],
        outputNames: [String],
        dynamicAxes: [String: [Int]] = [:],
        metadata: [String: String] = [:]
    ) throws {
        guard !modelName.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "onnx.modelName")
        }
        guard opsetVersion > 0 else {
            throw RLSwiftError.invalidVersion(String(opsetVersion))
        }
        try Self.validateNames(inputNames, field: "onnx.inputNames")
        try Self.validateNames(outputNames, field: "onnx.outputNames")
        self.modelName = modelName
        self.opsetVersion = opsetVersion
        self.inputNames = inputNames
        self.outputNames = outputNames
        self.dynamicAxes = dynamicAxes
        self.metadata = metadata
    }

    private static func validateNames(_ names: [String], field: String) throws {
        guard !names.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: field)
        }
        for name in names where name.isEmpty {
            throw RLSwiftError.emptyIdentifier(name: field)
        }
    }
}

/// Deterministic cache key metadata for a compiled TensorRT engine.
public struct TensorRTEngineCacheKey: Sendable, Equatable, Codable {
    /// Policy identifier.
    public let policyID: String

    /// Policy version.
    public let policyVersion: String

    /// TensorRT version used to build the engine.
    public let tensorRTVersion: String

    /// Precision or typing strategy, such as `typed-fp16` or `fp32`.
    public let precision: String

    /// Input tensor shape used for the engine.
    public let inputShape: [Int]

    /// Creates a TensorRT engine cache key.
    public init(policyID: String, policyVersion: String, tensorRTVersion: String, precision: String, inputShape: [Int]) throws {
        guard !policyID.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "engine.policyID")
        }
        guard !policyVersion.isEmpty else {
            throw RLSwiftError.invalidVersion(policyVersion)
        }
        guard !tensorRTVersion.isEmpty else {
            throw RLSwiftError.invalidVersion(tensorRTVersion)
        }
        guard !precision.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "engine.precision")
        }
        guard !inputShape.isEmpty else {
            throw RLSwiftError.dimensionMismatch(expected: 1, actual: 0)
        }
        self.policyID = policyID
        self.policyVersion = policyVersion
        self.tensorRTVersion = tensorRTVersion
        self.precision = precision
        self.inputShape = inputShape
    }

    /// Stable string suitable for filesystem cache paths.
    public var stableIdentifier: String {
        let shape = inputShape.map(String.init).joined(separator: "x")
        return [policyID, policyVersion, tensorRTVersion, precision, shape].joined(separator: "-")
    }
}

/// Manifest entry for a cached TensorRT engine file.
public struct TensorRTEngineCacheManifest: Sendable, Equatable, Codable {
    /// Cache key for the engine.
    public let key: TensorRTEngineCacheKey

    /// Path to the serialized engine.
    public let enginePath: String

    /// Source ONNX path used to build the engine.
    public let sourceONNXPath: String

    /// Build timestamp.
    public let builtAt: Date

    /// Creates a TensorRT engine cache manifest.
    public init(
        key: TensorRTEngineCacheKey,
        enginePath: String,
        sourceONNXPath: String,
        builtAt: Date = Date(timeIntervalSince1970: 0)
    ) throws {
        guard !enginePath.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "enginePath")
        }
        guard !sourceONNXPath.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "sourceONNXPath")
        }
        self.key = key
        self.enginePath = enginePath
        self.sourceONNXPath = sourceONNXPath
        self.builtAt = builtAt
    }
}

/// A bounded parameter used for domain randomization.
public struct DomainRandomizationParameter: Sendable, Equatable, Codable {
    /// Parameter name.
    public let name: String

    /// Inclusive lower bound.
    public let lowerBound: Double

    /// Inclusive upper bound.
    public let upperBound: Double

    /// Physical unit.
    public let unit: String

    /// Creates a domain-randomization parameter.
    public init(name: String, lowerBound: Double, upperBound: Double, unit: String = "") throws {
        guard !name.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "randomization.name")
        }
        guard lowerBound <= upperBound else {
            throw RLSwiftError.invalidBounds(index: 0, lower: lowerBound, upper: upperBound)
        }
        self.name = name
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.unit = unit
    }

    /// Samples this parameter with a deterministic random generator.
    public func sample(using generator: inout SeededGenerator) -> Double {
        let fraction = Double(generator.next()) / Double(UInt64.max)
        return lowerBound + fraction * (upperBound - lowerBound)
    }
}

/// A domain-randomization profile for simulator or policy evaluation.
public struct DomainRandomizationProfile: Sendable, Equatable, Codable {
    /// Parameters sampled together.
    public let parameters: [DomainRandomizationParameter]

    /// Creates a randomization profile.
    public init(parameters: [DomainRandomizationParameter]) throws {
        try Self.validateUnique(parameters.map(\.name))
        self.parameters = parameters
    }

    /// Samples all parameters deterministically from a seed.
    public func sample(seed: UInt64) -> [String: Double] {
        var generator = SeededGenerator(seed: seed)
        var values: [String: Double] = [:]
        for parameter in parameters {
            values[parameter.name] = parameter.sample(using: &generator)
        }
        return values
    }

    private static func validateUnique(_ names: [String]) throws {
        var seen: Set<String> = []
        for name in names {
            guard seen.insert(name).inserted else {
                throw RLSwiftError.duplicateIdentifier(name)
            }
        }
    }
}

/// A stage in a curriculum-learning schedule.
public struct CurriculumStage: Sendable, Equatable, Codable {
    /// Stage name.
    public let name: String

    /// Monotonic difficulty value.
    public let difficulty: Double

    /// Success rate required before advancing from this stage.
    public let successThreshold: Double

    /// Minimum completed episodes required before advancing.
    public let minimumEpisodes: Int

    /// Optional domain-randomization profile for the stage.
    public let randomization: DomainRandomizationProfile?

    /// Creates a curriculum stage.
    public init(
        name: String,
        difficulty: Double,
        successThreshold: Double,
        minimumEpisodes: Int,
        randomization: DomainRandomizationProfile? = nil
    ) throws {
        guard !name.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "curriculum.name")
        }
        guard (0...1).contains(successThreshold) else {
            throw RLSwiftError.invalidProbability(successThreshold)
        }
        guard minimumEpisodes >= 0 else {
            throw RLSwiftError.invalidSampleCount(minimumEpisodes)
        }
        self.name = name
        self.difficulty = difficulty
        self.successThreshold = successThreshold
        self.minimumEpisodes = minimumEpisodes
        self.randomization = randomization
    }
}

/// A deterministic curriculum schedule.
public struct CurriculumSchedule: Sendable, Equatable, Codable {
    /// Ordered stages.
    public let stages: [CurriculumStage]

    /// Creates a curriculum schedule.
    public init(stages: [CurriculumStage]) throws {
        guard !stages.isEmpty else {
            throw RLSwiftError.invalidCapacity(0)
        }
        self.stages = stages
    }

    /// Returns the active stage for an index.
    public func stage(at index: Int) throws -> CurriculumStage {
        guard index >= 0, index < stages.count else {
            throw RLSwiftError.dimensionMismatch(expected: stages.count, actual: index)
        }
        return stages[index]
    }

    /// Returns the next stage index after applying advancement criteria.
    public func nextStageIndex(currentIndex: Int, completedEpisodes: Int, successRate: Double) throws -> Int {
        let current = try stage(at: currentIndex)
        guard completedEpisodes >= current.minimumEpisodes, successRate >= current.successThreshold else {
            return currentIndex
        }
        return min(currentIndex + 1, stages.count - 1)
    }
}

/// A policy evaluation record suitable for dashboards and release gates.
public struct EvaluationRecord: Sendable, Equatable, Codable {
    /// Policy version evaluated.
    public let policyVersion: String

    /// Evaluation environment name.
    public let environment: String

    /// Number of episodes evaluated.
    public let episodeCount: Int

    /// Mean return across episodes.
    public let meanReturn: Double

    /// Success rate in the closed `0...1` range.
    public let successRate: Double

    /// Mean constraint cost across episodes.
    public let meanConstraintCost: Double

    /// Creates an evaluation record.
    public init(
        policyVersion: String,
        environment: String,
        episodeCount: Int,
        meanReturn: Double,
        successRate: Double,
        meanConstraintCost: Double
    ) throws {
        guard !policyVersion.isEmpty else {
            throw RLSwiftError.invalidVersion(policyVersion)
        }
        guard !environment.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "evaluation.environment")
        }
        guard episodeCount >= 0 else {
            throw RLSwiftError.invalidSampleCount(episodeCount)
        }
        guard (0...1).contains(successRate) else {
            throw RLSwiftError.invalidProbability(successRate)
        }
        self.policyVersion = policyVersion
        self.environment = environment
        self.episodeCount = episodeCount
        self.meanReturn = meanReturn
        self.successRate = successRate
        self.meanConstraintCost = meanConstraintCost
    }
}

/// Dashboard summary built from evaluation records.
public struct EvaluationDashboardSummary: Sendable, Equatable, Codable {
    /// Records included in the dashboard summary.
    public let records: [EvaluationRecord]

    /// Creates an evaluation dashboard summary.
    public init(records: [EvaluationRecord]) {
        self.records = records
    }

    /// Best record by success rate, then mean return.
    public var bestRecord: EvaluationRecord? {
        records.max { lhs, rhs in
            if lhs.successRate == rhs.successRate {
                return lhs.meanReturn < rhs.meanReturn
            }
            return lhs.successRate < rhs.successRate
        }
    }
}

/// Observation drift diagnostics for visual debugging.
public struct ObservationDriftSnapshot: Sendable, Equatable, Codable {
    /// Feature names in model-input order.
    public let featureNames: [String]

    /// Raw observed values.
    public let observedValues: [Double]

    /// Normalized z-like values based on a normalization snapshot.
    public let normalizedValues: [Double]

    /// Creates an observation drift snapshot.
    public init(featureNames: [String], observedValues: [Double], normalization: NormalizationSnapshot) throws {
        guard featureNames.count == observedValues.count else {
            throw RLSwiftError.dimensionMismatch(expected: featureNames.count, actual: observedValues.count)
        }
        self.featureNames = featureNames
        self.observedValues = observedValues
        self.normalizedValues = try normalization.normalize(observedValues)
    }

    /// Returns feature names whose absolute normalized value exceeds a threshold.
    public func driftedFeatures(threshold: Double) -> [String] {
        zip(featureNames, normalizedValues).compactMap { name, value in
            abs(value) > threshold ? name : nil
        }
    }
}

/// Action saturation diagnostics for visual debugging.
public struct ActionSaturationSnapshot: Sendable, Equatable, Codable {
    /// Action command values.
    public let commands: [Double]

    /// Action space used for saturation checks.
    public let actionSpace: ContinuousBoxSpace

    /// Creates an action saturation snapshot.
    public init(action: RobotAction, actionSpace: ContinuousBoxSpace) throws {
        guard action.commands.count == actionSpace.dimension else {
            throw RLSwiftError.dimensionMismatch(expected: actionSpace.dimension, actual: action.commands.count)
        }
        self.commands = action.commands
        self.actionSpace = actionSpace
    }

    /// Indices at or outside action bounds.
    public var saturatedIndices: [Int] {
        commands.indices.filter { index in
            commands[index] <= actionSpace.lowerBounds[index] || commands[index] >= actionSpace.upperBounds[index]
        }
    }
}

/// Prioritized replay diagnostics for rare or safety-critical events.
public struct PrioritizedReplayDebugSnapshot: Sendable, Equatable, Codable {
    /// Replay item index.
    public let index: Int

    /// Replay priority.
    public let priority: Double

    /// Maximum priority in the buffer.
    public let maximumPriority: Double

    /// Optional event label such as `collision`, `timeout`, or `recovery`.
    public let eventLabel: String?

    /// Creates a prioritized replay debug snapshot.
    public init(index: Int, priority: Double, maximumPriority: Double, eventLabel: String? = nil) throws {
        guard index >= 0 else {
            throw RLSwiftError.dimensionMismatch(expected: 0, actual: index)
        }
        guard priority > 0 else {
            throw RLSwiftError.invalidPriority(priority)
        }
        guard maximumPriority > 0 else {
            throw RLSwiftError.invalidPriority(maximumPriority)
        }
        self.index = index
        self.priority = priority
        self.maximumPriority = maximumPriority
        self.eventLabel = eventLabel
    }

    /// Priority as a fraction of the maximum buffer priority.
    public var relativePriority: Double {
        priority / maximumPriority
    }
}

private func stableShardHash(_ value: String, seed: UInt64) -> UInt64 {
    let prime: UInt64 = 0x00000100000001B3
    var hash: UInt64 = 0xcbf29ce484222325 ^ seed
    for byte in value.utf8 {
        hash ^= UInt64(byte)
        hash &*= prime
    }
    return hash
}
