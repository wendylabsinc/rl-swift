public import Foundation

/// Provenance metadata for a collected offline RL dataset.
public struct DatasetProvenance: Sendable, Equatable, Codable {
    /// Stable dataset identifier.
    public let datasetID: String

    /// Source system that produced the data, such as `sim`, `ros2`, or `wendyos`.
    public let sourceSystem: String

    /// Robot, simulator, or vehicle identifier.
    public let robotID: String

    /// Environment, task, or route name.
    public let environment: String

    /// Timestamp when collection started.
    public let collectionStartedAt: Date

    /// Timestamp when collection ended, if known.
    public let collectionEndedAt: Date?

    /// Additional metadata preserved with the dataset.
    public let metadata: [String: String]

    /// Creates dataset provenance.
    public init(
        datasetID: String,
        sourceSystem: String,
        robotID: String,
        environment: String,
        collectionStartedAt: Date = Date(timeIntervalSince1970: 0),
        collectionEndedAt: Date? = nil,
        metadata: [String: String] = [:]
    ) throws {
        guard !datasetID.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "datasetID")
        }
        guard !sourceSystem.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "sourceSystem")
        }
        guard !robotID.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "robotID")
        }
        guard !environment.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "environment")
        }
        self.datasetID = datasetID
        self.sourceSystem = sourceSystem
        self.robotID = robotID
        self.environment = environment
        self.collectionStartedAt = collectionStartedAt
        self.collectionEndedAt = collectionEndedAt
        self.metadata = metadata
    }
}

/// One replayable transition plus operational metadata captured at collection time.
public struct LoggedTransition<Observation: Sendable & Codable, Action: Sendable & Codable>: Sendable, Codable {
    /// Transition observed from the environment.
    public let transition: Transition<Observation, Action>

    /// Wall-clock timestamp when this transition was recorded.
    public let recordedAt: Date

    /// Control timing captured for the step.
    public let timing: ControlTiming?

    /// Constraint costs captured for constrained or safety-aware RL.
    public let constraints: ConstraintReport?

    /// Hardware supervisor decision, when the action came from a robot control loop.
    public let safetyDecision: SafetySupervisorDecision?

    /// Additional durable step metadata.
    public let metadata: [String: String]

    /// Creates a logged transition.
    public init(
        transition: Transition<Observation, Action>,
        recordedAt: Date = Date(timeIntervalSince1970: 0),
        timing: ControlTiming? = nil,
        constraints: ConstraintReport? = nil,
        safetyDecision: SafetySupervisorDecision? = nil,
        metadata: [String: String] = [:]
    ) {
        self.transition = transition
        self.recordedAt = recordedAt
        self.timing = timing
        self.constraints = constraints
        self.safetyDecision = safetyDecision
        self.metadata = metadata
    }
}

extension LoggedTransition: Equatable where Observation: Equatable, Action: Equatable {}

/// A manifest summarizing an offline dataset without loading all transition payloads.
public struct DatasetManifest: Sendable, Equatable, Codable {
    /// Manifest schema version.
    public let schemaVersion: String

    /// Dataset provenance.
    public let provenance: DatasetProvenance

    /// Optional model contract used to encode observations and actions.
    public let modelContract: ModelIOContract?

    /// Number of transitions in the dataset.
    public let transitionCount: Int

    /// Counts grouped by termination type.
    public let terminationCounts: [String: Int]

    /// Number of recorded hardware or envelope interventions.
    public let safetyInterventionCount: Int

    /// Total constraint cost across logged transitions.
    public let totalConstraintCost: Double

    /// Timestamp when the manifest was created.
    public let createdAt: Date

    /// Creates a dataset manifest.
    public init(
        schemaVersion: String = "1.0",
        provenance: DatasetProvenance,
        modelContract: ModelIOContract? = nil,
        transitionCount: Int,
        terminationCounts: [String: Int],
        safetyInterventionCount: Int,
        totalConstraintCost: Double,
        createdAt: Date = Date(timeIntervalSince1970: 0)
    ) throws {
        guard !schemaVersion.isEmpty else {
            throw RLSwiftError.invalidVersion(schemaVersion)
        }
        guard transitionCount >= 0 else {
            throw RLSwiftError.invalidSampleCount(transitionCount)
        }
        guard safetyInterventionCount >= 0 else {
            throw RLSwiftError.invalidSampleCount(safetyInterventionCount)
        }
        self.schemaVersion = schemaVersion
        self.provenance = provenance
        self.modelContract = modelContract
        self.transitionCount = transitionCount
        self.terminationCounts = terminationCounts
        self.safetyInterventionCount = safetyInterventionCount
        self.totalConstraintCost = totalConstraintCost
        self.createdAt = createdAt
    }

    /// Builds a manifest from logged transitions.
    public static func build<Observation: Sendable & Codable, Action: Sendable & Codable>(
        provenance: DatasetProvenance,
        modelContract: ModelIOContract? = nil,
        transitions: [LoggedTransition<Observation, Action>],
        createdAt: Date = Date(timeIntervalSince1970: 0)
    ) throws -> DatasetManifest {
        var terminations: [String: Int] = [:]
        var safetyInterventions = 0
        var constraintCost = 0.0
        for logged in transitions {
            terminations[logged.transition.termination.manifestKey, default: 0] += 1
            constraintCost += logged.constraints?.totalCost ?? 0
            if let safetyDecision = logged.safetyDecision {
                safetyInterventions += safetyDecision.interventions.count
            }
        }
        return try DatasetManifest(
            provenance: provenance,
            modelContract: modelContract,
            transitionCount: transitions.count,
            terminationCounts: terminations,
            safetyInterventionCount: safetyInterventions,
            totalConstraintCost: constraintCost,
            createdAt: createdAt
        )
    }
}

/// An offline dataset package that keeps transitions and manifest metadata together.
public struct OfflineDataset<Observation: Sendable & Codable, Action: Sendable & Codable>: Sendable, Codable {
    /// Dataset manifest.
    public let manifest: DatasetManifest

    /// Logged transitions in chronological storage order.
    public let transitions: [LoggedTransition<Observation, Action>]

    /// Creates an offline dataset and verifies manifest counts.
    public init(manifest: DatasetManifest, transitions: [LoggedTransition<Observation, Action>]) throws {
        guard manifest.transitionCount == transitions.count else {
            throw RLSwiftError.dimensionMismatch(expected: manifest.transitionCount, actual: transitions.count)
        }
        self.manifest = manifest
        self.transitions = transitions
    }

    /// Creates an offline dataset and derives its manifest.
    public init(
        provenance: DatasetProvenance,
        modelContract: ModelIOContract? = nil,
        transitions: [LoggedTransition<Observation, Action>],
        createdAt: Date = Date(timeIntervalSince1970: 0)
    ) throws {
        let manifest = try DatasetManifest.build(
            provenance: provenance,
            modelContract: modelContract,
            transitions: transitions,
            createdAt: createdAt
        )
        try self.init(manifest: manifest, transitions: transitions)
    }
}

extension OfflineDataset: Equatable where Observation: Equatable, Action: Equatable {}

private extension StepTermination {
    var manifestKey: String {
        switch self {
        case .continuing:
            return "continuing"
        case .terminated:
            return "terminated"
        case .truncated:
            return "truncated"
        case .interrupted:
            return "interrupted"
        }
    }
}
