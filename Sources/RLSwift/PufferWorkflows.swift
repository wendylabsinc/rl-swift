import Foundation

/// One named field inside a flattened observation or action tensor.
public struct StructuredTensorField: Sendable, Equatable, Codable {
    /// Stable field name used by environments, models, and manifests.
    public let name: String

    /// Tensor shape before flattening.
    public let shape: [Int]

    /// Number of scalar values represented by this field.
    public let count: Int

    /// Creates a named tensor field.
    public init(name: String, shape: [Int]) throws {
        guard !name.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "tensor.field")
        }
        guard !shape.isEmpty else {
            throw RLSwiftError.invalidCapacity(0)
        }
        var count = 1
        for dimension in shape {
            guard dimension > 0 else {
                throw RLSwiftError.invalidCapacity(dimension)
            }
            count *= dimension
        }
        self.name = name
        self.shape = shape
        self.count = count
    }
}

/// A deterministic schema for flattening structured observations or actions.
public struct StructuredTensorSchema: Sendable, Equatable, Codable {
    /// Fields in flattened model-input order.
    public let fields: [StructuredTensorField]

    /// Total scalar count of the flattened tensor.
    public let flattenedCount: Int

    private let offsets: [String: Int]

    /// Creates a structured tensor schema.
    public init(fields: [StructuredTensorField]) throws {
        guard !fields.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        var seen: Set<String> = []
        var offsets: [String: Int] = [:]
        var cursor = 0
        for field in fields {
            guard seen.insert(field.name).inserted else {
                throw RLSwiftError.duplicateIdentifier(field.name)
            }
            offsets[field.name] = cursor
            cursor += field.count
        }
        self.fields = fields
        self.flattenedCount = cursor
        self.offsets = offsets
    }

    /// Returns the flattened range occupied by a named field.
    public func range(for fieldName: String) throws -> Range<Int> {
        guard let offset = offsets[fieldName],
              let field = fields.first(where: { $0.name == fieldName }) else {
            throw RLSwiftError.emptyIdentifier(name: fieldName)
        }
        return offset..<(offset + field.count)
    }

    /// Flattens structured scalar arrays into model-input order.
    public func flatten(_ values: [String: [Double]]) throws -> [Double] {
        var flattened: [Double] = []
        flattened.reserveCapacity(flattenedCount)
        for field in fields {
            guard let fieldValues = values[field.name] else {
                throw RLSwiftError.emptyIdentifier(name: field.name)
            }
            guard fieldValues.count == field.count else {
                throw RLSwiftError.dimensionMismatch(expected: field.count, actual: fieldValues.count)
            }
            flattened.append(contentsOf: fieldValues)
        }
        return flattened
    }

    /// Restores a flattened tensor into named scalar arrays.
    public func unflatten(_ flattened: [Double]) throws -> [String: [Double]] {
        guard flattened.count == flattenedCount else {
            throw RLSwiftError.dimensionMismatch(expected: flattenedCount, actual: flattened.count)
        }
        var values: [String: [Double]] = [:]
        for field in fields {
            let fieldRange = try range(for: field.name)
            values[field.name] = Array(flattened[fieldRange])
        }
        return values
    }
}

/// Structured observation schema used by Puffer-style flattened environments.
public typealias StructuredObservationSchema = StructuredTensorSchema

/// Structured action schema used by Puffer-style flattened action spaces.
public typealias StructuredActionSchema = StructuredTensorSchema

/// Execution mode for vectorized rollout collection.
public enum VectorizationBackend: String, Sendable, Equatable, Codable {
    /// Single-process sequential stepping.
    case serial

    /// Concurrent Swift tasks or worker threads in one process.
    case threaded

    /// Multiple worker processes or remote workers.
    case multiprocessing

    /// CUDA-native stepping or rollout kernels.
    case cuda

    /// TensorRT-backed policy execution with vectorized environments.
    case tensorRT
}

/// Static configuration for a vectorized training or rollout run.
public struct VectorizationProfile: Sendable, Equatable, Codable {
    /// Vectorization backend.
    public let backend: VectorizationBackend

    /// Number of environment instances.
    public let environmentCount: Int

    /// Number of execution workers.
    public let workerCount: Int

    /// Number of environments advanced per policy batch.
    public let batchSize: Int

    /// Number of agents exposed by each environment instance.
    public let agentsPerEnvironment: Int

    /// Whether rollout collection uses asynchronous send/receive semantics.
    public let isAsynchronous: Bool

    /// Diagnostic accelerator label.
    public let accelerator: String

    /// Creates a vectorization profile.
    public init(
        backend: VectorizationBackend,
        environmentCount: Int,
        workerCount: Int = 1,
        batchSize: Int,
        agentsPerEnvironment: Int = 1,
        isAsynchronous: Bool = false,
        accelerator: String
    ) throws {
        guard environmentCount > 0 else {
            throw RLSwiftError.invalidCapacity(environmentCount)
        }
        guard workerCount > 0 else {
            throw RLSwiftError.invalidCapacity(workerCount)
        }
        guard batchSize > 0 else {
            throw RLSwiftError.invalidCapacity(batchSize)
        }
        guard batchSize <= environmentCount else {
            throw RLSwiftError.dimensionMismatch(expected: environmentCount, actual: batchSize)
        }
        guard environmentCount % batchSize == 0 else {
            throw RLSwiftError.dimensionMismatch(expected: environmentCount, actual: batchSize)
        }
        guard environmentCount % workerCount == 0 else {
            throw RLSwiftError.dimensionMismatch(expected: environmentCount, actual: workerCount)
        }
        guard agentsPerEnvironment > 0 else {
            throw RLSwiftError.invalidCapacity(agentsPerEnvironment)
        }
        guard !accelerator.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "accelerator")
        }
        self.backend = backend
        self.environmentCount = environmentCount
        self.workerCount = workerCount
        self.batchSize = batchSize
        self.agentsPerEnvironment = agentsPerEnvironment
        self.isAsynchronous = isAsynchronous
        self.accelerator = accelerator
    }

    /// Total number of agent slots in one vectorized step.
    public var totalAgentCount: Int {
        environmentCount * agentsPerEnvironment
    }

    /// Whether the profile targets an accelerator backend.
    public var usesAccelerator: Bool {
        backend == .cuda || backend == .tensorRT
    }

    /// Creates a serial Swift profile.
    public static func serial(environmentCount: Int, agentsPerEnvironment: Int = 1) throws -> VectorizationProfile {
        try VectorizationProfile(
            backend: .serial,
            environmentCount: environmentCount,
            batchSize: environmentCount,
            agentsPerEnvironment: agentsPerEnvironment,
            accelerator: "swift-cpu"
        )
    }
}

/// Metadata for a saved policy checkpoint.
public struct PolicyCheckpointManifest: Sendable, Equatable, Codable {
    /// Stable checkpoint identifier.
    public let checkpointID: String

    /// Policy metadata associated with the artifact.
    public let policyMetadata: PolicyMetadata

    /// Training step represented by the checkpoint.
    public let trainingStep: Int

    /// Artifact path or URI.
    public let artifactPath: String

    /// Scalar metrics captured at save time.
    public let metrics: [String: Double]

    /// Optional vectorization profile used to produce the checkpoint.
    public let vectorizationProfile: VectorizationProfile?

    /// Creates a policy checkpoint manifest.
    public init(
        checkpointID: String,
        policyMetadata: PolicyMetadata,
        trainingStep: Int,
        artifactPath: String,
        metrics: [String: Double] = [:],
        vectorizationProfile: VectorizationProfile? = nil
    ) throws {
        guard !checkpointID.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "checkpointID")
        }
        guard trainingStep >= 0 else {
            throw RLSwiftError.invalidSampleCount(trainingStep)
        }
        guard !artifactPath.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "checkpoint.artifactPath")
        }
        for metricName in metrics.keys {
            guard !metricName.isEmpty else {
                throw RLSwiftError.emptyIdentifier(name: "checkpoint.metric")
            }
        }
        self.checkpointID = checkpointID
        self.policyMetadata = policyMetadata
        self.trainingStep = trainingStep
        self.artifactPath = artifactPath
        self.metrics = metrics
        self.vectorizationProfile = vectorizationProfile
    }
}

/// One frozen opponent policy in a self-play pool.
public struct SelfPlayOpponent: Sendable, Equatable, Codable {
    /// Stable opponent identifier.
    public let id: String

    /// Checkpoint identifier or artifact path for the frozen policy.
    public let checkpointReference: String

    /// Elo-style rating.
    public let rating: Double

    /// Number of recorded games against this opponent.
    public let gamesPlayed: Int

    /// Creates a self-play opponent entry.
    public init(id: String, checkpointReference: String, rating: Double = 0, gamesPlayed: Int = 0) throws {
        guard !id.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "selfplay.opponentID")
        }
        guard !checkpointReference.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "selfplay.checkpointReference")
        }
        guard gamesPlayed >= 0 else {
            throw RLSwiftError.invalidSampleCount(gamesPlayed)
        }
        self.id = id
        self.checkpointReference = checkpointReference
        self.rating = rating
        self.gamesPlayed = gamesPlayed
    }
}

/// Rating update emitted after one self-play scoring window.
public struct SelfPlayRatingUpdate: Sendable, Equatable, Codable {
    /// Primary policy rating after the update.
    public let primaryRating: Double

    /// Opponent rating after the update.
    public let opponentRating: Double

    /// Primary score rate in the closed `0...1` range.
    public let primaryScore: Double
}

/// A bounded frozen-opponent pool for self-play curricula.
public struct SelfPlayOpponentPool: Sendable, Equatable, Codable {
    /// Maximum number of frozen opponents retained.
    public let maxSize: Int

    /// Current primary-policy rating.
    public private(set) var primaryRating: Double

    /// Frozen opponents ordered from oldest to newest.
    public private(set) var opponents: [SelfPlayOpponent]

    /// Creates a self-play opponent pool.
    public init(maxSize: Int, primaryRating: Double = 0, opponents: [SelfPlayOpponent] = []) throws {
        guard maxSize > 0 else {
            throw RLSwiftError.invalidCapacity(maxSize)
        }
        self.maxSize = maxSize
        self.primaryRating = primaryRating
        self.opponents = []
        for opponent in opponents {
            try add(opponent)
        }
    }

    /// Adds a new frozen opponent and evicts oldest entries if needed.
    public mutating func add(_ opponent: SelfPlayOpponent) throws {
        guard !opponents.contains(where: { $0.id == opponent.id }) else {
            throw RLSwiftError.duplicateIdentifier(opponent.id)
        }
        opponents.append(opponent)
        while opponents.count > maxSize {
            opponents.removeFirst()
        }
    }

    /// Selects a deterministic opponent, favoring newer entries while avoiding the newest frozen policies for large pools.
    public func sampledOpponent(seed: UInt64) throws -> SelfPlayOpponent {
        guard !opponents.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        let candidateCount = opponents.count < 6 ? opponents.count : opponents.count - 5
        let weightedSlots = (0..<candidateCount).flatMap { index in
            Array(repeating: index, count: index + 1)
        }
        var generator = SeededGenerator(seed: seed)
        let selectedSlot = Int(generator.next() % UInt64(weightedSlots.count))
        return opponents[weightedSlots[selectedSlot]]
    }

    /// Records a score rate against an opponent and updates both Elo-style ratings.
    public mutating func record(primaryScore: Double, against opponentID: String, kFactor: Double = 16) throws -> SelfPlayRatingUpdate {
        guard let index = opponents.firstIndex(where: { $0.id == opponentID }) else {
            throw RLSwiftError.emptyIdentifier(name: opponentID)
        }
        let updated = try Self.updatedRatings(
            primaryRating: primaryRating,
            opponentRating: opponents[index].rating,
            primaryScore: primaryScore,
            kFactor: kFactor
        )
        primaryRating = updated.primaryRating
        opponents[index] = try SelfPlayOpponent(
            id: opponents[index].id,
            checkpointReference: opponents[index].checkpointReference,
            rating: updated.opponentRating,
            gamesPlayed: opponents[index].gamesPlayed + 1
        )
        return updated
    }

    /// Computes Elo-style primary/opponent ratings from a score rate.
    public static func updatedRatings(
        primaryRating: Double,
        opponentRating: Double,
        primaryScore: Double,
        kFactor: Double = 16
    ) throws -> SelfPlayRatingUpdate {
        guard (0...1).contains(primaryScore) else {
            throw RLSwiftError.invalidProbability(primaryScore)
        }
        guard kFactor >= 0 else {
            throw RLSwiftError.invalidWeight(kFactor)
        }
        let expected = 1.0 / (1.0 + pow(10.0, (opponentRating - primaryRating) / 400.0))
        let delta = kFactor * (primaryScore - expected)
        return SelfPlayRatingUpdate(
            primaryRating: primaryRating + delta,
            opponentRating: opponentRating - delta,
            primaryScore: primaryScore
        )
    }
}
