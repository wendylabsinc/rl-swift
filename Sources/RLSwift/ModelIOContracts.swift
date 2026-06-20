public import Foundation

/// A source location for one scalar feature inside ``RobotObservation``.
public enum RobotObservationComponent: Sendable, Equatable, Codable {
    /// Reads a joint-position value by index.
    case jointPosition(index: Int)

    /// Reads a joint-velocity value by index.
    case jointVelocity(index: Int)

    /// Reads an end-effector or task-pose value by index.
    case endEffectorPose(index: Int)

    /// Reads a named scalar sensor value.
    case sensor(key: String)

    /// Extracts this component from an observation.
    public func value(from observation: RobotObservation) throws -> Double {
        switch self {
        case let .jointPosition(index):
            return try Self.value(at: index, in: observation.jointPositions)
        case let .jointVelocity(index):
            return try Self.value(at: index, in: observation.jointVelocities)
        case let .endEffectorPose(index):
            return try Self.value(at: index, in: observation.endEffectorPose)
        case let .sensor(key):
            guard let value = observation.sensorReadings[key] else {
                throw RLSwiftError.emptyIdentifier(name: key)
            }
            return value
        }
    }

    private static func value(at index: Int, in values: [Double]) throws -> Double {
        guard index >= 0, index < values.count else {
            throw RLSwiftError.dimensionMismatch(expected: values.count, actual: index + 1)
        }
        return values[index]
    }
}

/// A versioned scalar observation feature in model-input order.
public struct ObservationFeature: Sendable, Equatable, Codable {
    /// Stable feature name used in manifests and dashboards.
    public let name: String

    /// The zero-based model input index.
    public let index: Int

    /// The robot observation component used to produce this scalar.
    public let component: RobotObservationComponent

    /// Physical unit for the feature, such as `rad`, `m`, or `m/s`.
    public let unit: String

    /// Creates an observation feature definition.
    public init(name: String, index: Int, component: RobotObservationComponent, unit: String = "") throws {
        try Self.validateIdentifier(name, field: "feature.name")
        guard index >= 0 else {
            throw RLSwiftError.dimensionMismatch(expected: 0, actual: index)
        }
        self.name = name
        self.index = index
        self.component = component
        self.unit = unit
    }

    private static func validateIdentifier(_ value: String, field: String) throws {
        guard !value.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: field)
        }
    }
}

/// A serialized snapshot of streaming normalization statistics.
public struct NormalizationSnapshot: Sendable, Equatable, Codable {
    /// Number of samples used to estimate the statistics.
    public let count: Int

    /// Per-feature means in model-input order.
    public let mean: [Double]

    /// Per-feature variances in model-input order.
    public let variance: [Double]

    /// Creates a normalization snapshot from explicit statistics.
    public init(count: Int, mean: [Double], variance: [Double]) throws {
        guard count >= 0 else {
            throw RLSwiftError.invalidSampleCount(count)
        }
        guard mean.count == variance.count else {
            throw RLSwiftError.dimensionMismatch(expected: mean.count, actual: variance.count)
        }
        self.count = count
        self.mean = mean
        self.variance = variance
    }

    /// Captures the public state of an ``ObservationNormalizer``.
    public init(_ normalizer: ObservationNormalizer) throws {
        try self.init(count: normalizer.count, mean: normalizer.mean, variance: normalizer.variance)
    }

    /// Normalizes a feature vector with the captured statistics.
    public func normalize(_ values: [Double], epsilon: Double = 1e-8) throws -> [Double] {
        guard values.count == mean.count else {
            throw RLSwiftError.dimensionMismatch(expected: mean.count, actual: values.count)
        }
        return values.indices.map { index in
            (values[index] - mean[index]) / (variance[index] + epsilon).squareRoot()
        }
    }
}

/// A versioned scalar action output in model-output order.
public struct ActionSpecification: Sendable, Equatable, Codable {
    /// Stable action name used in manifests and dashboards.
    public let name: String

    /// The zero-based model output index.
    public let index: Int

    /// Physical unit for the action, such as `rad`, `rad/s`, or `Nm`.
    public let unit: String

    /// Inclusive lower bound in physical units.
    public let lowerBound: Double

    /// Inclusive upper bound in physical units.
    public let upperBound: Double

    /// Creates an action output definition.
    public init(name: String, index: Int, unit: String = "", lowerBound: Double, upperBound: Double) throws {
        guard !name.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "action.name")
        }
        guard index >= 0 else {
            throw RLSwiftError.dimensionMismatch(expected: 0, actual: index)
        }
        guard lowerBound <= upperBound else {
            throw RLSwiftError.invalidBounds(index: index, lower: lowerBound, upper: upperBound)
        }
        self.name = name
        self.index = index
        self.unit = unit
        self.lowerBound = lowerBound
        self.upperBound = upperBound
    }
}

/// TensorRT binding names expected by a serialized engine or ONNX export.
public struct TensorRTBindingNames: Sendable, Equatable, Codable {
    /// Name of the model input tensor.
    public let inputName: String

    /// Name of the model output tensor.
    public let outputName: String

    /// Optional optimization profile name.
    public let profileName: String?

    /// Creates TensorRT binding metadata.
    public init(inputName: String, outputName: String, profileName: String? = nil) throws {
        guard !inputName.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "tensorRT.inputName")
        }
        guard !outputName.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "tensorRT.outputName")
        }
        self.inputName = inputName
        self.outputName = outputName
        self.profileName = profileName
    }
}

/// Version and provenance metadata for a learned policy.
public struct PolicyMetadata: Sendable, Equatable, Codable {
    /// Stable policy identifier.
    public let policyID: String

    /// Semantic, build, or experiment version.
    public let version: String

    /// Creation timestamp for the policy artifact.
    public let createdAt: Date

    /// Optional training run identifier.
    public let trainingRunID: String?

    /// Additional metadata preserved in manifests.
    public let userInfo: [String: String]

    /// Creates policy metadata.
    public init(
        policyID: String,
        version: String,
        createdAt: Date = Date(timeIntervalSince1970: 0),
        trainingRunID: String? = nil,
        userInfo: [String: String] = [:]
    ) throws {
        guard !policyID.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "policyID")
        }
        guard !version.isEmpty else {
            throw RLSwiftError.invalidVersion(version)
        }
        self.policyID = policyID
        self.version = version
        self.createdAt = createdAt
        self.trainingRunID = trainingRunID
        self.userInfo = userInfo
    }
}

/// A complete versioned model IO contract for robot-policy deployment.
public struct ModelIOContract: Sendable, Equatable, Codable {
    /// Version of the contract schema.
    public let contractVersion: String

    /// Ordered observation features expected by the policy.
    public let observationFeatures: [ObservationFeature]

    /// Optional normalization statistics in observation-feature order.
    public let normalization: NormalizationSnapshot?

    /// Ordered action outputs emitted by the policy.
    public let actionSpecifications: [ActionSpecification]

    /// Control mode represented by action outputs.
    public let actionMode: RobotControlMode

    /// Optional TensorRT binding metadata for NVIDIA Linux deployment.
    public let tensorRTBindings: TensorRTBindingNames?

    /// Policy artifact metadata.
    public let policyMetadata: PolicyMetadata

    /// Creates and validates a model IO contract.
    public init(
        contractVersion: String,
        observationFeatures: [ObservationFeature],
        normalization: NormalizationSnapshot? = nil,
        actionSpecifications: [ActionSpecification],
        actionMode: RobotControlMode,
        tensorRTBindings: TensorRTBindingNames? = nil,
        policyMetadata: PolicyMetadata
    ) throws {
        guard !contractVersion.isEmpty else {
            throw RLSwiftError.invalidVersion(contractVersion)
        }
        try Self.validateUniqueOrdered(observationFeatures.map(\.name), indices: observationFeatures.map(\.index))
        try Self.validateUniqueOrdered(actionSpecifications.map(\.name), indices: actionSpecifications.map(\.index))
        if let normalization, normalization.mean.count != observationFeatures.count {
            throw RLSwiftError.dimensionMismatch(expected: observationFeatures.count, actual: normalization.mean.count)
        }
        self.contractVersion = contractVersion
        self.observationFeatures = observationFeatures.sorted { $0.index < $1.index }
        self.normalization = normalization
        self.actionSpecifications = actionSpecifications.sorted { $0.index < $1.index }
        self.actionMode = actionMode
        self.tensorRTBindings = tensorRTBindings
        self.policyMetadata = policyMetadata
    }

    /// Encodes a robot observation into model-input order.
    public func encode(_ observation: RobotObservation, applyNormalization: Bool = true) throws -> [Double] {
        let values = try observationFeatures.map { try $0.component.value(from: observation) }
        guard applyNormalization, let normalization else {
            return values
        }
        return try normalization.normalize(values)
    }

    /// Creates the physical action space described by this contract.
    public func actionSpace() throws -> ContinuousBoxSpace {
        try ContinuousBoxSpace(
            lowerBounds: actionSpecifications.map(\.lowerBound),
            upperBounds: actionSpecifications.map(\.upperBound)
        )
    }

    /// Decodes model output values into a bounded ``RobotAction``.
    public func decodeAction(_ values: [Double], clipToBounds: Bool = true) throws -> RobotAction {
        guard values.count == actionSpecifications.count else {
            throw RLSwiftError.dimensionMismatch(expected: actionSpecifications.count, actual: values.count)
        }
        let action = RobotAction(commands: values, mode: actionMode)
        guard clipToBounds else {
            return action
        }
        return try action.clipped(to: actionSpace())
    }

    private static func validateUniqueOrdered(_ names: [String], indices: [Int]) throws {
        var seenNames: Set<String> = []
        var seenIndices: Set<Int> = []
        for name in names {
            guard seenNames.insert(name).inserted else {
                throw RLSwiftError.duplicateIdentifier(name)
            }
        }
        for index in indices {
            guard seenIndices.insert(index).inserted else {
                throw RLSwiftError.duplicateIndex(index)
            }
        }
    }
}
