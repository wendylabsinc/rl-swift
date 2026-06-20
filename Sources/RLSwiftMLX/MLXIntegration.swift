/// Runtime support information for the RLSwift MLX backend on the current platform.
public struct MLXBackendSupport: Equatable, Sendable {
    /// Whether MLX-backed tensor integration is available in this build.
    public let isMLXAvailable: Bool

    /// A short explanation suitable for diagnostics or setup output.
    public let explanation: String

    /// Creates platform support information.
    public init(isMLXAvailable: Bool, explanation: String) {
        self.isMLXAvailable = isMLXAvailable
        self.explanation = explanation
    }

    /// Support information for the currently compiled platform.
#if SWIFTRL_ENABLE_MLX
    public static let current = MLXBackendSupport(
        isMLXAvailable: true,
        explanation: "MLX backend is available for this build."
    )
#else
    public static let current = MLXBackendSupport(
        isMLXAvailable: false,
        explanation: "MLX backend is disabled; build with the MLXBackend trait to enable it."
    )
#endif
}

#if SWIFTRL_ENABLE_MLX
public import MLX
import RLSwift

/// The tensor type used by RLSwift MLX integrations.
public typealias RLTensor = MLXArray

/// Dense MLX observation batch used by backend learners.
public struct MLXObservationBatch {
    /// Row-major scalar values used to create the MLX tensor.
    public let rowMajorValues: [Float]

    /// Number of observations in the batch.
    public let batchSize: Int

    /// Number of scalar features per observation.
    public let featureCount: Int

    /// Creates an MLX tensor from row-major `Float` observations.
    public init(rows: [[Float]]) throws {
        guard let first = rows.first else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        guard !first.isEmpty else {
            throw RLSwiftError.dimensionMismatch(expected: 1, actual: 0)
        }
        for row in rows where row.count != first.count {
            throw RLSwiftError.dimensionMismatch(expected: first.count, actual: row.count)
        }
        batchSize = rows.count
        featureCount = first.count
        rowMajorValues = rows.flatMap { $0 }
    }

    /// Tensor shape as ordinary Swift integers for tests, logging, and manifests.
    public var shape: [Int] {
        [batchSize, featureCount]
    }
}

/// Encodes domain observations into MLX tensors for function approximation.
public protocol MLXObservationEncoder<Observation> {
    /// The observation type accepted by the encoder.
    associatedtype Observation

    /// Converts an observation into an MLX tensor.
    func encode(_ observation: Observation) throws -> RLTensor
}
#endif
