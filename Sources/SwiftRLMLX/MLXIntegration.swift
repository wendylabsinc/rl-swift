/// Runtime support information for the SwiftRL MLX backend on the current platform.
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

/// The tensor type used by SwiftRL MLX integrations.
public typealias RLTensor = MLXArray

/// Encodes domain observations into MLX tensors for function approximation.
public protocol MLXObservationEncoder<Observation> {
    /// The observation type accepted by the encoder.
    associatedtype Observation

    /// Converts an observation into an MLX tensor.
    func encode(_ observation: Observation) throws -> RLTensor
}
#endif
