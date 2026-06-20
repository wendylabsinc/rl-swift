/// Compute backend for performance-critical RL kernels.
public enum NativeKernelBackend: String, Sendable, Equatable, Codable {
    /// Portable Swift CPU implementation.
    case swiftCPU

    /// Apple MLX-backed implementation.
    case mlx

    /// CUDA-native implementation for NVIDIA Linux.
    case cuda

    /// TensorRT execution-engine implementation.
    case tensorRT
}

/// Fused operation that a native backend can implement.
public enum KernelFusionOperation: String, Sendable, Equatable, Codable {
    /// Observation normalization.
    case normalizeObservations

    /// Generalized advantage estimation.
    case estimateAdvantages

    /// Clipped PPO objective evaluation.
    case clippedPPOObjective

    /// Action sampling from policy outputs.
    case sampleActions

    /// Recurrent policy forward pass.
    case recurrentPolicyForward
}

/// Execution plan for a fused native-kernel training path.
public struct NativeKernelPlan: Sendable, Equatable, Codable {
    /// Backend that should execute the fused operations.
    public let backend: NativeKernelBackend

    /// Operations included in the fused plan.
    public let operations: [KernelFusionOperation]

    /// Numeric precision label, such as `fp32`, `bf16`, or `fp16`.
    public let precision: String

    /// Whether backend tensors are allocated once and reused across steps.
    public let usesStaticMemory: Bool

    /// Whether the backend should use CUDA graph or equivalent command capture.
    public let usesGraphCapture: Bool

    /// Creates a native-kernel execution plan.
    public init(
        backend: NativeKernelBackend,
        operations: [KernelFusionOperation],
        precision: String,
        usesStaticMemory: Bool,
        usesGraphCapture: Bool
    ) throws {
        guard !operations.isEmpty else {
            throw RLSwiftError.invalidCapacity(0)
        }
        guard !precision.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "precision")
        }
        self.backend = backend
        self.operations = operations
        self.precision = precision
        self.usesStaticMemory = usesStaticMemory
        self.usesGraphCapture = usesGraphCapture
    }

    /// Recommended CUDA plan for high-throughput PPO training.
    public static func cudaPPO(precision: String = "bf16") throws -> NativeKernelPlan {
        try NativeKernelPlan(
            backend: .cuda,
            operations: [
                .normalizeObservations,
                .estimateAdvantages,
                .clippedPPOObjective,
                .sampleActions,
                .recurrentPolicyForward,
            ],
            precision: precision,
            usesStaticMemory: true,
            usesGraphCapture: true
        )
    }

    /// Portable Swift fallback plan for deterministic CPU tests.
    public static func swiftReference() throws -> NativeKernelPlan {
        try NativeKernelPlan(
            backend: .swiftCPU,
            operations: [.estimateAdvantages, .clippedPPOObjective],
            precision: "fp64",
            usesStaticMemory: false,
            usesGraphCapture: false
        )
    }
}
