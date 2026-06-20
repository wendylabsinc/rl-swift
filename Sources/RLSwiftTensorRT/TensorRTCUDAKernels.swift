#if os(Linux) && SWIFTRL_ENABLE_TENSORRT
public import RLSwift
import RLSwiftCUDANative

/// Error raised by the native CUDA kernel executor.
public enum TensorRTCUDAKernelError: Error, Equatable, Sendable {
    /// The CUDA runtime, NVRTC compiler, or compiled kernel module was unavailable.
    case runtimeUnavailable(String)

    /// The kernel rejected invalid arguments before launch.
    case invalidArguments(String)

    /// The kernel launch or host/device copy failed.
    case launchFailed(String)
}

/// Runtime availability reported by the CUDA native-kernel executor.
public struct TensorRTCUDAKernelRuntimeStatus: Equatable, Sendable {
    /// Whether CUDA kernels can be compiled and launched in this process.
    public let isAvailable: Bool

    /// Last runtime diagnostic emitted by the native CUDA shim.
    public let message: String

    /// Creates a CUDA runtime status value.
    public init(isAvailable: Bool, message: String) {
        self.isAvailable = isAvailable
        self.message = message
    }
}

/// High-throughput CUDA kernels used by the TensorRT/NVIDIA Linux backend.
public enum TensorRTCUDAKernelExecutor {
    /// Returns whether CUDA Driver API and NVRTC-backed kernels are available.
    public static func runtimeStatus() -> TensorRTCUDAKernelRuntimeStatus {
        let available = rlsw_cuda_is_available() == 1
        return TensorRTCUDAKernelRuntimeStatus(
            isAvailable: available,
            message: lastErrorMessage()
        )
    }

    /// Steps a batch of `LineWorld` states in one CUDA launch.
    public static func stepLineWorld(
        actions: [LineWorldAction],
        positions: inout [Int32],
        stepIndices: inout [Int32],
        length: Int,
        maxSteps: Int
    ) throws -> VectorizedStepResult<LineWorldObservation> {
        guard !actions.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        guard positions.count == actions.count else {
            throw RLSwiftError.dimensionMismatch(expected: actions.count, actual: positions.count)
        }
        guard stepIndices.count == actions.count else {
            throw RLSwiftError.dimensionMismatch(expected: actions.count, actual: stepIndices.count)
        }
        guard length >= 2 else {
            throw RLSwiftError.invalidCapacity(length)
        }
        guard maxSteps > 0 else {
            throw RLSwiftError.invalidHorizon(maxSteps)
        }

        let actionCodes = actions.map { action in
            switch action {
            case .left:
                return Int32(0)
            case .right:
                return Int32(1)
            }
        }
        var rewards = Array(repeating: Float(0), count: actions.count)
        var terminalBytes = Array(repeating: UInt8(0), count: actions.count)
        var terminationCodes = Array(repeating: Int32(0), count: actions.count)
        let status = actionCodes.withUnsafeBufferPointer { actionPointer in
            positions.withUnsafeMutableBufferPointer { positionPointer in
                stepIndices.withUnsafeMutableBufferPointer { stepPointer in
                    rewards.withUnsafeMutableBufferPointer { rewardPointer in
                        terminalBytes.withUnsafeMutableBufferPointer { terminalPointer in
                            terminationCodes.withUnsafeMutableBufferPointer { terminationPointer in
                                rlsw_cuda_lineworld_step(
                                    actionPointer.baseAddress,
                                    positionPointer.baseAddress,
                                    stepPointer.baseAddress,
                                    rewardPointer.baseAddress,
                                    terminalPointer.baseAddress,
                                    terminationPointer.baseAddress,
                                    Int32(actions.count),
                                    Int32(length),
                                    Int32(maxSteps)
                                )
                            }
                        }
                    }
                }
            }
        }
        try throwIfNeeded(status)

        let observations = positions.indices.map { index in
            LineWorldObservation(position: Int(positions[index]), goal: length - 1, stepIndex: Int(stepIndices[index]))
        }
        let terminations = terminationCodes.map { code in
            switch code {
            case 1:
                return StepTermination.terminated(reason: "goal")
            case 2:
                return StepTermination.truncated(reason: "max_steps")
            default:
                return StepTermination.continuing
            }
        }
        return VectorizedStepResult(
            observations: observations,
            rewards: rewards.map(Double.init),
            terminalFlags: terminalBytes.map { $0 != 0 },
            terminations: terminations
        )
    }

    /// Evaluates the clipped PPO objective using a fused CUDA reduction kernel.
    public static func ppoObjective(
        samples: [PPOClippedObjectiveSample],
        configuration: PPOConfiguration
    ) throws -> PPOObjectiveBreakdown {
        guard !samples.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        let oldLogProbabilities = samples.map { Float($0.oldLogProbability) }
        let newLogProbabilities = samples.map { Float($0.newLogProbability) }
        let advantages = samples.map { Float($0.advantage) }
        let returns = samples.map { Float($0.returnEstimate) }
        let valueEstimates = samples.map { Float($0.valueEstimate) }
        let entropies = samples.map { Float($0.entropy) }
        var result = rlsw_cuda_ppo_objective_result()

        let status = oldLogProbabilities.withUnsafeBufferPointer { oldPointer in
            newLogProbabilities.withUnsafeBufferPointer { newPointer in
                advantages.withUnsafeBufferPointer { advantagePointer in
                    returns.withUnsafeBufferPointer { returnPointer in
                        valueEstimates.withUnsafeBufferPointer { valuePointer in
                            entropies.withUnsafeBufferPointer { entropyPointer in
                                rlsw_cuda_ppo_objective(
                                    oldPointer.baseAddress,
                                    newPointer.baseAddress,
                                    advantagePointer.baseAddress,
                                    returnPointer.baseAddress,
                                    valuePointer.baseAddress,
                                    entropyPointer.baseAddress,
                                    Int32(samples.count),
                                    Float(configuration.clipRange),
                                    Float(configuration.valueLossCoefficient),
                                    Float(configuration.entropyCoefficient),
                                    &result
                                )
                            }
                        }
                    }
                }
            }
        }
        try throwIfNeeded(status)

        return PPOObjectiveBreakdown(
            policyLoss: result.policy_loss,
            valueLoss: result.value_loss,
            entropyBonus: result.entropy_bonus,
            totalLoss: result.total_loss,
            meanApproximateKL: result.mean_approximate_kl,
            clippedFraction: result.clipped_fraction
        )
    }

    private static func throwIfNeeded(_ status: Int32) throws {
        switch status {
        case 0:
            return
        case 1:
            throw TensorRTCUDAKernelError.invalidArguments(lastErrorMessage())
        case 2:
            throw TensorRTCUDAKernelError.runtimeUnavailable(lastErrorMessage())
        default:
            throw TensorRTCUDAKernelError.launchFailed(lastErrorMessage())
        }
    }

    private static func lastErrorMessage() -> String {
        guard let pointer = rlsw_cuda_last_error_message() else {
            return "CUDA native kernel runtime returned no diagnostic."
        }
        return String(cString: pointer)
    }
}
#endif
