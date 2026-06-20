/// A deployment backend available to an RL policy.
public enum DeploymentBackend: Sendable, Equatable, Codable {
    /// Pure Swift policy logic without an accelerator-specific tensor backend.
    case coreSwift

    /// Apple MLX backend for Apple silicon devices.
    case mlx

    /// NVIDIA TensorRT backend for Linux hosts with CUDA.
    case tensorRT

    /// A user-provided backend identified by name.
    case custom(String)

    /// Whether this backend is expected to run on Apple device platforms.
    public var supportsAppleDevices: Bool {
        switch self {
        case .coreSwift, .mlx:
            return true
        case .tensorRT, .custom:
            return false
        }
    }

    /// Whether this backend is expected to require NVIDIA Linux.
    public var requiresNVIDIALinux: Bool {
        switch self {
        case .tensorRT:
            return true
        case .coreSwift, .mlx, .custom:
            return false
        }
    }
}

/// A deterministic target description for policy deployment.
public struct DeploymentTarget: Sendable, Equatable, Codable {
    /// Stable target name.
    public let name: String

    /// Backend used for inference.
    public let backend: DeploymentBackend

    /// Operating-system or platform identifier.
    public let platform: String

    /// Accelerator identifier, such as `metal`, `cuda`, or `cpu`.
    public let accelerator: String

    /// Minimum Swift version expected by the target package.
    public let minimumSwiftVersion: String

    /// Creates a deployment target.
    public init(
        name: String,
        backend: DeploymentBackend,
        platform: String,
        accelerator: String,
        minimumSwiftVersion: String = "6.3"
    ) throws {
        guard !name.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "deploymentTarget.name")
        }
        guard !platform.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "deploymentTarget.platform")
        }
        guard !accelerator.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "deploymentTarget.accelerator")
        }
        guard !minimumSwiftVersion.isEmpty else {
            throw RLSwiftError.invalidVersion(minimumSwiftVersion)
        }
        self.name = name
        self.backend = backend
        self.platform = platform
        self.accelerator = accelerator
        self.minimumSwiftVersion = minimumSwiftVersion
    }

    /// A default Apple-device target backed by MLX.
    public static func appleMLX(platform: String = "apple") throws -> DeploymentTarget {
        try DeploymentTarget(name: "apple-mlx", backend: .mlx, platform: platform, accelerator: "metal")
    }

    /// A default NVIDIA Linux target backed by TensorRT.
    public static func nvidiaTensorRT(platform: String = "linux") throws -> DeploymentTarget {
        try DeploymentTarget(name: "nvidia-tensorrt", backend: .tensorRT, platform: platform, accelerator: "cuda")
    }
}

/// A complete deterministic deployment plan for a policy artifact.
public struct DeploymentPlan: Sendable, Equatable, Codable {
    /// Target hardware and backend.
    public let target: DeploymentTarget

    /// Model IO contract deployed with the policy.
    public let modelContract: ModelIOContract

    /// Deterministic seed used for reproducible rollout behavior.
    public let deterministicSeed: UInt64

    /// Optional engine cache key for compiled backends.
    public let engineCacheKey: String?

    /// Additional deployment metadata.
    public let metadata: [String: String]

    /// Creates a deployment plan.
    public init(
        target: DeploymentTarget,
        modelContract: ModelIOContract,
        deterministicSeed: UInt64,
        engineCacheKey: String? = nil,
        metadata: [String: String] = [:]
    ) {
        self.target = target
        self.modelContract = modelContract
        self.deterministicSeed = deterministicSeed
        self.engineCacheKey = engineCacheKey
        self.metadata = metadata
    }

    /// Whether the plan contains TensorRT binding metadata when the backend requires it.
    public var hasRequiredBackendMetadata: Bool {
        switch target.backend {
        case .tensorRT:
            return modelContract.tensorRTBindings != nil
        case .coreSwift, .mlx, .custom:
            return true
        }
    }
}

/// A deterministic policy-version traffic split for staged rollouts.
public struct PolicyVersionRollout: Sendable, Equatable, Codable {
    /// Currently deployed policy version.
    public let currentVersion: String

    /// Candidate policy version under evaluation.
    public let candidateVersion: String

    /// Fraction of stable unit IDs routed to the candidate version.
    public let candidateTrafficFraction: Double

    /// Rollout seed mixed into stable routing.
    public let seed: UInt64

    /// Creates a deterministic rollout split.
    public init(
        currentVersion: String,
        candidateVersion: String,
        candidateTrafficFraction: Double,
        seed: UInt64 = 0
    ) throws {
        guard !currentVersion.isEmpty else {
            throw RLSwiftError.invalidVersion(currentVersion)
        }
        guard !candidateVersion.isEmpty else {
            throw RLSwiftError.invalidVersion(candidateVersion)
        }
        guard (0...1).contains(candidateTrafficFraction) else {
            throw RLSwiftError.invalidProbability(candidateTrafficFraction)
        }
        self.currentVersion = currentVersion
        self.candidateVersion = candidateVersion
        self.candidateTrafficFraction = candidateTrafficFraction
        self.seed = seed
    }

    /// Selects a policy version for a stable robot, device, or environment ID.
    public func selectedVersion(for unitID: String) -> String {
        stableUnitInterval(unitID, seed: seed) < candidateTrafficFraction ? candidateVersion : currentVersion
    }
}

/// A summarized telemetry view for robot-policy operation.
public struct AutonomyTelemetrySummary: Sendable, Equatable, Codable {
    /// Number of recorded control steps.
    public let stepCount: Int

    /// Mean closed-loop latency in seconds.
    public let meanClosedLoopLatency: Double

    /// Number of latency deadline misses.
    public let deadlineMissCount: Int

    /// Counts grouped by safety-intervention name.
    public let safetyInterventionCounts: [String: Int]

    /// Accumulated constraint costs grouped by constraint name.
    public let constraintCosts: [String: Double]

    /// Counts grouped by policy version.
    public let policyVersionCounts: [String: Int]
}

/// A streaming accumulator for real-time autonomy telemetry.
public struct AutonomyTelemetryAccumulator: Sendable, Equatable {
    private var stepCount: Int = 0
    private var latencyTotal: Double = 0
    private var deadlineMissCount: Int = 0
    private var safetyInterventionCounts: [String: Int] = [:]
    private var constraintCosts: [String: Double] = [:]
    private var policyVersionCounts: [String: Int] = [:]

    /// Creates an empty telemetry accumulator.
    public init() {}

    /// Records one control step into the accumulator.
    public mutating func record(
        timing: ControlTiming,
        maximumLatency: Double? = nil,
        safetyDecision: SafetySupervisorDecision? = nil,
        constraints: ConstraintReport? = nil,
        policyVersion: String? = nil
    ) throws {
        stepCount += 1
        latencyTotal += timing.closedLoopLatency
        if let maximumLatency, try timing.missesDeadline(maximumLatency: maximumLatency) {
            deadlineMissCount += 1
        }
        for intervention in safetyDecision?.interventions ?? [] {
            safetyInterventionCounts[intervention.metricKey, default: 0] += 1
        }
        for signal in constraints?.signals ?? [] where signal.cost > 0 {
            constraintCosts[signal.name, default: 0] += signal.cost
        }
        if let policyVersion {
            policyVersionCounts[policyVersion, default: 0] += 1
        }
    }

    /// Returns the current telemetry summary.
    public var summary: AutonomyTelemetrySummary {
        AutonomyTelemetrySummary(
            stepCount: stepCount,
            meanClosedLoopLatency: stepCount == 0 ? 0 : latencyTotal / Double(stepCount),
            deadlineMissCount: deadlineMissCount,
            safetyInterventionCounts: safetyInterventionCounts,
            constraintCosts: constraintCosts,
            policyVersionCounts: policyVersionCounts
        )
    }
}

private func stableUnitInterval(_ value: String, seed: UInt64) -> Double {
    let basis: UInt64 = 0xcbf29ce484222325 ^ seed
    let prime: UInt64 = 0x00000100000001B3
    var hash = basis
    for byte in value.utf8 {
        hash ^= UInt64(byte)
        hash &*= prime
    }
    return Double(hash) / Double(UInt64.max)
}

private extension SafetySupervisorIntervention {
    var metricKey: String {
        switch self {
        case .emergencyStop:
            return "emergency_stop"
        case .staleSensor:
            return "stale_sensor"
        case .deadlineMissed:
            return "deadline_missed"
        case .safetyEnvelope:
            return "safety_envelope"
        }
    }
}
