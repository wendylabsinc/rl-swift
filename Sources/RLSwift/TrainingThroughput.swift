/// Throughput statistics for rollout collection or policy training.
public struct TrainingThroughputReport: Sendable, Equatable, Codable {
    /// Number of environment steps processed.
    public let stepCount: Int

    /// Number of agent samples processed.
    public let sampleCount: Int

    /// Wall-clock seconds represented by this report.
    public let elapsedSeconds: Double

    /// Number of environments or simulation instances contributing samples.
    public let environmentCount: Int

    /// Backend or accelerator label used for diagnostics.
    public let accelerator: String

    /// Creates a throughput report.
    public init(
        stepCount: Int,
        sampleCount: Int,
        elapsedSeconds: Double,
        environmentCount: Int,
        accelerator: String
    ) throws {
        guard stepCount >= 0 else {
            throw RLSwiftError.invalidSampleCount(stepCount)
        }
        guard sampleCount >= 0 else {
            throw RLSwiftError.invalidSampleCount(sampleCount)
        }
        guard elapsedSeconds > 0 else {
            throw RLSwiftError.invalidDuration(name: "elapsedSeconds", value: elapsedSeconds)
        }
        guard environmentCount > 0 else {
            throw RLSwiftError.invalidCapacity(environmentCount)
        }
        guard !accelerator.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "accelerator")
        }
        self.stepCount = stepCount
        self.sampleCount = sampleCount
        self.elapsedSeconds = elapsedSeconds
        self.environmentCount = environmentCount
        self.accelerator = accelerator
    }

    /// Environment steps processed per second.
    public var stepsPerSecond: Double {
        Double(stepCount) / elapsedSeconds
    }

    /// Agent samples processed per second.
    public var samplesPerSecond: Double {
        Double(sampleCount) / elapsedSeconds
    }

    /// Average samples contributed per environment.
    public var samplesPerEnvironment: Double {
        Double(sampleCount) / Double(environmentCount)
    }
}

/// Incremental counter used by training loops to publish throughput reports.
public struct ThroughputMeter: Sendable, Equatable {
    private var steps = 0
    private var samples = 0

    /// Creates an empty throughput meter.
    public init() {}

    /// Number of environment steps recorded so far.
    public var stepCount: Int {
        steps
    }

    /// Number of agent samples recorded so far.
    public var sampleCount: Int {
        samples
    }

    /// Records a positive batch of steps and samples.
    public mutating func record(steps: Int, samples: Int) throws {
        guard steps >= 0 else {
            throw RLSwiftError.invalidSampleCount(steps)
        }
        guard samples >= 0 else {
            throw RLSwiftError.invalidSampleCount(samples)
        }
        self.steps += steps
        self.samples += samples
    }

    /// Builds a throughput report from the current counters.
    public func report(
        elapsedSeconds: Double,
        environmentCount: Int,
        accelerator: String
    ) throws -> TrainingThroughputReport {
        try TrainingThroughputReport(
            stepCount: steps,
            sampleCount: samples,
            elapsedSeconds: elapsedSeconds,
            environmentCount: environmentCount,
            accelerator: accelerator
        )
    }
}
