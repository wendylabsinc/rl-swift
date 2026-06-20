/// One ordered result returned by a vectorized environment step.
public struct VectorizedEnvironmentStep<Observation: Sendable>: Sendable {
    /// Environment index inside the vectorized runner.
    public let environmentIndex: Int

    /// Step result emitted by that environment.
    public let result: StepResult<Observation>
}

extension VectorizedEnvironmentStep: Equatable where Observation: Equatable {}

/// Ordered batch returned by an asynchronous vectorized environment step.
public struct AsyncVectorizedStepBatch<Observation: Sendable>: Sendable {
    /// Per-environment results in environment-index order.
    public let results: [VectorizedEnvironmentStep<Observation>]

    /// Creates an ordered vectorized step batch.
    public init(results: [VectorizedEnvironmentStep<Observation>]) throws {
        guard !results.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        self.results = results.sorted { $0.environmentIndex < $1.environmentIndex }
    }

    /// Observations in environment-index order.
    public var observations: [Observation] {
        results.map(\.result.observation)
    }

    /// Rewards in environment-index order.
    public var rewards: [Double] {
        results.map(\.result.reward)
    }

    /// Termination states in environment-index order.
    public var terminations: [StepTermination] {
        results.map(\.result.termination)
    }

    /// Number of environments whose latest step ended an episode.
    public var terminalCount: Int {
        results.reduce(0) { $0 + ($1.result.isTerminal ? 1 : 0) }
    }
}

extension AsyncVectorizedStepBatch: Equatable where Observation: Equatable {}

/// Actor-isolated environment worker used by ``AsyncVectorizedEnvironmentRunner``.
public actor AsyncEnvironmentWorker<Wrapped: Environment> {
    private var environment: Wrapped

    /// Environment index inside the owning runner.
    public let id: Int

    /// Creates a worker around one mutable environment instance.
    public init(id: Int, environment: Wrapped) {
        self.id = id
        self.environment = environment
    }

    /// Resets the wrapped environment.
    public func reset() -> Wrapped.Observation {
        environment.reset()
    }

    /// Steps the wrapped environment.
    public func step(_ action: Wrapped.Action) throws -> StepResult<Wrapped.Observation> {
        try environment.step(action)
    }
}

/// In-process asynchronous vectorized environment runner backed by Swift actors.
public struct AsyncVectorizedEnvironmentRunner<Wrapped: Environment>: Sendable {
    private let workers: [AsyncEnvironmentWorker<Wrapped>]

    /// Runtime profile describing the runner.
    public let profile: VectorizationProfile

    /// Number of environment workers.
    public var count: Int {
        workers.count
    }

    /// Creates a runner from independent environment instances.
    public init(
        environments: [Wrapped],
        batchSize: Int? = nil,
        accelerator: String = "swift-concurrency"
    ) throws {
        guard !environments.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        workers = environments.enumerated().map { index, environment in
            AsyncEnvironmentWorker(id: index, environment: environment)
        }
        profile = try VectorizationProfile(
            backend: .threaded,
            environmentCount: environments.count,
            workerCount: environments.count,
            batchSize: batchSize ?? environments.count,
            isAsynchronous: true,
            accelerator: accelerator
        )
    }

    /// Resets every environment concurrently and returns observations in worker order.
    public func resetAll() async -> [Wrapped.Observation] {
        await withTaskGroup(of: (Int, Wrapped.Observation).self) { group in
            for (index, worker) in workers.enumerated() {
                group.addTask {
                    (index, await worker.reset())
                }
            }
            var ordered = Array<Wrapped.Observation?>(repeating: nil, count: workers.count)
            for await (index, observation) in group {
                ordered[index] = observation
            }
            return ordered.map { $0! }
        }
    }

    /// Steps every environment concurrently using one action per worker.
    public func stepAll(_ actions: [Wrapped.Action]) async throws -> AsyncVectorizedStepBatch<Wrapped.Observation> {
        guard actions.count == workers.count else {
            throw RLSwiftError.dimensionMismatch(expected: workers.count, actual: actions.count)
        }
        let steps = try await withThrowingTaskGroup(
            of: (Int, StepResult<Wrapped.Observation>).self
        ) { group in
            for (index, action) in actions.enumerated() {
                let worker = workers[index]
                group.addTask {
                    (index, try await worker.step(action))
                }
            }
            var ordered = Array<StepResult<Wrapped.Observation>?>(repeating: nil, count: workers.count)
            for try await (index, result) in group {
                ordered[index] = result
            }
            return ordered.enumerated().map { index, result in
                VectorizedEnvironmentStep(environmentIndex: index, result: result!)
            }
        }
        return try AsyncVectorizedStepBatch(results: steps)
    }
}
