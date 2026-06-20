import Foundation
import RLSwift

@main
struct VectorizedPPOExample {
    static func main() throws {
        do {
            let configuration = try ExampleConfiguration.parse(CommandLine.arguments.dropFirst())
            let result = try VectorizedPPORunner(configuration: configuration).run()
            print(result.summaryText)
            guard result.finalLoss.isFinite else {
                throw ExampleError.trainingProducedInvalidLoss
            }
        } catch let error as ExampleError {
            FileHandle.standardError.write(Data((error.message + "\n").utf8))
            throw ExitCode.failure
        }
    }
}

struct ExampleConfiguration: Sendable {
    var iterations = 4
    var environmentCount = 8
    var rolloutLength = 4
    var seed: UInt64 = 13

    static func parse(_ arguments: ArraySlice<String>) throws -> ExampleConfiguration {
        var configuration = ExampleConfiguration()
        var index = arguments.startIndex
        while index < arguments.endIndex {
            let argument = arguments[index]
            let valueIndex = arguments.index(after: index)
            guard valueIndex < arguments.endIndex else {
                throw ExampleError.usage("Missing value for \(argument)")
            }
            let value = arguments[valueIndex]
            switch argument {
            case "--iterations":
                configuration.iterations = try parsePositiveInt(value, name: "--iterations")
            case "--envs":
                configuration.environmentCount = try parsePositiveInt(value, name: "--envs")
            case "--rollout":
                configuration.rolloutLength = try parsePositiveInt(value, name: "--rollout")
            case "--seed":
                guard let parsed = UInt64(value) else {
                    throw ExampleError.usage("--seed must be an unsigned integer")
                }
                configuration.seed = parsed
            default:
                throw ExampleError.usage("Unknown argument \(argument)")
            }
            index = arguments.index(after: valueIndex)
        }
        return configuration
    }

    private static func parsePositiveInt(_ value: String, name: String) throws -> Int {
        guard let parsed = Int(value), parsed > 0 else {
            throw ExampleError.usage("\(name) must be a positive integer")
        }
        return parsed
    }
}

enum ExampleError: Error {
    case trainingProducedInvalidLoss
    case usage(String)

    var message: String {
        switch self {
        case .trainingProducedInvalidLoss:
            return "Training produced a non-finite loss."
        case let .usage(message):
            return """
            \(message)
            Usage: swift run vectorized-ppo [--iterations 4] [--envs 8] [--rollout 4] [--seed 13]
            """
        }
    }
}

enum ExitCode: Error {
    case failure
}

struct CollectedStep: Sendable {
    let observation: [Double]
    let actionIndex: Int
    let oldLogProbability: Double
    let reward: Double
    let valueEstimate: Double
    let entropy: Double
    let termination: StepTermination
}

struct VectorizedPPOResult: Sendable {
    let configuration: ExampleConfiguration
    let profile: VectorizationProfile
    let throughput: TrainingThroughputReport
    let checkpoint: PolicyCheckpointManifest
    let dashboard: TrainingDashboardSnapshot
    let finalLoss: Double
    let sampleCount: Int

    var summaryText: String {
        let dashboardText = (try? dashboard.renderTextTable()) ?? ""
        return """
        RLSwift VectorizedPPO example
        iterations: \(configuration.iterations)
        environments: \(configuration.environmentCount)
        rolloutLength: \(configuration.rolloutLength)
        seed: \(configuration.seed)
        vectorization: \(profile.backend.rawValue) envs=\(profile.environmentCount) batch=\(profile.batchSize) agents=\(profile.totalAgentCount)
        samples: \(sampleCount)
        stepsPerSecond: \(format(throughput.stepsPerSecond))
        finalLoss: \(format(finalLoss))
        checkpoint: \(checkpoint.checkpointID) step=\(checkpoint.trainingStep) path=\(checkpoint.artifactPath)
        dashboard:
        \(dashboardText)
        """
    }
}

struct VectorizedPPORunner {
    let configuration: ExampleConfiguration
    let schema: StructuredObservationSchema

    init(configuration: ExampleConfiguration) throws {
        self.configuration = configuration
        schema = try StructuredObservationSchema(fields: [
            try StructuredTensorField(name: "position", shape: [1]),
            try StructuredTensorField(name: "goal", shape: [1]),
            try StructuredTensorField(name: "step", shape: [1]),
        ])
    }

    func run() throws -> VectorizedPPOResult {
        let profile = try VectorizationProfile(
            backend: .serial,
            environmentCount: configuration.environmentCount,
            batchSize: configuration.environmentCount,
            accelerator: "swift-cpu"
        )
        var runner = try VectorizedEnvironmentRunner(makeEnvironments())
        var observations = runner.resetAll()
        let ppoConfiguration = try PPOConfiguration(
            discount: 0.95,
            gaeLambda: 0.9,
            clipRange: 0.2,
            valueLossCoefficient: 0.5,
            entropyCoefficient: 0.01,
            learningRate: 0.05,
            epochs: 2,
            minibatchSize: max(1, configuration.environmentCount)
        )
        let model = try DenseDiscreteActorCriticModel(
            observationDimension: schema.flattenedCount,
            hiddenUnitCount: 8,
            actionCount: LineWorldAction.allCases.count,
            seed: configuration.seed
        )
        var trainer = NeuralPPOTrainer(model: model, configuration: ppoConfiguration)
        var meter = ThroughputMeter()
        var lossPoints: [MetricSeriesPoint] = []
        var rewardPoints: [MetricSeriesPoint] = []
        var finalLoss = 0.0
        var sampleCount = 0

        for iteration in 0..<configuration.iterations {
            let collected = try collectRollout(
                runner: &runner,
                observations: &observations,
                model: trainer.model,
                meter: &meter
            )
            let samples = try makeTrainingSamples(
                collected: collected,
                lastObservations: observations,
                model: trainer.model,
                configuration: ppoConfiguration
            )
            let summary = try trainer.update(samples: samples)
            finalLoss = summary.finalObjective?.totalLoss ?? finalLoss
            sampleCount += summary.sampleCount
            let rewardMean = collected.flatMap { $0 }.map(\.reward).reduce(0, +) / Double(samples.count)
            lossPoints.append(try MetricSeriesPoint(step: iteration, value: finalLoss))
            rewardPoints.append(try MetricSeriesPoint(step: iteration, value: rewardMean))
            observations = runner.resetAll()
        }

        let syntheticElapsedSeconds = Double(max(1, configuration.iterations * configuration.rolloutLength)) / 100
        let throughput = try meter.report(
            elapsedSeconds: syntheticElapsedSeconds,
            environmentCount: configuration.environmentCount,
            accelerator: profile.accelerator
        )
        let policyMetadata = try PolicyMetadata(
            policyID: "lineworld-vectorized-ppo",
            version: "example",
            trainingRunID: "seed-\(configuration.seed)"
        )
        let checkpoint = try PolicyCheckpointManifest(
            checkpointID: "lineworld-\(configuration.seed)-\(meter.stepCount)",
            policyMetadata: policyMetadata,
            trainingStep: meter.stepCount,
            artifactPath: "checkpoints/lineworld-\(configuration.seed)-\(meter.stepCount).bin",
            metrics: [
                "loss": finalLoss,
                "steps_per_second": throughput.stepsPerSecond,
            ],
            vectorizationProfile: profile
        )
        let dashboard = try TrainingDashboardSnapshot(title: "VectorizedPPO", series: [
            try TrainingMetricSeries(name: "loss", points: lossPoints),
            try TrainingMetricSeries(name: "reward", points: rewardPoints),
        ])
        return VectorizedPPOResult(
            configuration: configuration,
            profile: profile,
            throughput: throughput,
            checkpoint: checkpoint,
            dashboard: dashboard,
            finalLoss: finalLoss,
            sampleCount: sampleCount
        )
    }

    private func makeEnvironments() throws -> [LineWorldEnvironment] {
        try (0..<configuration.environmentCount).map { _ in
            try LineWorldEnvironment(length: configuration.rolloutLength + 1, maxSteps: configuration.rolloutLength)
        }
    }

    private func collectRollout(
        runner: inout VectorizedEnvironmentRunner<LineWorldEnvironment>,
        observations: inout [LineWorldObservation],
        model: DenseDiscreteActorCriticModel,
        meter: inout ThroughputMeter
    ) throws -> [[CollectedStep]] {
        var collected = Array(repeating: [CollectedStep](), count: configuration.environmentCount)
        for _ in 0..<configuration.rolloutLength {
            let features = try observations.map(features)
            var actions = Array(repeating: LineWorldAction.right, count: configuration.environmentCount)
            for environmentIndex in 0..<configuration.environmentCount {
                let prediction = try model.prediction(for: features[environmentIndex])
                let actionIndex = 1
                let oldLogProbability = try prediction.logProbability(actionIndex: actionIndex)
                collected[environmentIndex].append(CollectedStep(
                    observation: features[environmentIndex],
                    actionIndex: actionIndex,
                    oldLogProbability: oldLogProbability,
                    reward: 0,
                    valueEstimate: prediction.valueEstimate,
                    entropy: prediction.entropy,
                    termination: .continuing
                ))
                actions[environmentIndex] = .right
            }
            let result = try runner.step(actions)
            for environmentIndex in 0..<configuration.environmentCount {
                let previous = collected[environmentIndex].removeLast()
                collected[environmentIndex].append(CollectedStep(
                    observation: previous.observation,
                    actionIndex: previous.actionIndex,
                    oldLogProbability: previous.oldLogProbability,
                    reward: result.rewards[environmentIndex],
                    valueEstimate: previous.valueEstimate,
                    entropy: previous.entropy,
                    termination: result.terminations[environmentIndex]
                ))
            }
            observations = result.observations
            try meter.record(steps: configuration.environmentCount, samples: configuration.environmentCount)
        }
        return collected
    }

    private func makeTrainingSamples(
        collected: [[CollectedStep]],
        lastObservations: [LineWorldObservation],
        model: DenseDiscreteActorCriticModel,
        configuration: PPOConfiguration
    ) throws -> [PPOTrainingSample] {
        var samples: [PPOTrainingSample] = []
        for environmentIndex in collected.indices {
            let steps = collected[environmentIndex]
            let lastValue: Double
            if steps.last?.termination.endsEpisode == true {
                lastValue = 0
            } else {
                lastValue = try model.prediction(for: features(lastObservations[environmentIndex])).valueEstimate
            }
            let trajectory = steps.map { step in
                PPOTrajectoryStep(
                    reward: step.reward,
                    valueEstimate: step.valueEstimate,
                    logProbability: step.oldLogProbability,
                    entropy: step.entropy,
                    termination: step.termination
                )
            }
            let advantageBatch = try PPOAdvantageEstimator.generalizedAdvantageEstimate(
                steps: trajectory,
                lastValue: lastValue,
                configuration: configuration
            )
            for index in steps.indices {
                samples.append(try PPOTrainingSample(
                    observation: steps[index].observation,
                    actionIndex: steps[index].actionIndex,
                    oldLogProbability: steps[index].oldLogProbability,
                    advantage: advantageBatch.advantages[index],
                    returnEstimate: advantageBatch.returns[index]
                ))
            }
        }
        return samples
    }

    private func features(_ observation: LineWorldObservation) throws -> [Double] {
        let goal = max(1, observation.goal)
        return try schema.flatten([
            "position": [Double(observation.position) / Double(goal)],
            "goal": [Double(observation.goal)],
            "step": [Double(observation.stepIndex)],
        ])
    }
}

private func format(_ value: Double) -> String {
    String(format: "%.4f", value)
}
