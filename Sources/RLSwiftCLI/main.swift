import Foundation
import RLSwift

@main
struct RLSwiftCLI {
    static func main() throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        let command = arguments.first ?? "help"
        switch command {
        case "catalog":
            try printCatalog()
        case "train":
            try trainLineWorld(arguments: Array(arguments.dropFirst()))
        case "sweep":
            try printSweep()
        case "protein":
            try printProteinSuggestions()
        case "experiment":
            try printExperimentConfiguration()
        case "checkpoint":
            try printCheckpointRecord()
        case "eval":
            try evaluateLineWorld(arguments: Array(arguments.dropFirst()))
        case "visualize":
            try printDashboard()
        default:
            printHelp()
        }
    }

    private static func printProteinSuggestions() throws {
        let tuner = try ProteinTuner(
            parameters: [
                try ProteinParameter(name: "learningRate", lowerBound: 1e-5, upperBound: 1e-3, scale: .logarithmic),
                try ProteinParameter(name: "entropy", lowerBound: 0.001, upperBound: 0.05),
            ],
            seed: 11
        )
        for suggestion in try tuner.suggest(count: 3) {
            print("\(suggestion.id)\tseed=\(suggestion.seed)\tparams=\(suggestion.parameters)")
        }
    }

    private static func printExperimentConfiguration() throws {
        print(try encodedJSON(defaultExperimentConfiguration()))
    }

    private static func printCheckpointRecord() throws {
        let configuration = try defaultExperimentConfiguration()
        let metadata = try PolicyMetadata(policyID: "lineworld", version: "0.1.0", trainingRunID: configuration.experimentID)
        let manifest = try PolicyCheckpointManifest(
            checkpointID: "lineworld-000001",
            policyMetadata: metadata,
            trainingStep: 1,
            artifactPath: "\(configuration.checkpointDirectory)/lineworld-000001.bin",
            metrics: ["return": 1],
            vectorizationProfile: configuration.vectorizationProfile
        )
        let summary = try ExperimentEvaluator.evaluateLineWorldRightPolicy(episodes: 2)
        print(try encodedJSON(ExperimentCheckpointRecord(
            configuration: configuration,
            manifest: manifest,
            evaluationSummary: summary
        )))
    }

    private static func evaluateLineWorld(arguments: [String]) throws {
        let episodes = value(after: "--episodes", in: arguments).flatMap(Int.init) ?? 4
        let summary = try ExperimentEvaluator.evaluateLineWorldRightPolicy(episodes: episodes)
        print("episodes=\(summary.episodeCount)")
        print("meanReturn=\(String(format: "%.4f", summary.meanReturn))")
        print("successRate=\(String(format: "%.4f", summary.successRate))")
        print("meanEpisodeLength=\(String(format: "%.2f", summary.meanEpisodeLength))")
    }

    private static func printCatalog() throws {
        for entry in try BuiltInEnvironmentCatalog.allEntries() {
            print("\(entry.id.rawValue)\t\(entry.displayName)\tmultiAgent=\(entry.supportsMultiAgent)")
        }
    }

    private static func trainLineWorld(arguments: [String]) throws {
        let episodes = value(after: "--episodes", in: arguments).flatMap(Int.init) ?? 24
        var environment = try LineWorldEnvironment(length: 5, maxSteps: 16)
        var agent = try TabularQAgent<LineWorldObservation, LineWorldAction>(
            actions: LineWorldAction.allCases,
            learningRate: 0.5,
            discount: 0.95,
            epsilon: 0.2,
            seed: 7
        )
        var meter = ThroughputMeter()
        var returns: [MetricSeriesPoint] = []

        for episode in 0..<episodes {
            var observation = environment.reset()
            var totalReward = 0.0
            for _ in 0..<16 {
                let action = try agent.action(for: observation)
                let result = try environment.step(action)
                let transition = Transition(
                    observation: observation,
                    action: action,
                    reward: result.reward,
                    nextObservation: result.observation,
                    isTerminal: result.isTerminal,
                    termination: result.termination
                )
                agent.observe(transition)
                try meter.record(steps: 1, samples: 1)
                totalReward += result.reward
                observation = result.observation
                if result.isTerminal {
                    break
                }
            }
            returns.append(try MetricSeriesPoint(step: episode, value: totalReward))
        }

        let report = try meter.report(elapsedSeconds: max(1, Double(episodes)), environmentCount: 1, accelerator: "swift-cpu")
        let series = try TrainingMetricSeries(name: "return", points: returns)
        print("episodes=\(episodes)")
        print("stepsPerSecond=\(String(format: "%.2f", report.stepsPerSecond))")
        print("latestReturn=\(String(format: "%.4f", series.latestValue ?? 0))")
        print("sparkline=\(try series.sparkline())")
    }

    private static func printSweep() throws {
        let plan = try SweepPlan.grid(
            parameters: [
                try SweepParameter(name: "learningRate", values: [0.001, 0.0003]),
                try SweepParameter(name: "clipRange", values: [0.1, 0.2]),
            ],
            seeds: [1, 2]
        )
        print("trials=\(plan.trials.count)")
        for trial in plan.trials.prefix(4) {
            print("\(trial.id)\tseed=\(trial.seed)\tparams=\(trial.parameters)")
        }
    }

    private static func printDashboard() throws {
        let points = try [0.1, 0.2, 0.15, 0.4, 0.6].enumerated().map {
            try MetricSeriesPoint(step: $0.offset, value: $0.element)
        }
        let dashboard = try TrainingDashboardSnapshot(
            title: "RLSwift Training",
            series: [try TrainingMetricSeries(name: "return", points: points)]
        )
        print(try dashboard.renderTextTable())
    }

    private static func printHelp() {
        print("usage: rl-swift [catalog|train|eval|experiment|checkpoint|sweep|protein|visualize]")
    }

    private static func value(after flag: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: flag), index + 1 < arguments.count else {
            return nil
        }
        return arguments[index + 1]
    }

    private static func defaultExperimentConfiguration() throws -> ExperimentConfiguration {
        try ExperimentConfiguration(
            experimentID: "lineworld-ppo-smoke",
            environmentID: .lineWorld,
            seed: 7,
            episodeCount: 8,
            maxEpisodeSteps: 16,
            ppoConfiguration: try PPOConfiguration(),
            vectorizationProfile: VectorizationProfile.serial(environmentCount: 1),
            checkpointDirectory: "checkpoints"
        )
    }

    private static func encodedJSON<Value: Encodable>(_ value: Value) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(value)
        return String(decoding: data, as: UTF8.self)
    }
}
