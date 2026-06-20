import Foundation
import Testing
@testable import RLSwift

@Suite struct PufferParityTests {
    @Test func ppoComputesAdvantagesAndClippedObjective() throws {
        let configuration = try PPOConfiguration(
            discount: 0.9,
            gaeLambda: 0.8,
            clipRange: 0.2,
            valueLossCoefficient: 0.5,
            entropyCoefficient: 0.01,
            learningRate: 0.001,
            epochs: 2,
            minibatchSize: 4
        )
        let steps = [
            PPOTrajectoryStep(reward: 1, valueEstimate: 0.5, logProbability: -0.2, entropy: 0.3),
            PPOTrajectoryStep(
                reward: 2,
                valueEstimate: 0.25,
                logProbability: -0.1,
                entropy: 0.2,
                termination: .terminated(reason: "goal")
            ),
        ]

        let batch = try PPOAdvantageEstimator.generalizedAdvantageEstimate(
            steps: steps,
            lastValue: 10,
            configuration: configuration
        )
        #expect(batch.advantages.map(round4) == [1.985, 1.75])
        #expect(batch.returns.map(round4) == [2.485, 2])

        let sampleA = PPOClippedObjectiveSample(
            oldLogProbability: 0,
            newLogProbability: log(1.5),
            advantage: 2,
            returnEstimate: 3,
            valueEstimate: 1,
            entropy: 0.1
        )
        let sampleB = PPOClippedObjectiveSample(
            oldLogProbability: 0,
            newLogProbability: log(0.9),
            advantage: -1,
            returnEstimate: 0,
            valueEstimate: 1,
            entropy: 0.3
        )
        #expect(round4(sampleA.probabilityRatio) == 1.5)
        #expect(round4(sampleA.clippedRatio(clipRange: configuration.clipRange)) == 1.2)

        let objective = try PPOClippedObjective.evaluate(samples: [sampleA, sampleB], configuration: configuration)
        #expect(round4(objective.policyLoss) == -0.75)
        #expect(round4(objective.valueLoss) == 1.25)
        #expect(round4(objective.entropyBonus) == 0.2)
        #expect(round4(objective.totalLoss) == -0.127)
        #expect(round4(objective.meanApproximateKL) == -0.1501)
        #expect(round4(objective.clippedFraction) == 0.5)

        #expect(try PPOAdvantageBatch(advantages: [1], returns: [2]).returns == [2])
    }

    @Test func ppoValidationBranchesAreCovered() throws {
        #expect(throws: RLSwiftError.invalidProbability(1.1)) {
            _ = try PPOConfiguration(discount: 1.1)
        }
        #expect(throws: RLSwiftError.invalidProbability(-0.1)) {
            _ = try PPOConfiguration(gaeLambda: -0.1)
        }
        #expect(throws: RLSwiftError.invalidWeight(-0.2)) {
            _ = try PPOConfiguration(clipRange: -0.2)
        }
        #expect(throws: RLSwiftError.invalidWeight(-0.5)) {
            _ = try PPOConfiguration(valueLossCoefficient: -0.5)
        }
        #expect(throws: RLSwiftError.invalidWeight(-0.01)) {
            _ = try PPOConfiguration(entropyCoefficient: -0.01)
        }
        #expect(throws: RLSwiftError.invalidWeight(0)) {
            _ = try PPOConfiguration(learningRate: 0)
        }
        #expect(throws: RLSwiftError.invalidHorizon(0)) {
            _ = try PPOConfiguration(epochs: 0)
        }
        #expect(throws: RLSwiftError.invalidCapacity(0)) {
            _ = try PPOConfiguration(minibatchSize: 0)
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try PPOAdvantageBatch(advantages: [1, 2], returns: [1])
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try PPOAdvantageEstimator.generalizedAdvantageEstimate(
                steps: [],
                lastValue: 0,
                configuration: try PPOConfiguration()
            )
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try PPOClippedObjective.evaluate(samples: [], configuration: try PPOConfiguration())
        }
    }

    @Test func throughputMeterReportsTrainingRates() throws {
        var meter = ThroughputMeter()
        try meter.record(steps: 10, samples: 40)
        try meter.record(steps: 5, samples: 20)

        #expect(meter.stepCount == 15)
        #expect(meter.sampleCount == 60)

        let report = try meter.report(elapsedSeconds: 3, environmentCount: 5, accelerator: "mlx")
        #expect(report.stepCount == 15)
        #expect(report.sampleCount == 60)
        #expect(report.stepsPerSecond == 5)
        #expect(report.samplesPerSecond == 20)
        #expect(report.samplesPerEnvironment == 12)

        #expect(throws: RLSwiftError.invalidSampleCount(-1)) {
            try meter.record(steps: -1, samples: 0)
        }
        #expect(throws: RLSwiftError.invalidSampleCount(-2)) {
            try meter.record(steps: 0, samples: -2)
        }
        #expect(throws: RLSwiftError.invalidSampleCount(-1)) {
            _ = try TrainingThroughputReport(
                stepCount: -1,
                sampleCount: 0,
                elapsedSeconds: 1,
                environmentCount: 1,
                accelerator: "cpu"
            )
        }
        #expect(throws: RLSwiftError.invalidSampleCount(-3)) {
            _ = try TrainingThroughputReport(
                stepCount: 0,
                sampleCount: -3,
                elapsedSeconds: 1,
                environmentCount: 1,
                accelerator: "cpu"
            )
        }
        #expect(throws: RLSwiftError.invalidDuration(name: "elapsedSeconds", value: 0)) {
            _ = try TrainingThroughputReport(
                stepCount: 0,
                sampleCount: 0,
                elapsedSeconds: 0,
                environmentCount: 1,
                accelerator: "cpu"
            )
        }
        #expect(throws: RLSwiftError.invalidCapacity(0)) {
            _ = try TrainingThroughputReport(
                stepCount: 0,
                sampleCount: 0,
                elapsedSeconds: 1,
                environmentCount: 0,
                accelerator: "cpu"
            )
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "accelerator")) {
            _ = try TrainingThroughputReport(
                stepCount: 0,
                sampleCount: 0,
                elapsedSeconds: 1,
                environmentCount: 1,
                accelerator: ""
            )
        }
    }

    @Test func nativeKernelPlansDescribeSwiftMLXCUDAAndTensorRTPaths() throws {
        let cuda = try NativeKernelPlan.cudaPPO()
        let swift = try NativeKernelPlan.swiftReference()
        let mlx = try NativeKernelPlan(
            backend: .mlx,
            operations: [.normalizeObservations],
            precision: "fp32",
            usesStaticMemory: true,
            usesGraphCapture: false
        )
        let tensorRT = try NativeKernelPlan(
            backend: .tensorRT,
            operations: [.sampleActions],
            precision: "fp16",
            usesStaticMemory: true,
            usesGraphCapture: true
        )

        #expect(cuda.backend == .cuda)
        #expect(cuda.operations.contains(.recurrentPolicyForward))
        #expect(swift.backend == .swiftCPU)
        #expect(swift.precision == "fp64")
        #expect(mlx.operations == [.normalizeObservations])
        #expect(tensorRT.usesGraphCapture)

        #expect(throws: RLSwiftError.invalidCapacity(0)) {
            _ = try NativeKernelPlan(
                backend: .cuda,
                operations: [],
                precision: "bf16",
                usesStaticMemory: true,
                usesGraphCapture: true
            )
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "precision")) {
            _ = try NativeKernelPlan(
                backend: .cuda,
                operations: [.clippedPPOObjective],
                precision: "",
                usesStaticMemory: true,
                usesGraphCapture: true
            )
        }
    }

    @Test func builtInEnvironmentCatalogAndValidationBranchesAreCovered() throws {
        let entries = try BuiltInEnvironmentCatalog.allEntries()
        #expect(entries.map(\.id) == [.lineWorld, .binaryBandit, .matrixGame])
        #expect(try BuiltInEnvironmentCatalog.entry(for: .lineWorld).displayName == "LineWorld")
        #expect(try BuiltInEnvironmentCatalog.entry(for: .binaryBandit).displayName == "BinaryBandit")
        #expect(try BuiltInEnvironmentCatalog.entry(for: .matrixGame).supportsMultiAgent)

        #expect(throws: RLSwiftError.emptyIdentifier(name: "environment.displayName")) {
            _ = try EnvironmentCatalogEntry(
                id: .lineWorld,
                displayName: "",
                observationSpace: "obs",
                actionSpace: "act",
                defaultMaxSteps: 1,
                supportsMultiAgent: false,
                tags: []
            )
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "environment.observationSpace")) {
            _ = try EnvironmentCatalogEntry(
                id: .lineWorld,
                displayName: "name",
                observationSpace: "",
                actionSpace: "act",
                defaultMaxSteps: 1,
                supportsMultiAgent: false,
                tags: []
            )
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "environment.actionSpace")) {
            _ = try EnvironmentCatalogEntry(
                id: .lineWorld,
                displayName: "name",
                observationSpace: "obs",
                actionSpace: "",
                defaultMaxSteps: 1,
                supportsMultiAgent: false,
                tags: []
            )
        }
        #expect(throws: RLSwiftError.invalidHorizon(0)) {
            _ = try EnvironmentCatalogEntry(
                id: .lineWorld,
                displayName: "name",
                observationSpace: "obs",
                actionSpace: "act",
                defaultMaxSteps: 0,
                supportsMultiAgent: false,
                tags: []
            )
        }
    }

    @Test func lineWorldAndBanditRunAsBuiltInE2EEnvironments() throws {
        var line = try LineWorldEnvironment(length: 3, maxSteps: 2)
        #expect(line.observation == LineWorldObservation(position: 0, goal: 2, stepIndex: 0))
        #expect(line.reset() == LineWorldObservation(position: 0, goal: 2, stepIndex: 0))
        let left = try line.step(.left)
        #expect(left.observation.position == 0)
        #expect(!left.isTerminal)
        let goal = try line.step(.right)
        #expect(goal.observation.position == 1)
        #expect(goal.termination.isTruncated)

        var goalLine = try LineWorldEnvironment(length: 2, maxSteps: 4)
        _ = goalLine.reset()
        let terminal = try goalLine.step(.right)
        #expect(terminal.reward == 1)
        #expect(terminal.termination.reason == "goal")

        #expect(throws: RLSwiftError.invalidCapacity(1)) {
            _ = try LineWorldEnvironment(length: 1)
        }
        #expect(throws: RLSwiftError.invalidHorizon(0)) {
            _ = try LineWorldEnvironment(maxSteps: 0)
        }

        var bandit = try BinaryBanditEnvironment(rewardingAction: .optionA, maxPulls: 2)
        #expect(bandit.observation == BinaryBanditObservation(pullCount: 0))
        #expect(bandit.reset() == BinaryBanditObservation(pullCount: 0))
        let rewarded = try bandit.step(.optionA)
        #expect(rewarded.reward == 1)
        #expect(!rewarded.isTerminal)
        let unrewarded = try bandit.step(.optionB)
        #expect(unrewarded.reward == 0)
        #expect(unrewarded.termination.isTruncated)
        #expect(throws: RLSwiftError.invalidHorizon(0)) {
            _ = try BinaryBanditEnvironment(maxPulls: 0)
        }
    }

    @Test func multiAgentMatrixGameCoversGameScaleBranches() throws {
        var game = try MatrixGameEnvironment(agentIDs: ["a", "b"], maxRounds: 4)
        #expect(game.agentIDs == ["a", "b"])
        #expect(game.reset() == ["a": MatrixGameObservation(round: 0), "b": MatrixGameObservation(round: 0)])

        let bothCooperate = try game.step(["a": .cooperate, "b": .cooperate])
        #expect(bothCooperate.rewards == ["a": 2, "b": 2])
        #expect(!bothCooperate.allAgentsDone)
        let bothDefect = try game.step(["a": .defect, "b": .defect])
        #expect(bothDefect.rewards == ["a": 0, "b": 0])
        let firstDefects = try game.step(["a": .defect, "b": .cooperate])
        #expect(firstDefects.rewards == ["a": 3, "b": -1])
        let secondDefects = try game.step(["a": .cooperate, "b": .defect])
        #expect(secondDefects.rewards == ["a": -1, "b": 3])
        #expect(secondDefects.allAgentsDone)

        let explicit = MultiAgentStepResult(
            observations: ["a": MatrixGameObservation(round: 1)],
            rewards: ["a": 1],
            terminations: ["a": .terminated(reason: "done")],
            info: ["a": ["policy": "shared"]]
        )
        #expect(explicit.allAgentsDone)
        #expect(explicit.info["a"] == ["policy": "shared"])
        let empty = MultiAgentStepResult<MatrixGameObservation>(observations: [:], rewards: [:], terminations: [:])
        #expect(!empty.allAgentsDone)

        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try MatrixGameEnvironment(agentIDs: ["a"])
        }
        #expect(throws: RLSwiftError.duplicateIdentifier("agentID")) {
            _ = try MatrixGameEnvironment(agentIDs: ["a", "a"])
        }
        #expect(throws: RLSwiftError.invalidHorizon(0)) {
            _ = try MatrixGameEnvironment(maxRounds: 0)
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try game.step(["a": .cooperate])
        }
    }

    @Test func sweepsAndTuningBuildGridAndParetoFrontier() throws {
        let learningRate = try SweepParameter(name: "lr", values: [0.1, 0.01])
        let clip = try SweepParameter(name: "clip", values: [0.2])
        let plan = try SweepPlan.grid(parameters: [learningRate, clip], seeds: [7, 9])
        #expect(plan.trials.count == 4)
        #expect(plan.trials[0].parameters == ["lr": 0.1, "clip": 0.2])
        #expect(try SweepPlan(trials: [plan.trials[0]]).trials[0].id == "seed-7-trial-0")

        let resultA = try SweepResult(trial: plan.trials[0], score: 10, cost: 5)
        let resultB = try SweepResult(trial: plan.trials[1], score: 11, cost: 5)
        let resultC = try SweepResult(trial: plan.trials[2], score: 11, cost: 4)
        let frontier = try SweepTuner.paretoFrontier([resultA, resultB, resultC])
        #expect(frontier == [resultC])

        let cheapTrial = try SweepTrial(id: "cheap", parameters: ["lr": 0.1], seed: 1)
        let fastTrial = try SweepTrial(id: "fast", parameters: ["lr": 0.2], seed: 1)
        let fastTieTrial = try SweepTrial(id: "fast-tie", parameters: ["lr": 0.3], seed: 1)
        let cheap = try SweepResult(trial: cheapTrial, score: 10, cost: 1)
        let fast = try SweepResult(trial: fastTrial, score: 11, cost: 2)
        let fastTie = try SweepResult(trial: fastTieTrial, score: 11, cost: 2)
        let tradeoffFrontier = try SweepTuner.paretoFrontier([cheap, fast, fastTie])
        #expect(tradeoffFrontier.map(\.trial.id) == ["fast", "fast-tie", "cheap"])

        #expect(throws: RLSwiftError.emptyIdentifier(name: "sweep.parameter")) {
            _ = try SweepParameter(name: "", values: [1])
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try SweepParameter(name: "x", values: [])
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "sweep.trialID")) {
            _ = try SweepTrial(id: "", parameters: ["x": 1], seed: 0)
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try SweepTrial(id: "t", parameters: [:], seed: 0)
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try SweepPlan(trials: [])
        }
        #expect(throws: RLSwiftError.duplicateIdentifier(plan.trials[0].id)) {
            _ = try SweepPlan(trials: [plan.trials[0], plan.trials[0]])
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try SweepPlan.grid(parameters: [], seeds: [1])
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try SweepPlan.grid(parameters: [learningRate], seeds: [])
        }
        #expect(throws: RLSwiftError.duplicateIdentifier("lr")) {
            _ = try SweepPlan.grid(parameters: [learningRate, learningRate], seeds: [1])
        }
        #expect(throws: RLSwiftError.invalidDuration(name: "sweep.cost", value: -1)) {
            _ = try SweepResult(trial: plan.trials[0], score: 0, cost: -1)
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try SweepTuner.paretoFrontier([])
        }
    }

    @Test func visualizationRendersLocalDashboardSnapshots() throws {
        let emptySeries = try TrainingMetricSeries(name: "empty", points: [])
        #expect(emptySeries.latestValue == nil)
        #expect(try emptySeries.sparkline() == "")

        let flat = try TrainingMetricSeries(
            name: "flat",
            points: [
                try MetricSeriesPoint(step: 0, value: 2),
                try MetricSeriesPoint(step: 1, value: 2),
            ]
        )
        #expect(try flat.sparkline() == "++")

        let improving = try TrainingMetricSeries(
            name: "return",
            points: [
                try MetricSeriesPoint(step: 0, value: 0),
                try MetricSeriesPoint(step: 1, value: 0.5),
                try MetricSeriesPoint(step: 2, value: 1),
            ]
        )
        #expect(improving.latestValue == 1)
        #expect(try improving.sparkline(width: 2) == " @")

        let dashboard = try TrainingDashboardSnapshot(title: "Training", series: [improving])
        let table = try dashboard.renderTextTable()
        #expect(table.contains("Training"))
        #expect(table.contains("return 1.0000"))

        let emptyDashboard = try TrainingDashboardSnapshot(title: "Empty", series: [emptySeries])
        let emptyTable = try emptyDashboard.renderTextTable()
        #expect(emptyTable.contains("empty 0.0000"))

        #expect(throws: RLSwiftError.invalidSampleCount(-1)) {
            _ = try MetricSeriesPoint(step: -1, value: 0)
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "metric.name")) {
            _ = try TrainingMetricSeries(name: "", points: [])
        }
        #expect(throws: RLSwiftError.invalidCapacity(0)) {
            _ = try improving.sparkline(width: 0)
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "dashboard.title")) {
            _ = try TrainingDashboardSnapshot(title: "", series: [improving])
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try TrainingDashboardSnapshot(title: "Training", series: [])
        }
    }
}

private func round4(_ value: Double) -> Double {
    (value * 10_000).rounded() / 10_000
}
