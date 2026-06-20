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
        #expect(try PPOAnnealingSchedule(finalMultiplier: 0.5).finalMultiplier == 0.5)
        #expect(throws: RLSwiftError.invalidWeight(0)) {
            _ = try PPOAnnealingSchedule(finalMultiplier: 0)
        }
        let schedule = try PPOAnnealingSchedule(finalMultiplier: 0.25)
        #expect(try schedule.multiplier(epoch: 0, totalEpochs: 3) == 1)
        #expect(try schedule.multiplier(epoch: 2, totalEpochs: 3) == 0.25)
        #expect(try schedule.multiplier(epoch: 0, totalEpochs: 1) == 0.25)
        #expect(throws: RLSwiftError.invalidHorizon(0)) {
            _ = try schedule.multiplier(epoch: 0, totalEpochs: 0)
        }
        #expect(throws: RLSwiftError.invalidSampleCount(-1)) {
            _ = try schedule.multiplier(epoch: -1, totalEpochs: 2)
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 3)) {
            _ = try schedule.multiplier(epoch: 2, totalEpochs: 2)
        }

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
        #expect(throws: RLSwiftError.invalidWeight(-0.1)) {
            _ = try PPOConfiguration(valueClipRange: -0.1)
        }
        #expect(throws: RLSwiftError.invalidWeight(0)) {
            _ = try PPOConfiguration(learningRate: 0)
        }
        #expect(throws: RLSwiftError.invalidWeight(0)) {
            _ = try PPOConfiguration(maximumGradientNorm: 0)
        }
        #expect(throws: RLSwiftError.invalidWeight(-1)) {
            _ = try PPOConfiguration(trajectoryPriorityAlpha: -1)
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

        let scheduled = try PPOConfiguration(
            entropyCoefficient: 0.02,
            learningRate: 0.1,
            learningRateSchedule: PPOAnnealingSchedule(finalMultiplier: 0.5),
            entropyCoefficientSchedule: PPOAnnealingSchedule(finalMultiplier: 0.25),
            epochs: 3
        )
        #expect(try scheduled.learningRate(epoch: 2) == 0.05)
        #expect(try scheduled.entropyCoefficient(epoch: 2) == 0.005)
        #expect(try scheduled.scheduled(epoch: 2).learningRate == 0.05)

        let unclippedValueSample = PPOClippedObjectiveSample(
            oldLogProbability: 0,
            newLogProbability: 0,
            advantage: 1,
            returnEstimate: 0,
            valueEstimate: 2,
            entropy: 0
        )
        #expect(unclippedValueSample.clippedValueEstimate(valueClipRange: 0.5) == nil)
        #expect(unclippedValueSample.valueLoss(valueClipRange: nil) == 2)
        let clippedValueSample = PPOClippedObjectiveSample(
            oldLogProbability: 0,
            newLogProbability: 0,
            advantage: 1,
            returnEstimate: 0,
            valueEstimate: 2,
            entropy: 0,
            oldValueEstimate: 0
        )
        #expect(clippedValueSample.clippedValueEstimate(valueClipRange: 0.5) == 0.5)
        #expect(clippedValueSample.valueLoss(valueClipRange: 0.5) == 2)

        let clipped = try PPOGradientClipper.clipped([3, 4], maximumNorm: 2)
        #expect(round4(clipped.summary.originalNorm) == 5)
        #expect(round4(clipped.summary.clippedNorm) == 2)
        #expect(clipped.values.map(round4) == [1.2, 1.6])
        #expect(try PPOGradientClipper.clipped([1, 0], maximumNorm: 2).summary.scale == 1)
        #expect(throws: RLSwiftError.invalidWeight(0)) {
            _ = try PPOGradientClipper.clipped([1], maximumNorm: 0)
        }
    }

    @Test func ppoActionDistributionsCoverDiscreteMultiDiscreteAndContinuousSpaces() throws {
        let categorical = PPOActionDistribution.categorical(logits: [0, 1])
        #expect(round4(try categorical.logProbability(of: .discrete(1))) == -0.3133)
        #expect(round4(try categorical.entropy()) == 0.5822)

        let multi = PPOActionDistribution.multiCategorical(logitsByDimension: [[0, 1], [1, 0]])
        #expect(round4(try multi.logProbability(of: .multiDiscrete([1, 0]))) == -0.6265)
        #expect(round4(try multi.entropy()) == 1.1644)

        let gaussian = PPOActionDistribution.diagonalGaussian(mean: [0, 1], logStandardDeviation: [0, 0])
        #expect(round4(try gaussian.logProbability(of: .continuous([0, 1]))) == -1.8379)
        #expect(round4(try gaussian.entropy()) == 2.8379)

        let prediction = try PPOPolicyValuePrediction(logits: [0, 1], valueEstimate: 0)
        #expect(round4(try prediction.logProbability(of: .discrete(1))) == -0.3133)
        #expect(prediction.actionDistribution == categorical)

        #expect(throws: RLSwiftError.emptyActionSpace) {
            _ = try PPOActionDistribution.categorical(logits: []).logProbability(of: .discrete(0))
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 3)) {
            _ = try categorical.logProbability(of: .discrete(2))
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 0)) {
            _ = try categorical.logProbability(of: .continuous([0]))
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try multi.logProbability(of: .multiDiscrete([0]))
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 3)) {
            _ = try multi.logProbability(of: .multiDiscrete([0, 2]))
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try PPOActionDistribution.multiCategorical(logitsByDimension: []).entropy()
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 0)) {
            _ = try PPOActionDistribution.diagonalGaussian(mean: [], logStandardDeviation: []).entropy()
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try PPOActionDistribution.diagonalGaussian(mean: [0, 1], logStandardDeviation: [0])
                .entropy()
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try gaussian.logProbability(of: .continuous([0]))
        }
    }

    @Test func neuralPPOTrainerRunsDenseActorCriticOptimizationLoop() throws {
        let configuration = try PPOConfiguration(
            clipRange: 0.2,
            valueLossCoefficient: 0.5,
            entropyCoefficient: 0.01,
            valueClipRange: 0.2,
            learningRate: 0.1,
            learningRateSchedule: PPOAnnealingSchedule(finalMultiplier: 0.5),
            entropyCoefficientSchedule: PPOAnnealingSchedule(finalMultiplier: 0.5),
            maximumGradientNorm: 1,
            trajectoryPriorityAlpha: 1,
            epochs: 2,
            minibatchSize: 2
        )
        let model = try DenseDiscreteActorCriticModel(
            observationDimension: 2,
            hiddenUnitCount: 3,
            actionCount: 2,
            seed: 99
        )
        var trainer = NeuralPPOTrainer(model: model, configuration: configuration)
        let samples = [
            try PPOTrainingSample(
                observation: [1, 0],
                actionIndex: 1,
                oldLogProbability: log(0.5),
                advantage: 1.2,
                returnEstimate: 1,
                oldValueEstimate: 0
            ),
            try PPOTrainingSample(
                observation: [0, 1],
                actionIndex: 0,
                oldLogProbability: log(0.4),
                advantage: -0.4,
                returnEstimate: -0.2,
                oldValueEstimate: 0
            ),
            try PPOTrainingSample(
                observation: [1, 1],
                actionIndex: 1,
                oldLogProbability: -10,
                advantage: 0.8,
                returnEstimate: 0.6,
                oldValueEstimate: 0
            ),
        ]

        let before = try trainer.model.prediction(for: [1, 0])
        let summary = try trainer.update(samples: samples)
        let after = try trainer.model.prediction(for: [1, 0])

        #expect(summary.sampleCount == 3)
        #expect(summary.epochCount == 2)
        #expect(summary.optimizerSteps.count == 4)
        #expect(summary.optimizerSteps.map(\.sampleCount) == [2, 1, 2, 1])
        #expect(summary.optimizerSteps.map(\.epoch) == [0, 0, 1, 1])
        #expect(summary.optimizerSteps.map(\.minibatchIndex) == [0, 1, 0, 1])
        #expect(summary.finalObjective != nil)
        #expect(before.probabilities != after.probabilities)

        let segmentA = try PPOTrajectorySegment(id: "a", samples: Array(samples.prefix(2)), priority: 1)
        let segmentB = try PPOTrajectorySegment(id: "b", samples: Array(samples.suffix(1)), priority: 4)
        let sampled = try PPOTrajectorySegmentSampler(
            segments: [segmentA, segmentB],
            priorityAlpha: configuration.trajectoryPriorityAlpha
        ).sample(count: 3, seed: 9)
        #expect(sampled.count == 3)
        let segmentSummary = try trainer.update(segments: [segmentA, segmentB], sampledSegmentCount: 1, seed: 1)
        #expect(segmentSummary.sampleCount > 0)
        let defaultSegmentSummary = try trainer.update(segments: [segmentA, segmentB], seed: 2)
        #expect(defaultSegmentSummary.sampleCount > 0)
    }

    @Test func neuralPPOPredictionModelAndTrainerValidationBranchesAreCovered() throws {
        let prediction = try PPOPolicyValuePrediction(logits: [1, 2], valueEstimate: 0.25)
        #expect(round4(prediction.probabilities.reduce(0, +)) == 1)
        #expect(round4(try prediction.logProbability(actionIndex: 1)) == -0.3133)
        #expect(round4(prediction.entropy) == 0.5822)
        #expect(throws: RLSwiftError.emptyActionSpace) {
            _ = try PPOPolicyValuePrediction(logits: [], valueEstimate: 0)
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 3)) {
            _ = try prediction.logProbability(actionIndex: 2)
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 0)) {
            _ = try prediction.logProbability(actionIndex: -1)
        }

        #expect(try PPOTrainingSample(
            observation: [1],
            actionIndex: 0,
            oldLogProbability: 0,
            advantage: 1,
            returnEstimate: 1
        ).actionIndex == 0)
        #expect(try PPOTrainingSample(
            observation: [1],
            action: .multiDiscrete([0, 1]),
            oldLogProbability: 0,
            advantage: 1,
            returnEstimate: 1
        ).actionIndex == -1)
        #expect(try PPOTrainingSample(
            observation: [1],
            action: .continuous([0.5]),
            oldLogProbability: 0,
            advantage: 1,
            returnEstimate: 1
        ).actionIndex == -1)
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 0)) {
            _ = try PPOTrainingSample(
                observation: [],
                actionIndex: 0,
                oldLogProbability: 0,
                advantage: 1,
                returnEstimate: 1
            )
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try PPOTrainingSample(
                observation: [1],
                action: .multiDiscrete([]),
                oldLogProbability: 0,
                advantage: 1,
                returnEstimate: 1
            )
        }
        #expect(throws: RLSwiftError.invalidCapacity(-1)) {
            _ = try PPOTrainingSample(
                observation: [1],
                action: .multiDiscrete([0, -1]),
                oldLogProbability: 0,
                advantage: 1,
                returnEstimate: 1
            )
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try PPOTrainingSample(
                observation: [1],
                action: .continuous([]),
                oldLogProbability: 0,
                advantage: 1,
                returnEstimate: 1
            )
        }
        #expect(throws: RLSwiftError.invalidCapacity(-1)) {
            _ = try PPOTrainingSample(
                observation: [1],
                actionIndex: -1,
                oldLogProbability: 0,
                advantage: 1,
                returnEstimate: 1
            )
        }

        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 0)) {
            _ = try DenseDiscreteActorCriticModel(observationDimension: 0, hiddenUnitCount: 1, actionCount: 1)
        }
        #expect(throws: RLSwiftError.invalidCapacity(0)) {
            _ = try DenseDiscreteActorCriticModel(observationDimension: 1, hiddenUnitCount: 0, actionCount: 1)
        }
        #expect(throws: RLSwiftError.emptyActionSpace) {
            _ = try DenseDiscreteActorCriticModel(observationDimension: 1, hiddenUnitCount: 1, actionCount: 0)
        }
        #expect(throws: RLSwiftError.invalidWeight(-0.1)) {
            _ = try DenseDiscreteActorCriticModel(
                observationDimension: 1,
                hiddenUnitCount: 1,
                actionCount: 1,
                weightScale: -0.1
            )
        }

        var model = try DenseDiscreteActorCriticModel(
            observationDimension: 1,
            hiddenUnitCount: 1,
            actionCount: 1,
            seed: 1,
            weightScale: 0
        )
        let configuration = try PPOConfiguration(epochs: 1, minibatchSize: 1)
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 2)) {
            _ = try model.prediction(for: [1, 2])
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try model.update(minibatch: [], configuration: configuration)
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 2)) {
            _ = try model.update(
                minibatch: [
                    try PPOTrainingSample(
                        observation: [1],
                        actionIndex: 1,
                        oldLogProbability: 0,
                        advantage: 1,
                        returnEstimate: 1
                    ),
                ],
                configuration: configuration
            )
        }
        var trainer = NeuralPPOTrainer(model: model, configuration: configuration)
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try trainer.update(samples: [])
        }
        let sample = try PPOTrainingSample(
            observation: [1],
            actionIndex: 0,
            oldLogProbability: 0,
            advantage: 1,
            returnEstimate: 1
        )
        #expect(throws: RLSwiftError.emptyIdentifier(name: "ppo.segmentID")) {
            _ = try PPOTrajectorySegment(id: "", samples: [sample], priority: 1)
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try PPOTrajectorySegment(id: "a", samples: [], priority: 1)
        }
        #expect(throws: RLSwiftError.invalidPriority(0)) {
            _ = try PPOTrajectorySegment(id: "a", samples: [sample], priority: 0)
        }
        let segment = try PPOTrajectorySegment(id: "a", samples: [sample], priority: 1)
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try PPOTrajectorySegmentSampler(segments: [], priorityAlpha: 0)
        }
        #expect(throws: RLSwiftError.invalidWeight(-1)) {
            _ = try PPOTrajectorySegmentSampler(segments: [segment], priorityAlpha: -1)
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try PPOTrajectorySegmentSampler(segments: [segment], priorityAlpha: 0).sample(count: 0, seed: 0)
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

    @Test func proteinStyleTunerExploresAndRefinesBoundedParameters() throws {
        let learningRate = try ProteinParameter(
            name: "learningRate",
            lowerBound: 1e-5,
            upperBound: 1e-3,
            scale: .logarithmic
        )
        let entropy = try ProteinParameter(name: "entropy", lowerBound: 0.0, upperBound: 0.1)
        #expect(round4(try learningRate.value(unit: 0.5)) == 0.0001)
        #expect(round4(try learningRate.unit(value: 0.0001)) == 0.5)
        #expect(try entropy.value(unit: 0.25) == 0.025)
        #expect(try entropy.unit(value: 0.025) == 0.25)

        let tuner = try ProteinTuner(parameters: [learningRate, entropy], seed: 3)
        let suggestions = try tuner.suggest(count: 2)
        #expect(suggestions.count == 2)
        #expect(suggestions[0].id == "protein-0")
        #expect(suggestions[0].parameters.keys.contains("learningRate"))

        let observationA = try ProteinObservation(suggestion: suggestions[0], score: 10, cost: 5)
        let observationB = try ProteinObservation(suggestion: suggestions[1], score: 11, cost: 4)
        let frontier = try ProteinTuner.paretoFrontier([observationA, observationB])
        #expect(frontier == [observationB])
        let refined = try tuner.suggest(completed: [observationA, observationB], count: 2)
        #expect(refined.map(\.id) == ["protein-2", "protein-3"])
        let partialSuggestion = try ProteinSuggestion(id: "partial", parameters: ["learningRate": 0.0001], seed: 4)
        let partialObservation = try ProteinObservation(suggestion: partialSuggestion, score: 9, cost: 1)
        let fallbackRefined = try tuner.suggest(completed: [partialObservation], count: 1)
        #expect(fallbackRefined[0].parameters.keys.contains("entropy"))
        let tieA = try ProteinObservation(suggestion: suggestions[0], score: 12, cost: 3)
        let tieB = try ProteinObservation(suggestion: suggestions[1], score: 12, cost: 3)
        #expect(try ProteinTuner.paretoFrontier([tieA, tieB]).count == 2)
        let tradeoffA = try ProteinObservation(suggestion: suggestions[0], score: 10, cost: 1)
        let tradeoffB = try ProteinObservation(suggestion: suggestions[1], score: 11, cost: 2)
        #expect(try ProteinTuner.paretoFrontier([tradeoffA, tradeoffB]).map(\.score) == [11, 10])

        #expect(throws: RLSwiftError.emptyIdentifier(name: "protein.parameter")) {
            _ = try ProteinParameter(name: "", lowerBound: 0, upperBound: 1)
        }
        #expect(throws: RLSwiftError.invalidBounds(index: 0, lower: 1, upper: 1)) {
            _ = try ProteinParameter(name: "bad", lowerBound: 1, upperBound: 1)
        }
        #expect(throws: RLSwiftError.invalidBounds(index: 0, lower: 0, upper: 1)) {
            _ = try ProteinParameter(name: "bad", lowerBound: 0, upperBound: 1, scale: .logarithmic)
        }
        #expect(throws: RLSwiftError.invalidProbability(1.5)) {
            _ = try entropy.value(unit: 1.5)
        }
        #expect(throws: RLSwiftError.invalidBounds(index: 0, lower: 0, upper: 0.1)) {
            _ = try entropy.unit(value: 0.2)
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "protein.suggestionID")) {
            _ = try ProteinSuggestion(id: "", parameters: ["x": 1], seed: 0)
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try ProteinSuggestion(id: "x", parameters: [:], seed: 0)
        }
        #expect(throws: RLSwiftError.invalidDuration(name: "protein.cost", value: -1)) {
            _ = try ProteinObservation(suggestion: suggestions[0], score: 0, cost: -1)
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try ProteinTuner(parameters: [])
        }
        #expect(throws: RLSwiftError.duplicateIdentifier("entropy")) {
            _ = try ProteinTuner(parameters: [entropy, entropy])
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try tuner.suggest(count: 0)
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try ProteinTuner.paretoFrontier([])
        }
    }

    @Test func experimentRecordsAndEvaluationSummariesSupportCLIWorkflows() throws {
        let configuration = try ExperimentConfiguration(
            experimentID: "lineworld-smoke",
            environmentID: .lineWorld,
            seed: 7,
            episodeCount: 3,
            maxEpisodeSteps: 8,
            ppoConfiguration: try PPOConfiguration(),
            vectorizationProfile: try VectorizationProfile.serial(environmentCount: 1),
            checkpointDirectory: "checkpoints"
        )
        let summary = try ExperimentEvaluator.evaluateLineWorldRightPolicy(episodes: 2, length: 3, maxSteps: 4)
        #expect(summary.episodeCount == 2)
        #expect(summary.meanReturn == 0.99)
        #expect(summary.successRate == 1)
        #expect(summary.meanEpisodeLength == 2)

        let metadata = try PolicyMetadata(policyID: "lineworld", version: "1.0.0", trainingRunID: configuration.experimentID)
        let manifest = try PolicyCheckpointManifest(
            checkpointID: "ckpt",
            policyMetadata: metadata,
            trainingStep: 1,
            artifactPath: "checkpoints/ckpt.bin"
        )
        let record = ExperimentCheckpointRecord(configuration: configuration, manifest: manifest, evaluationSummary: summary)
        #expect(record.configuration.experimentID == "lineworld-smoke")
        #expect(record.evaluationSummary == summary)

        #expect(throws: RLSwiftError.emptyIdentifier(name: "experimentID")) {
            _ = try ExperimentConfiguration(
                experimentID: "",
                environmentID: .lineWorld,
                seed: 0,
                episodeCount: 1,
                maxEpisodeSteps: 1,
                ppoConfiguration: try PPOConfiguration(),
                checkpointDirectory: "checkpoints"
            )
        }
        #expect(throws: RLSwiftError.invalidHorizon(0)) {
            _ = try ExperimentConfiguration(
                experimentID: "x",
                environmentID: .lineWorld,
                seed: 0,
                episodeCount: 0,
                maxEpisodeSteps: 1,
                ppoConfiguration: try PPOConfiguration(),
                checkpointDirectory: "checkpoints"
            )
        }
        #expect(throws: RLSwiftError.invalidHorizon(0)) {
            _ = try ExperimentConfiguration(
                experimentID: "x",
                environmentID: .lineWorld,
                seed: 0,
                episodeCount: 1,
                maxEpisodeSteps: 0,
                ppoConfiguration: try PPOConfiguration(),
                checkpointDirectory: "checkpoints"
            )
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "checkpointDirectory")) {
            _ = try ExperimentConfiguration(
                experimentID: "x",
                environmentID: .lineWorld,
                seed: 0,
                episodeCount: 1,
                maxEpisodeSteps: 1,
                ppoConfiguration: try PPOConfiguration(),
                checkpointDirectory: ""
            )
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try EvaluationSummary(episodeCount: 0, meanReturn: 0, successRate: 0, meanEpisodeLength: 0)
        }
        #expect(throws: RLSwiftError.invalidProbability(1.2)) {
            _ = try EvaluationSummary(episodeCount: 1, meanReturn: 0, successRate: 1.2, meanEpisodeLength: 0)
        }
        #expect(throws: RLSwiftError.invalidDuration(name: "meanEpisodeLength", value: -1)) {
            _ = try EvaluationSummary(episodeCount: 1, meanReturn: 0, successRate: 0, meanEpisodeLength: -1)
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try ExperimentEvaluator.evaluateLineWorldRightPolicy(episodes: 0)
        }
    }

    @Test func structuredTensorSchemasFlattenPufferStyleSpaces() throws {
        let pose = try StructuredTensorField(name: "pose", shape: [2])
        let sensors = try StructuredTensorField(name: "sensors", shape: [2, 2])
        let schema = try StructuredObservationSchema(fields: [pose, sensors])

        #expect(schema.flattenedCount == 6)
        #expect(try schema.range(for: "pose") == 0..<2)
        #expect(try schema.range(for: "sensors") == 2..<6)

        let flattened = try schema.flatten([
            "pose": [1, 2],
            "sensors": [3, 4, 5, 6],
        ])
        #expect(flattened == [1, 2, 3, 4, 5, 6])
        let unflattened = try schema.unflatten(flattened)
        #expect(unflattened["pose"] == [1, 2])
        #expect(unflattened["sensors"] == [3, 4, 5, 6])

        let actionSchema = try StructuredActionSchema(fields: [
            try StructuredTensorField(name: "move", shape: [1]),
            try StructuredTensorField(name: "grip", shape: [1]),
        ])
        #expect(try actionSchema.flatten(["move": [0.25], "grip": [1]]) == [0.25, 1])

        #expect(throws: RLSwiftError.emptyIdentifier(name: "tensor.field")) {
            _ = try StructuredTensorField(name: "", shape: [1])
        }
        #expect(throws: RLSwiftError.invalidCapacity(0)) {
            _ = try StructuredTensorField(name: "bad", shape: [])
        }
        #expect(throws: RLSwiftError.invalidCapacity(-1)) {
            _ = try StructuredTensorField(name: "bad", shape: [2, -1])
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try StructuredTensorSchema(fields: [])
        }
        #expect(throws: RLSwiftError.duplicateIdentifier("pose")) {
            _ = try StructuredTensorSchema(fields: [pose, pose])
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "missing")) {
            _ = try schema.range(for: "missing")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "sensors")) {
            _ = try schema.flatten(["pose": [1, 2]])
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 4, actual: 1)) {
            _ = try schema.flatten(["pose": [1, 2], "sensors": [3]])
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 6, actual: 2)) {
            _ = try schema.unflatten([1, 2])
        }
    }

    @Test func vectorizationProfilesDescribeSerialAsyncAndAcceleratedRollouts() throws {
        let serial = try VectorizationProfile.serial(environmentCount: 4, agentsPerEnvironment: 2)
        #expect(serial.backend == .serial)
        #expect(serial.totalAgentCount == 8)
        #expect(!serial.usesAccelerator)

        let threaded = try VectorizationProfile(
            backend: .threaded,
            environmentCount: 8,
            workerCount: 4,
            batchSize: 2,
            isAsynchronous: true,
            accelerator: "swift-concurrency"
        )
        #expect(threaded.isAsynchronous)
        #expect(threaded.workerCount == 4)

        let cuda = try VectorizationProfile(
            backend: .cuda,
            environmentCount: 16,
            workerCount: 8,
            batchSize: 4,
            accelerator: "cuda"
        )
        let tensorRT = try VectorizationProfile(
            backend: .tensorRT,
            environmentCount: 16,
            workerCount: 4,
            batchSize: 8,
            accelerator: "tensorrt"
        )
        #expect(cuda.usesAccelerator)
        #expect(tensorRT.usesAccelerator)

        #expect(throws: RLSwiftError.invalidCapacity(0)) {
            _ = try VectorizationProfile(backend: .serial, environmentCount: 0, batchSize: 1, accelerator: "cpu")
        }
        #expect(throws: RLSwiftError.invalidCapacity(0)) {
            _ = try VectorizationProfile(backend: .serial, environmentCount: 1, workerCount: 0, batchSize: 1, accelerator: "cpu")
        }
        #expect(throws: RLSwiftError.invalidCapacity(0)) {
            _ = try VectorizationProfile(backend: .serial, environmentCount: 1, batchSize: 0, accelerator: "cpu")
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 3)) {
            _ = try VectorizationProfile(backend: .serial, environmentCount: 2, batchSize: 3, accelerator: "cpu")
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 6, actual: 4)) {
            _ = try VectorizationProfile(backend: .serial, environmentCount: 6, batchSize: 4, accelerator: "cpu")
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 6, actual: 4)) {
            _ = try VectorizationProfile(backend: .multiprocessing, environmentCount: 6, workerCount: 4, batchSize: 3, accelerator: "cpu")
        }
        #expect(throws: RLSwiftError.invalidCapacity(0)) {
            _ = try VectorizationProfile(backend: .serial, environmentCount: 1, batchSize: 1, agentsPerEnvironment: 0, accelerator: "cpu")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "accelerator")) {
            _ = try VectorizationProfile(backend: .serial, environmentCount: 1, batchSize: 1, accelerator: "")
        }
    }

    @Test func asyncVectorizedRunnerStepsEnvironmentsInWorkerOrder() async throws {
        let runner = try AsyncVectorizedEnvironmentRunner(environments: [
            try LineWorldEnvironment(length: 3, maxSteps: 2),
            try LineWorldEnvironment(length: 3, maxSteps: 2),
        ])
        #expect(runner.count == 2)
        #expect(runner.profile.isAsynchronous)

        let resets = await runner.resetAll()
        #expect(resets.map(\.position) == [0, 0])

        let first = try await runner.stepAll([.right, .left])
        #expect(first.observations.map(\.position) == [1, 0])
        #expect(first.rewards == [-0.01, -0.01])
        #expect(first.terminations == [.continuing, .continuing])
        #expect(first.terminalCount == 0)

        let second = try await runner.stepAll([.right, .right])
        #expect(second.observations.map(\.position) == [2, 1])
        #expect(second.terminalCount == 2)

        let unordered = try AsyncVectorizedStepBatch(results: [
            VectorizedEnvironmentStep(
                environmentIndex: 1,
                result: StepResult(observation: LineWorldObservation(position: 1, goal: 2, stepIndex: 1), reward: 0, isTerminal: false)
            ),
            VectorizedEnvironmentStep(
                environmentIndex: 0,
                result: StepResult(observation: LineWorldObservation(position: 0, goal: 2, stepIndex: 1), reward: 1, isTerminal: true)
            ),
        ])
        #expect(unordered.observations.map(\.position) == [0, 1])

        do {
            _ = try await runner.stepAll([.right])
            Issue.record("Expected action count mismatch")
        } catch let error as RLSwiftError {
            #expect(error == .dimensionMismatch(expected: 2, actual: 1))
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try AsyncVectorizedEnvironmentRunner<LineWorldEnvironment>(environments: [])
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try AsyncVectorizedStepBatch<LineWorldObservation>(results: [])
        }
    }

    @Test func minGRUReferenceCellAndRecurrentActorCriticCarryState() throws {
        let cell = try MinGRUCell(
            inputDimension: 2,
            hiddenDimension: 1,
            updateWeights: [0, 0],
            updateBiases: [0],
            candidateWeights: [1, -1],
            candidateBiases: [0]
        )
        let state = try cell.initialState()
        #expect(state.hidden == [0])

        let next = try cell.step(input: [2, 0], state: state)
        #expect(next.hidden.map(round4) == [0.482])
        let carried = try cell.step(input: [2, 0], state: next)
        #expect(carried.hidden.map(round4) == [0.723])

        let model = try MinGRUDiscreteActorCriticModel(
            cell: cell,
            actionCount: 2,
            actorWeights: [1, -1],
            actorBiases: [0, 0],
            valueWeights: [2],
            valueBias: 0
        )
        let prediction = try model.prediction(for: [2, 0], state: try model.initialState())
        #expect(prediction.policyValuePrediction.logits.map(round4) == [0.482, -0.482])
        #expect(round4(prediction.policyValuePrediction.valueEstimate) == 0.964)
        #expect(prediction.nextState == next)

        #expect(throws: RLSwiftError.invalidCapacity(0)) {
            _ = try MinGRUState(hidden: [])
        }
        #expect(throws: RLSwiftError.invalidCapacity(0)) {
            _ = try MinGRUState.zeros(hiddenDimension: 0)
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 0)) {
            _ = try MinGRUCell(
                inputDimension: 0,
                hiddenDimension: 1,
                updateWeights: [],
                updateBiases: [0],
                candidateWeights: [],
                candidateBiases: [0]
            )
        }
        #expect(throws: RLSwiftError.invalidCapacity(0)) {
            _ = try MinGRUCell(
                inputDimension: 1,
                hiddenDimension: 0,
                updateWeights: [],
                updateBiases: [],
                candidateWeights: [],
                candidateBiases: []
            )
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try MinGRUCell(
                inputDimension: 2,
                hiddenDimension: 1,
                updateWeights: [0],
                updateBiases: [0],
                candidateWeights: [0, 0],
                candidateBiases: [0]
            )
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try MinGRUCell(
                inputDimension: 2,
                hiddenDimension: 1,
                updateWeights: [0, 0],
                updateBiases: [0],
                candidateWeights: [0],
                candidateBiases: [0]
            )
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 0)) {
            _ = try MinGRUCell(
                inputDimension: 1,
                hiddenDimension: 1,
                updateWeights: [0],
                updateBiases: [],
                candidateWeights: [0],
                candidateBiases: [0]
            )
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 0)) {
            _ = try MinGRUCell(
                inputDimension: 1,
                hiddenDimension: 1,
                updateWeights: [0],
                updateBiases: [0],
                candidateWeights: [0],
                candidateBiases: []
            )
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try cell.step(input: [1], state: state)
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 2)) {
            _ = try cell.step(input: [1, 1], state: try MinGRUState(hidden: [0, 0]))
        }
        #expect(throws: RLSwiftError.emptyActionSpace) {
            _ = try MinGRUDiscreteActorCriticModel(
                cell: cell,
                actionCount: 0,
                actorWeights: [],
                actorBiases: [],
                valueWeights: [0],
                valueBias: 0
            )
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try MinGRUDiscreteActorCriticModel(
                cell: cell,
                actionCount: 2,
                actorWeights: [0],
                actorBiases: [0, 0],
                valueWeights: [0],
                valueBias: 0
            )
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try MinGRUDiscreteActorCriticModel(
                cell: cell,
                actionCount: 2,
                actorWeights: [0, 0],
                actorBiases: [0],
                valueWeights: [0],
                valueBias: 0
            )
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 0)) {
            _ = try MinGRUDiscreteActorCriticModel(
                cell: cell,
                actionCount: 2,
                actorWeights: [0, 0],
                actorBiases: [0, 0],
                valueWeights: [],
                valueBias: 0
            )
        }
    }

    @Test func checkpointManifestsAndSelfPlayPoolsCoverPufferTrainingBookkeeping() throws {
        let metadata = try PolicyMetadata(policyID: "lineworld", version: "1.0.0", trainingRunID: "run-1")
        let profile = try VectorizationProfile.serial(environmentCount: 2)
        let manifest = try PolicyCheckpointManifest(
            checkpointID: "ckpt-0001",
            policyMetadata: metadata,
            trainingStep: 128,
            artifactPath: "checkpoints/ckpt-0001.bin",
            metrics: ["return": 1.0],
            vectorizationProfile: profile
        )
        #expect(manifest.policyMetadata.policyID == "lineworld")
        #expect(manifest.vectorizationProfile == profile)

        #expect(throws: RLSwiftError.emptyIdentifier(name: "checkpointID")) {
            _ = try PolicyCheckpointManifest(
                checkpointID: "",
                policyMetadata: metadata,
                trainingStep: 0,
                artifactPath: "path"
            )
        }
        #expect(throws: RLSwiftError.invalidSampleCount(-1)) {
            _ = try PolicyCheckpointManifest(
                checkpointID: "id",
                policyMetadata: metadata,
                trainingStep: -1,
                artifactPath: "path"
            )
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "checkpoint.artifactPath")) {
            _ = try PolicyCheckpointManifest(
                checkpointID: "id",
                policyMetadata: metadata,
                trainingStep: 0,
                artifactPath: ""
            )
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "checkpoint.metric")) {
            _ = try PolicyCheckpointManifest(
                checkpointID: "id",
                policyMetadata: metadata,
                trainingStep: 0,
                artifactPath: "path",
                metrics: ["": 1]
            )
        }

        let opponents = [
            try SelfPlayOpponent(id: "a", checkpointReference: "a.bin"),
            try SelfPlayOpponent(id: "b", checkpointReference: "b.bin"),
            try SelfPlayOpponent(id: "c", checkpointReference: "c.bin"),
            try SelfPlayOpponent(id: "d", checkpointReference: "d.bin"),
        ]
        var pool = try SelfPlayOpponentPool(maxSize: 3, opponents: opponents)
        #expect(pool.opponents.map(\.id) == ["b", "c", "d"])
        #expect(try pool.sampledOpponent(seed: 1).id != "")

        let update = try pool.record(primaryScore: 1, against: "b", kFactor: 16)
        #expect(update.primaryRating == 8)
        #expect(update.opponentRating == -8)
        #expect(pool.primaryRating == 8)
        #expect(pool.opponents[0].gamesPlayed == 1)

        var largePool = try SelfPlayOpponentPool(maxSize: 8)
        for index in 0..<7 {
            try largePool.add(try SelfPlayOpponent(id: "o\(index)", checkpointReference: "o\(index).bin"))
        }
        let sampled = try largePool.sampledOpponent(seed: 3)
        #expect(["o0", "o1"].contains(sampled.id))

        let emptyPool = try SelfPlayOpponentPool(maxSize: 2)
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try emptyPool.sampledOpponent(seed: 1)
        }
        #expect(throws: RLSwiftError.invalidCapacity(0)) {
            _ = try SelfPlayOpponentPool(maxSize: 0)
        }
        #expect(throws: RLSwiftError.duplicateIdentifier("a")) {
            _ = try SelfPlayOpponentPool(maxSize: 4, opponents: [opponents[0], opponents[0]])
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "missing")) {
            _ = try pool.record(primaryScore: 0.5, against: "missing")
        }
        #expect(throws: RLSwiftError.invalidProbability(1.1)) {
            _ = try SelfPlayOpponentPool.updatedRatings(primaryRating: 0, opponentRating: 0, primaryScore: 1.1)
        }
        #expect(throws: RLSwiftError.invalidWeight(-1)) {
            _ = try SelfPlayOpponentPool.updatedRatings(primaryRating: 0, opponentRating: 0, primaryScore: 0.5, kFactor: -1)
        }

        #expect(try SelfPlayOpponent(id: "ok", checkpointReference: "ok.bin", gamesPlayed: 0).id == "ok")
        #expect(throws: RLSwiftError.emptyIdentifier(name: "selfplay.opponentID")) {
            _ = try SelfPlayOpponent(id: "", checkpointReference: "ok.bin")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "selfplay.checkpointReference")) {
            _ = try SelfPlayOpponent(id: "ok", checkpointReference: "")
        }
        #expect(throws: RLSwiftError.invalidSampleCount(-1)) {
            _ = try SelfPlayOpponent(id: "ok", checkpointReference: "ok.bin", gamesPlayed: -1)
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
