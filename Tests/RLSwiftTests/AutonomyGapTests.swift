import Foundation
import Testing
@testable import RLSwift

@Suite struct AutonomyGapTests {
    @Test func modelIOContractPinsObservationAndActionOrder() throws {
        let normalization = try NormalizationSnapshot(count: 10, mean: [1, 0, 0.5, 2], variance: [1, 4, 0.25, 1])
        let bindings = try TensorRTBindingNames(inputName: "obs", outputName: "act", profileName: "fast")
        let metadata = try PolicyMetadata(policyID: "reach-policy", version: "1.0.0", trainingRunID: "run-7")
        let contract = try ModelIOContract(
            contractVersion: "1",
            observationFeatures: [
                try ObservationFeature(name: "force", index: 3, component: .sensor(key: "force"), unit: "N"),
                try ObservationFeature(name: "q0", index: 0, component: .jointPosition(index: 0), unit: "rad"),
                try ObservationFeature(name: "dq0", index: 1, component: .jointVelocity(index: 0), unit: "rad/s"),
                try ObservationFeature(name: "ee-x", index: 2, component: .endEffectorPose(index: 0), unit: "m"),
            ],
            normalization: normalization,
            actionSpecifications: [
                try ActionSpecification(name: "joint-0", index: 0, unit: "rad/s", lowerBound: -1, upperBound: 1),
                try ActionSpecification(name: "joint-1", index: 1, unit: "rad/s", lowerBound: -0.5, upperBound: 0.5),
            ],
            actionMode: .velocity,
            tensorRTBindings: bindings,
            policyMetadata: metadata
        )
        let observation = try RobotObservation(
            jointPositions: [2],
            jointVelocities: [4],
            endEffectorPose: [1],
            sensorReadings: ["force": 5]
        )

        #expect(try contract.encode(observation, applyNormalization: false) == [2, 4, 1, 5])
        let normalized = try contract.encode(observation)
        #expect(normalized.map { round($0 * 100) / 100 } == [1, 2, 1, 3])
        #expect(try contract.decodeAction([2, -2]) == RobotAction(commands: [1, -0.5], mode: .velocity))
        #expect(try contract.actionSpace().contains([0.2, 0.3]))
        #expect(contract.tensorRTBindings?.profileName == "fast")

        let encoded = try JSONEncoder().encode(contract)
        let decoded = try JSONDecoder().decode(ModelIOContract.self, from: encoded)
        #expect(decoded == contract)
    }

    @Test func modelIOContractValidatesDuplicateAndMissingFields() throws {
        let metadata = try PolicyMetadata(policyID: "p", version: "v")
        let feature = try ObservationFeature(name: "q0", index: 0, component: .jointPosition(index: 0))
        let action = try ActionSpecification(name: "a0", index: 0, lowerBound: -1, upperBound: 1)

        #expect(throws: RLSwiftError.duplicateIdentifier("q0")) {
            _ = try ModelIOContract(
                contractVersion: "1",
                observationFeatures: [feature, feature],
                actionSpecifications: [action],
                actionMode: .position,
                policyMetadata: metadata
            )
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "policyID")) {
            _ = try PolicyMetadata(policyID: "", version: "v")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "missing")) {
            _ = try RobotObservationComponent.sensor(key: "missing").value(
                from: try RobotObservation(jointPositions: [], jointVelocities: [])
            )
        }
    }

    @Test func hardwareSafetySupervisorHaltsAndShieldsOutsidePolicy() throws {
        let space = try ContinuousBoxSpace(lowerBounds: [-1, -1], upperBounds: [1, 1])
        let envelope = try RobotSafetyEnvelope(commandSpace: space, maximumDelta: [0.25, 0.25])
        let failsafe = RobotAction(commands: [0, 0], mode: .velocity)
        let supervisor = try HardwareSafetySupervisor(
            envelope: envelope,
            failsafeAction: failsafe,
            maximumSensorAge: 0.1,
            maximumClosedLoopLatency: 0.2
        )
        let previous = RobotAction(commands: [0, 0], mode: .velocity)
        let requested = RobotAction(commands: [0.8, -2], mode: .velocity)
        let nominal = try supervisor.assess(
            SafetySupervisorInput(
                requestedAction: requested,
                previousAction: previous,
                timing: try ControlTiming(stepIndex: 1, deltaTime: 0.02, sensorAge: 0.01, actionLatency: 0.02)
            )
        )

        #expect(!nominal.shouldHalt)
        #expect(nominal.commandedAction == RobotAction(commands: [0.25, -0.25], mode: .velocity))
        #expect(nominal.interventions.count == 1)
        #expect(nominal.safetyAssessment?.didIntervene == true)

        let halted = try supervisor.assess(
            SafetySupervisorInput(
                requestedAction: requested,
                timing: try ControlTiming(stepIndex: 2, deltaTime: 0.02, sensorAge: 0.2, actionLatency: 0.05),
                emergencyStop: EmergencyStopState(isEngaged: true, reason: "operator")
            )
        )
        #expect(halted.shouldHalt)
        #expect(halted.commandedAction == failsafe)
        #expect(halted.termination.isInterrupted)
        #expect(halted.interventions.contains(.emergencyStop(reason: "operator")))
        #expect(halted.interventions.contains(.staleSensor(age: 0.2, maximumAge: 0.1)))
        #expect(halted.interventions.contains(.deadlineMissed(latency: 0.25, maximumLatency: 0.2)))

        #expect(throws: RLSwiftError.invalidDuration(name: "maximumSensorAge", value: -1)) {
            _ = try HardwareSafetySupervisor(envelope: envelope, maximumSensorAge: -1)
        }
    }

    @Test func offlineDatasetManifestPreservesReplayMetadata() throws {
        let provenance = try DatasetProvenance(
            datasetID: "ds-1",
            sourceSystem: "wendyos",
            robotID: "arm-1",
            environment: "reach"
        )
        let constraint = try ConstraintSignal(name: "force", value: 12, limit: 10, relation: .lessThanOrEqual, weight: 2)
        let transition = Transition(
            observation: try RobotObservation(jointPositions: [0], jointVelocities: [0]),
            action: RobotAction(commands: [0.1], mode: .velocity),
            reward: 1,
            nextObservation: try RobotObservation(jointPositions: [0.1], jointVelocities: [0]),
            isTerminal: true,
            termination: .interrupted(reason: "safety")
        )
        let safetyDecision = SafetySupervisorDecision(
            requestedAction: RobotAction(commands: [2], mode: .velocity),
            commandedAction: RobotAction(commands: [1], mode: .velocity),
            shouldHalt: false,
            termination: .continuing,
            interventions: [.safetyEnvelope([.commandClipped(index: 0, requested: 2, applied: 1)])],
            safetyAssessment: nil
        )
        let logged = LoggedTransition(
            transition: transition,
            timing: try ControlTiming(stepIndex: 0, deltaTime: 0.02),
            constraints: ConstraintReport([constraint]),
            safetyDecision: safetyDecision,
            metadata: ["bag": "001"]
        )
        let dataset = try OfflineDataset(provenance: provenance, transitions: [logged])

        #expect(dataset.manifest.transitionCount == 1)
        #expect(dataset.manifest.terminationCounts["interrupted"] == 1)
        #expect(dataset.manifest.safetyInterventionCount == 1)
        #expect(dataset.manifest.totalConstraintCost == 4)

        let data = try JSONEncoder().encode(dataset)
        let decoded = try JSONDecoder().decode(OfflineDataset<RobotObservation, RobotAction>.self, from: data)
        #expect(decoded == dataset)
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try OfflineDataset(manifest: try DatasetManifest(
                provenance: provenance,
                transitionCount: 2,
                terminationCounts: [:],
                safetyInterventionCount: 0,
                totalConstraintCost: 0
            ), transitions: [logged])
        }
    }

    @Test func deploymentPlansAndTelemetryAreDeterministic() throws {
        let metadata = try PolicyMetadata(policyID: "p", version: "1")
        let contract = try ModelIOContract(
            contractVersion: "1",
            observationFeatures: [try ObservationFeature(name: "q0", index: 0, component: .jointPosition(index: 0))],
            actionSpecifications: [try ActionSpecification(name: "a0", index: 0, lowerBound: -1, upperBound: 1)],
            actionMode: .position,
            tensorRTBindings: try TensorRTBindingNames(inputName: "obs", outputName: "act"),
            policyMetadata: metadata
        )
        let plan = DeploymentPlan(
            target: try .nvidiaTensorRT(),
            modelContract: contract,
            deterministicSeed: 42,
            engineCacheKey: "engine"
        )
        #expect(plan.target.backend.requiresNVIDIALinux)
        #expect(plan.hasRequiredBackendMetadata)
        #expect(try DeploymentTarget.appleMLX().backend.supportsAppleDevices)

        let rollout = try PolicyVersionRollout(currentVersion: "1", candidateVersion: "2", candidateTrafficFraction: 0.5, seed: 7)
        #expect(rollout.selectedVersion(for: "robot-a") == rollout.selectedVersion(for: "robot-a"))
        #expect(throws: RLSwiftError.invalidProbability(1.5)) {
            _ = try PolicyVersionRollout(currentVersion: "1", candidateVersion: "2", candidateTrafficFraction: 1.5)
        }

        var telemetry = AutonomyTelemetryAccumulator()
        let constraints = ConstraintReport([
            try ConstraintSignal(name: "torque", value: 3, limit: 2, relation: .lessThanOrEqual),
        ])
        let decision = SafetySupervisorDecision(
            requestedAction: RobotAction(commands: [2], mode: .torque),
            commandedAction: RobotAction(commands: [1], mode: .torque),
            shouldHalt: false,
            termination: .continuing,
            interventions: [.safetyEnvelope([.commandClipped(index: 0, requested: 2, applied: 1)])],
            safetyAssessment: nil
        )
        try telemetry.record(
            timing: try ControlTiming(stepIndex: 0, deltaTime: 0.02, sensorAge: 0.08, actionLatency: 0.04),
            maximumLatency: 0.1,
            safetyDecision: decision,
            constraints: constraints,
            policyVersion: "2"
        )
        #expect(telemetry.summary.stepCount == 1)
        #expect(telemetry.summary.deadlineMissCount == 1)
        #expect(telemetry.summary.safetyInterventionCounts["safety_envelope"] == 1)
        #expect(telemetry.summary.constraintCosts["torque"] == 1)
        #expect(telemetry.summary.policyVersionCounts["2"] == 1)
    }

    @Test func integrationPlanningCoversAdaptersRolloutsExportsAndDebugging() throws {
        let ros = try RobotIntegrationAdapterConfiguration.ros2(
            namespace: "/arm",
            observationTopic: "/obs",
            actionTopic: "/cmd"
        )
        let sim = try RobotIntegrationAdapterConfiguration.simulator(
            endpoint: "localhost:9000",
            observationStream: "state",
            actionStream: "action"
        )
        let wendy = try RobotIntegrationAdapterConfiguration.wendyOS(
            device: "wendy-1",
            observationStream: "observations",
            actionStream: "actions"
        )
        let mujoco = try RobotIntegrationAdapterConfiguration.mujoco(
            modelPath: "humanoid.xml",
            metadata: ["task": "walk"]
        )
        let isaac = try RobotIntegrationAdapterConfiguration.isaacSim(
            endpoint: "http://127.0.0.1:8211",
            robotPath: "/World/Carter",
            metadata: ["task": "nav"]
        )
        #expect([ros.kind, sim.kind, wendy.kind, mujoco.kind, isaac.kind] == [.ros2, .simulator, .wendyOS, .simulator, .simulator])
        #expect(mujoco.endpoint == "humanoid.xml")
        #expect(mujoco.observationChannel == "qpos,qvel,sensors")
        #expect(mujoco.actionChannel == "ctrl")
        #expect(mujoco.metadata["simulator"] == "mujoco")
        #expect(mujoco.metadata["modelPath"] == "humanoid.xml")
        #expect(mujoco.metadata["task"] == "walk")
        #expect(isaac.endpoint == "http://127.0.0.1:8211")
        #expect(isaac.observationChannel == "observation")
        #expect(isaac.actionChannel == "action")
        #expect(isaac.metadata["simulator"] == "isaac-sim")
        #expect(isaac.metadata["robotPath"] == "/World/Carter")
        #expect(isaac.metadata["task"] == "nav")

        var vectorized = try VectorizedEnvironmentRunner([CounterEnvironment(), CounterEnvironment()])
        #expect(vectorized.count == 2)
        #expect(vectorized.resetAll() == [0, 0])
        let stepped = try vectorized.step([1, 2])
        #expect(stepped.observations == [1, 2])
        #expect(stepped.terminalFlags == [false, true])
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try vectorized.step([1])
        }

        let shard = try RolloutShardAssignment(shardIndex: 0, shardCount: 2, seed: 5)
        #expect(shard.owns(environmentID: "env-a") == shard.owns(environmentID: "env-a"))

        let onnx = try ONNXExportDescriptor(
            modelName: "policy",
            opsetVersion: 18,
            inputNames: ["obs"],
            outputNames: ["act"],
            dynamicAxes: ["obs": [0]]
        )
        #expect(onnx.dynamicAxes["obs"] == [0])
        let key = try TensorRTEngineCacheKey(
            policyID: "policy",
            policyVersion: "1",
            tensorRTVersion: "11.1",
            precision: "typed-fp16",
            inputShape: [1, 4]
        )
        let cache = try TensorRTEngineCacheManifest(key: key, enginePath: "policy.engine", sourceONNXPath: "policy.onnx")
        #expect(cache.key.stableIdentifier == "policy-1-11.1-typed-fp16-1x4")

        let profile = try DomainRandomizationProfile(parameters: [
            try DomainRandomizationParameter(name: "mass", lowerBound: 0.8, upperBound: 1.2),
        ])
        #expect(profile.sample(seed: 9)["mass"] == profile.sample(seed: 9)["mass"])
        let easy = try CurriculumStage(name: "easy", difficulty: 0.1, successThreshold: 0.8, minimumEpisodes: 10, randomization: profile)
        let hard = try CurriculumStage(name: "hard", difficulty: 1.0, successThreshold: 0.9, minimumEpisodes: 20)
        let schedule = try CurriculumSchedule(stages: [easy, hard])
        #expect(try schedule.nextStageIndex(currentIndex: 0, completedEpisodes: 9, successRate: 1) == 0)
        #expect(try schedule.nextStageIndex(currentIndex: 0, completedEpisodes: 10, successRate: 0.8) == 1)

        let dashboard = EvaluationDashboardSummary(records: [
            try EvaluationRecord(policyVersion: "1", environment: "reach", episodeCount: 10, meanReturn: 5, successRate: 0.8, meanConstraintCost: 1),
            try EvaluationRecord(policyVersion: "2", environment: "reach", episodeCount: 10, meanReturn: 4, successRate: 0.9, meanConstraintCost: 0.5),
        ])
        #expect(dashboard.bestRecord?.policyVersion == "2")

        let drift = try ObservationDriftSnapshot(
            featureNames: ["x", "y"],
            observedValues: [3, 1],
            normalization: try NormalizationSnapshot(count: 5, mean: [0, 0], variance: [1, 4])
        )
        #expect(drift.driftedFeatures(threshold: 1) == ["x"])
        let saturation = try ActionSaturationSnapshot(
            action: RobotAction(commands: [-1, 0.2, 1], mode: .position),
            actionSpace: try ContinuousBoxSpace(lowerBounds: [-1, -1, -1], upperBounds: [1, 1, 1])
        )
        #expect(saturation.saturatedIndices == [0, 2])
        let replayDebug = try PrioritizedReplayDebugSnapshot(index: 3, priority: 5, maximumPriority: 10, eventLabel: "collision")
        #expect(replayDebug.relativePriority == 0.5)
        #expect(replayDebug.eventLabel == "collision")
    }

    @Test func modelIOValidationBranchesAreCovered() throws {
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 2)) {
            _ = try RobotObservationComponent.jointPosition(index: 1).value(
                from: try RobotObservation(jointPositions: [0], jointVelocities: [0])
            )
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "feature.name")) {
            _ = try ObservationFeature(name: "", index: 0, component: .jointPosition(index: 0))
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 0, actual: -1)) {
            _ = try ObservationFeature(name: "q", index: -1, component: .jointPosition(index: 0))
        }
        #expect(throws: RLSwiftError.invalidSampleCount(-1)) {
            _ = try NormalizationSnapshot(count: -1, mean: [], variance: [])
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 0)) {
            _ = try NormalizationSnapshot(count: 0, mean: [0], variance: [])
        }
        var normalizer = try ObservationNormalizer(dimension: 1)
        try normalizer.update(with: [2])
        #expect(try NormalizationSnapshot(normalizer).mean == [2])
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 2)) {
            _ = try NormalizationSnapshot(count: 1, mean: [0], variance: [1]).normalize([0, 1])
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "action.name")) {
            _ = try ActionSpecification(name: "", index: 0, lowerBound: -1, upperBound: 1)
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 0, actual: -1)) {
            _ = try ActionSpecification(name: "a", index: -1, lowerBound: -1, upperBound: 1)
        }
        #expect(throws: RLSwiftError.invalidBounds(index: 0, lower: 2, upper: 1)) {
            _ = try ActionSpecification(name: "a", index: 0, lowerBound: 2, upperBound: 1)
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "tensorRT.inputName")) {
            _ = try TensorRTBindingNames(inputName: "", outputName: "out")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "tensorRT.outputName")) {
            _ = try TensorRTBindingNames(inputName: "in", outputName: "")
        }
        #expect(throws: RLSwiftError.invalidVersion("")) {
            _ = try PolicyMetadata(policyID: "p", version: "")
        }

        let metadata = try PolicyMetadata(policyID: "p", version: "1")
        let feature = try ObservationFeature(name: "q", index: 0, component: .jointPosition(index: 0))
        let action = try ActionSpecification(name: "a", index: 0, lowerBound: -1, upperBound: 1)
        #expect(throws: RLSwiftError.invalidVersion("")) {
            _ = try ModelIOContract(
                contractVersion: "",
                observationFeatures: [feature],
                actionSpecifications: [action],
                actionMode: .position,
                policyMetadata: metadata
            )
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 2)) {
            _ = try ModelIOContract(
                contractVersion: "1",
                observationFeatures: [feature],
                normalization: try NormalizationSnapshot(count: 1, mean: [0, 0], variance: [1, 1]),
                actionSpecifications: [action],
                actionMode: .position,
                policyMetadata: metadata
            )
        }
        let contract = try ModelIOContract(
            contractVersion: "1",
            observationFeatures: [feature],
            actionSpecifications: [action],
            actionMode: .position,
            policyMetadata: metadata
        )
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 2)) {
            _ = try contract.decodeAction([0, 1])
        }
        #expect(try contract.decodeAction([2], clipToBounds: false) == RobotAction(commands: [2], mode: .position))
        #expect(throws: RLSwiftError.duplicateIndex(0)) {
            _ = try ModelIOContract(
                contractVersion: "1",
                observationFeatures: [
                    try ObservationFeature(name: "q0", index: 0, component: .jointPosition(index: 0)),
                    try ObservationFeature(name: "q1", index: 0, component: .jointPosition(index: 1)),
                ],
                actionSpecifications: [action],
                actionMode: .position,
                policyMetadata: metadata
            )
        }
    }

    @Test func deploymentAndTelemetryValidationBranchesAreCovered() throws {
        #expect(!DeploymentBackend.tensorRT.supportsAppleDevices)
        #expect(!DeploymentBackend.custom("backend").supportsAppleDevices)
        #expect(!DeploymentBackend.coreSwift.requiresNVIDIALinux)
        #expect(!DeploymentBackend.mlx.requiresNVIDIALinux)
        #expect(!DeploymentBackend.custom("backend").requiresNVIDIALinux)
        #expect(throws: RLSwiftError.emptyIdentifier(name: "deploymentTarget.name")) {
            _ = try DeploymentTarget(name: "", backend: .coreSwift, platform: "linux", accelerator: "cpu")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "deploymentTarget.platform")) {
            _ = try DeploymentTarget(name: "target", backend: .coreSwift, platform: "", accelerator: "cpu")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "deploymentTarget.accelerator")) {
            _ = try DeploymentTarget(name: "target", backend: .coreSwift, platform: "linux", accelerator: "")
        }
        #expect(throws: RLSwiftError.invalidVersion("")) {
            _ = try DeploymentTarget(name: "target", backend: .coreSwift, platform: "linux", accelerator: "cpu", minimumSwiftVersion: "")
        }

        let metadata = try PolicyMetadata(policyID: "p", version: "1")
        let contractWithoutBindings = try ModelIOContract(
            contractVersion: "1",
            observationFeatures: [try ObservationFeature(name: "q", index: 0, component: .jointPosition(index: 0))],
            actionSpecifications: [try ActionSpecification(name: "a", index: 0, lowerBound: -1, upperBound: 1)],
            actionMode: .position,
            policyMetadata: metadata
        )
        let tensorRTPlan = DeploymentPlan(
            target: try .nvidiaTensorRT(),
            modelContract: contractWithoutBindings,
            deterministicSeed: 1
        )
        #expect(!tensorRTPlan.hasRequiredBackendMetadata)
        let corePlan = DeploymentPlan(
            target: try DeploymentTarget(name: "core", backend: .coreSwift, platform: "linux", accelerator: "cpu"),
            modelContract: contractWithoutBindings,
            deterministicSeed: 1
        )
        #expect(corePlan.hasRequiredBackendMetadata)

        #expect(throws: RLSwiftError.invalidVersion("")) {
            _ = try PolicyVersionRollout(currentVersion: "", candidateVersion: "2", candidateTrafficFraction: 0.5)
        }
        #expect(throws: RLSwiftError.invalidVersion("")) {
            _ = try PolicyVersionRollout(currentVersion: "1", candidateVersion: "", candidateTrafficFraction: 0.5)
        }
        #expect(try PolicyVersionRollout(currentVersion: "1", candidateVersion: "2", candidateTrafficFraction: 0).selectedVersion(for: "robot") == "1")
        #expect(try PolicyVersionRollout(currentVersion: "1", candidateVersion: "2", candidateTrafficFraction: 1).selectedVersion(for: "robot") == "2")

        var emptyTelemetry = AutonomyTelemetryAccumulator()
        #expect(emptyTelemetry.summary.meanClosedLoopLatency == 0)
        try emptyTelemetry.record(timing: try ControlTiming(stepIndex: 0, deltaTime: 0.01))
        #expect(emptyTelemetry.summary.deadlineMissCount == 0)

        var telemetry = AutonomyTelemetryAccumulator()
        let haltDecision = SafetySupervisorDecision(
            requestedAction: RobotAction(commands: [0], mode: .velocity),
            commandedAction: nil,
            shouldHalt: true,
            termination: .interrupted(reason: "halt"),
            interventions: [
                .emergencyStop(reason: nil),
                .staleSensor(age: 1, maximumAge: 0.5),
                .deadlineMissed(latency: 1, maximumLatency: 0.2),
            ],
            safetyAssessment: nil
        )
        let satisfied = ConstraintReport([
            try ConstraintSignal(name: "ok", value: 0, limit: 1, relation: .lessThanOrEqual),
        ])
        try telemetry.record(
            timing: try ControlTiming(stepIndex: 1, deltaTime: 0.01),
            safetyDecision: haltDecision,
            constraints: satisfied
        )
        #expect(telemetry.summary.safetyInterventionCounts["emergency_stop"] == 1)
        #expect(telemetry.summary.safetyInterventionCounts["stale_sensor"] == 1)
        #expect(telemetry.summary.safetyInterventionCounts["deadline_missed"] == 1)
        #expect(telemetry.summary.constraintCosts.isEmpty)
    }

    @Test func integrationPlanningValidationBranchesAreCovered() throws {
        #expect(throws: RLSwiftError.emptyIdentifier(name: "adapter.endpoint")) {
            _ = try RobotIntegrationAdapterConfiguration(kind: .ros2, endpoint: "", observationChannel: "obs", actionChannel: "act")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "adapter.observationChannel")) {
            _ = try RobotIntegrationAdapterConfiguration(kind: .ros2, endpoint: "endpoint", observationChannel: "", actionChannel: "act")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "adapter.actionChannel")) {
            _ = try RobotIntegrationAdapterConfiguration(kind: .ros2, endpoint: "endpoint", observationChannel: "obs", actionChannel: "")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "mujoco.modelPath")) {
            _ = try RobotIntegrationAdapterConfiguration.mujoco(modelPath: "")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "isaacSim.robotPath")) {
            _ = try RobotIntegrationAdapterConfiguration.isaacSim(endpoint: "http://localhost", robotPath: "")
        }
        #expect(throws: RLSwiftError.invalidCapacity(0)) {
            _ = try VectorizedEnvironmentRunner<CounterEnvironment>([])
        }
        #expect(throws: RLSwiftError.invalidCapacity(0)) {
            _ = try RolloutShardAssignment(shardIndex: 0, shardCount: 0)
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 2)) {
            _ = try RolloutShardAssignment(shardIndex: 2, shardCount: 2)
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "onnx.modelName")) {
            _ = try ONNXExportDescriptor(modelName: "", opsetVersion: 1, inputNames: ["in"], outputNames: ["out"])
        }
        #expect(throws: RLSwiftError.invalidVersion("0")) {
            _ = try ONNXExportDescriptor(modelName: "m", opsetVersion: 0, inputNames: ["in"], outputNames: ["out"])
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "onnx.inputNames")) {
            _ = try ONNXExportDescriptor(modelName: "m", opsetVersion: 1, inputNames: [], outputNames: ["out"])
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "onnx.inputNames")) {
            _ = try ONNXExportDescriptor(modelName: "m", opsetVersion: 1, inputNames: [""], outputNames: ["out"])
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "onnx.outputNames")) {
            _ = try ONNXExportDescriptor(modelName: "m", opsetVersion: 1, inputNames: ["in"], outputNames: [""])
        }
        #expect(try ONNXExportDescriptor(modelName: "m", opsetVersion: 1, inputNames: ["in0", "in1"], outputNames: ["out"]).inputNames.count == 2)

        #expect(throws: RLSwiftError.emptyIdentifier(name: "engine.policyID")) {
            _ = try TensorRTEngineCacheKey(policyID: "", policyVersion: "1", tensorRTVersion: "11", precision: "fp32", inputShape: [1])
        }
        #expect(throws: RLSwiftError.invalidVersion("")) {
            _ = try TensorRTEngineCacheKey(policyID: "p", policyVersion: "", tensorRTVersion: "11", precision: "fp32", inputShape: [1])
        }
        #expect(throws: RLSwiftError.invalidVersion("")) {
            _ = try TensorRTEngineCacheKey(policyID: "p", policyVersion: "1", tensorRTVersion: "", precision: "fp32", inputShape: [1])
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "engine.precision")) {
            _ = try TensorRTEngineCacheKey(policyID: "p", policyVersion: "1", tensorRTVersion: "11", precision: "", inputShape: [1])
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 0)) {
            _ = try TensorRTEngineCacheKey(policyID: "p", policyVersion: "1", tensorRTVersion: "11", precision: "fp32", inputShape: [])
        }
        let key = try TensorRTEngineCacheKey(policyID: "p", policyVersion: "1", tensorRTVersion: "11", precision: "fp32", inputShape: [1])
        #expect(throws: RLSwiftError.emptyIdentifier(name: "enginePath")) {
            _ = try TensorRTEngineCacheManifest(key: key, enginePath: "", sourceONNXPath: "m.onnx")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "sourceONNXPath")) {
            _ = try TensorRTEngineCacheManifest(key: key, enginePath: "m.engine", sourceONNXPath: "")
        }

        #expect(throws: RLSwiftError.emptyIdentifier(name: "randomization.name")) {
            _ = try DomainRandomizationParameter(name: "", lowerBound: 0, upperBound: 1)
        }
        #expect(throws: RLSwiftError.invalidBounds(index: 0, lower: 2, upper: 1)) {
            _ = try DomainRandomizationParameter(name: "mass", lowerBound: 2, upperBound: 1)
        }
        let parameter = try DomainRandomizationParameter(name: "mass", lowerBound: 0, upperBound: 1)
        #expect(throws: RLSwiftError.duplicateIdentifier("mass")) {
            _ = try DomainRandomizationProfile(parameters: [parameter, parameter])
        }

        #expect(throws: RLSwiftError.emptyIdentifier(name: "curriculum.name")) {
            _ = try CurriculumStage(name: "", difficulty: 0, successThreshold: 0.5, minimumEpisodes: 1)
        }
        #expect(throws: RLSwiftError.invalidProbability(1.2)) {
            _ = try CurriculumStage(name: "stage", difficulty: 0, successThreshold: 1.2, minimumEpisodes: 1)
        }
        #expect(throws: RLSwiftError.invalidSampleCount(-1)) {
            _ = try CurriculumStage(name: "stage", difficulty: 0, successThreshold: 0.5, minimumEpisodes: -1)
        }
        #expect(throws: RLSwiftError.invalidCapacity(0)) {
            _ = try CurriculumSchedule(stages: [])
        }
        let stage = try CurriculumStage(name: "stage", difficulty: 0, successThreshold: 0.5, minimumEpisodes: 1)
        let schedule = try CurriculumSchedule(stages: [stage])
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 1)) {
            _ = try schedule.stage(at: 1)
        }

        #expect(throws: RLSwiftError.invalidVersion("")) {
            _ = try EvaluationRecord(policyVersion: "", environment: "env", episodeCount: 1, meanReturn: 0, successRate: 0, meanConstraintCost: 0)
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "evaluation.environment")) {
            _ = try EvaluationRecord(policyVersion: "1", environment: "", episodeCount: 1, meanReturn: 0, successRate: 0, meanConstraintCost: 0)
        }
        #expect(throws: RLSwiftError.invalidSampleCount(-1)) {
            _ = try EvaluationRecord(policyVersion: "1", environment: "env", episodeCount: -1, meanReturn: 0, successRate: 0, meanConstraintCost: 0)
        }
        #expect(throws: RLSwiftError.invalidProbability(-0.1)) {
            _ = try EvaluationRecord(policyVersion: "1", environment: "env", episodeCount: 1, meanReturn: 0, successRate: -0.1, meanConstraintCost: 0)
        }
        let tied = EvaluationDashboardSummary(records: [
            try EvaluationRecord(policyVersion: "1", environment: "env", episodeCount: 1, meanReturn: 1, successRate: 0.5, meanConstraintCost: 0),
            try EvaluationRecord(policyVersion: "2", environment: "env", episodeCount: 1, meanReturn: 2, successRate: 0.5, meanConstraintCost: 0),
        ])
        #expect(tied.bestRecord?.policyVersion == "2")

        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 2)) {
            _ = try ObservationDriftSnapshot(
                featureNames: ["x"],
                observedValues: [1, 2],
                normalization: try NormalizationSnapshot(count: 1, mean: [0], variance: [1])
            )
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 2, actual: 1)) {
            _ = try ActionSaturationSnapshot(
                action: RobotAction(commands: [0], mode: .position),
                actionSpace: try ContinuousBoxSpace(lowerBounds: [0, 0], upperBounds: [1, 1])
            )
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 0, actual: -1)) {
            _ = try PrioritizedReplayDebugSnapshot(index: -1, priority: 1, maximumPriority: 1)
        }
        #expect(throws: RLSwiftError.invalidPriority(0)) {
            _ = try PrioritizedReplayDebugSnapshot(index: 0, priority: 0, maximumPriority: 1)
        }
        #expect(throws: RLSwiftError.invalidPriority(0)) {
            _ = try PrioritizedReplayDebugSnapshot(index: 0, priority: 1, maximumPriority: 0)
        }
    }

    @Test func offlineDatasetAndSupervisorValidationBranchesAreCovered() throws {
        #expect(throws: RLSwiftError.emptyIdentifier(name: "datasetID")) {
            _ = try DatasetProvenance(datasetID: "", sourceSystem: "sim", robotID: "r", environment: "e")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "sourceSystem")) {
            _ = try DatasetProvenance(datasetID: "d", sourceSystem: "", robotID: "r", environment: "e")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "robotID")) {
            _ = try DatasetProvenance(datasetID: "d", sourceSystem: "sim", robotID: "", environment: "e")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "environment")) {
            _ = try DatasetProvenance(datasetID: "d", sourceSystem: "sim", robotID: "r", environment: "")
        }
        let provenance = try DatasetProvenance(datasetID: "d", sourceSystem: "sim", robotID: "r", environment: "e")
        #expect(throws: RLSwiftError.invalidVersion("")) {
            _ = try DatasetManifest(
                schemaVersion: "",
                provenance: provenance,
                transitionCount: 0,
                terminationCounts: [:],
                safetyInterventionCount: 0,
                totalConstraintCost: 0
            )
        }
        #expect(throws: RLSwiftError.invalidSampleCount(-1)) {
            _ = try DatasetManifest(
                provenance: provenance,
                transitionCount: -1,
                terminationCounts: [:],
                safetyInterventionCount: 0,
                totalConstraintCost: 0
            )
        }
        #expect(throws: RLSwiftError.invalidSampleCount(-1)) {
            _ = try DatasetManifest(
                provenance: provenance,
                transitionCount: 0,
                terminationCounts: [:],
                safetyInterventionCount: -1,
                totalConstraintCost: 0
            )
        }

        let observation = try RobotObservation(jointPositions: [0], jointVelocities: [0])
        let action = RobotAction(commands: [0], mode: .position)
        let transitions = [
            LoggedTransition(transition: Transition(observation: observation, action: action, reward: 0, nextObservation: observation, isTerminal: false, termination: .continuing)),
            LoggedTransition(transition: Transition(observation: observation, action: action, reward: 0, nextObservation: observation, isTerminal: true, termination: .terminated(reason: "done"))),
            LoggedTransition(transition: Transition(observation: observation, action: action, reward: 0, nextObservation: observation, isTerminal: true, termination: .truncated(reason: "limit"))),
        ]
        let manifest = try DatasetManifest.build(provenance: provenance, transitions: transitions)
        #expect(manifest.terminationCounts["continuing"] == 1)
        #expect(manifest.terminationCounts["terminated"] == 1)
        #expect(manifest.terminationCounts["truncated"] == 1)
        #expect(manifest.totalConstraintCost == 0)

        let space = try ContinuousBoxSpace(lowerBounds: [-1], upperBounds: [1])
        let envelope = try RobotSafetyEnvelope(commandSpace: space)
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 2)) {
            _ = try HardwareSafetySupervisor(envelope: envelope, failsafeAction: RobotAction(commands: [0, 0], mode: .position))
        }
    }
}

private struct CounterEnvironment: Environment {
    private var state = 0

    mutating func reset() -> Int {
        state = 0
        return state
    }

    mutating func step(_ action: Int) throws -> StepResult<Int> {
        state += action
        return StepResult(
            observation: state,
            reward: Double(action),
            isTerminal: state >= 2,
            termination: state >= 2 ? .terminated(reason: "done") : .continuing
        )
    }
}
