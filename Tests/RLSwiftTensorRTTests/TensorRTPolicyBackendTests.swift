import Foundation
import Testing
import RLSwift
@testable import RLSwiftTensorRT
#if os(Linux) && SWIFTRL_ENABLE_TENSORRT
import TensorRT

@Suite struct TensorRTPolicyBackendTests {
    @Test func buildsInferenceBatchWithFloat32HostInput() async throws {
        let backend = try makeBackend(inputShape: [1, 3], outputShape: [1, 2])

        let batch = try await backend.inferenceBatch(for: [1, 2, 3])
        let value = try #require(batch.inputs["observation"])

        #expect(batch.profileName == nil)
        #expect(value.descriptor.name == "observation")
        #expect(value.descriptor.shape.dimensions == [1, 3])
        #expect(try floats(from: value) == [1, 2, 3])
    }

    @Test func preparesDynamicShapeAndProfileOnce() async throws {
        let context = RecordingExecutionContext { batch in
            #expect(batch.profileName == "fast")
            return InferenceResult(
                outputs: [
                    "action": TensorValue(
                        descriptor: descriptor("action", [2]),
                        storage: .host(data([0.25, -0.5]))
                    ),
                ],
                duration: .milliseconds(3),
                metadata: ["backend": "mock"],
                profileUsed: "fast"
            )
        }
        let backend = try TensorRTPolicyBackend(
            context: context,
            inputDescriptor: descriptor("observation", [-1, 3]),
            outputDescriptor: descriptor("action", [-1]),
            configuration: TensorRTPolicyConfiguration(
                inputName: "observation",
                outputName: "action",
                inputShape: [2, 3],
                outputShape: [2],
                profileName: "fast"
            )
        )

        let first = try await backend.inference(for: [1, 2, 3, 4, 5, 6])
        let second = try await backend.inference(for: [6, 5, 4, 3, 2, 1])
        let snapshot = await context.snapshot()

        #expect(first.values == [0.25, -0.5])
        #expect(first.duration == .milliseconds(3))
        #expect(first.metadata == ["backend": "mock"])
        #expect(first.profileUsed == "fast")
        #expect(second.values == [0.25, -0.5])
        #expect(snapshot.profileNames == ["fast"])
        #expect(snapshot.reshapeBindings == [["observation": [2, 3]]])
        #expect(snapshot.synchronousFlags == [true, true])
    }

    @Test func predictsAndCreatesRobotActionsFromFeaturesAndObservations() async throws {
        let context = RecordingExecutionContext { batch in
            let input = try floats(from: #require(batch.inputs["observation"]))
            return InferenceResult(outputs: [
                "action": TensorValue(
                    descriptor: descriptor("action", [2]),
                    storage: .host(data([input[0] + input[1], input[3] + input[4]]))
                ),
            ])
        }
        let backend = try makeBackend(
            context: context,
            inputShape: [1, 5],
            outputShape: [2],
            controlMode: .velocity
        )
        let observation = try RobotObservation(
            jointPositions: [1],
            jointVelocities: [2],
            endEffectorPose: [3],
            sensorReadings: ["force": 4]
        )

        #expect(try await backend.predict([1, 2, 3, 4, 5]) == [3, 9])
        #expect(try await backend.action(for: [1, 2, 3, 4, 5]) == RobotAction(commands: [3, 9], mode: .velocity))

        let observationAction = try await backend.action(for: observation, sensorOrder: ["force", "missing"])
        let observationOutput = try await backend.policyOutput(for: observation, sensorOrder: ["force", "missing"])

        #expect(observationAction == RobotAction(commands: [3, 4], mode: .velocity))
        #expect(observationOutput.action == observationAction)
        #expect(observationOutput.rawValues == [3, 4])
    }

    @Test func warmsUpPreparedContext() async throws {
        let context = RecordingExecutionContext { _ in
            InferenceResult(outputs: [
                "action": TensorValue(descriptor: descriptor("action", [1]), storage: .host(data([1]))),
            ])
        }
        let backend = try makeBackend(context: context, inputShape: [1], outputShape: [1])

        let summary = try await backend.warmup(iterations: 4)
        let snapshot = await context.snapshot()

        #expect(summary.samples == [.milliseconds(4)])
        #expect(snapshot.warmupIterations == [4])
    }

    @Test func rejectsInvalidConfigurationDescriptorsAndDataTypes() async throws {
        let context = RecordingExecutionContext { _ in InferenceResult(outputs: [:]) }

        #expect(throws: TensorRTBackendError.emptyBindingName) {
            _ = try TensorRTPolicyBackend(
                context: context,
                inputDescriptor: descriptor("observation", [1]),
                outputDescriptor: descriptor("action", [1]),
                configuration: TensorRTPolicyConfiguration(inputName: "", outputName: "action")
            )
        }
        #expect(throws: TensorRTBackendError.descriptorNameMismatch(expected: "observation", actual: "state")) {
            _ = try TensorRTPolicyBackend(
                context: context,
                inputDescriptor: descriptor("state", [1]),
                outputDescriptor: descriptor("action", [1]),
                configuration: TensorRTPolicyConfiguration(inputName: "observation", outputName: "action")
            )
        }
        #expect(throws: TensorRTBackendError.unsupportedDataType(binding: "observation", dataType: "int32")) {
            _ = try TensorRTPolicyBackend(
                context: context,
                inputDescriptor: descriptor("observation", [1], dataType: .int32),
                outputDescriptor: descriptor("action", [1]),
                configuration: TensorRTPolicyConfiguration(inputName: "observation", outputName: "action")
            )
        }
    }

    @Test func rejectsMissingDynamicInputShapeAndInputCountMismatch() async throws {
        let dynamicBackend = try makeBackend(inputShape: [-1, 3], outputShape: [1])
        await #expect(throws: TensorRTBackendError.dynamicInputRequiresShape(binding: "observation")) {
            _ = try await dynamicBackend.inferenceBatch(for: [1, 2, 3])
        }

        let staticBackend = try makeBackend(inputShape: [1, 3], outputShape: [1])
        await #expect(throws: TensorRTBackendError.tensorElementCountMismatch(binding: "observation", expected: 3, actual: 2)) {
            _ = try await staticBackend.inferenceBatch(for: [1, 2])
        }
    }

    @Test func rejectsMissingOutputUnsupportedStorageAndBadOutputBytes() async throws {
        let missingOutputBackend = try makeBackend(context: RecordingExecutionContext { _ in
            InferenceResult(outputs: [:])
        })
        await #expect(throws: TensorRTBackendError.missingOutput(name: "action")) {
            _ = try await missingOutputBackend.inference(for: [1])
        }

        let deviceOutputBackend = try makeBackend(context: RecordingExecutionContext { _ in
            InferenceResult(outputs: [
                "action": TensorValue(
                    descriptor: descriptor("action", [1]),
                    storage: .deviceBuffer(address: 0, length: MemoryLayout<Float>.stride)
                ),
            ])
        })
        await #expect(throws: TensorRTBackendError.unsupportedOutputStorage(binding: "action")) {
            _ = try await deviceOutputBackend.inference(for: [1])
        }

        let badBytesBackend = try makeBackend(context: RecordingExecutionContext { _ in
            InferenceResult(outputs: [
                "action": TensorValue(descriptor: descriptor("action", [1]), storage: .host(Data([1, 2, 3]))),
            ])
        })
        await #expect(throws: TensorRTBackendError.invalidOutputByteCount(binding: "action", actual: 3)) {
            _ = try await badBytesBackend.inference(for: [1])
        }
    }

    @Test func rejectsOutputDataTypeAndOutputCountMismatch() async throws {
        let wrongTypeBackend = try makeBackend(context: RecordingExecutionContext { _ in
            InferenceResult(outputs: [
                "action": TensorValue(
                    descriptor: descriptor("action", [1], dataType: .int32),
                    storage: .host(data([1]))
                ),
            ])
        })
        await #expect(throws: TensorRTBackendError.unsupportedDataType(binding: "action", dataType: "int32")) {
            _ = try await wrongTypeBackend.inference(for: [1])
        }

        let wrongCountBackend = try makeBackend(
            context: RecordingExecutionContext { _ in
                InferenceResult(outputs: [
                    "action": TensorValue(descriptor: descriptor("action", [2]), storage: .host(data([1, 2]))),
                ])
            },
            configuredOutputShape: [3]
        )
        await #expect(throws: TensorRTBackendError.tensorElementCountMismatch(binding: "action", expected: 3, actual: 2)) {
            _ = try await wrongCountBackend.inference(for: [1])
        }
    }

    @Test func decodesDeferredDynamicOutputsWithoutExpectedShape() async throws {
        let backend = try makeBackend(
            context: RecordingExecutionContext { _ in
                InferenceResult(outputs: [
                    "action": TensorValue(
                        descriptor: descriptor("action", [-1]),
                        storage: .deferred { data([1, 2, 3]) }
                    ),
                ])
            },
            outputShape: [-1]
        )

        #expect(try await backend.inference(for: [1]).values == [1, 2, 3])
    }

    @Test func createsBackendFromEngineDescriptionAndReportsMissingBindings() throws {
        let input = TensorBinding(descriptor: descriptor("observation", [1]), role: .input)
        let output = TensorBinding(descriptor: descriptor("action", [1]), role: .output)
        let engine = Engine(description: EngineDescription(inputs: [input], outputs: [output], precision: .fp32))

        _ = try TensorRTPolicyBackend.makeBackend(
            from: engine,
            policyConfiguration: TensorRTPolicyConfiguration(inputName: "observation", outputName: "action")
        )

        #expect(throws: TensorRTBackendError.missingEngineBinding(name: "missing", role: "input")) {
            _ = try TensorRTPolicyBackend.makeBackend(
                from: engine,
                policyConfiguration: TensorRTPolicyConfiguration(inputName: "missing", outputName: "action")
            )
        }
        #expect(throws: TensorRTBackendError.missingEngineBinding(name: "missing", role: "output")) {
            _ = try TensorRTPolicyBackend.makeBackend(
                from: engine,
                policyConfiguration: TensorRTPolicyConfiguration(inputName: "observation", outputName: "missing")
            )
        }
    }

    @Test func storesCUDAKernelPlanMetadata() throws {
        let cudaPlan = try TensorRTCUDAKernelPlan(
            nativePlan: try .cudaPPO(),
            engineCacheKey: "policy-engine",
            rolloutBufferCount: 3
        )
        let tensorRTNativePlan = try NativeKernelPlan(
            backend: .tensorRT,
            operations: [.sampleActions],
            precision: "fp16",
            usesStaticMemory: true,
            usesGraphCapture: true
        )
        let tensorRTPlan = try TensorRTCUDAKernelPlan(nativePlan: tensorRTNativePlan)

        #expect(cudaPlan.nativePlan.backend == .cuda)
        #expect(cudaPlan.engineCacheKey == "policy-engine")
        #expect(cudaPlan.rolloutBufferCount == 3)
        #expect(tensorRTPlan.nativePlan.backend == .tensorRT)

        #expect(throws: RLSwiftError.emptyIdentifier(name: "cudaOrTensorRTBackend")) {
            _ = try TensorRTCUDAKernelPlan(nativePlan: try .swiftReference())
        }
        #expect(throws: RLSwiftError.invalidCapacity(0)) {
            _ = try TensorRTCUDAKernelPlan(nativePlan: try .cudaPPO(), rolloutBufferCount: 0)
        }
    }

    @Test func nativeCUDAKernelsReportRuntimeStatus() {
        let status = TensorRTCUDAKernelExecutor.runtimeStatus()

        #expect(!status.message.isEmpty)
    }

    @Test func nativeCUDAKernelsMatchSwiftReferenceWhenAvailable() throws {
        let status = TensorRTCUDAKernelExecutor.runtimeStatus()
        let requiresCUDA = ProcessInfo.processInfo.environment["SWIFTRL_REQUIRE_CUDA_KERNELS"] == "1"
        if requiresCUDA {
            #expect(status.isAvailable)
        }
        guard status.isAvailable else {
            #expect(status.message.contains("Unable") || status.message.contains("failed") || !status.message.isEmpty)
            return
        }

        var positions: [Int32] = [0, 1, 3]
        var stepIndices: [Int32] = [0, 1, 3]
        let lineWorld = try TensorRTCUDAKernelExecutor.stepLineWorld(
            actions: [.right, .left, .right],
            positions: &positions,
            stepIndices: &stepIndices,
            length: 4,
            maxSteps: 4
        )
        #expect(positions == [1, 0, 3])
        #expect(stepIndices == [1, 2, 4])
        #expect(lineWorld.observations == [
            LineWorldObservation(position: 1, goal: 3, stepIndex: 1),
            LineWorldObservation(position: 0, goal: 3, stepIndex: 2),
            LineWorldObservation(position: 3, goal: 3, stepIndex: 4),
        ])
        #expect(lineWorld.terminalFlags == [false, false, true])
        #expect(lineWorld.terminations == [.continuing, .continuing, .terminated(reason: "goal")])
        #expect(lineWorld.rewards.map(round4) == [-0.01, -0.01, 1])

        let configuration = try PPOConfiguration(
            clipRange: 0.2,
            valueLossCoefficient: 0.5,
            entropyCoefficient: 0.01
        )
        let samples = [
            PPOClippedObjectiveSample(
                oldLogProbability: 0,
                newLogProbability: log(1.5),
                advantage: 2,
                returnEstimate: 3,
                valueEstimate: 1,
                entropy: 0.1
            ),
            PPOClippedObjectiveSample(
                oldLogProbability: 0,
                newLogProbability: log(0.9),
                advantage: -1,
                returnEstimate: 0,
                valueEstimate: 1,
                entropy: 0.3
            ),
            PPOClippedObjectiveSample(
                oldLogProbability: -0.2,
                newLogProbability: -0.25,
                advantage: 0.5,
                returnEstimate: 0.75,
                valueEstimate: 0.25,
                entropy: 0.2
            ),
        ]
        let swift = try PPOClippedObjective.evaluate(samples: samples, configuration: configuration)
        let cuda = try TensorRTCUDAKernelExecutor.ppoObjective(samples: samples, configuration: configuration)
        #expect(round4(cuda.policyLoss) == round4(swift.policyLoss))
        #expect(round4(cuda.valueLoss) == round4(swift.valueLoss))
        #expect(round4(cuda.entropyBonus) == round4(swift.entropyBonus))
        #expect(round4(cuda.totalLoss) == round4(swift.totalLoss))
        #expect(round4(cuda.meanApproximateKL) == round4(swift.meanApproximateKL))
        #expect(round4(cuda.clippedFraction) == round4(swift.clippedFraction))
    }

    @Test func nativeCUDAKernelValidationBranchesAreCoveredWhenRuntimeIsUnavailableOrInvalid() throws {
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            var positions: [Int32] = []
            var steps: [Int32] = []
            _ = try TensorRTCUDAKernelExecutor.stepLineWorld(
                actions: [],
                positions: &positions,
                stepIndices: &steps,
                length: 4,
                maxSteps: 4
            )
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 0)) {
            var positions: [Int32] = []
            var steps: [Int32] = [0]
            _ = try TensorRTCUDAKernelExecutor.stepLineWorld(
                actions: [.right],
                positions: &positions,
                stepIndices: &steps,
                length: 4,
                maxSteps: 4
            )
        }
        #expect(throws: RLSwiftError.dimensionMismatch(expected: 1, actual: 0)) {
            var positions: [Int32] = [0]
            var steps: [Int32] = []
            _ = try TensorRTCUDAKernelExecutor.stepLineWorld(
                actions: [.right],
                positions: &positions,
                stepIndices: &steps,
                length: 4,
                maxSteps: 4
            )
        }
        #expect(throws: RLSwiftError.invalidCapacity(1)) {
            var positions: [Int32] = [0]
            var steps: [Int32] = [0]
            _ = try TensorRTCUDAKernelExecutor.stepLineWorld(
                actions: [.right],
                positions: &positions,
                stepIndices: &steps,
                length: 1,
                maxSteps: 4
            )
        }
        #expect(throws: RLSwiftError.invalidHorizon(0)) {
            var positions: [Int32] = [0]
            var steps: [Int32] = [0]
            _ = try TensorRTCUDAKernelExecutor.stepLineWorld(
                actions: [.right],
                positions: &positions,
                stepIndices: &steps,
                length: 4,
                maxSteps: 0
            )
        }
        #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try TensorRTCUDAKernelExecutor.ppoObjective(samples: [], configuration: try PPOConfiguration())
        }
    }
}

private struct RecordingSnapshot: Equatable, Sendable {
    var profileNames: [String]
    var reshapeBindings: [[String: [Int]]]
    var synchronousFlags: [Bool]
    var warmupIterations: [Int]
}

private actor RecordingExecutionContext: ExecutionContexting {
    private var profileNames: [String] = []
    private var reshapeBindings: [[String: TensorShape]] = []
    private var synchronousFlags: [Bool] = []
    private var warmupIterations: [Int] = []
    private let handler: @Sendable (InferenceBatch) async throws -> InferenceResult

    init(handler: @escaping @Sendable (InferenceBatch) async throws -> InferenceResult) {
        self.handler = handler
    }

    func enqueue(_ batch: InferenceBatch, synchronously: Bool) async throws -> InferenceResult {
        synchronousFlags.append(synchronously)
        return try await handler(batch)
    }

    func setOptimizationProfile(_ profile: OptimizationProfile) async throws {
        profileNames.append(profile.name)
    }

    func setOptimizationProfile(named name: String) async throws {
        profileNames.append(name)
    }

    func reshape(bindings: [String: TensorShape]) async throws {
        reshapeBindings.append(bindings)
    }

    func warmup(iterations: Int) async throws -> WarmupSummary {
        warmupIterations.append(iterations)
        return WarmupSummary(samples: [.milliseconds(iterations)])
    }

    func snapshot() -> RecordingSnapshot {
        RecordingSnapshot(
            profileNames: profileNames,
            reshapeBindings: reshapeBindings.mapValues(),
            synchronousFlags: synchronousFlags,
            warmupIterations: warmupIterations
        )
    }
}

private extension Array where Element == [String: TensorShape] {
    func mapValues() -> [[String: [Int]]] {
        map { bindings in
            bindings.reduce(into: [:]) { result, pair in
                result[pair.key] = pair.value.dimensions
            }
        }
    }
}

private func makeBackend(
    context: any ExecutionContexting = RecordingExecutionContext { _ in
        InferenceResult(outputs: [
            "action": TensorValue(descriptor: descriptor("action", [1]), storage: .host(data([1]))),
        ])
    },
    inputShape: [Int] = [1],
    outputShape: [Int] = [1],
    configuredInputShape: [Int]? = nil,
    configuredOutputShape: [Int]? = nil,
    controlMode: RobotControlMode = .torque
) throws -> TensorRTPolicyBackend {
    try TensorRTPolicyBackend(
        context: context,
        inputDescriptor: descriptor("observation", inputShape),
        outputDescriptor: descriptor("action", outputShape),
        configuration: TensorRTPolicyConfiguration(
            inputName: "observation",
            outputName: "action",
            inputShape: configuredInputShape,
            outputShape: configuredOutputShape,
            controlMode: controlMode
        )
    )
}

private func descriptor(
    _ name: String,
    _ shape: [Int],
    dataType: TensorDataType = .float32
) -> TensorDescriptor {
    TensorDescriptor(name: name, shape: TensorShape(shape), dataType: dataType)
}

private func data(_ values: [Float]) -> Data {
    values.withUnsafeBufferPointer { Data(buffer: $0) }
}

private func floats(from value: TensorValue) throws -> [Float] {
    guard case .host(let data) = value.storage else {
        return []
    }
    var values = [Float](repeating: 0, count: data.count / MemoryLayout<Float>.stride)
    _ = values.withUnsafeMutableBufferPointer { buffer in
        data.copyBytes(to: UnsafeMutableRawBufferPointer(buffer))
    }
    return values
}

private func round4(_ value: Double) -> Double {
    (value * 10_000).rounded() / 10_000
}
#else
@Suite struct TensorRTPolicyBackendSupportTests {
    @Test func reportsTensorRTUnavailableOnAppleDevelopmentHosts() {
        let support = TensorRTBackendSupport.current

        #expect(!support.isNativeTensorRTAvailable)
        #expect(support.explanation.contains("Linux-only"))
    }

    @Test func storesPolicyConfiguration() {
        let configuration = TensorRTPolicyConfiguration(
            inputName: "observation",
            outputName: "action",
            inputShape: [1, 8],
            outputShape: [1, 2],
            profileName: "fast",
            synchronously: false,
            controlMode: .position
        )

        #expect(configuration.inputName == "observation")
        #expect(configuration.outputName == "action")
        #expect(configuration.inputShape == [1, 8])
        #expect(configuration.outputShape == [1, 2])
        #expect(configuration.profileName == "fast")
        #expect(!configuration.synchronously)
        #expect(configuration.controlMode == .position)
    }

    @Test func storesInferenceAndPolicyOutputs() {
        let inference = TensorRTInferenceOutput(
            values: [0.1, -0.2],
            duration: .milliseconds(2),
            metadata: ["engine": "mock"],
            profileUsed: "fast"
        )
        let action = RobotAction(commands: [0.1, -0.2], mode: .torque)
        let policy = TensorRTPolicyOutput(
            action: action,
            rawValues: inference.values,
            duration: inference.duration,
            metadata: inference.metadata,
            profileUsed: inference.profileUsed
        )
        let explicitSupport = TensorRTBackendSupport(
            isNativeTensorRTAvailable: false,
            explanation: "not available"
        )

        #expect(inference.values == [0.1, -0.2])
        #expect(inference.duration == .milliseconds(2))
        #expect(policy.action == action)
        #expect(policy.rawValues == [0.1, -0.2])
        #expect(policy.metadata == ["engine": "mock"])
        #expect(policy.profileUsed == "fast")
        #expect(explicitSupport.explanation == "not available")
    }

    @Test func storesCUDAKernelPlanMetadata() throws {
        let cudaPlan = try TensorRTCUDAKernelPlan(
            nativePlan: try .cudaPPO(),
            engineCacheKey: "policy-engine",
            rolloutBufferCount: 3
        )
        let tensorRTNativePlan = try NativeKernelPlan(
            backend: .tensorRT,
            operations: [.sampleActions],
            precision: "fp16",
            usesStaticMemory: true,
            usesGraphCapture: true
        )
        let tensorRTPlan = try TensorRTCUDAKernelPlan(nativePlan: tensorRTNativePlan)

        #expect(cudaPlan.nativePlan.backend == .cuda)
        #expect(cudaPlan.engineCacheKey == "policy-engine")
        #expect(cudaPlan.rolloutBufferCount == 3)
        #expect(tensorRTPlan.nativePlan.backend == .tensorRT)

        #expect(throws: RLSwiftError.emptyIdentifier(name: "cudaOrTensorRTBackend")) {
            _ = try TensorRTCUDAKernelPlan(nativePlan: try .swiftReference())
        }
        #expect(throws: RLSwiftError.invalidCapacity(0)) {
            _ = try TensorRTCUDAKernelPlan(nativePlan: try .cudaPPO(), rolloutBufferCount: 0)
        }
    }
}
#endif
