public import SwiftRL
#if os(Linux) && SWIFTRL_ENABLE_TENSORRT
public import Foundation
public import TensorRT
#endif

/// Errors thrown by the SwiftRL TensorRT backend when model IO cannot be mapped
/// to SwiftRL policy inputs or robot actions.
public enum TensorRTBackendError: Error, Equatable, Sendable {
    /// Indicates that an input or output binding name was empty.
    case emptyBindingName

    /// Indicates that an engine did not expose the requested input or output binding.
    case missingEngineBinding(name: String, role: String)

    /// Indicates that a provided descriptor did not match the configured binding name.
    case descriptorNameMismatch(expected: String, actual: String)

    /// Indicates that a binding used a tensor data type this backend cannot decode.
    case unsupportedDataType(binding: String, dataType: String)

    /// Indicates that a dynamic input binding needs a concrete input shape before inference.
    case dynamicInputRequiresShape(binding: String)

    /// Indicates that an input or output tensor element count did not match the expected shape.
    case tensorElementCountMismatch(binding: String, expected: Int, actual: Int)

    /// Indicates that a result did not contain the configured output binding.
    case missingOutput(name: String)

    /// Indicates that an output byte buffer could not be decoded as complete `Float32` values.
    case invalidOutputByteCount(binding: String, actual: Int)

    /// Indicates that an output tensor was returned in storage the backend does not yet decode.
    case unsupportedOutputStorage(binding: String)
}

/// Configuration for running a TensorRT engine as a SwiftRL policy backend.
public struct TensorRTPolicyConfiguration: Equatable, Sendable {
    /// The TensorRT input binding name that receives the flattened observation vector.
    public let inputName: String

    /// The TensorRT output binding name that returns action logits or continuous commands.
    public let outputName: String

    /// A concrete input shape for dynamic TensorRT bindings.
    public let inputShape: [Int]?

    /// An optional expected output shape for dynamic TensorRT outputs.
    public let outputShape: [Int]?

    /// The optimization profile to select before inference.
    public let profileName: String?

    /// Whether `enqueue` should wait for completion before returning.
    public let synchronously: Bool

    /// The robot-control interpretation for decoded output values.
    public let controlMode: RobotControlMode

    /// Creates TensorRT policy configuration.
    public init(
        inputName: String,
        outputName: String,
        inputShape: [Int]? = nil,
        outputShape: [Int]? = nil,
        profileName: String? = nil,
        synchronously: Bool = true,
        controlMode: RobotControlMode = .torque
    ) {
        self.inputName = inputName
        self.outputName = outputName
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.profileName = profileName
        self.synchronously = synchronously
        self.controlMode = controlMode
    }
}

/// Raw TensorRT inference output decoded into Swift values.
public struct TensorRTInferenceOutput: Equatable, Sendable {
    /// The decoded `Float32` values returned by the configured TensorRT output binding.
    public let values: [Float]

    /// The execution duration reported by TensorRT, if available.
    public let duration: Duration?

    /// Backend metadata returned by TensorRT.
    public let metadata: [String: String]

    /// The optimization profile TensorRT reported using for the request.
    public let profileUsed: String?

    /// Creates a decoded inference output.
    public init(values: [Float], duration: Duration?, metadata: [String: String], profileUsed: String?) {
        self.values = values
        self.duration = duration
        self.metadata = metadata
        self.profileUsed = profileUsed
    }
}

/// TensorRT inference output converted into a SwiftRL robot action.
public struct TensorRTPolicyOutput: Equatable, Sendable {
    /// The decoded robot action.
    public let action: RobotAction

    /// The raw `Float32` values returned by the TensorRT output binding.
    public let rawValues: [Float]

    /// The execution duration reported by TensorRT, if available.
    public let duration: Duration?

    /// Backend metadata returned by TensorRT.
    public let metadata: [String: String]

    /// The optimization profile TensorRT reported using for the request.
    public let profileUsed: String?

    /// Creates a policy output.
    public init(
        action: RobotAction,
        rawValues: [Float],
        duration: Duration?,
        metadata: [String: String],
        profileUsed: String?
    ) {
        self.action = action
        self.rawValues = rawValues
        self.duration = duration
        self.metadata = metadata
        self.profileUsed = profileUsed
    }
}

/// Runtime support information for the SwiftRL TensorRT backend on the current platform.
public struct TensorRTBackendSupport: Equatable, Sendable {
    /// Whether the native TensorRT-backed policy actor is available in this build.
    public let isNativeTensorRTAvailable: Bool

    /// A short explanation suitable for diagnostics or setup output.
    public let explanation: String

    /// Creates platform support information.
    public init(isNativeTensorRTAvailable: Bool, explanation: String) {
        self.isNativeTensorRTAvailable = isNativeTensorRTAvailable
        self.explanation = explanation
    }

    /// Support information for the currently compiled platform.
#if os(Linux) && SWIFTRL_ENABLE_TENSORRT
    public static let current = TensorRTBackendSupport(
        isNativeTensorRTAvailable: true,
        explanation: "TensorRT backend is available for NVIDIA Linux builds with CUDA and TensorRT installed."
    )
#else
    public static let current = TensorRTBackendSupport(
        isNativeTensorRTAvailable: false,
        explanation: "TensorRT backend is Linux-only; use SwiftRL or SwiftRLMLX on Apple devices and SwiftRLTensorRT on NVIDIA Linux."
    )
#endif
}

#if os(Linux) && SWIFTRL_ENABLE_TENSORRT
/// A TensorRT-backed SwiftRL policy adapter for low-latency robot inference.
///
/// The backend is an actor because TensorRT execution contexts own mutable GPU
/// state such as optimization profiles, dynamic shapes, CUDA streams, and
/// warm-up state. Tests can inject `MockExecutionContext` from the TensorRT
/// package, while Linux deployments can create the backend from serialized
/// engines or ONNX models.
public actor TensorRTPolicyBackend {
    /// The TensorRT context used for inference.
    public let context: any ExecutionContexting

    /// The descriptor for the configured input binding.
    public let inputDescriptor: TensorDescriptor

    /// The descriptor for the configured output binding.
    public let outputDescriptor: TensorDescriptor

    /// The policy configuration.
    public let configuration: TensorRTPolicyConfiguration

    private var prepared = false

    /// Creates a backend from an existing TensorRT execution context.
    public init(
        context: any ExecutionContexting,
        inputDescriptor: TensorDescriptor,
        outputDescriptor: TensorDescriptor,
        configuration: TensorRTPolicyConfiguration
    ) throws {
        try Self.validateConfiguration(configuration)
        try Self.validateDescriptor(inputDescriptor, matches: configuration.inputName)
        try Self.validateDescriptor(outputDescriptor, matches: configuration.outputName)
        try Self.validateFloat32(inputDescriptor)
        try Self.validateFloat32(outputDescriptor)

        self.context = context
        self.inputDescriptor = inputDescriptor
        self.outputDescriptor = outputDescriptor
        self.configuration = configuration
    }

#if os(Linux)
    /// Creates a backend from a serialized TensorRT engine file.
    public static func loadSerializedEngine(
        from url: URL,
        policyConfiguration: TensorRTPolicyConfiguration,
        loadConfiguration: EngineLoadConfiguration = EngineLoadConfiguration(),
        queue: ExecutionQueue = .automatic,
        allocator: MemoryAllocator = .default
    ) throws -> TensorRTPolicyBackend {
        let engine = try Engine.load(from: url, configuration: loadConfiguration)
        return try makeBackend(
            from: engine,
            policyConfiguration: policyConfiguration,
            queue: queue,
            allocator: allocator
        )
    }

    /// Builds a TensorRT engine from ONNX and creates a backend from it.
    public static func buildONNXEngine(
        from url: URL,
        policyConfiguration: TensorRTPolicyConfiguration,
        buildOptions: EngineBuildOptions = EngineBuildOptions(),
        runtime: TensorRTRuntime = TensorRTRuntime(),
        queue: ExecutionQueue = .automatic,
        allocator: MemoryAllocator = .default
    ) throws -> TensorRTPolicyBackend {
        let engine = try runtime.buildEngine(onnxURL: url, options: buildOptions)
        return try makeBackend(
            from: engine,
            policyConfiguration: policyConfiguration,
            queue: queue,
            allocator: allocator
        )
    }
#endif

    /// Creates a backend from an already materialized TensorRT engine.
    public static func makeBackend(
        from engine: Engine,
        policyConfiguration: TensorRTPolicyConfiguration,
        queue: ExecutionQueue = .automatic,
        allocator: MemoryAllocator = .default
    ) throws -> TensorRTPolicyBackend {
        guard let input = engine.description.inputs.first(where: { $0.name == policyConfiguration.inputName }) else {
            throw TensorRTBackendError.missingEngineBinding(name: policyConfiguration.inputName, role: "input")
        }
        guard let output = engine.description.outputs.first(where: { $0.name == policyConfiguration.outputName }) else {
            throw TensorRTBackendError.missingEngineBinding(name: policyConfiguration.outputName, role: "output")
        }

        return try TensorRTPolicyBackend(
            context: engine.makeExecutionContext(queue: queue, allocator: allocator),
            inputDescriptor: input.descriptor,
            outputDescriptor: output.descriptor,
            configuration: policyConfiguration
        )
    }

    /// Builds a TensorRT batch from flattened `Float32` input values.
    public func inferenceBatch(for input: [Float]) throws -> InferenceBatch {
        let shape = try concreteInputShape()
        try validateElementCount(input.count, matches: shape, binding: configuration.inputName)

        let descriptor = TensorDescriptor(
            name: inputDescriptor.name,
            shape: shape,
            dataType: inputDescriptor.dataType,
            format: inputDescriptor.format,
            dynamicAxes: inputDescriptor.dynamicAxes,
            strides: inputDescriptor.strides,
            quantization: inputDescriptor.quantization
        )
        let value = TensorValue(descriptor: descriptor, storage: .host(Self.data(from: input)))
        return InferenceBatch(inputs: [configuration.inputName: value], profileName: configuration.profileName)
    }

    /// Runs inference and decodes the configured output binding as `Float32` values.
    public func inference(for input: [Float]) async throws -> TensorRTInferenceOutput {
        try await prepareIfNeeded()

        let batch = try inferenceBatch(for: input)
        let result = try await context.enqueue(batch, synchronously: configuration.synchronously)
        guard let output = result.outputs[configuration.outputName] else {
            throw TensorRTBackendError.missingOutput(name: configuration.outputName)
        }

        let values = try Self.decodeFloat32(output)
        try validateOutputElementCount(values.count, descriptor: output.descriptor)

        return TensorRTInferenceOutput(
            values: values,
            duration: result.duration,
            metadata: result.metadata,
            profileUsed: result.profileUsed
        )
    }

    /// Runs inference and returns only the decoded `Float32` output values.
    public func predict(_ input: [Float]) async throws -> [Float] {
        try await inference(for: input).values
    }

    /// Runs inference from flattened `Double` features and returns a robot action.
    public func action(for features: [Double]) async throws -> RobotAction {
        let output = try await policyOutput(for: features)
        return output.action
    }

    /// Runs inference from flattened `Double` features and returns full policy output metadata.
    public func policyOutput(for features: [Double]) async throws -> TensorRTPolicyOutput {
        let inferenceOutput = try await inference(for: features.map(Float.init))
        let action = RobotAction(
            commands: inferenceOutput.values.map(Double.init),
            mode: configuration.controlMode
        )
        return TensorRTPolicyOutput(
            action: action,
            rawValues: inferenceOutput.values,
            duration: inferenceOutput.duration,
            metadata: inferenceOutput.metadata,
            profileUsed: inferenceOutput.profileUsed
        )
    }

    /// Runs inference from a robot observation using deterministic sensor ordering.
    public func action(
        for observation: RobotObservation,
        sensorOrder: [String]
    ) async throws -> RobotAction {
        try await action(for: observation.flattenedFeatures(sensorOrder: sensorOrder))
    }

    /// Runs inference from a robot observation and returns full policy output metadata.
    public func policyOutput(
        for observation: RobotObservation,
        sensorOrder: [String]
    ) async throws -> TensorRTPolicyOutput {
        try await policyOutput(for: observation.flattenedFeatures(sensorOrder: sensorOrder))
    }

    /// Warms up the underlying TensorRT execution context.
    public func warmup(iterations: Int = 10) async throws -> WarmupSummary {
        try await prepareIfNeeded()
        return try await context.warmup(iterations: iterations)
    }

    private func prepareIfNeeded() async throws {
        guard !prepared else {
            return
        }
        if let profileName = configuration.profileName {
            try await context.setOptimizationProfile(named: profileName)
        }
        if inputDescriptor.shape.isDynamic || configuration.inputShape != nil {
            try await context.reshape(bindings: [configuration.inputName: concreteInputShape()])
        }
        prepared = true
    }

    private func concreteInputShape() throws -> TensorShape {
        if let inputShape = configuration.inputShape {
            return TensorShape(inputShape)
        }
        guard !inputDescriptor.shape.isDynamic else {
            throw TensorRTBackendError.dynamicInputRequiresShape(binding: configuration.inputName)
        }
        return inputDescriptor.shape
    }

    private func validateOutputElementCount(_ count: Int, descriptor: TensorDescriptor) throws {
        if let outputShape = configuration.outputShape {
            try validateElementCount(count, matches: TensorShape(outputShape), binding: configuration.outputName)
            return
        }
        guard !descriptor.shape.isDynamic else {
            return
        }
        try validateElementCount(count, matches: descriptor.shape, binding: configuration.outputName)
    }

    private func validateElementCount(_ count: Int, matches shape: TensorShape, binding: String) throws {
        let expected = shape.elementCount
        guard expected == count else {
            throw TensorRTBackendError.tensorElementCountMismatch(
                binding: binding,
                expected: expected,
                actual: count
            )
        }
    }

    private static func validateConfiguration(_ configuration: TensorRTPolicyConfiguration) throws {
        guard !configuration.inputName.isEmpty, !configuration.outputName.isEmpty else {
            throw TensorRTBackendError.emptyBindingName
        }
    }

    private static func validateDescriptor(_ descriptor: TensorDescriptor, matches name: String) throws {
        guard descriptor.name == name else {
            throw TensorRTBackendError.descriptorNameMismatch(expected: name, actual: descriptor.name)
        }
    }

    private static func validateFloat32(_ descriptor: TensorDescriptor) throws {
        guard descriptor.dataType == .float32 else {
            throw TensorRTBackendError.unsupportedDataType(
                binding: descriptor.name,
                dataType: descriptor.dataType.rawValue
            )
        }
    }

    private static func data(from values: [Float]) -> Data {
        values.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private static func decodeFloat32(_ value: TensorValue) throws -> [Float] {
        guard value.descriptor.dataType == .float32 else {
            throw TensorRTBackendError.unsupportedDataType(
                binding: value.descriptor.name,
                dataType: value.descriptor.dataType.rawValue
            )
        }

        let data: Data
        switch value.storage {
        case .host(let hostData):
            data = hostData
        case .deferred(let thunk):
            data = thunk()
        default:
            throw TensorRTBackendError.unsupportedOutputStorage(binding: value.descriptor.name)
        }

        guard data.count.isMultiple(of: MemoryLayout<Float>.stride) else {
            throw TensorRTBackendError.invalidOutputByteCount(binding: value.descriptor.name, actual: data.count)
        }

        var values = [Float](repeating: 0, count: data.count / MemoryLayout<Float>.stride)
        values.withUnsafeMutableBufferPointer { buffer in
            data.copyBytes(to: UnsafeMutableRawBufferPointer(buffer))
        }
        return values
    }
}
#endif
