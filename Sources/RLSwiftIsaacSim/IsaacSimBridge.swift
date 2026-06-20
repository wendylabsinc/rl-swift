public import Foundation
public import RLSwift
#if canImport(FoundationNetworking)
public import FoundationNetworking
#endif

/// Errors thrown by the RLSwift Isaac Sim bridge.
public enum IsaacSimBridgeError: Error, Equatable, Sendable {
    /// The HTTP transport returned a non-2xx status.
    case httpStatus(code: Int, body: String)

    /// The transport returned a response that was not HTTP.
    case nonHTTPResponse
}

/// Runtime support information for the RLSwift Isaac Sim bridge.
public struct IsaacSimBackendSupport: Equatable, Sendable {
    /// Whether the Swift bridge types are available in this build.
    public let isBridgeAvailable: Bool

    /// A short explanation suitable for diagnostics or setup output.
    public let explanation: String

    /// Creates Isaac Sim support information.
    public init(isBridgeAvailable: Bool, explanation: String) {
        self.isBridgeAvailable = isBridgeAvailable
        self.explanation = explanation
    }

    /// Support information for the currently compiled platform.
    public static let current = IsaacSimBackendSupport(
        isBridgeAvailable: true,
        explanation: "Isaac Sim support is available through the JSON/HTTP bridge client; run Isaac Sim or Isaac Lab in a sidecar process."
    )
}

/// Configuration for a JSON/HTTP bridge exposed by an Isaac Sim sidecar.
public struct IsaacSimBridgeConfiguration: Sendable, Equatable, Codable {
    /// Base URL for the bridge service.
    public let baseURL: URL

    /// USD scene or stage path requested by reset calls.
    public let scenePath: String?

    /// Prim path for the robot controlled by RLSwift.
    public let robotPath: String

    /// Relative or absolute endpoint for health checks.
    public let healthPath: String

    /// Relative or absolute endpoint for episode resets.
    public let resetPath: String

    /// Relative or absolute endpoint for environment steps.
    public let stepPath: String

    /// Request timeout in seconds.
    public let timeout: TimeInterval

    /// Additional bridge metadata.
    public let metadata: [String: String]

    /// Relative or absolute endpoint for batch environment resets.
    public let batchResetPath: String

    /// Relative or absolute endpoint for batch environment steps.
    public let batchStepPath: String

    /// Creates a bridge configuration.
    public init(
        baseURL: URL,
        scenePath: String? = nil,
        robotPath: String = "/World/Robot",
        healthPath: String = "/health",
        resetPath: String = "/reset",
        stepPath: String = "/step",
        timeout: TimeInterval = 30,
        metadata: [String: String] = [:],
        batchResetPath: String = "/batch/reset",
        batchStepPath: String = "/batch/step"
    ) throws {
        let scheme = baseURL.scheme?.lowercased()
        guard scheme == "http" || scheme == "https" else {
            throw RLSwiftError.emptyIdentifier(name: "isaacSim.baseURL")
        }
        guard !robotPath.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "isaacSim.robotPath")
        }
        guard !healthPath.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "isaacSim.healthPath")
        }
        guard !resetPath.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "isaacSim.resetPath")
        }
        guard !stepPath.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "isaacSim.stepPath")
        }
        guard !batchResetPath.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "isaacSim.batchResetPath")
        }
        guard !batchStepPath.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "isaacSim.batchStepPath")
        }
        guard timeout > 0 else {
            throw RLSwiftError.invalidDuration(name: "isaacSim.timeout", value: timeout)
        }
        self.baseURL = baseURL
        self.scenePath = scenePath
        self.robotPath = robotPath
        self.healthPath = healthPath
        self.resetPath = resetPath
        self.stepPath = stepPath
        self.timeout = timeout
        self.metadata = metadata
        self.batchResetPath = batchResetPath
        self.batchStepPath = batchStepPath
    }

    /// Builds a URL for a configured endpoint path.
    public func url(for path: String) throws -> URL {
        guard !path.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "isaacSim.path")
        }
        if let absolute = URL(string: path), absolute.scheme != nil {
            return absolute
        }
        let relativePath = path.drop { $0 == "/" }
        return baseURL.appendingPathComponent(String(relativePath))
    }
}

/// Options sent with an Isaac Sim reset request.
public struct IsaacSimResetOptions: Sendable, Equatable, Codable {
    /// Standard reset options with no seed or randomization overrides.
    public static let standard: IsaacSimResetOptions = {
        try! IsaacSimResetOptions()
    }()

    /// Deterministic seed for the simulator or task wrapper.
    public let seed: UInt64?

    /// Stable episode identifier used for logs, replay, or offline datasets.
    public let episodeID: String?

    /// Numeric domain-randomization parameters for the sidecar.
    public let randomization: [String: Double]

    /// Additional reset metadata.
    public let metadata: [String: String]

    /// Creates reset options.
    public init(
        seed: UInt64? = nil,
        episodeID: String? = nil,
        randomization: [String: Double] = [:],
        metadata: [String: String] = [:]
    ) throws {
        if let episodeID, episodeID.isEmpty {
            throw RLSwiftError.emptyIdentifier(name: "isaacSim.episodeID")
        }
        self.seed = seed
        self.episodeID = episodeID
        self.randomization = randomization
        self.metadata = metadata
    }
}

/// Options sent with an Isaac Sim step request.
public struct IsaacSimStepOptions: Sendable, Equatable, Codable {
    /// Standard step options for one non-rendering physics step.
    public static let standard: IsaacSimStepOptions = {
        try! IsaacSimStepOptions()
    }()

    /// Number of physics steps the sidecar should advance for one policy action.
    public let physicsSteps: Int

    /// Whether the sidecar should render sensors that require a render pass.
    public let render: Bool

    /// Additional step metadata.
    public let metadata: [String: String]

    /// Creates step options.
    public init(physicsSteps: Int = 1, render: Bool = false, metadata: [String: String] = [:]) throws {
        guard physicsSteps > 0 else {
            throw RLSwiftError.invalidHorizon(physicsSteps)
        }
        self.physicsSteps = physicsSteps
        self.render = render
        self.metadata = metadata
    }
}

/// One Isaac Sim or Isaac Lab environment controlled through a batch bridge.
public struct IsaacSimEnvironmentHandle: Sendable, Equatable, Codable {
    /// Stable sidecar environment identifier.
    public let id: String

    /// Prim path for the robot inside this environment.
    public let robotPath: String

    /// Optional Isaac Lab task or scenario name.
    public let taskName: String?

    /// Additional environment metadata.
    public let metadata: [String: String]

    /// Creates an environment handle.
    public init(id: String, robotPath: String, taskName: String? = nil, metadata: [String: String] = [:]) throws {
        guard !id.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "isaacSim.environmentID")
        }
        guard !robotPath.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "isaacSim.robotPath")
        }
        if let taskName, taskName.isEmpty {
            throw RLSwiftError.emptyIdentifier(name: "isaacSim.taskName")
        }
        self.id = id
        self.robotPath = robotPath
        self.taskName = taskName
        self.metadata = metadata
    }
}

/// Action for one sidecar environment in a batch step request.
public struct IsaacSimBatchAction: Sendable, Equatable, Codable {
    /// Stable sidecar environment identifier.
    public let environmentID: String

    /// Action for that environment.
    public let action: IsaacSimAction

    /// Creates one batch action.
    public init(environmentID: String, action: IsaacSimAction) throws {
        guard !environmentID.isEmpty else {
            throw RLSwiftError.emptyIdentifier(name: "isaacSim.environmentID")
        }
        self.environmentID = environmentID
        self.action = action
    }
}

/// Batch reset response keyed by environment id.
public struct IsaacSimBatchResetResponse: Sendable, Equatable, Codable {
    /// Initial observations returned by the sidecar.
    public let observations: [String: IsaacSimObservation]

    /// Creates a batch reset response.
    public init(observations: [String: IsaacSimObservation]) {
        self.observations = observations
    }
}

/// Batch step response keyed by environment id.
public struct IsaacSimBatchStepResponse: Sendable, Equatable, Codable {
    /// Step responses returned by the sidecar.
    public let responses: [String: IsaacSimStepResponse]

    /// Creates a batch step response.
    public init(responses: [String: IsaacSimStepResponse]) {
        self.responses = responses
    }

    /// Converts sidecar responses into RLSwift step results.
    public var stepResults: [String: StepResult<IsaacSimObservation>] {
        Dictionary(uniqueKeysWithValues: responses.map { key, response in
            (key, response.stepResult)
        })
    }
}

/// Action command sent to Isaac Sim.
public struct IsaacSimAction: Sendable, Equatable, Codable {
    /// Action values in the sidecar's expected control order.
    public let commands: [Double]

    /// Additional action metadata.
    public let metadata: [String: String]

    /// Creates one Isaac Sim action.
    public init(commands: [Double], metadata: [String: String] = [:]) {
        self.commands = commands
        self.metadata = metadata
    }
}

/// Observation returned by an Isaac Sim bridge.
public struct IsaacSimObservation: Sendable, Equatable, Codable {
    /// Simulation time in seconds.
    public let time: Double

    /// Sidecar step index.
    public let stepIndex: Int

    /// Flattened model-input features in policy order.
    public let features: [Double]

    /// Named sensor vectors for debugging or custom encoders.
    public let sensorReadings: [String: [Double]]

    /// Additional observation metadata.
    public let metadata: [String: String]

    /// Creates one Isaac Sim observation.
    public init(
        time: Double,
        stepIndex: Int,
        features: [Double],
        sensorReadings: [String: [Double]] = [:],
        metadata: [String: String] = [:]
    ) {
        self.time = time
        self.stepIndex = stepIndex
        self.features = features
        self.sensorReadings = sensorReadings
        self.metadata = metadata
    }

    /// Flattened model-input features.
    public var flattenedFeatures: [Double] {
        features
    }
}

/// Episode status returned by an Isaac Sim bridge step.
public enum IsaacSimEpisodeStatus: String, Sendable, Equatable, Codable {
    /// The episode should continue.
    case continuing

    /// The task reached a natural terminal condition.
    case terminated

    /// The episode stopped because of a time or external rollout limit.
    case truncated

    /// The episode stopped because a supervisor or safety system interrupted it.
    case interrupted

    /// Converts bridge status into RLSwift termination semantics.
    public func termination(reason: String?) -> StepTermination {
        switch self {
        case .continuing:
            return .continuing
        case .terminated:
            return .terminated(reason: reason ?? "terminated")
        case .truncated:
            return .truncated(reason: reason ?? "truncated")
        case .interrupted:
            return .interrupted(reason: reason ?? "interrupted")
        }
    }
}

/// Step response returned by an Isaac Sim bridge.
public struct IsaacSimStepResponse: Sendable, Equatable, Codable {
    /// Observation after the action was applied.
    public let observation: IsaacSimObservation

    /// Scalar reward emitted by the sidecar.
    public let reward: Double

    /// Episode status emitted by the sidecar.
    public let status: IsaacSimEpisodeStatus

    /// Optional human-readable status reason.
    public let reason: String?

    /// Additional step metadata.
    public let info: [String: String]

    /// Creates a step response.
    public init(
        observation: IsaacSimObservation,
        reward: Double,
        status: IsaacSimEpisodeStatus,
        reason: String? = nil,
        info: [String: String] = [:]
    ) {
        self.observation = observation
        self.reward = reward
        self.status = status
        self.reason = reason
        self.info = info
    }

    /// Converts this bridge response into an RLSwift step result.
    public var stepResult: StepResult<IsaacSimObservation> {
        let termination = status.termination(reason: reason)
        return StepResult(
            observation: observation,
            reward: reward,
            isTerminal: termination.endsEpisode,
            info: info,
            termination: termination
        )
    }
}

/// Health information returned by an Isaac Sim bridge.
public struct IsaacSimBridgeHealth: Sendable, Equatable, Codable {
    /// Whether the sidecar is ready to accept reset/step calls.
    public let isReady: Bool

    /// Isaac Sim or Isaac Lab version reported by the sidecar.
    public let simulatorVersion: String?

    /// Enabled sidecar extensions or capabilities.
    public let extensions: [String]

    /// Creates a bridge health response.
    public init(isReady: Bool, simulatorVersion: String? = nil, extensions: [String] = []) {
        self.isReady = isReady
        self.simulatorVersion = simulatorVersion
        self.extensions = extensions
    }
}

/// HTTP method used by the bridge transport.
public enum IsaacSimHTTPMethod: String, Sendable, Equatable {
    /// GET request.
    case get = "GET"

    /// POST request.
    case post = "POST"
}

/// HTTP request passed from the bridge client to a transport.
public struct IsaacSimHTTPRequest: Sendable, Equatable {
    /// Request method.
    public let method: IsaacSimHTTPMethod

    /// Request URL.
    public let url: URL

    /// Request body.
    public let body: Data?

    /// Request headers.
    public let headers: [String: String]

    /// Request timeout in seconds.
    public let timeout: TimeInterval

    /// Creates a bridge HTTP request.
    public init(
        method: IsaacSimHTTPMethod,
        url: URL,
        body: Data? = nil,
        headers: [String: String] = [:],
        timeout: TimeInterval
    ) {
        self.method = method
        self.url = url
        self.body = body
        self.headers = headers
        self.timeout = timeout
    }
}

/// HTTP response returned by a bridge transport.
public struct IsaacSimHTTPResponse: Sendable, Equatable {
    /// HTTP status code.
    public let statusCode: Int

    /// Response body.
    public let body: Data

    /// Creates a bridge HTTP response.
    public init(statusCode: Int, body: Data) {
        self.statusCode = statusCode
        self.body = body
    }
}

/// Transport used by ``IsaacSimBridgeClient``.
public protocol IsaacSimBridgeTransport: Sendable {
    /// Sends one HTTP request and returns the response.
    func send(_ request: IsaacSimHTTPRequest) async throws -> IsaacSimHTTPResponse
}

/// URLSession-backed bridge transport.
public final class IsaacSimURLSessionTransport: IsaacSimBridgeTransport, @unchecked Sendable {
    private let session: URLSession

    /// Creates a URLSession-backed transport.
    public init(session: URLSession = .shared) {
        self.session = session
    }

    /// Sends one HTTP request.
    public func send(_ request: IsaacSimHTTPRequest) async throws -> IsaacSimHTTPResponse {
        var urlRequest = URLRequest(url: request.url)
        urlRequest.httpMethod = request.method.rawValue
        urlRequest.httpBody = request.body
        urlRequest.timeoutInterval = request.timeout
        for header in request.headers {
            urlRequest.setValue(header.value, forHTTPHeaderField: header.key)
        }
        let (data, response) = try await session.data(for: urlRequest)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw IsaacSimBridgeError.nonHTTPResponse
        }
        return IsaacSimHTTPResponse(statusCode: httpResponse.statusCode, body: data)
    }
}

/// JSON/HTTP client for an Isaac Sim sidecar.
public struct IsaacSimBridgeClient<Transport: IsaacSimBridgeTransport>: Sendable {
    private let configuration: IsaacSimBridgeConfiguration
    private let transport: Transport
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    /// Creates a bridge client.
    public init(configuration: IsaacSimBridgeConfiguration, transport: Transport) {
        self.configuration = configuration
        self.transport = transport
        encoder = JSONEncoder()
        decoder = JSONDecoder()
    }

    /// Reads bridge health.
    public func health() async throws -> IsaacSimBridgeHealth {
        try await get(configuration.healthPath)
    }

    /// Resets the Isaac Sim environment and returns the initial observation.
    public func reset(seed: UInt64? = nil) async throws -> IsaacSimObservation {
        try await reset(options: IsaacSimResetOptions(seed: seed))
    }

    /// Resets the Isaac Sim environment with explicit episode options.
    public func reset(options: IsaacSimResetOptions) async throws -> IsaacSimObservation {
        try await post(
            configuration.resetPath,
            body: IsaacSimResetRequest(
                scenePath: configuration.scenePath,
                robotPath: configuration.robotPath,
                seed: options.seed,
                episodeID: options.episodeID,
                randomization: options.randomization,
                metadata: configuration.metadata.merging(options.metadata) { _, override in override }
            )
        )
    }

    /// Applies an action and returns an RLSwift step result.
    public func step(
        _ action: IsaacSimAction,
        options: IsaacSimStepOptions = .standard
    ) async throws -> StepResult<IsaacSimObservation> {
        let response: IsaacSimStepResponse = try await post(
            configuration.stepPath,
            body: IsaacSimStepRequest(
                robotPath: configuration.robotPath,
                action: action,
                physicsSteps: options.physicsSteps,
                render: options.render,
                metadata: configuration.metadata.merging(options.metadata) { _, override in override }
            )
        )
        return response.stepResult
    }

    /// Resets multiple Isaac Sim or Isaac Lab environments through a batch endpoint.
    public func resetMany(
        _ environments: [IsaacSimEnvironmentHandle],
        options: IsaacSimResetOptions = .standard
    ) async throws -> IsaacSimBatchResetResponse {
        guard !environments.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        return try await post(
            configuration.batchResetPath,
            body: IsaacSimBatchResetRequest(
                scenePath: configuration.scenePath,
                environments: environments,
                seed: options.seed,
                episodeID: options.episodeID,
                randomization: options.randomization,
                metadata: configuration.metadata.merging(options.metadata) { _, override in override }
            )
        )
    }

    /// Steps multiple Isaac Sim or Isaac Lab environments through a batch endpoint.
    public func stepMany(
        _ actions: [IsaacSimBatchAction],
        options: IsaacSimStepOptions = .standard
    ) async throws -> IsaacSimBatchStepResponse {
        guard !actions.isEmpty else {
            throw RLSwiftError.invalidSampleCount(0)
        }
        return try await post(
            configuration.batchStepPath,
            body: IsaacSimBatchStepRequest(
                actions: actions,
                physicsSteps: options.physicsSteps,
                render: options.render,
                metadata: configuration.metadata.merging(options.metadata) { _, override in override }
            )
        )
    }

    private func get<Response: Decodable>(_ path: String) async throws -> Response {
        let request = try IsaacSimHTTPRequest(
            method: .get,
            url: configuration.url(for: path),
            headers: ["Accept": "application/json"],
            timeout: configuration.timeout
        )
        let response = try await transport.send(request)
        try Self.validate(response)
        return try decoder.decode(Response.self, from: response.body)
    }

    private func post<Body: Encodable, Response: Decodable>(_ path: String, body: Body) async throws -> Response {
        let encodedBody = try encoder.encode(body)
        let request = try IsaacSimHTTPRequest(
            method: .post,
            url: configuration.url(for: path),
            body: encodedBody,
            headers: ["Accept": "application/json", "Content-Type": "application/json"],
            timeout: configuration.timeout
        )
        let response = try await transport.send(request)
        try Self.validate(response)
        return try decoder.decode(Response.self, from: response.body)
    }

    private static func validate(_ response: IsaacSimHTTPResponse) throws {
        guard 200..<300 ~= response.statusCode else {
            throw IsaacSimBridgeError.httpStatus(
                code: response.statusCode,
                body: String(decoding: response.body, as: UTF8.self)
            )
        }
    }
}

private struct IsaacSimResetRequest: Sendable, Equatable, Codable {
    let scenePath: String?
    let robotPath: String
    let seed: UInt64?
    let episodeID: String?
    let randomization: [String: Double]
    let metadata: [String: String]
}

private struct IsaacSimStepRequest: Sendable, Equatable, Codable {
    let robotPath: String
    let action: IsaacSimAction
    let physicsSteps: Int
    let render: Bool
    let metadata: [String: String]
}

private struct IsaacSimBatchResetRequest: Sendable, Equatable, Codable {
    let scenePath: String?
    let environments: [IsaacSimEnvironmentHandle]
    let seed: UInt64?
    let episodeID: String?
    let randomization: [String: Double]
    let metadata: [String: String]
}

private struct IsaacSimBatchStepRequest: Sendable, Equatable, Codable {
    let actions: [IsaacSimBatchAction]
    let physicsSteps: Int
    let render: Bool
    let metadata: [String: String]
}
