import Foundation
import Testing
import RLSwift
@testable import RLSwiftIsaacSim
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

@Suite struct IsaacSimBridgeTests {
    @Test func reportsBridgeSupport() {
        let support = IsaacSimBackendSupport.current
        let explicit = IsaacSimBackendSupport(isBridgeAvailable: false, explanation: "disabled")

        #expect(support.isBridgeAvailable)
        #expect(support.explanation.contains("sidecar"))
        #expect(!explicit.isBridgeAvailable)
        #expect(explicit.explanation == "disabled")
    }

    @Test func validatesConfigurationAndBuildsURLs() throws {
        let configuration = try IsaacSimBridgeConfiguration(
            baseURL: try #require(URL(string: "http://127.0.0.1:8211/api")),
            scenePath: "warehouse.usd",
            robotPath: "/World/Carter",
            healthPath: "status",
            resetPath: "/episode/reset",
            stepPath: "https://bridge.example/step",
            timeout: 5,
            metadata: ["task": "nav"]
        )

        #expect(configuration.scenePath == "warehouse.usd")
        #expect(configuration.robotPath == "/World/Carter")
        #expect(try configuration.url(for: "status").absoluteString == "http://127.0.0.1:8211/api/status")
        #expect(try configuration.url(for: "/episode/reset").absoluteString == "http://127.0.0.1:8211/api/episode/reset")
        #expect(try configuration.url(for: "https://bridge.example/step").absoluteString == "https://bridge.example/step")
        #expect(throws: RLSwiftError.emptyIdentifier(name: "isaacSim.path")) {
            _ = try configuration.url(for: "")
        }

        #expect(throws: RLSwiftError.emptyIdentifier(name: "isaacSim.baseURL")) {
            _ = try IsaacSimBridgeConfiguration(baseURL: URL(fileURLWithPath: "/tmp"))
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "isaacSim.robotPath")) {
            _ = try IsaacSimBridgeConfiguration(baseURL: try #require(URL(string: "http://localhost")), robotPath: "")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "isaacSim.healthPath")) {
            _ = try IsaacSimBridgeConfiguration(baseURL: try #require(URL(string: "http://localhost")), healthPath: "")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "isaacSim.resetPath")) {
            _ = try IsaacSimBridgeConfiguration(baseURL: try #require(URL(string: "http://localhost")), resetPath: "")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "isaacSim.stepPath")) {
            _ = try IsaacSimBridgeConfiguration(baseURL: try #require(URL(string: "http://localhost")), stepPath: "")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "isaacSim.batchResetPath")) {
            _ = try IsaacSimBridgeConfiguration(baseURL: try #require(URL(string: "http://localhost")), batchResetPath: "")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "isaacSim.batchStepPath")) {
            _ = try IsaacSimBridgeConfiguration(baseURL: try #require(URL(string: "http://localhost")), batchStepPath: "")
        }
        #expect(throws: RLSwiftError.invalidDuration(name: "isaacSim.timeout", value: 0)) {
            _ = try IsaacSimBridgeConfiguration(baseURL: try #require(URL(string: "http://localhost")), timeout: 0)
        }
    }

    @Test func stepResponsesMapEpisodeStatusToRLSwiftTermination() {
        let observation = IsaacSimObservation(time: 1, stepIndex: 4, features: [1, 2], sensorReadings: ["imu": [3]], metadata: ["frame": "4"])
        let continuing = IsaacSimStepResponse(observation: observation, reward: 0.5, status: .continuing).stepResult
        let terminated = IsaacSimStepResponse(observation: observation, reward: 1, status: .terminated).stepResult
        let truncated = IsaacSimStepResponse(observation: observation, reward: 1, status: .truncated, reason: "horizon").stepResult
        let defaultTruncated = IsaacSimStepResponse(observation: observation, reward: 1, status: .truncated).stepResult
        let interrupted = IsaacSimStepResponse(observation: observation, reward: -1, status: .interrupted).stepResult

        #expect(observation.flattenedFeatures == [1, 2])
        #expect(!continuing.isTerminal)
        #expect(terminated.termination == .terminated(reason: "terminated"))
        #expect(truncated.termination == .truncated(reason: "horizon"))
        #expect(defaultTruncated.termination == .truncated(reason: "truncated"))
        #expect(interrupted.termination == .interrupted(reason: "interrupted"))
    }

    @Test func bridgeClientSendsHealthResetAndStepRequests() async throws {
        let encoder = JSONEncoder()
        let health = try encoder.encode(IsaacSimBridgeHealth(isReady: true, simulatorVersion: "6.0", extensions: ["ros2"]))
        let resetObservation = try encoder.encode(IsaacSimObservation(time: 0, stepIndex: 0, features: [0, 1]))
        let stepResponse = try encoder.encode(IsaacSimStepResponse(
            observation: IsaacSimObservation(time: 0.02, stepIndex: 1, features: [1, 2]),
            reward: 2,
            status: .terminated,
            reason: "goal",
            info: ["contact": "clear"]
        ))
        let transport = RecordingTransport(responses: [
            IsaacSimHTTPResponse(statusCode: 200, body: health),
            IsaacSimHTTPResponse(statusCode: 200, body: resetObservation),
            IsaacSimHTTPResponse(statusCode: 200, body: stepResponse),
        ])
        let configuration = try IsaacSimBridgeConfiguration(
            baseURL: try #require(URL(string: "http://127.0.0.1:8211")),
            scenePath: "warehouse.usd",
            robotPath: "/World/Carter",
            metadata: ["task": "nav"]
        )
        let client = IsaacSimBridgeClient(configuration: configuration, transport: transport)

        #expect(try await client.health() == IsaacSimBridgeHealth(isReady: true, simulatorVersion: "6.0", extensions: ["ros2"]))
        #expect(try await client.reset(options: IsaacSimResetOptions(seed: 42, metadata: ["task": "reset"])).features == [0, 1])
        let result = try await client.step(
            IsaacSimAction(commands: [0.25, -0.1], metadata: ["mode": "velocity"]),
            options: IsaacSimStepOptions(physicsSteps: 2, metadata: ["task": "step"])
        )

        #expect(result.reward == 2)
        #expect(result.termination == .terminated(reason: "goal"))
        #expect(result.info == ["contact": "clear"])

        let requests = await transport.requests
        #expect(requests.map(\.method) == [.get, .post, .post])
        #expect(requests.map { $0.url.path } == ["/health", "/reset", "/step"])
        #expect(requests[0].headers["Accept"] == "application/json")
        #expect(requests[1].headers["Content-Type"] == "application/json")
        let resetBody = try JSONDecoder().decode(ResetBodySnapshot.self, from: try #require(requests[1].body))
        let stepBody = try JSONDecoder().decode(StepBodySnapshot.self, from: try #require(requests[2].body))
        #expect(resetBody.scenePath == "warehouse.usd")
        #expect(resetBody.robotPath == "/World/Carter")
        #expect(resetBody.seed == 42)
        #expect(resetBody.metadata["task"] == "reset")
        #expect(stepBody.robotPath == "/World/Carter")
        #expect(stepBody.action.commands == [0.25, -0.1])
        #expect(stepBody.action.metadata["mode"] == "velocity")
        #expect(stepBody.physicsSteps == 2)
        #expect(!stepBody.render)
        #expect(stepBody.metadata["task"] == "step")
    }

    @Test func bridgeClientSeedResetConvenienceSendsResetRequest() async throws {
        let resetObservation = try JSONEncoder().encode(IsaacSimObservation(time: 0, stepIndex: 0, features: [9]))
        let transport = RecordingTransport(responses: [
            IsaacSimHTTPResponse(statusCode: 200, body: resetObservation),
        ])
        let configuration = try IsaacSimBridgeConfiguration(baseURL: try #require(URL(string: "http://127.0.0.1:8211")))
        let client = IsaacSimBridgeClient(configuration: configuration, transport: transport)

        #expect(try await client.reset(seed: 99).features == [9])

        let request = try #require(await transport.requests.first)
        let resetBody = try JSONDecoder().decode(ResetBodySnapshot.self, from: try #require(request.body))
        #expect(request.url.path == "/reset")
        #expect(resetBody.seed == 99)
        #expect(resetBody.robotPath == "/World/Robot")
        #expect(resetBody.randomization.isEmpty)
        #expect(resetBody.metadata.isEmpty)
    }

    @Test func validatesOptionsAndSendsBatchRequests() async throws {
        let resetOptions = try IsaacSimResetOptions(
            seed: 7,
            episodeID: "episode-7",
            randomization: ["friction": 0.4],
            metadata: ["curriculum": "rough", "task": "batch-reset"]
        )
        let stepOptions = try IsaacSimStepOptions(
            physicsSteps: 4,
            render: true,
            metadata: ["phase": "train", "task": "batch-step"]
        )
        let envA = try IsaacSimEnvironmentHandle(
            id: "env-0",
            robotPath: "/World/envs/env_0/Robot",
            taskName: "Navigation",
            metadata: ["split": "train"]
        )
        let envB = try IsaacSimEnvironmentHandle(id: "env-1", robotPath: "/World/envs/env_1/Robot")
        let actionA = try IsaacSimBatchAction(environmentID: "env-0", action: IsaacSimAction(commands: [0.1]))
        let actionB = try IsaacSimBatchAction(environmentID: "env-1", action: IsaacSimAction(commands: [-0.1]))
        let encoder = JSONEncoder()
        let batchReset = try encoder.encode(IsaacSimBatchResetResponse(observations: [
            "env-0": IsaacSimObservation(time: 0, stepIndex: 0, features: [1]),
            "env-1": IsaacSimObservation(time: 0, stepIndex: 0, features: [2]),
        ]))
        let batchStep = try encoder.encode(IsaacSimBatchStepResponse(responses: [
            "env-0": IsaacSimStepResponse(
                observation: IsaacSimObservation(time: 0.04, stepIndex: 4, features: [3]),
                reward: 1,
                status: .continuing
            ),
            "env-1": IsaacSimStepResponse(
                observation: IsaacSimObservation(time: 0.04, stepIndex: 4, features: [4]),
                reward: -1,
                status: .interrupted,
                reason: "safety"
            ),
        ]))
        let transport = RecordingTransport(responses: [
            IsaacSimHTTPResponse(statusCode: 200, body: batchReset),
            IsaacSimHTTPResponse(statusCode: 200, body: batchStep),
        ])
        let configuration = try IsaacSimBridgeConfiguration(
            baseURL: try #require(URL(string: "http://127.0.0.1:8211/api")),
            scenePath: "warehouse.usd",
            robotPath: "/World/Carter",
            metadata: ["task": "nav"],
            batchResetPath: "reset_many",
            batchStepPath: "step_many"
        )
        let client = IsaacSimBridgeClient(configuration: configuration, transport: transport)

        let resetResponse = try await client.resetMany([envA, envB], options: resetOptions)
        let stepResponse = try await client.stepMany([actionA, actionB], options: stepOptions)

        #expect(resetResponse.observations["env-0"]?.features == [1])
        #expect(resetResponse.observations["env-1"]?.features == [2])
        #expect(stepResponse.stepResults["env-0"]?.termination == .continuing)
        #expect(stepResponse.stepResults["env-1"]?.termination == .interrupted(reason: "safety"))

        let requests = await transport.requests
        #expect(requests.map { $0.url.path } == ["/api/reset_many", "/api/step_many"])
        let resetBody = try JSONDecoder().decode(BatchResetBodySnapshot.self, from: try #require(requests[0].body))
        let stepBody = try JSONDecoder().decode(BatchStepBodySnapshot.self, from: try #require(requests[1].body))
        #expect(resetBody.scenePath == "warehouse.usd")
        #expect(resetBody.environments == [envA, envB])
        #expect(resetBody.seed == 7)
        #expect(resetBody.episodeID == "episode-7")
        #expect(resetBody.randomization == ["friction": 0.4])
        #expect(resetBody.metadata["task"] == "batch-reset")
        #expect(resetBody.metadata["curriculum"] == "rough")
        #expect(stepBody.actions == [actionA, actionB])
        #expect(stepBody.physicsSteps == 4)
        #expect(stepBody.render)
        #expect(stepBody.metadata["task"] == "batch-step")
        #expect(stepBody.metadata["phase"] == "train")

        await #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try await client.resetMany([])
        }
        await #expect(throws: RLSwiftError.invalidSampleCount(0)) {
            _ = try await client.stepMany([])
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "isaacSim.episodeID")) {
            _ = try IsaacSimResetOptions(episodeID: "")
        }
        #expect(throws: RLSwiftError.invalidHorizon(0)) {
            _ = try IsaacSimStepOptions(physicsSteps: 0)
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "isaacSim.environmentID")) {
            _ = try IsaacSimEnvironmentHandle(id: "", robotPath: "/World/Robot")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "isaacSim.robotPath")) {
            _ = try IsaacSimEnvironmentHandle(id: "env", robotPath: "")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "isaacSim.taskName")) {
            _ = try IsaacSimEnvironmentHandle(id: "env", robotPath: "/World/Robot", taskName: "")
        }
        #expect(throws: RLSwiftError.emptyIdentifier(name: "isaacSim.environmentID")) {
            _ = try IsaacSimBatchAction(environmentID: "", action: IsaacSimAction(commands: []))
        }
    }

    @Test func bridgeClientRejectsHTTPFailures() async throws {
        let transport = RecordingTransport(responses: [
            IsaacSimHTTPResponse(statusCode: 503, body: Data("busy".utf8)),
        ])
        let configuration = try IsaacSimBridgeConfiguration(baseURL: try #require(URL(string: "http://127.0.0.1:8211")))
        let client = IsaacSimBridgeClient(configuration: configuration, transport: transport)

        await #expect(throws: IsaacSimBridgeError.httpStatus(code: 503, body: "busy")) {
            _ = try await client.health()
        }
    }

    @Test func urlSessionTransportMapsHTTPAndRejectsNonHTTPResponses() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        let session = URLSession(configuration: configuration)
        let transport = IsaacSimURLSessionTransport(session: session)
        StubURLProtocol.setHandler { request in
            #expect(request.httpMethod == "POST")
            #expect(request.value(forHTTPHeaderField: "X-Test") == "1")
            return (
                HTTPURLResponse(url: try #require(request.url), statusCode: 202, httpVersion: nil, headerFields: nil)!,
                Data("{}".utf8)
            )
        }

        let response = try await transport.send(IsaacSimHTTPRequest(
            method: .post,
            url: try #require(URL(string: "http://bridge.test/step")),
            body: Data([1, 2, 3]),
            headers: ["X-Test": "1"],
            timeout: 1
        ))
        #expect(response == IsaacSimHTTPResponse(statusCode: 202, body: Data("{}".utf8)))

        StubURLProtocol.setHandler { request in
            (URLResponse(url: try #require(request.url), mimeType: nil, expectedContentLength: 0, textEncodingName: nil), Data())
        }
        await #expect(throws: IsaacSimBridgeError.nonHTTPResponse) {
            _ = try await transport.send(IsaacSimHTTPRequest(
                method: .get,
                url: try #require(URL(string: "http://bridge.test/health")),
                timeout: 1
            ))
        }
    }
}

actor RecordingTransport: IsaacSimBridgeTransport {
    private var remainingResponses: [IsaacSimHTTPResponse]
    private(set) var requests: [IsaacSimHTTPRequest] = []

    init(responses: [IsaacSimHTTPResponse]) {
        remainingResponses = responses
    }

    func send(_ request: IsaacSimHTTPRequest) async throws -> IsaacSimHTTPResponse {
        requests.append(request)
        return remainingResponses.removeFirst()
    }
}

struct ResetBodySnapshot: Decodable {
    let scenePath: String?
    let robotPath: String
    let seed: UInt64?
    let episodeID: String?
    let randomization: [String: Double]
    let metadata: [String: String]
}

struct StepBodySnapshot: Decodable {
    let robotPath: String
    let action: IsaacSimAction
    let physicsSteps: Int
    let render: Bool
    let metadata: [String: String]
}

struct BatchResetBodySnapshot: Decodable, Equatable {
    let scenePath: String?
    let environments: [IsaacSimEnvironmentHandle]
    let seed: UInt64?
    let episodeID: String?
    let randomization: [String: Double]
    let metadata: [String: String]
}

struct BatchStepBodySnapshot: Decodable, Equatable {
    let actions: [IsaacSimBatchAction]
    let physicsSteps: Int
    let render: Bool
    let metadata: [String: String]
}

final class StubURLProtocol: URLProtocol {
    private static let storage = StubURLProtocolStorage()

    static func setHandler(_ handler: @escaping @Sendable (URLRequest) throws -> (URLResponse, Data)) {
        storage.setHandler(handler)
    }

    override class func canInit(with request: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        do {
            let (response, data) = try Self.storage.handle(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}

final class StubURLProtocolStorage: @unchecked Sendable {
    private let lock = NSLock()
    private var handler: @Sendable (URLRequest) throws -> (URLResponse, Data) = { request in
        (HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!, Data())
    }

    func setHandler(_ handler: @escaping @Sendable (URLRequest) throws -> (URLResponse, Data)) {
        lock.lock()
        self.handler = handler
        lock.unlock()
    }

    func handle(_ request: URLRequest) throws -> (URLResponse, Data) {
        lock.lock()
        let handler = handler
        lock.unlock()
        return try handler(request)
    }
}
