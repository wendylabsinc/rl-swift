import Foundation
import RLSwift

@main
struct RobotGridWorldExample {
    static func main() throws {
        do {
            let configuration = try ExampleConfiguration.parse(CommandLine.arguments.dropFirst())
            let result = try RobotGridWorldRunner(configuration: configuration).run()
            print(result.summaryText)
            guard result.evaluation.succeeded else {
                throw ExampleError.trainingDidNotConverge(result.evaluation.path)
            }
        } catch let error as ExampleError {
            FileHandle.standardError.write(Data((error.message + "\n").utf8))
            throw ExitCode.failure
        }
    }
}

struct ExampleConfiguration: Sendable {
    var episodes = 240
    var maxSteps = 48
    var seed: UInt64 = 11

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
            case "--episodes":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ExampleError.usage("--episodes must be a positive integer")
                }
                configuration.episodes = parsed
            case "--max-steps":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ExampleError.usage("--max-steps must be a positive integer")
                }
                configuration.maxSteps = parsed
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
}

enum ExampleError: Error {
    case trainingDidNotConverge([GridState])
    case usage(String)

    var message: String {
        switch self {
        case let .trainingDidNotConverge(path):
            return "Training did not converge. Greedy path: \(path.map(\.description).joined(separator: " -> "))"
        case let .usage(message):
            return """
            \(message)
            Usage: swift run robot-grid-world [--episodes 240] [--max-steps 48] [--seed 11]
            """
        }
    }
}

enum ExitCode: Error {
    case failure
}

struct GridState: Hashable, Sendable, Codable, CustomStringConvertible {
    let x: Int
    let y: Int

    var description: String {
        "(\(x),\(y))"
    }
}

enum GridAction: String, CaseIterable, Hashable, Sendable, Codable {
    case right
    case down
    case left
    case up

    var delta: (x: Int, y: Int) {
        switch self {
        case .right:
            return (1, 0)
        case .down:
            return (0, 1)
        case .left:
            return (-1, 0)
        case .up:
            return (0, -1)
        }
    }

    var robotAction: RobotAction {
        let delta = delta
        return RobotAction(commands: [Double(delta.x), Double(delta.y)], mode: .velocity)
    }
}

struct GridWorld: Environment {
    let width: Int
    let height: Int
    let start: GridState
    let goal: GridState
    let hazards: Set<GridState>
    let maxSteps: Int
    private(set) var stepIndex = 0
    private(set) var position: GridState

    init(maxSteps: Int) {
        width = 5
        height = 5
        start = GridState(x: 0, y: 0)
        goal = GridState(x: 4, y: 4)
        hazards = [
            GridState(x: 2, y: 0),
            GridState(x: 2, y: 1),
            GridState(x: 2, y: 2),
            GridState(x: 2, y: 3),
        ]
        self.maxSteps = maxSteps
        position = start
    }

    mutating func reset() -> GridState {
        stepIndex = 0
        position = start
        return position
    }

    mutating func step(_ action: GridAction) throws -> StepResult<GridState> {
        stepIndex += 1
        let candidate = GridState(x: position.x + action.delta.x, y: position.y + action.delta.y)
        let next = contains(candidate) ? candidate : position
        position = next

        if next == goal {
            return StepResult(
                observation: next,
                reward: 1,
                isTerminal: true,
                info: ["event": "goal"],
                termination: .terminated(reason: "goal")
            )
        }
        if hazards.contains(next) {
            return StepResult(
                observation: next,
                reward: -1,
                isTerminal: true,
                info: ["event": "hazard"],
                termination: .interrupted(reason: "entered-hazard")
            )
        }
        if stepIndex >= maxSteps {
            return StepResult(
                observation: next,
                reward: -0.25,
                isTerminal: true,
                info: ["event": "time-limit"],
                termination: .truncated(reason: "time-limit")
            )
        }

        let wallPenalty = candidate == position && !contains(candidate) ? -0.08 : 0
        let shaping = -0.02 - 0.01 * Double(distanceToGoal(from: next))
        return StepResult(
            observation: next,
            reward: shaping + wallPenalty,
            isTerminal: false,
            info: ["event": "step"],
            termination: .continuing
        )
    }

    func distanceToGoal(from state: GridState) -> Int {
        abs(goal.x - state.x) + abs(goal.y - state.y)
    }

    func robotObservation(for state: GridState, stepIndex: Int) throws -> RobotObservation {
        try RobotObservation(
            jointPositions: [Double(state.x), Double(state.y)],
            jointVelocities: [0, 0],
            endEffectorPose: [Double(goal.x - state.x), Double(goal.y - state.y)],
            sensorReadings: ["manhattan_distance": Double(distanceToGoal(from: state))],
            stepIndex: stepIndex
        )
    }

    func render(path: [GridState] = []) -> String {
        let visited = Set(path)
        return (0..<height).map { y in
            (0..<width).map { x in
                let state = GridState(x: x, y: y)
                if state == start {
                    return "S"
                }
                if state == goal {
                    return "G"
                }
                if hazards.contains(state) {
                    return "#"
                }
                if visited.contains(state) {
                    return "*"
                }
                return "."
            }.joined(separator: " ")
        }.joined(separator: "\n")
    }

    private func contains(_ state: GridState) -> Bool {
        (0..<width).contains(state.x) && (0..<height).contains(state.y)
    }
}

struct EvaluationResult: Sendable {
    let succeeded: Bool
    let totalReward: Double
    let steps: Int
    let path: [GridState]
    let terminations: [StepTermination]
}

struct ExampleRunResult: Sendable {
    let configuration: ExampleConfiguration
    let evaluation: EvaluationResult
    let trainingReturn: Double
    let replayCount: Int
    let prioritizedReplayCount: Int
    let dataset: OfflineDataset<GridState, GridAction>
    let deploymentPlan: DeploymentPlan
    let telemetry: AutonomyTelemetrySummary
    let normalizedStartFeatures: [Double]
    let renderedGrid: String

    var summaryText: String {
        """
        RLSwift RobotGridWorld example
        episodes: \(configuration.episodes)
        maxSteps: \(configuration.maxSteps)
        seed: \(configuration.seed)
        trainingMeanReturnLast20: \(format(trainingReturn))
        evaluationSucceeded: \(evaluation.succeeded)
        evaluationReturn: \(format(evaluation.totalReward))
        evaluationSteps: \(evaluation.steps)
        evaluationPath: \(evaluation.path.map(\.description).joined(separator: " -> "))
        replayCount: \(replayCount)
        prioritizedReplayCount: \(prioritizedReplayCount)
        datasetTransitions: \(dataset.manifest.transitionCount)
        datasetTerminations: \(format(dataset.manifest.terminationCounts))
        datasetConstraintCost: \(format(dataset.manifest.totalConstraintCost))
        deploymentTarget: \(deploymentPlan.target.name)
        deploymentHasBackendMetadata: \(deploymentPlan.hasRequiredBackendMetadata)
        telemetrySteps: \(telemetry.stepCount)
        telemetryMeanLatency: \(format(telemetry.meanClosedLoopLatency))
        telemetryDeadlineMisses: \(telemetry.deadlineMissCount)
        telemetryPolicyVersions: \(format(telemetry.policyVersionCounts))
        normalizedStartFeatures: \(normalizedStartFeatures.map(format).joined(separator: ", "))
        learnedGrid:
        \(renderedGrid)
        """
    }
}

struct RobotGridWorldRunner {
    let configuration: ExampleConfiguration

    func run() throws -> ExampleRunResult {
        var environment = GridWorld(maxSteps: configuration.maxSteps)
        var agent = try TabularQAgent<GridState, GridAction>(
            actions: GridAction.allCases,
            learningRate: 0.35,
            discount: 0.94,
            epsilon: 0.18,
            seed: configuration.seed
        )
        var replay = try ReplayBuffer<Transition<GridState, GridAction>>(
            capacity: max(32, configuration.episodes * configuration.maxSteps),
            seed: configuration.seed &+ 1
        )
        var prioritizedReplay = try PrioritizedReplayBuffer<LoggedTransition<GridState, GridAction>>(
            capacity: max(32, configuration.episodes * configuration.maxSteps),
            exponent: 0.7,
            seed: configuration.seed &+ 2
        )
        var normalizer = try ObservationNormalizer(dimension: 5)
        var telemetry = AutonomyTelemetryAccumulator()
        let safetySupervisor = try makeSafetySupervisor()
        let rollout = try PolicyVersionRollout(
            currentVersion: "gridworld-tabular-v1",
            candidateVersion: "gridworld-tabular-v2",
            candidateTrafficFraction: 0.25,
            seed: configuration.seed
        )

        var loggedTransitions: [LoggedTransition<GridState, GridAction>] = []
        var returns: [Double] = []
        var globalStep = 0

        for episodeIndex in 0..<configuration.episodes {
            var state = environment.reset()
            var episode = Episode<GridState, GridAction>()
            var previousRobotAction: RobotAction?

            while !(episode.transitions.last?.isTerminal ?? false) {
                let action = try agent.action(for: state)
                let robotObservation = try environment.robotObservation(for: state, stepIndex: globalStep)
                let rawFeatures = try makeModelContract().encode(robotObservation, applyNormalization: false)
                try normalizer.update(with: rawFeatures)

                let timing = try ControlTiming(
                    stepIndex: globalStep,
                    deltaTime: 0.05,
                    sensorAge: 0.002 + Double(globalStep % 5) * 0.001,
                    actionLatency: 0.003
                )
                let safetyDecision = try safetySupervisor.assess(
                    SafetySupervisorInput(
                        requestedAction: action.robotAction,
                        previousAction: previousRobotAction,
                        timing: timing
                    )
                )
                previousRobotAction = safetyDecision.commandedAction ?? previousRobotAction

                let step = try environment.step(action)
                let transition = Transition(
                    observation: state,
                    action: action,
                    reward: step.reward,
                    nextObservation: step.observation,
                    isTerminal: step.isTerminal,
                    termination: step.termination
                )
                agent.observe(transition)
                episode.append(transition)
                replay.append(transition)

                let constraints = try constraintReport(environment: environment, state: step.observation)
                let policyVersion = rollout.selectedVersion(for: "episode-\(episodeIndex)")
                try telemetry.record(
                    timing: timing,
                    maximumLatency: 0.02,
                    safetyDecision: safetyDecision,
                    constraints: constraints,
                    policyVersion: policyVersion
                )

                let logged = LoggedTransition(
                    transition: transition,
                    recordedAt: Date(timeIntervalSince1970: TimeInterval(globalStep)),
                    timing: timing,
                    constraints: constraints,
                    safetyDecision: safetyDecision,
                    metadata: [
                        "episode": "\(episodeIndex)",
                        "event": step.info["event"] ?? "unknown",
                        "policyVersion": policyVersion,
                    ]
                )
                loggedTransitions.append(logged)
                try prioritizedReplay.append(
                    logged,
                    priority: max(0.01, abs(step.reward) + constraints.totalCost + (step.termination.isInterrupted ? 1 : 0))
                )

                state = step.observation
                globalStep += 1
            }
            returns.append(episode.totalReward)
        }

        let finalContract = try makeModelContract(normalization: NormalizationSnapshot(normalizer))
        let dataset = try OfflineDataset(
            provenance: DatasetProvenance(
                datasetID: "gridworld-\(configuration.seed)",
                sourceSystem: "swift-example",
                robotID: "sim-gridbot-1",
                environment: "RobotGridWorld",
                collectionStartedAt: Date(timeIntervalSince1970: 0),
                collectionEndedAt: Date(timeIntervalSince1970: TimeInterval(globalStep)),
                metadata: ["episodes": "\(configuration.episodes)"]
            ),
            modelContract: finalContract,
            transitions: loggedTransitions,
            createdAt: Date(timeIntervalSince1970: TimeInterval(globalStep))
        )
        let deploymentPlan = DeploymentPlan(
            target: try DeploymentTarget.appleMLX(platform: "macos-ios-visionos"),
            modelContract: finalContract,
            deterministicSeed: configuration.seed,
            metadata: ["example": "RobotGridWorld"]
        )

        let evaluation = try evaluate(agent: agent, maxSteps: configuration.maxSteps)
        let startObservation = try environment.robotObservation(for: environment.start, stepIndex: 0)
        let normalizedStartFeatures = try finalContract.encode(startObservation)
        let meanReturn = returns.suffix(20).reduce(0, +) / Double(min(20, returns.count))

        return ExampleRunResult(
            configuration: configuration,
            evaluation: evaluation,
            trainingReturn: meanReturn,
            replayCount: replay.count,
            prioritizedReplayCount: prioritizedReplay.count,
            dataset: dataset,
            deploymentPlan: deploymentPlan,
            telemetry: telemetry.summary,
            normalizedStartFeatures: normalizedStartFeatures,
            renderedGrid: environment.render(path: evaluation.path)
        )
    }

    private func evaluate(
        agent: TabularQAgent<GridState, GridAction>,
        maxSteps: Int
    ) throws -> EvaluationResult {
        var environment = GridWorld(maxSteps: maxSteps)
        var state = environment.reset()
        var totalReward = 0.0
        var path = [state]
        var terminations: [StepTermination] = []

        for _ in 0..<maxSteps {
            let action = greedyAction(for: state, using: agent)
            let step = try environment.step(action)
            totalReward += step.reward
            state = step.observation
            path.append(state)
            terminations.append(step.termination)
            if step.isTerminal {
                return EvaluationResult(
                    succeeded: state == environment.goal,
                    totalReward: totalReward,
                    steps: path.count - 1,
                    path: path,
                    terminations: terminations
                )
            }
        }

        return EvaluationResult(
            succeeded: false,
            totalReward: totalReward,
            steps: path.count - 1,
            path: path,
            terminations: terminations
        )
    }
}

private func makeModelContract(normalization: NormalizationSnapshot? = nil) throws -> ModelIOContract {
    try ModelIOContract(
        contractVersion: "1.0",
        observationFeatures: [
            ObservationFeature(name: "grid_x", index: 0, component: .jointPosition(index: 0), unit: "cell"),
            ObservationFeature(name: "grid_y", index: 1, component: .jointPosition(index: 1), unit: "cell"),
            ObservationFeature(name: "goal_dx", index: 2, component: .endEffectorPose(index: 0), unit: "cell"),
            ObservationFeature(name: "goal_dy", index: 3, component: .endEffectorPose(index: 1), unit: "cell"),
            ObservationFeature(name: "manhattan_distance", index: 4, component: .sensor(key: "manhattan_distance"), unit: "cell"),
        ],
        normalization: normalization,
        actionSpecifications: [
            ActionSpecification(name: "velocity_x", index: 0, unit: "cell/s", lowerBound: -1, upperBound: 1),
            ActionSpecification(name: "velocity_y", index: 1, unit: "cell/s", lowerBound: -1, upperBound: 1),
        ],
        actionMode: .velocity,
        tensorRTBindings: TensorRTBindingNames(inputName: "observations", outputName: "actions", profileName: "gridworld"),
        policyMetadata: PolicyMetadata(
            policyID: "robot-gridworld-tabular",
            version: "1.0.0",
            createdAt: Date(timeIntervalSince1970: 0),
            trainingRunID: "example-\(normalization?.count ?? 0)",
            userInfo: ["algorithm": "tabular-q-learning"]
        )
    )
}

private func makeSafetySupervisor() throws -> HardwareSafetySupervisor {
    let commandSpace = try ContinuousBoxSpace(lowerBounds: [-1, -1], upperBounds: [1, 1])
    let envelope = try RobotSafetyEnvelope(commandSpace: commandSpace, maximumDelta: [1, 1])
    return try HardwareSafetySupervisor(
        envelope: envelope,
        failsafeAction: RobotAction(commands: [0, 0], mode: .velocity),
        maximumSensorAge: 0.05,
        maximumClosedLoopLatency: 0.03
    )
}

private func constraintReport(environment: GridWorld, state: GridState) throws -> ConstraintReport {
    let distance = Double(environment.distanceToGoal(from: state))
    let hazardClearance = environment.hazards.map {
        abs($0.x - state.x) + abs($0.y - state.y)
    }.min() ?? Int.max
    return ConstraintReport([
        try ConstraintSignal(
            name: "distance_to_goal",
            value: distance,
            limit: 2,
            relation: .lessThanOrEqual,
            weight: 0.02
        ),
        try ConstraintSignal(
            name: "hazard_clearance",
            value: Double(hazardClearance),
            limit: 1,
            relation: .greaterThanOrEqual,
            weight: 0.5
        ),
    ])
}

private func greedyAction(
    for state: GridState,
    using agent: TabularQAgent<GridState, GridAction>
) -> GridAction {
    var bestAction = GridAction.allCases[0]
    var bestValue = agent.qValue(for: state, action: bestAction)
    for action in GridAction.allCases.dropFirst() {
        let value = agent.qValue(for: state, action: action)
        if value > bestValue {
            bestValue = value
            bestAction = action
        }
    }
    return bestAction
}

private func format(_ value: Double) -> String {
    String(format: "%.4f", value)
}

private func format(_ values: [String: Int]) -> String {
    values
        .sorted { $0.key < $1.key }
        .map { "\($0.key)=\($0.value)" }
        .joined(separator: ",")
}
