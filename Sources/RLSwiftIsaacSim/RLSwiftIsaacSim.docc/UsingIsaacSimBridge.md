# Using the Isaac Sim Bridge

Use the Isaac Sim bridge when simulation runs in NVIDIA Isaac Sim or Isaac Lab
and Swift should drive policy, evaluation, logging, or deployment code over a
small JSON/HTTP contract.

## Overview

Isaac Sim usually runs as an Omniverse application, extension, or standalone
Python process. `RLSwiftIsaacSim` keeps that runtime out of the Swift package.
Instead, Swift sends health, reset, step, batch reset, and batch step requests
to a sidecar process and receives typed observations and step results.

This split keeps the Swift policy loop portable across Apple development
machines, NVIDIA Linux boxes, and CI while leaving USD loading, RTX sensors,
Isaac Lab tasks, ROS bridges, and simulator-specific reward logic in Python.

## Add the Product

Import the bridge in the target that owns simulator evaluation or training:

```swift
import RLSwift
import RLSwiftIsaacSim
```

The bridge has no Isaac Sim binary dependency, so tests can run without
Omniverse installed:

```sh
swift test --filter IsaacSimBridgeTests
swift package generate-documentation --target RLSwiftIsaacSim
```

Use ``IsaacSimBackendSupport/current`` when printing setup output. A ready Swift
bridge still needs a running sidecar before calls to ``IsaacSimBridgeClient``
can succeed.

## Describe a Bridge in Core RLSwift

Use dependency-light adapter metadata when configuration code should not import
the bridge product:

```swift
let adapter = try RobotIntegrationAdapterConfiguration.isaacSim(
    endpoint: "http://127.0.0.1:8211",
    robotPath: "/World/Go2",
    observationStream: "observation",
    actionStream: "action",
    metadata: [
        "scene": "warehouse.usd",
        "task": "velocity_tracking"
    ]
)
```

Use this descriptor in manifests, CLIs, and deployment planning. Use
``IsaacSimBridgeClient`` for live HTTP calls.

## Build a Client

Point the configuration at your sidecar base URL and robot prim path:

```swift
let configuration = try IsaacSimBridgeConfiguration(
    baseURL: URL(string: "http://127.0.0.1:8211")!,
    scenePath: "/sim/scenes/go2_warehouse.usd",
    robotPath: "/World/Go2",
    metadata: [
        "policy": "go2-walk-v3"
    ]
)

let client = IsaacSimBridgeClient(
    configuration: configuration,
    transport: IsaacSimURLSessionTransport()
)

let health = try await client.health()
guard health.isReady else {
    throw RLSwiftError.emptyIdentifier(name: "isaacSim.notReady")
}
```

Endpoint paths default to `/health`, `/reset`, `/step`, `/batch/reset`, and
`/batch/step`. Override them in ``IsaacSimBridgeConfiguration`` when your
extension exposes a different route layout.

## Reset With Reproducibility Metadata

Use ``IsaacSimResetOptions`` to carry all episode-level state that a sidecar
needs for reproducible evaluation:

```swift
let resetOptions = try IsaacSimResetOptions(
    seed: 42,
    episodeID: "eval-go2-00042",
    randomization: [
        "floor_friction": 0.85,
        "payload_mass_kg": 1.2
    ],
    metadata: [
        "split": "validation"
    ]
)

let observation = try await client.reset(options: resetOptions)
print(observation.features)
```

Keep randomization keys stable across Swift, Python, and offline logs. That
makes failures replayable and lets dataset tools reconstruct the episode
context.

## Step a Policy

Send one action in the sidecar's expected command order:

```swift
let action = IsaacSimAction(
    commands: Array(repeating: 0, count: 12),
    metadata: [
        "mode": "joint_position"
    ]
)

let step = try await client.step(
    action,
    options: IsaacSimStepOptions(
        physicsSteps: 4,
        render: true,
        metadata: [
            "sensor_frame": "camera_front"
        ]
    )
)

switch step.termination {
case .continuing:
    print("reward", step.reward)
case let .terminated(reason), let .truncated(reason), let .interrupted(reason):
    print("episode ended:", reason)
}
```

Use `physicsSteps` to keep the policy control period explicit. Use `render:
true` when the sidecar should synchronize cameras, lidar, or other render-pass
sensors with the step.

## Run Isaac Lab Style Batches

For vectorized tasks, register handles for each environment and call the batch
endpoints:

```swift
let environments = try (0..<4).map { index in
    try IsaacSimEnvironmentHandle(
        id: "env-\(index)",
        robotPath: "/World/envs/env_\(index)/Go2",
        taskName: "Go2Velocity",
        metadata: [
            "shard": "\(index)"
        ]
    )
}

let resetBatch = try await client.resetMany(
    environments,
    options: IsaacSimResetOptions(seed: 7)
)

let batchActions = try environments.map { handle in
    try IsaacSimBatchAction(
        environmentID: handle.id,
        action: IsaacSimAction(commands: Array(repeating: 0, count: 12))
    )
}

let batchStep = try await client.stepMany(
    batchActions,
    options: IsaacSimStepOptions(physicsSteps: 2)
)

let stepResults = batchStep.stepResults
print(resetBatch.observations.keys.sorted())
print(stepResults.keys.sorted())
```

The sidecar owns environment creation and simulator stepping. Swift keeps typed
IDs, actions, rewards, and termination states aligned with RLSwift rollout
collectors.

## Implement the Python Sidecar

The sidecar contract is intentionally small:

- `GET /health` returns ``IsaacSimBridgeHealth``.
- `POST /reset` accepts scene, robot, seed, episode id, randomization, and
  metadata, then returns ``IsaacSimObservation``.
- `POST /step` accepts robot path, ``IsaacSimAction``, physics steps, render
  intent, and metadata, then returns ``IsaacSimStepResponse``.
- `POST /batch/reset` and `POST /batch/step` mirror those calls for many
  environments.

Sidecar responsibilities usually include:

- Loading USD scenes and robot assets.
- Mapping Swift action order to simulator actuator names.
- Applying domain randomization.
- Advancing physics and render sensors.
- Computing task-specific reward and termination.
- Returning flattened model features plus named sensor vectors for debugging.

Keep the feature vector order pinned with `ModelIOContract` so Swift and Python
cannot silently disagree about model inputs.

## Production Boundaries

Use this bridge for simulator-driven policy loops. Do not treat an Isaac Sim
sidecar as the real robot safety boundary. When moving to hardware:

- Keep emergency stop, command freshness, joint limits, and rate limits outside
  the learned policy.
- Preserve termination semantics: task success, time-limit truncation, and
  safety interruption should remain different states.
- Log reset options, randomization, actions, observations, rewards, and
  termination reasons into offline datasets.
- Reuse the same action order and model IO metadata from simulator evaluation.

## Troubleshooting

- HTTP 404: check endpoint paths in ``IsaacSimBridgeConfiguration``.
- HTTP 500: surface the Python exception body; the Swift bridge preserves
  non-2xx response bodies in ``IsaacSimBridgeError/httpStatus(code:body:)``.
- Empty or shuffled feature vectors: validate the sidecar against
  `ModelIOContract`.
- Unstable camera or lidar observations: set ``IsaacSimStepOptions/render`` to
  `true` for steps that must synchronize render-pass sensors.
- Batch result mismatch: every ``IsaacSimBatchAction/environmentID`` should
  correspond to a sidecar environment returned by `resetMany`.
