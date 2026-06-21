# Using the MuJoCo Bridge

Use the MuJoCo bridge when you want a native physics environment that runs
inside the Swift process and still presents the standard RLSwift
`Environment` shape.

## Overview

`RLSwiftMuJoCo` wraps one MJCF/XML model as an RLSwift environment. The target
loads a MuJoCo model, owns the matching `mjData`, writes actuator controls,
advances physics for a configured frame skip, and returns observations with
time, `qpos`, `qvel`, sensor data, actuator activations, controls, and optional
contact reports.

This is the right bridge for fast local smoke tests, repeatable rollout
collection, contact debugging, and policies that should be evaluated against
the exact native MuJoCo model used by robotics teams.

## Add the Product

Add `RLSwiftMuJoCo` to your package dependency and import it from the target
that owns simulator rollouts:

```swift
import RLSwift
import RLSwiftMuJoCo
```

The product is optional because MuJoCo is a native dependency. Enable it with
the `MuJoCoBackend` package trait and make sure `pkg-config` can find
`mujoco.pc`:

```sh
pkg-config --modversion mujoco
swift test --disable-default-traits --traits MuJoCoBackend
```

For CI or local machines without a system install, use the package helper:

```sh
eval "$(./scripts/install-mujoco-sdk.sh)"
swift test --disable-default-traits --traits MuJoCoBackend
```

Without the trait, support and configuration types still compile, but native
simulation constructors throw ``MuJoCoBackendError/backendUnavailable(_:)``.
Use ``MuJoCoBackendSupport/current`` when printing setup diagnostics.

## Describe a Simulator Without Linking MuJoCo

Core planning code can keep dependency-light metadata in the base `RLSwift`
target. That is useful for CLI configuration, deployment manifests, or tests
that should not link MuJoCo:

```swift
let adapter = try RobotIntegrationAdapterConfiguration.mujoco(
    modelPath: "models/go2.xml",
    observationStream: "qpos,qvel,sensors",
    actionStream: "ctrl",
    metadata: [
        "robot": "unitree-go2",
        "task": "rough-terrain"
    ]
)
```

Use the adapter as descriptive metadata. Use ``MuJoCoEnvironment`` or
``MuJoCoSimulation`` when you need physics.

## Run an Environment Loop

For the common case, configure ``MuJoCoEnvironment`` and use it through the same
loop as any other RLSwift environment:

```swift
var environment = try MuJoCoEnvironment(
    configuration: MuJoCoEnvironmentConfiguration(
        modelPath: "models/go2.xml",
        frameSkip: 4,
        maxEpisodeSteps: 1_000,
        rewardSource: .constant(0),
        actionMode: .normalizedToControlRange,
        includeContacts: true
    )
)

let initialObservation = environment.reset()
let action = MuJoCoAction(controls: Array(repeating: 0, count: 12))
let step = try environment.step(action)

print(initialObservation.qpos)
print(step.observation.contacts)
```

Use ``MuJoCoActionMode/normalizedToControlRange`` when a policy emits values in
`[-1, 1]` and the model actuators have control limits. Use
``MuJoCoActionMode/clipped`` when the policy already emits model-unit commands
but should be bounded before reaching `mjData.ctrl`.

## Inspect Model Metadata

Before connecting a trained policy, check that the model dimensions and actuator
ordering match the policy contract:

```swift
if let summary = environment.modelSummary {
    print("qpos:", summary.qposCount)
    print("qvel:", summary.qvelCount)

    for actuator in summary.actuators {
        print(
            actuator.index,
            actuator.name ?? "<unnamed>",
            actuator.minimumControl as Any,
            actuator.maximumControl as Any
        )
    }
}
```

Keep this actuator order aligned with your model export, ONNX/TensorRT binding
metadata, and real robot command order. A mismatch here is a deployment bug, not
a training detail.

## Reset From Keyframes or Saved State

MuJoCo keyframes are useful for curriculum starts and regression tests:

```swift
let resetObservation = try environment.reset(to: .keyframe(0))
print(resetObservation.time)
```

For deterministic replay, use the lower-level ``MuJoCoSimulation`` owner:

```swift
let simulation = try MuJoCoSimulation(modelPath: "models/go2.xml")
_ = simulation.reset()

let snapshot = simulation.stateSnapshot(signature: .integration)
_ = try simulation.step(
    controls: Array(repeating: 0, count: simulation.summary.actuatorCount),
    frameSkip: 4,
    actionMode: .normalizedToControlRange,
    includeContacts: true
)

let restored = try simulation.restore(snapshot: snapshot, includeContacts: true)
print(restored.qpos)
```

Store snapshots alongside failing actions when you need a compact bug report
that can reproduce a bad contact, unstable gait, or reward spike.

## Use Contacts for Robotics Debugging

Set ``MuJoCoEnvironmentConfiguration/includeContacts`` to `true` when debugging
locomotion, manipulation, or collision-heavy reset states. Each
``MuJoCoContact`` includes geom ids, optional geom names, contact distance,
position, normal, and a six-value contact-frame force vector.

Contacts are diagnostic data. Keep policy inputs explicit and stable; avoid
silently changing the observation vector just because contact reporting was
enabled for a debugging run.

## Connect to Training

`MuJoCoEnvironment` conforms to `Environment`, so you can use it with
RLSwift rollout utilities and agents that operate over generic environments.
For production tasks, wrap the environment when you need custom reward shaping,
termination conditions, randomized resets, or conversion from `qpos`/`qvel` to a
project-specific `RobotObservation`.

Recommended split:

- Use `RLSwiftMuJoCo` for fast native physics loops and reproducible state.
- Keep reward and termination logic in task-specific Swift code.
- Keep real hardware safety outside the learned policy with
  `HardwareSafetySupervisor` and `RobotSafetyEnvelope`.
- Record seeds, snapshots, actions, and termination reasons in offline datasets.

## Troubleshooting

- `backendUnavailable`: rebuild with `--traits MuJoCoBackend`.
- `module CMuJoCo not found`: make sure `PKG_CONFIG_PATH` contains the directory
  with `mujoco.pc`.
- `libmujoco` load failures on Linux: include the MuJoCo `lib` directory in
  `LD_LIBRARY_PATH`.
- `invalidControlCount`: align policy output size with
  ``MuJoCoModelSummary/actuatorCount``.
- Unexpected actuator motion: inspect ``MuJoCoActuatorSummary`` and verify that
  your action order matches MuJoCo's actuator order.
