# RLSwiftIsaacSim

Connect RLSwift policy loops to NVIDIA Isaac Sim sidecars.

## Overview

RLSwiftIsaacSim is the opt-in bridge surface for NVIDIA Isaac Sim and Isaac Lab
workflows. Isaac Sim is usually launched as an Omniverse application or
standalone Python process, so this target does not try to link Omniverse
libraries into Swift. Instead, it defines a JSON/HTTP sidecar contract that can
be implemented by a Python extension, standalone Isaac Sim script, Isaac Lab
task wrapper, or ROS-adjacent bridge.

Use ``IsaacSimBridgeClient`` when a Swift training or evaluation loop needs to
call health, reset, and step endpoints. The step response converts directly into
RLSwift `StepResult` semantics so task termination, time-limit truncation, and
safety interruption remain explicit.

The bridge includes robotics-critical controls for seeded resets, episode ids,
domain-randomization parameters, physics substeps, render-pass requests, and
Isaac Lab-style batch reset/step calls. That keeps vectorized training,
evaluation replay, camera/lidar synchronization, and curriculum metadata in the
typed Swift contract instead of ad hoc JSON dictionaries.

The core `RLSwift` target also includes
`RobotIntegrationAdapterConfiguration.isaacSim(endpoint:robotPath:observationStream:actionStream:metadata:)`
for dependency-light deployment planning metadata.

```sh
swift build
swift test --filter IsaacSimBridgeTests
swift package generate-documentation --target RLSwiftIsaacSim
```

## Bridge Contract

The bridge expects:

- `GET /health` returning ``IsaacSimBridgeHealth``.
- `POST /reset` accepting ``IsaacSimResetOptions`` fields and returning
  ``IsaacSimObservation``.
- `POST /step` accepting ``IsaacSimAction`` metadata and returning
  ``IsaacSimStepResponse``.
- `POST /batch/reset` accepting environment handles and reset options, returning
  ``IsaacSimBatchResetResponse``.
- `POST /batch/step` accepting batch actions and step options, returning
  ``IsaacSimBatchStepResponse``.

Endpoint paths are configurable through ``IsaacSimBridgeConfiguration``. A
sidecar can expose richer simulator behavior behind that small protocol:
loading USD scenes, selecting Isaac Lab tasks, domain randomization, sensor
streaming, curriculum state, or robot-specific reward logic.

## Topics

### Platform Support

- ``IsaacSimBackendSupport``
- ``IsaacSimBridgeError``

### Bridge Client

- ``IsaacSimBridgeConfiguration``
- ``IsaacSimBridgeClient``
- ``IsaacSimBridgeTransport``
- ``IsaacSimURLSessionTransport``
- ``IsaacSimHTTPRequest``
- ``IsaacSimHTTPResponse``
- ``IsaacSimHTTPMethod``

### Environment Data

- ``IsaacSimAction``
- ``IsaacSimResetOptions``
- ``IsaacSimStepOptions``
- ``IsaacSimEnvironmentHandle``
- ``IsaacSimBatchAction``
- ``IsaacSimObservation``
- ``IsaacSimEpisodeStatus``
- ``IsaacSimStepResponse``
- ``IsaacSimBatchResetResponse``
- ``IsaacSimBatchStepResponse``
- ``IsaacSimBridgeHealth``
