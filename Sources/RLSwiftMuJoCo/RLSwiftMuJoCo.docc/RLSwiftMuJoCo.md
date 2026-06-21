# RLSwiftMuJoCo

Run RLSwift environment loops against MuJoCo physics models.

## Overview

RLSwiftMuJoCo is the opt-in physics simulation backend for RLSwift. When the
`MuJoCoBackend` package trait is enabled and MuJoCo headers plus libraries are
visible to `pkg-config`, this target imports the native MuJoCo C API through the
`CMuJoCo` system-library target.

Use this product when a robot policy should train, evaluate, or smoke-test
against MJCF/XML models while preserving the same RLSwift `Environment` loop
used by built-in or hardware-facing environments. The backend owns one
MuJoCo model/data pair, resets simulation state, writes actuator controls,
advances physics for a configurable frame skip, and returns position, velocity,
sensor, and time observations.

The robotics-critical surface includes deterministic state snapshots for replay
and branch evaluation, keyframe reset support, actuator control-range metadata,
control clipping or `[-1, 1]` normalization, and optional contact reports with
geom names, contact poses, normals, and force vectors. Those pieces are needed
for safe policy bounds, reproducible bug reports, curriculum resets, sim-to-real
debugging, and contact-rich locomotion or manipulation tasks.

Without the trait, the public configuration, observation, action, reward, and
support types still compile so downstream packages can keep one API surface and
report that native MuJoCo support is unavailable.

```sh
swift build --disable-default-traits --traits MuJoCoBackend
swift test --disable-default-traits --traits MuJoCoBackend
```

## Adapter Split

The core `RLSwift` target includes
`RobotIntegrationAdapterConfiguration.mujoco(modelPath:observationStream:actionStream:metadata:)`
for dependency-light planning metadata. This target provides the native
simulation runtime behind that descriptor.

Reward handling is intentionally generic. ``MuJoCoRewardSource`` covers smoke
tests and simple tasks, while production tasks can wrap ``MuJoCoEnvironment`` or
use the lower-level native simulation owner to compute domain-specific rewards,
termination conditions, curriculum updates, and reset randomization.

For normalized policy outputs, set ``MuJoCoEnvironmentConfiguration/actionMode``
to ``MuJoCoActionMode/normalizedToControlRange``. For replay, capture a
``MuJoCoStateSnapshot`` from ``MuJoCoSimulation/stateSnapshot(signature:)`` and
restore it with ``MuJoCoSimulation/restore(snapshot:includeContacts:)``.

## Topics

### Articles

- <doc:UsingMuJoCoBridge>

### Platform Support

- ``MuJoCoBackendSupport``
- ``MuJoCoBackendError``

### Environment Loop

- ``MuJoCoEnvironmentConfiguration``
- ``MuJoCoEnvironment``
- ``MuJoCoAction``
- ``MuJoCoActionMode``
- ``MuJoCoObservation``
- ``MuJoCoModelSummary``
- ``MuJoCoActuatorSummary``
- ``MuJoCoSensorSummary``
- ``MuJoCoContact``
- ``MuJoCoRewardSource``
- ``MuJoCoStateSignature``
- ``MuJoCoStateSnapshot``
- ``MuJoCoResetMode``
