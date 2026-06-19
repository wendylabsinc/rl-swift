# SwiftRL

SwiftRL is a Swift 6.3+ reinforcement learning package with opt-in MLX and
TensorRT backends for Apple and NVIDIA Linux deployments.

## Toolchain

This package is pinned to Swift 6.3.2 through `.swift-version`.

```sh
swiftly install 6.3.2 --use
swift --version
```

## Build and Test

```sh
swift test
./scripts/check-coverage.sh
swift package generate-documentation --target SwiftRL
swift package generate-documentation --target SwiftRLMLX
swift package generate-documentation --target SwiftRLTensorRT
```

## Backend Traits

SwiftPM traits are the right fit for backend selection in Swift 6.3+. This
package keeps the core `SwiftRL` product dependency-light and puts tensor
integration behind separate products and traits:

- `MLXBackend` builds `SwiftRLMLX` and is enabled by default for Apple
  development.
- `TensorRTBackend` builds the native TensorRT path on NVIDIA Linux.
- `--disable-default-traits` gives a core-only or TensorRT-only build.

Useful commands:

```sh
swift package show-traits
swift build --disable-default-traits
swift test --traits MLXBackend
swift test --disable-default-traits --traits TensorRTBackend
```

## Apple Device Support

SwiftRL is intended to run on more than macOS. The package declares support for
macOS 14, iOS 17, tvOS 17, and visionOS 1, and iPadOS is covered by SwiftPM's
iOS platform target. The MLX Swift dependency used by this package declares the
same Apple platform set and links Apple's Metal and Accelerate frameworks on
Apple platforms.

That means the core RL loop, robot observation/action models, safety envelope,
replay buffer, normalization, and optional MLX-facing tensor contracts can be compiled
for recent iPhone, iPad, and Apple Vision Pro devices, not just Macs. In
practice, robot RL on mobile or spatial devices should usually use the device
for inference, adaptation, telemetry, simulation, or supervised evaluation, with
heavy offline training and large replay workloads kept on a Mac or server. Any
real robot control loop should keep hard safety limits outside the learned policy
as well as inside `RobotSafetyEnvelope`, because OS scheduling, thermal limits,
sensor latency, and wireless transport can all affect closed-loop behavior on
mobile devices.

## TensorRT on DGX and NVIDIA Linux

TensorRT is possible, but it should be treated as an NVIDIA Linux deployment
backend rather than an Apple-device backend. SwiftRL exposes a separate
`SwiftRLTensorRT` product that depends on
`https://github.com/wendylabsinc/tensorrt-swift` for Linux builds only. That
keeps iOS, iPadOS, visionOS, and macOS development from pulling CUDA/TensorRT
modules while allowing a DGX Spark, DGX workstation, Jetson-class robot, or
NVIDIA Linux host to load TensorRT engines for policy inference.

The intended flow is:

1. Train, fine-tune, or evaluate with `SwiftRL` and `SwiftRLMLX` on Apple
   hardware where that is convenient.
2. Export the policy to ONNX or a serialized TensorRT engine.
3. Build and run the `SwiftRLTensorRT` product with
   `--disable-default-traits --traits TensorRTBackend` on NVIDIA Linux with CUDA
   and TensorRT installed.
4. Keep hard robot safety limits, watchdogs, and emergency-stop handling outside
   the learned policy even when TensorRT inference is fast.

On Apple platforms, `TensorRTBackendSupport.current` reports that native
TensorRT is unavailable. On Linux builds where the `TensorRT` module is
importable, `TensorRTPolicyBackend` is compiled and can select optimization
profiles, reshape dynamic inputs, enqueue TensorRT inference, decode `Float32`
outputs, and convert them into `RobotAction` values.

## Robot and Autonomy Gap Analysis

Robots and autonomous systems need more than a basic reward/action loop. The
critical missing features are the pieces that keep data trustworthy and execution
bounded when a learned policy is connected to hardware:

- Episode endings must distinguish natural task termination, time-limit
  truncation, and safety interruption. SwiftRL models that with
  `StepTermination` on both `StepResult` and `Transition`.
- Control-loop data must carry timing and latency metadata, because stale
  observations and delayed commands can invalidate otherwise-correct RL updates.
  `ControlTiming` tracks control period, sensor age, action latency, and
  deadline misses.
- Robot learning often has constraints in addition to reward. `ConstraintSignal`
  and `ConstraintReport` represent safety and performance limits as explicit
  costs for constrained RL, offline filtering, and deployment monitoring.
- A safety shield must be auditable. `RobotSafetyEnvelope.assess` returns a
  `RobotSafetyAssessment` with every `SafetyIntervention`, so logs can tell
  whether a policy action was clipped or rate-limited before reaching hardware.
- Actuators need smooth commands. `ActionSmoother` provides a deterministic
  low-pass command filter that can sit before the safety envelope.
- Real robot failures are rare but important. `PrioritizedReplayBuffer` allows
  scarce safety, collision, timeout, or recovery events to be sampled more often
  than ordinary transitions.

### Need-to-Haves

- Versioned model IO contracts that pin observation order, normalization state,
  action units, TensorRT binding names, and policy metadata.
- Hardware-facing safety supervisors outside the policy: watchdogs, emergency
  stops, sensor freshness checks, actuator envelopes, and command-rate limits.
- Durable offline data pipelines with dataset manifests, provenance, timestamps,
  termination causes, and replayable safety interventions.
- Deterministic deployment backends for the target hardware: MLX on Apple
  devices and TensorRT on NVIDIA Linux.
- Real-time observability for latency, deadline misses, intervention counts,
  constraint costs, and policy version rollouts.

### Nice-to-Haves

- ROS 2, simulator, and WendyOS adapters for common robot integration paths.
- Vectorized environments and distributed rollout collection for larger training
  runs.
- ONNX export helpers and TensorRT engine cache management.
- Curriculum learning, domain randomization, and evaluation dashboards.
- Visual debugging tools for observation drift, action saturation, and rare
  prioritized replay events.

## Package Surface

- Generic `Environment` and `Agent` protocols.
- `StepTermination`, `Transition`, `StepResult`, and `Episode` for rollout accounting.
- `ControlTiming` for closed-loop latency and deadline metadata.
- Continuous `ContinuousBoxSpace` bounds for robot commands and observations.
- `RobotAction`, `RobotObservation`, `RobotSafetyEnvelope`, and `RobotSafetyAssessment` for robot control loops.
- `ActionSmoother` for low-pass command filtering.
- `ConstraintSignal` and `ConstraintReport` for constrained robot learning.
- `ObservationNormalizer` for streaming sensor-feature statistics.
- `RewardBreakdown` and `RewardComponent` for shaped robot rewards.
- Deterministic `ReplayBuffer` and `PrioritizedReplayBuffer` sampling.
- `EpsilonGreedyPolicy` and `SoftmaxPolicy` for discrete action spaces.
- `TabularQAgent` for small state-action spaces.
- `MLXBackendSupport`, `MLXObservationEncoder`, and `RLTensor` for MLX tensor integration.
- `TensorRTBackendSupport`, `TensorRTPolicyConfiguration`, and
  `TensorRTPolicyBackend` for NVIDIA Linux TensorRT inference.

Public interfaces are documented with DocC comments and a DocC catalog at
`Sources/SwiftRL/SwiftRL.docc`,
`Sources/SwiftRLMLX/SwiftRLMLX.docc`, and
`Sources/SwiftRLTensorRT/SwiftRLTensorRT.docc`.
