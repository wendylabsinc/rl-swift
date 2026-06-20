# RLSwift

[![E2E Integration Tests](https://github.com/wendylabsinc/rl-swift/actions/workflows/e2e-integration.yml/badge.svg)](https://github.com/wendylabsinc/rl-swift/actions/workflows/e2e-integration.yml)
![Swift 6.3+](https://img.shields.io/badge/Swift-6.3%2B-orange)
![Apple Platforms](https://img.shields.io/badge/Apple-macOS%2014%20%7C%20iOS%2017%20%7C%20iPadOS%2017%20%7C%20tvOS%2017%20%7C%20visionOS%201-blue)
![Linux](https://img.shields.io/badge/Linux-Ubuntu%2024.04%20%7C%20NVIDIA%20TensorRT-green)

RLSwift is a Swift 6.3+ reinforcement learning package with opt-in MLX and
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
(cd Examples/RobotGridWorld && swift run robot-grid-world)
swift package generate-documentation --target RLSwift
swift package generate-documentation --target RLSwiftMLX
swift package generate-documentation --target RLSwiftTensorRT
```

## Swift 6.3 Performance Notes

RLSwift uses Swift 6 ownership features where they are portable across the
declared platform matrix. Hot vector transforms in action scaling, normalization,
model IO, and control smoothing use internal move-only scratch storage
(`~Copyable`) to build a single owned output array without intermediate `map`
closures or accidental scratch copies.

`Span` and `InlineArray` were evaluated for public vector APIs and fixed-size
robot vectors, but in the current Apple Swift 6.3.2 toolchain they are gated to
OS 26 availability. Exposing them unconditionally would break the package's
macOS 14, iOS 17, iPadOS 17, tvOS 17, visionOS 1, and Linux support. They should
remain candidates for future availability-gated APIs once SwiftPM and the
supported deployment targets make that surface practical.

## Backend Traits

SwiftPM traits are the right fit for backend selection in Swift 6.3+. This
package keeps the core `RLSwift` product dependency-light and puts tensor
integration behind separate products and traits:

- `MLXBackend` builds `RLSwiftMLX` and is enabled by default for Apple
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

RLSwift is intended to run on more than macOS. The package declares support for
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
backend rather than an Apple-device backend. RLSwift exposes a separate
`RLSwiftTensorRT` product that depends on
`https://github.com/wendylabsinc/tensorrt-swift` for Linux builds only. That
keeps iOS, iPadOS, visionOS, and macOS development from pulling CUDA/TensorRT
modules while allowing a DGX Spark, DGX workstation, Jetson-class robot, or
NVIDIA Linux host to load TensorRT engines for policy inference.

The intended flow is:

1. Train, fine-tune, or evaluate with `RLSwift` and `RLSwiftMLX` on Apple
   hardware where that is convenient.
2. Export the policy to ONNX or a serialized TensorRT engine.
3. Build and run the `RLSwiftTensorRT` product with
   `--disable-default-traits --traits TensorRTBackend` on NVIDIA Linux with CUDA
   and TensorRT installed.
4. Keep hard robot safety limits, watchdogs, and emergency-stop handling outside
   the learned policy even when TensorRT inference is fast.

On Apple platforms, `TensorRTBackendSupport.current` reports that native
TensorRT is unavailable. On Linux builds where the `TensorRT` module is
importable, `TensorRTPolicyBackend` is compiled and can select optimization
profiles, reshape dynamic inputs, enqueue TensorRT inference, decode `Float32`
outputs, and convert them into `RobotAction` values.

## Examples

Runnable example projects live in `Examples/`. Start with `RobotGridWorld`,
which trains a tabular Q-learning policy in a deterministic navigation task and
exercises the robot/autonomy API surface around the loop:

```sh
cd Examples/RobotGridWorld
swift run robot-grid-world
```

## Robot and Autonomy Features

RLSwift includes robot and autonomous-system support beyond a basic
reward/action loop. These features keep rollout data trustworthy and execution
bounded when a learned policy is connected to hardware:

- Episode endings must distinguish natural task termination, time-limit
  truncation, and safety interruption. RLSwift models that with
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

| Feature area | Capability | RLSwift API surface |
| --- | --- | --- |
| Model IO contracts | Versioned observation order, normalization state, action units, TensorRT binding names, and policy metadata. | `ModelIOContract`, `ObservationFeature`, `NormalizationSnapshot`, `ActionSpecification`, `TensorRTBindingNames`, `PolicyMetadata` |
| Safety supervision | Hardware-facing watchdog checks, emergency stops, sensor freshness, actuator envelopes, and command-rate limits outside the learned policy. | `HardwareSafetySupervisor`, `SafetySupervisorInput`, `SafetySupervisorDecision`, `EmergencyStopState` |
| Offline datasets | Durable manifests, provenance, timestamps, termination causes, constraint costs, and replayable safety interventions. | `OfflineDataset`, `LoggedTransition`, `DatasetManifest`, `DatasetProvenance` |
| Deployment planning | Deterministic backend selection for MLX on Apple devices, TensorRT on NVIDIA Linux, and core Swift fallback deployments. | `DeploymentTarget`, `DeploymentPlan`, `DeploymentBackend` |
| Observability | Real-time summaries for latency, deadline misses, intervention counts, constraint costs, and policy-version rollout decisions. | `AutonomyTelemetryAccumulator`, `AutonomyTelemetrySummary`, `PolicyVersionRollout` |
| Robot adapters | Dependency-light ROS 2, simulator, and WendyOS adapter descriptors. | `RobotIntegrationAdapterConfiguration` |
| Rollout collection | Vectorized environments and distributed rollout sharding. | `VectorizedEnvironmentRunner`, `RolloutShardAssignment` |
| Export and engine cache | ONNX export descriptors and TensorRT engine cache metadata. | `ONNXExportDescriptor`, `TensorRTEngineCacheKey`, `TensorRTEngineCacheManifest` |
| Training workflows | Curriculum learning, domain randomization, and evaluation dashboard summaries. | `CurriculumStage`, `CurriculumSchedule`, `DomainRandomizationParameter`, `DomainRandomizationProfile`, `EvaluationRecord`, `EvaluationDashboardSummary` |
| Visual debugging | Observation drift, action saturation, and rare prioritized replay event snapshots. | `ObservationDriftSnapshot`, `ActionSaturationSnapshot`, `PrioritizedReplayDebugSnapshot` |

## Package Surface

- Generic `Environment` and `Agent` protocols.
- `StepTermination`, `Transition`, `StepResult`, and `Episode` for rollout accounting.
- `ControlTiming` for closed-loop latency and deadline metadata.
- `ModelIOContract` and related IO metadata types for versioned policy inputs
  and outputs.
- Continuous `ContinuousBoxSpace` bounds for robot commands and observations.
- `RobotAction`, `RobotObservation`, `RobotSafetyEnvelope`, and `RobotSafetyAssessment` for robot control loops.
- `HardwareSafetySupervisor` and related decision types for deployment-time
  safety checks outside the policy.
- `OfflineDataset`, `DatasetManifest`, and `LoggedTransition` for durable robot
  datasets.
- `DeploymentPlan`, `DeploymentTarget`, and `AutonomyTelemetryAccumulator` for
  deployment and observability.
- Adapter, vectorized rollout, export/cache, curriculum, evaluation, and visual
  debug descriptors for autonomy workflows.
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
`Sources/RLSwift/RLSwift.docc`,
`Sources/RLSwiftMLX/RLSwiftMLX.docc`, and
`Sources/RLSwiftTensorRT/RLSwiftTensorRT.docc`.
