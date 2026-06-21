# RLSwift

[![E2E Integration Tests](https://github.com/wendylabsinc/rl-swift/actions/workflows/e2e-integration.yml/badge.svg)](https://github.com/wendylabsinc/rl-swift/actions/workflows/e2e-integration.yml)
![Swift 6.3+](https://img.shields.io/badge/Swift-6.3%2B-orange)
![Apple Platforms](https://img.shields.io/badge/Apple-macOS%2014%20%7C%20iOS%2017%20%7C%20iPadOS%2017%20%7C%20tvOS%2017%20%7C%20visionOS%201-blue)
![Linux](https://img.shields.io/badge/Linux-Ubuntu%2024.04%20%7C%20NVIDIA%20TensorRT-green)

RLSwift is a Swift 6.3+ reinforcement learning package with opt-in MLX,
TensorRT, MuJoCo, and Isaac Sim surfaces for Apple, NVIDIA Linux, and robotics
simulation workflows.

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
swift run rl-swift catalog
swift run rl-swift train --episodes 24
swift run rl-swift eval --episodes 4
swift run rl-swift experiment
swift run rl-swift checkpoint
swift run rl-swift sweep
swift run rl-swift protein
swift run rl-swift visualize
(cd Examples/RobotGridWorld && swift run robot-grid-world)
(cd Examples/VectorizedPPO && swift run vectorized-ppo --iterations 2 --envs 4 --rollout 3)
swift package generate-documentation --target RLSwift
swift package generate-documentation --target RLSwiftMLX
swift package generate-documentation --target RLSwiftTensorRT
swift package --disable-default-traits --traits MuJoCoBackend generate-documentation --target RLSwiftMuJoCo
swift package generate-documentation --target RLSwiftIsaacSim
```

## Documentation Site

DocC reference pages and simulator bridge articles are published to GitHub Pages
from the `Publish DocC` workflow. The site builds static documentation for
`RLSwift`, `RLSwiftIsaacSim`, and `RLSwiftMuJoCo`.

To build the same site locally:

```sh
eval "$(./scripts/install-mujoco-sdk.sh)"
./scripts/build-docc-site.sh --output-path .build/docc-site --hosting-base-path rl-swift
```

The generated site entry point is `.build/docc-site/index.html`.

## CLI Workflow

The root package includes an `rl-swift` executable for local smoke tests and CI
logs:

```sh
swift run rl-swift catalog
swift run rl-swift train --episodes 24
swift run rl-swift eval --episodes 4
swift run rl-swift experiment
swift run rl-swift checkpoint
swift run rl-swift sweep
swift run rl-swift protein
swift run rl-swift visualize
```

Use `swift run --disable-default-traits rl-swift catalog` for core-only Linux
checks that should not compile MLX.

## PufferLib Comparison

RLSwift now covers the core workflow shape that makes PufferLib useful, but it
is intentionally Swift-native and robot/autonomy-oriented rather than a Python
drop-in replacement.

| PufferLib capability | RLSwift status | API or command |
| --- | --- | --- |
| Training throughput reporting | Implemented as backend-agnostic counters and rates. | `ThroughputMeter`, `TrainingThroughputReport` |
| PPO learner math | Implemented for GAE, clipped PPO objective evaluation, value clipping, categorical/multidiscrete/continuous action log-probability math, coefficient annealing, gradient clipping, prioritized trajectory segments, and a deterministic neural actor-critic PPO optimizer loop. | `PPOConfiguration`, `PPOAnnealingSchedule`, `PPOActionDistribution`, `PPOGradientClipper`, `PPOTrajectorySegmentSampler`, `PPOAdvantageEstimator`, `PPOClippedObjective`, `NeuralPPOTrainer`, `DenseDiscreteActorCriticModel` |
| Structured spaces | Implemented for flattening and unflattening named observation/action fields in model order. | `StructuredTensorSchema`, `StructuredObservationSchema`, `StructuredActionSchema` |
| Vectorized rollout profiles | Implemented as serial/threaded/multiprocessing/CUDA/TensorRT metadata, the sequential runner, and an in-process async Swift actor runner. | `VectorizationProfile`, `VectorizedEnvironmentRunner`, `AsyncVectorizedEnvironmentRunner`, `AsyncEnvironmentWorker` |
| Checkpoint bookkeeping | Implemented as policy/checkpoint manifests with metrics and vectorization provenance. | `PolicyMetadata`, `PolicyCheckpointManifest` |
| Self-play pools | Implemented as bounded frozen-opponent metadata with deterministic sampling and Elo-style updates. | `SelfPlayOpponentPool`, `SelfPlayOpponent`, `SelfPlayRatingUpdate` |
| CUDA-native execution | Implemented for TensorRT trait builds on Linux with native CUDA kernels for batched LineWorld stepping and fused PPO objective evaluation. | `NativeKernelPlan`, `TensorRTCUDAKernelExecutor`, `TensorRTCUDAKernelPlan`, `RLSwiftTensorRT` |
| Built-in environments | Implemented as a small deterministic suite for smoke tests and examples. | `BuiltInEnvironmentCatalog`, `LineWorldEnvironment`, `BinaryBanditEnvironment`, `MatrixGameEnvironment` |
| CLI workflow | Implemented for catalog, train, eval, experiment config, checkpoint manifest, sweep, Protein-style tuning, and visualization commands. | `swift run rl-swift ...` |
| Sweeps and tuning | Implemented deterministic grid sweeps, Pareto frontier selection, and a pure Swift Protein-style bounded tuner. | `SweepPlan`, `SweepTuner`, `ProteinTuner` |
| Visualization | Implemented lightweight terminal dashboards and sparklines for logs. | `TrainingMetricSeries`, `TrainingDashboardSnapshot` |
| Multi-agent simulation | Implemented protocol and matrix-game environment for game-scale API coverage. | `MultiAgentEnvironment`, `MatrixGameEnvironment` |
| Physics simulator bridge | Implemented as an optional MuJoCo product with state replay, keyframe reset, control-range transforms, contact reporting, and dependency-light simulator adapter metadata in core. | `RLSwiftMuJoCo`, `MuJoCoEnvironment`, `MuJoCoSimulation`, `RobotIntegrationAdapterConfiguration.mujoco(...)` |
| Isaac Sim sidecar bridge | Implemented as a pure Swift JSON/HTTP client for standalone Isaac Sim, Isaac Lab, or extension sidecars, including seeded reset, domain-randomization metadata, physics-step options, and batch reset/step. | `RLSwiftIsaacSim`, `IsaacSimBridgeClient`, `IsaacSimBatchStepResponse`, `RobotIntegrationAdapterConfiguration.isaacSim(...)` |
| Recurrent/entity encoders | Recurrent state and a MinGRU actor-critic reference model are implemented; PufferLib-style entity encoders remain a future model-family expansion. | `RecurrentPolicyValueModel`, `MinGRUCell`, `MinGRUState`, `MinGRUDiscreteActorCriticModel` |
| Distributed async rollout workers | In-process async rollout workers are implemented with Swift actors; multiprocess or remote worker orchestration remains a future distributed runtime layer. | `AsyncVectorizedEnvironmentRunner`, `AsyncVectorizedStepBatch` |

## Simulator Gap Analysis

The robotics need-to-have surface for simulator integration is:

| Need | MuJoCo coverage | Isaac Sim coverage |
| --- | --- | --- |
| Reproducible reset and replay | `MuJoCoResetMode.keyframe`, `MuJoCoStateSnapshot`, and restore support. | `IsaacSimResetOptions` with seed, episode id, and randomization metadata. |
| Safe policy action bounds | `MuJoCoActionMode.clipped` and `normalizedToControlRange` use native actuator control ranges. | `IsaacSimStepOptions` carries physics-step and render intent; sidecars keep robot-specific command bounds. |
| Contact-rich debugging | `MuJoCoContact` reports geom ids/names, contact pose, normal, and force. | Sidecars can return named sensor vectors and step `info`; camera/lidar rendering is requested through step options. |
| Vectorized simulator training | One native MuJoCo environment per model/data pair, compatible with `VectorizedEnvironmentRunner`. | `resetMany` and `stepMany` model Isaac Lab-style parallel environments over HTTP. |
| Dependency isolation | Optional `MuJoCoBackend` trait and system-library target. | Dependency-light Swift HTTP contract; Isaac Sim stays in Python/Omniverse. |

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
- `MuJoCoBackend` builds `RLSwiftMuJoCo` when MuJoCo headers and libraries are
  installed.
- `--disable-default-traits` gives a core-only, TensorRT-only, or MuJoCo-only
  build.

Useful commands:

```sh
swift package show-traits
swift build --disable-default-traits
swift test --traits MLXBackend
swift test --disable-default-traits --traits TensorRTBackend
swift test --disable-default-traits --traits MuJoCoBackend
```

## MuJoCo Simulation

`RLSwiftMuJoCo` wraps MJCF/XML models as RLSwift environments. It loads one
model, owns its MuJoCo data buffer, writes actuator controls from
`MuJoCoAction`, advances physics for the configured frame skip, and returns
`MuJoCoObservation` values containing time, `qpos`, `qvel`, sensor data,
actuator activations, controls, and optional contacts.

Need-to-have robotics features are implemented:

| Feature | API |
| --- | --- |
| Actuator range metadata | `MuJoCoActuatorSummary`, `MuJoCoModelSummary.actuators` |
| Control clipping and normalized policy outputs | `MuJoCoActionMode.clipped`, `MuJoCoActionMode.normalizedToControlRange` |
| Keyframe reset | `MuJoCoEnvironment.reset(to: .keyframe(...))` |
| Deterministic replay | `MuJoCoSimulation.stateSnapshot(signature:)`, `MuJoCoSimulation.restore(snapshot:)` |
| Contact debugging | `MuJoCoContact`, `MuJoCoEnvironmentConfiguration.includeContacts` |

The target is intentionally optional because MuJoCo is a native dependency. Core
planning code can still describe a MuJoCo runtime without linking MuJoCo:

```swift
let adapter = try RobotIntegrationAdapterConfiguration.mujoco(
    modelPath: "humanoid.xml"
)
```

Native MuJoCo tests require MuJoCo's `mujoco.pc`, headers, and shared library to
be visible to SwiftPM:

```sh
pkg-config --modversion mujoco
swift test --disable-default-traits --traits MuJoCoBackend
```

For local or CI environments without a system MuJoCo install, use the bundled
installer script:

```sh
eval "$(./scripts/install-mujoco-sdk.sh)"
swift test --disable-default-traits --traits MuJoCoBackend
```

## Isaac Sim Bridge

`RLSwiftIsaacSim` connects Swift policy loops to NVIDIA Isaac Sim or Isaac Lab
through a small JSON/HTTP sidecar contract. Isaac Sim stays in its normal
Omniverse/Python process, while Swift sends health, reset, and step requests and
receives `IsaacSimObservation` plus `StepResult`-compatible reward and
termination semantics.

Need-to-have robotics features are implemented:

| Feature | API |
| --- | --- |
| Seeded reset and replay metadata | `IsaacSimResetOptions` |
| Domain randomization payloads | `IsaacSimResetOptions.randomization` |
| Physics substeps and render synchronization | `IsaacSimStepOptions` |
| Isaac Lab-style vectorized environments | `IsaacSimEnvironmentHandle`, `resetMany`, `stepMany` |
| Batch step conversion into RLSwift results | `IsaacSimBatchStepResponse.stepResults` |

Core planning code can describe the simulator without importing the bridge
product:

```swift
let adapter = try RobotIntegrationAdapterConfiguration.isaacSim(
    endpoint: "http://127.0.0.1:8211",
    robotPath: "/World/Carter"
)
```

The bridge target is dependency-light and does not require Isaac Sim to be
installed for Swift tests:

```sh
swift test --filter IsaacSimBridgeTests
swift package generate-documentation --target RLSwiftIsaacSim
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

Runnable example projects live in `Examples/`.

`RobotGridWorld` trains a tabular Q-learning policy in a deterministic
navigation task and exercises the robot/autonomy API surface around the loop:

```sh
cd Examples/RobotGridWorld
swift run robot-grid-world
```

`VectorizedPPO` runs a Puffer-style collect/update loop with structured
observations, vectorized LineWorld environments, neural PPO, throughput metrics,
dashboard output, and checkpoint metadata:

```sh
cd Examples/VectorizedPPO
swift run vectorized-ppo --iterations 2 --envs 4 --rollout 3
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
| Robot adapters | Dependency-light ROS 2, MuJoCo simulator, Isaac Sim bridge, generic simulator, and WendyOS adapter descriptors. | `RobotIntegrationAdapterConfiguration` |
| Rollout collection | Vectorized environments, vectorization profiles, structured flattened spaces, and distributed rollout sharding. | `VectorizedEnvironmentRunner`, `VectorizationProfile`, `StructuredObservationSchema`, `StructuredActionSchema`, `RolloutShardAssignment` |
| Export and engine cache | ONNX export descriptors and TensorRT engine cache metadata. | `ONNXExportDescriptor`, `TensorRTEngineCacheKey`, `TensorRTEngineCacheManifest` |
| Training workflows | Curriculum learning, domain randomization, evaluation dashboards, checkpoint manifests, and self-play opponent pools. | `CurriculumStage`, `CurriculumSchedule`, `DomainRandomizationParameter`, `DomainRandomizationProfile`, `EvaluationRecord`, `EvaluationDashboardSummary`, `PolicyCheckpointManifest`, `SelfPlayOpponentPool` |
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
- Structured observation/action schemas, vectorization profiles, checkpoint
  manifests, async vectorized environment workers, and self-play opponent pools
  for Puffer-style training workflows.
- Built-in smoke-test environments with catalog metadata.
- PPO advantage estimation, clipped objective evaluation, action-distribution
  scoring, schedule/gradient clipping helpers, prioritized trajectory segments,
  recurrent MinGRU policy/value APIs, and a deterministic neural actor-critic
  PPO optimizer loop.
- Experiment configuration, checkpoint records, and built-in evaluation
  summaries for CLI and CI workflows.
- Throughput metering for rollout collection and training loops.
- Native kernel planning metadata for Swift CPU, MLX, CUDA, and TensorRT paths.
- Deterministic sweep plans, Pareto frontier tuning, Protein-style suggestions,
  text dashboards, and the `rl-swift` CLI.
- Multi-agent environment protocols and a matrix-game environment.
- `ActionSmoother` for low-pass command filtering.
- `ConstraintSignal` and `ConstraintReport` for constrained robot learning.
- `ObservationNormalizer` for streaming sensor-feature statistics.
- `RewardBreakdown` and `RewardComponent` for shaped robot rewards.
- Deterministic `ReplayBuffer` and `PrioritizedReplayBuffer` sampling.
- `EpsilonGreedyPolicy` and `SoftmaxPolicy` for discrete action spaces.
- `TabularQAgent` for small state-action spaces.
- `MLXBackendSupport`, `MLXObservationBatch`, `MLXObservationEncoder`, and
  `RLTensor` for MLX tensor integration.
- `TensorRTBackendSupport`, `TensorRTPolicyConfiguration`,
  `TensorRTCUDAKernelPlan`, and `TensorRTPolicyBackend` for NVIDIA Linux
  TensorRT inference.
- `MuJoCoBackendSupport`, `MuJoCoEnvironment`, `MuJoCoObservation`, and
  `MuJoCoRewardSource` for native MuJoCo simulator loops, plus
  `MuJoCoStateSnapshot`, `MuJoCoActionMode`, and `MuJoCoContact` for replay,
  bounded control, and contact debugging.
- `IsaacSimBackendSupport`, `IsaacSimBridgeClient`, `IsaacSimObservation`, and
  `IsaacSimStepResponse` for Isaac Sim sidecar loops, plus
  `IsaacSimResetOptions`, `IsaacSimStepOptions`, `IsaacSimEnvironmentHandle`,
  and batch reset/step response types for Isaac Lab-style parallel simulation.

Public interfaces are documented with DocC comments and a DocC catalog at
`Sources/RLSwift/RLSwift.docc`,
`Sources/RLSwiftMLX/RLSwiftMLX.docc`,
`Sources/RLSwiftTensorRT/RLSwiftTensorRT.docc`,
`Sources/RLSwiftMuJoCo/RLSwiftMuJoCo.docc`, and
`Sources/RLSwiftIsaacSim/RLSwiftIsaacSim.docc`.
