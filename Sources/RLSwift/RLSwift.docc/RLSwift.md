# RLSwift

Build reinforcement learning agents in Swift with opt-in tensor backends.

## Overview

RLSwift provides a small, composable reinforcement learning core for Swift 6.3
and newer. The package includes protocols for environments and agents,
transition and episode accounting, deterministic replay memory, discrete
policies, tabular Q-learning, and robot/autonomy support types.

The core abstractions are generic over observation and action types so that
simple tabular agents and backend-powered function approximators can share the
same environment loop.

Tensor integrations live in separate products. `RLSwiftMLX` provides MLX tensor
adapters when the `MLXBackend` package trait is enabled. `RLSwiftTensorRT`
provides a TensorRT backend on Linux builds when the `TensorRTBackend` trait is
enabled. This keeps the core RLSwift product dependency-light while allowing
iOS, iPadOS, visionOS, macOS, DGX, Jetson, and NVIDIA Linux deployments to select
the backend they need.

Hot vector transforms use Swift 6 ownership features where they are portable
across the declared deployment targets. Internally, RLSwift uses move-only
scratch storage for action scaling, normalization, model IO, and command
smoothing so those paths build one owned output array without intermediate
`map` closures. `Span` and `InlineArray` were evaluated for public vector APIs,
but this Swift 6.3.2 toolchain gates them to OS 26 availability, so the stable
public API avoids them until that surface can coexist with macOS 14, iOS 17,
iPadOS 17, tvOS 17, visionOS 1, and Linux support.

## Apple Device Support

RLSwift is not limited to macOS. The package declares support for macOS 14,
iOS 17, tvOS 17, and visionOS 1; iPadOS builds use the iOS platform target in
SwiftPM. The MLX Swift dependency declares the same Apple platform set and uses
Metal and Accelerate on Apple platforms.

On iPhone, iPad, or Apple Vision Pro hardware, the practical split is usually to
run policy inference, online adaptation, telemetry collection, simulation, or
evaluation on device, while keeping large-scale training and replay-heavy jobs on
a Mac or server. Real robot deployments should keep hard safety limits outside
the learned policy in addition to using ``RobotSafetyEnvelope``, because mobile
OS scheduling, thermal throttling, sensor latency, and wireless transport can
affect closed-loop robot behavior.

## Robot and Autonomy Features

Robots and autonomous systems need explicit support for safety, timing, and
logged-data quality. RLSwift includes those features in the core package:

- Episode-ending semantics are represented by ``StepTermination`` so algorithms
  can distinguish task termination from time-limit truncation and supervisory
  interruption.
- ``ControlTiming`` records control period, observation age, command latency, and
  deadline misses for closed-loop robot data.
- ``ConstraintSignal`` and ``ConstraintReport`` turn safety and performance
  limits into explicit cost signals for constrained RL and deployment monitoring.
- ``RobotSafetyEnvelope`` can produce a ``RobotSafetyAssessment`` with each
  ``SafetyIntervention`` applied to a command before it reaches hardware.
- ``ActionSmoother`` filters actuator commands before safety clipping or
  execution.
- ``PrioritizedReplayBuffer`` lets rare but important autonomy events be sampled
  more often than ordinary transitions.

The package also includes concrete autonomy infrastructure for production-style
policy workflows: ``ModelIOContract`` pins model input/output order and
deployment metadata; ``HardwareSafetySupervisor`` keeps emergency-stop,
freshness, deadline, and envelope checks outside the policy; ``OfflineDataset``
stores replayable logged data with provenance; and
``AutonomyTelemetryAccumulator`` summarizes latency, safety, constraints, and
policy-version rollout behavior. Dependency-light descriptors cover ROS 2,
simulator, WendyOS, vectorized rollout, ONNX export, TensorRT engine cache,
curriculum, domain randomization, evaluation, and visual debugging workflows.

## Topics

### Environment Loops

- ``Environment``
- ``Agent``
- ``StepResult``
- ``StepTermination``
- ``Transition``
- ``Episode``
- ``ControlTiming``

### Action Selection

- ``DiscreteActionSpace``
- ``ContinuousBoxSpace``
- ``EpsilonGreedyPolicy``
- ``SoftmaxPolicy``
- ``SeededGenerator``

### Robotics

- ``RobotAction``
- ``RobotControlMode``
- ``RobotObservation``
- ``ModelIOContract``
- ``ObservationFeature``
- ``RobotObservationComponent``
- ``NormalizationSnapshot``
- ``ActionSpecification``
- ``TensorRTBindingNames``
- ``PolicyMetadata``
- ``RobotSafetyEnvelope``
- ``RobotSafetyAssessment``
- ``SafetyIntervention``
- ``EmergencyStopState``
- ``HardwareSafetySupervisor``
- ``SafetySupervisorInput``
- ``SafetySupervisorDecision``
- ``SafetySupervisorIntervention``
- ``ActionSmoother``
- ``ObservationNormalizer``
- ``RewardBreakdown``
- ``RewardComponent``
- ``ConstraintRelation``
- ``ConstraintSignal``
- ``ConstraintReport``

### Datasets and Deployment

- ``DatasetProvenance``
- ``LoggedTransition``
- ``DatasetManifest``
- ``OfflineDataset``
- ``DeploymentBackend``
- ``DeploymentTarget``
- ``DeploymentPlan``
- ``PolicyVersionRollout``
- ``AutonomyTelemetryAccumulator``
- ``AutonomyTelemetrySummary``

### Integration and Debugging

- ``RobotIntegrationKind``
- ``RobotIntegrationAdapterConfiguration``
- ``VectorizedEnvironmentRunner``
- ``VectorizedStepResult``
- ``RolloutShardAssignment``
- ``ONNXExportDescriptor``
- ``TensorRTEngineCacheKey``
- ``TensorRTEngineCacheManifest``
- ``DomainRandomizationParameter``
- ``DomainRandomizationProfile``
- ``CurriculumStage``
- ``CurriculumSchedule``
- ``EvaluationRecord``
- ``EvaluationDashboardSummary``
- ``ObservationDriftSnapshot``
- ``ActionSaturationSnapshot``
- ``PrioritizedReplayDebugSnapshot``

### Learning

- ``TabularQAgent``
- ``ReplayBuffer``
- ``PrioritizedReplayBuffer``

### Errors

- ``RLSwiftError``
