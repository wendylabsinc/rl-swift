# RLSwift

Build reinforcement learning agents in Swift with opt-in tensor backends.

## Overview

RLSwift provides a small, composable reinforcement learning core for Swift 6.3
and newer. The package includes protocols for environments and agents,
transition and episode accounting, deterministic replay memory, discrete
policies, tabular Q-learning, PPO objective utilities, asynchronous vectorized
environment workers, recurrent MinGRU policy/value APIs, a deterministic neural
PPO optimizer loop, built-in smoke-test environments, training-throughput
instrumentation, sweep helpers, Protein-style tuning, terminal visualization,
multi-agent simulation protocols, and robot/autonomy support types.

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

## Training Workflows

RLSwift includes PufferLib-inspired workflow building blocks for Swift training
loops: ``ThroughputMeter`` records environment and sample throughput,
``PPOAdvantageEstimator`` computes generalized advantage estimates,
``PPOClippedObjective`` evaluates clipped PPO losses,
``NeuralPPOTrainer`` runs minibatch PPO epochs over trainable actor-critic
models, ``DenseDiscreteActorCriticModel`` provides a compact reference neural
policy/value implementation, ``StructuredObservationSchema`` and
``StructuredActionSchema`` flatten named spaces for model inputs and outputs,
``VectorizationProfile`` records rollout execution shape,
``AsyncVectorizedEnvironmentRunner`` provides in-process Swift actor workers,
``PolicyCheckpointManifest`` captures checkpoint provenance,
``SelfPlayOpponentPool`` tracks frozen opponents for self-play curricula,
``MinGRUCell`` provides a recurrent reference cell,
``ExperimentConfiguration`` and ``EvaluationSummary`` capture CLI-friendly run
records, ``SweepPlan`` builds deterministic hyperparameter grids,
``ProteinTuner`` suggests bounded hyperparameters, and
``TrainingDashboardSnapshot`` renders terminal-friendly metric summaries for CI
or local runs.

## Topics

### Environment Loops

- ``Environment``
- ``Agent``
- ``StepResult``
- ``StepTermination``
- ``Transition``
- ``Episode``
- ``ControlTiming``

### Built-In Environments

- ``BuiltInEnvironmentID``
- ``EnvironmentCatalogEntry``
- ``BuiltInEnvironmentCatalog``
- ``LineWorldAction``
- ``LineWorldObservation``
- ``LineWorldEnvironment``
- ``BinaryBanditAction``
- ``BinaryBanditObservation``
- ``BinaryBanditEnvironment``

### Multi-Agent Simulation

- ``MultiAgentEnvironment``
- ``MultiAgentStepResult``
- ``MatrixGameAction``
- ``MatrixGameObservation``
- ``MatrixGameEnvironment``

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
- ``AsyncEnvironmentWorker``
- ``AsyncVectorizedEnvironmentRunner``
- ``AsyncVectorizedStepBatch``
- ``VectorizedEnvironmentStep``
- ``StructuredTensorField``
- ``StructuredTensorSchema``
- ``StructuredObservationSchema``
- ``StructuredActionSchema``
- ``VectorizationBackend``
- ``VectorizationProfile``
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
- ``PPOConfiguration``
- ``PPOAnnealingSchedule``
- ``PPOAction``
- ``PPOActionDistribution``
- ``PPOGradientClipSummary``
- ``PPOClippedGradientVector``
- ``PPOGradientClipper``
- ``PPOTrajectoryStep``
- ``PPOTrajectorySegment``
- ``PPOTrajectorySegmentSampler``
- ``PPOAdvantageBatch``
- ``PPOAdvantageEstimator``
- ``PPOClippedObjectiveSample``
- ``PPOObjectiveBreakdown``
- ``PPOClippedObjective``
- ``PPOTrainingSample``
- ``PPOPolicyValuePrediction``
- ``PPOPolicyValueModel``
- ``PPOOptimizerStepSummary``
- ``PPOTrainingSummary``
- ``NeuralPPOTrainer``
- ``DenseDiscreteActorCriticModel``
- ``RecurrentPolicyValuePrediction``
- ``RecurrentPolicyValueModel``
- ``MinGRUState``
- ``MinGRUCell``
- ``MinGRUDiscreteActorCriticModel``
- ``ExperimentConfiguration``
- ``ExperimentCheckpointRecord``
- ``EvaluationSummary``
- ``ExperimentEvaluator``
- ``PolicyCheckpointManifest``
- ``SelfPlayOpponent``
- ``SelfPlayRatingUpdate``
- ``SelfPlayOpponentPool``
- ``TrainingThroughputReport``
- ``ThroughputMeter``
- ``NativeKernelBackend``
- ``KernelFusionOperation``
- ``NativeKernelPlan``

### Sweeps and Visualization

- ``SweepParameter``
- ``SweepTrial``
- ``SweepPlan``
- ``SweepResult``
- ``SweepTuner``
- ``ProteinParameterScale``
- ``ProteinParameter``
- ``ProteinSuggestion``
- ``ProteinObservation``
- ``ProteinTuner``
- ``MetricSeriesPoint``
- ``TrainingMetricSeries``
- ``TrainingDashboardSnapshot``

### Errors

- ``RLSwiftError``
