# SwiftRL

Build reinforcement learning agents in Swift with opt-in tensor backends.

## Overview

SwiftRL provides a small, composable reinforcement learning core for Swift 6.3
and newer. The package includes protocols for environments and agents,
transition and episode accounting, deterministic replay memory, discrete
policies, tabular Q-learning, and robot/autonomy support types.

The core abstractions are generic over observation and action types so that
simple tabular agents and backend-powered function approximators can share the
same environment loop.

Tensor integrations live in separate products. `SwiftRLMLX` provides MLX tensor
adapters when the `MLXBackend` package trait is enabled. `SwiftRLTensorRT`
provides a TensorRT backend on Linux builds when the `TensorRTBackend` trait is
enabled. This keeps the core SwiftRL product dependency-light while allowing
iOS, iPadOS, visionOS, macOS, DGX, Jetson, and NVIDIA Linux deployments to select
the backend they need.

## Apple Device Support

SwiftRL is not limited to macOS. The package declares support for macOS 14,
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

## Robot and Autonomy Gap Analysis

Robots and autonomous systems need explicit support for safety, timing, and
logged-data quality. SwiftRL covers the most important gaps in the core package:

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

Need-to-have autonomy capabilities beyond the core loop include versioned model
IO contracts, durable offline datasets, hardware-facing safety supervisors,
deterministic deployment backends, and real-time observability for latency and
interventions. Nice-to-have extensions include ROS 2 or simulator adapters,
vectorized rollout collection, ONNX export helpers, curriculum learning, domain
randomization, and visual debugging tools.

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
- ``RobotSafetyEnvelope``
- ``RobotSafetyAssessment``
- ``SafetyIntervention``
- ``ActionSmoother``
- ``ObservationNormalizer``
- ``RewardBreakdown``
- ``RewardComponent``
- ``ConstraintRelation``
- ``ConstraintSignal``
- ``ConstraintReport``

### Learning

- ``TabularQAgent``
- ``ReplayBuffer``
- ``PrioritizedReplayBuffer``

### Errors

- ``SwiftRLError``
