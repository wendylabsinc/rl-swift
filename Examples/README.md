# RLSwift Examples

The examples are standalone Swift packages that depend on the root package by
local path. Run them from their own directories so SwiftPM keeps example build
artifacts separate from the library build.

## RobotGridWorld

`RobotGridWorld` trains a tabular Q-learning agent in a small deterministic
navigation task and exercises the robot/autonomy APIs around the training loop:

- `Environment`, `StepResult`, `Transition`, `Episode`, and `TabularQAgent`
- `ReplayBuffer` and `PrioritizedReplayBuffer`
- `RobotObservation`, `ModelIOContract`, and `ObservationNormalizer`
- `HardwareSafetySupervisor`, `ControlTiming`, and `ConstraintReport`
- `OfflineDataset`, `DeploymentPlan`, and `AutonomyTelemetryAccumulator`

Run it with the default deterministic configuration:

```sh
cd Examples/RobotGridWorld
swift run robot-grid-world
```

Useful flags:

```sh
swift run robot-grid-world --episodes 300 --max-steps 48 --seed 17
```

## VectorizedPPO

`VectorizedPPO` runs a compact PPO collect/update loop over vectorized
LineWorld environments and exercises the Puffer-style workflow APIs:

- `StructuredObservationSchema` and `StructuredTensorField`
- `VectorizedEnvironmentRunner` and `VectorizationProfile`
- `DenseDiscreteActorCriticModel`, `PPOAdvantageEstimator`, and `NeuralPPOTrainer`
- `ThroughputMeter`, `TrainingDashboardSnapshot`, and `PolicyCheckpointManifest`

Run it with the default deterministic configuration:

```sh
cd Examples/VectorizedPPO
swift run vectorized-ppo
```

Useful flags:

```sh
swift run vectorized-ppo --iterations 4 --envs 8 --rollout 4 --seed 13
```
