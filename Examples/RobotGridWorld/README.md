# RobotGridWorld

`RobotGridWorld` is a runnable RLSwift example that trains a tabular
Q-learning policy for a small robot-navigation task. The example intentionally
keeps the environment simple while showing the surrounding production concerns:
model IO contracts, replay buffers, safety supervision, control timing,
constraint telemetry, offline dataset manifests, and deployment descriptors.

```sh
swift run robot-grid-world
```

The executable exits with a nonzero status if the trained greedy policy cannot
reach the goal, so it can also be used as a smoke test.

```sh
swift run robot-grid-world --episodes 300 --max-steps 48 --seed 17
```

Flags:

| Flag | Default | Description |
| --- | ---: | --- |
| `--episodes` | `240` | Number of training episodes. |
| `--max-steps` | `48` | Maximum steps per episode before truncation. |
| `--seed` | `11` | Deterministic seed for the agent and replay samplers. |
