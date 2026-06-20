# VectorizedPPO

`VectorizedPPO` is a runnable RLSwift example that uses the Puffer-style
training workflow primitives:

- `StructuredObservationSchema` for flattened model inputs.
- `VectorizedEnvironmentRunner` and `VectorizationProfile` for batched rollout collection.
- `DenseDiscreteActorCriticModel` and `NeuralPPOTrainer` for a full PPO update.
- `ThroughputMeter`, `TrainingDashboardSnapshot`, and `PolicyCheckpointManifest`
  for training logs and checkpoint metadata.

```sh
swift run vectorized-ppo
```

Useful flags:

```sh
swift run vectorized-ppo --iterations 4 --envs 8 --rollout 4 --seed 13
```

| Flag | Default | Description |
| --- | ---: | --- |
| `--iterations` | `4` | Number of collect/update cycles. |
| `--envs` | `8` | Number of vectorized LineWorld environments. |
| `--rollout` | `4` | Rollout horizon per environment. |
| `--seed` | `13` | Deterministic model seed. |
