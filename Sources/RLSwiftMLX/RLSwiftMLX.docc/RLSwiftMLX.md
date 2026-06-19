# RLSwiftMLX

Use MLX tensors with RLSwift.

## Overview

RLSwiftMLX is the opt-in MLX backend for RLSwift. It contains the public MLX
tensor alias and observation-encoder protocol that were intentionally kept out of
the core `RLSwift` target so non-MLX builds can use the reinforcement-learning
and robot-control types without importing MLX.

The package enables the `MLXBackend` trait by default. Disable default traits
when you want a TensorRT-only Linux build or a pure core build.

```sh
swift build --traits MLXBackend
swift build --disable-default-traits --traits TensorRTBackend
swift build --disable-default-traits
```

## Topics

### Platform Support

- ``MLXBackendSupport``

### Tensor Integration

When the `MLXBackend` trait is enabled, this target exposes `RLTensor` and
`MLXObservationEncoder` with DocC comments in source.
