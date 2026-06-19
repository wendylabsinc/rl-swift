# RLSwiftTensorRT

Run RLSwift policies through NVIDIA TensorRT on Linux.

## Overview

RLSwiftTensorRT is the opt-in NVIDIA backend for RLSwift. When the
`TensorRTBackend` package trait is enabled on Linux, it depends on the
`TensorRT` product from `wendylabsinc/tensorrt-swift`. Apple-platform
development remains focused on RLSwift and MLX while DGX, workstation, and
NVIDIA Linux deployments can use TensorRT.

Use this target when a trained policy is exported to ONNX or a serialized
TensorRT engine and needs low-latency inference near a robot or autonomous
system. On iOS, iPadOS, visionOS, and macOS, ``TensorRTBackendSupport/current``
reports that the native TensorRT backend is unavailable; those platforms should
use RLSwift/MLX for local inference, adaptation, simulation, telemetry, or
evaluation.

On NVIDIA Linux builds, `TensorRTPolicyBackend` is compiled and can:

- load serialized TensorRT engines,
- build engines from ONNX,
- select optimization profiles,
- reshape dynamic input bindings,
- decode `Float32` policy outputs, and
- convert continuous outputs into RLSwift robot actions.

```sh
swift build --disable-default-traits --traits TensorRTBackend
swift test --disable-default-traits --traits TensorRTBackend
```

## Deployment Split

TensorRT is possible for this library, but it should be treated as a deployment
backend rather than a replacement for the Apple MLX path. A practical robot stack
can train or fine-tune with MLX on Apple hardware, export a model to ONNX, build
a TensorRT engine on an NVIDIA Linux machine, and run the resulting policy
through RLSwiftTensorRT for low-latency inference.

The native backend requires Linux with CUDA and TensorRT installed. A DGX Spark,
DGX workstation, Jetson-class device, or NVIDIA Linux host is a better fit than
macOS because TensorRT links `libnvinfer`, `libnvinfer_plugin`,
`libnvonnxparser`, and CUDA driver libraries.

## Topics

### Platform Support

- ``TensorRTBackendSupport``

### Policy Configuration

- ``TensorRTPolicyConfiguration``
- ``TensorRTInferenceOutput``
- ``TensorRTPolicyOutput``

### Errors

- ``TensorRTBackendError``
