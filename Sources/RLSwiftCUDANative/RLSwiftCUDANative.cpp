#include "RLSwiftCUDANative.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <vector>

namespace {

using CUdevice = int;
using CUresult = int;
using CUdeviceptr = unsigned long long;
using CUcontext = void *;
using CUmodule = void *;
using CUfunction = void *;
using CUstream = void *;
using nvrtcProgram = void *;
using nvrtcResult = int;

constexpr CUresult CUDA_SUCCESS_VALUE = 0;
constexpr nvrtcResult NVRTC_SUCCESS_VALUE = 0;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

std::string last_error = "CUDA runtime has not been initialized.";

template <typename T>
bool resolve(void *library, const char *name, T *out) {
    void *symbol = dlsym(library, name);
    if (symbol == nullptr) {
        last_error = std::string("Missing symbol: ") + name;
        return false;
    }
    *out = reinterpret_cast<T>(symbol);
    return true;
}

const char *kernel_source() {
    return R"CUDA(
extern "C" __global__ void rlsw_lineworld_step_kernel(
    const int *actions,
    int *positions,
    int *step_indices,
    float *rewards,
    unsigned char *terminals,
    int *termination_codes,
    int environment_count,
    int length,
    int max_steps
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= environment_count) {
        return;
    }

    int position = positions[index];
    int step = step_indices[index] + 1;
    int action = actions[index];
    if (action == 0) {
        position = position > 0 ? position - 1 : 0;
    } else {
        int goal = length - 1;
        position = position < goal ? position + 1 : goal;
    }

    positions[index] = position;
    step_indices[index] = step;
    if (position == length - 1) {
        rewards[index] = 1.0f;
        terminals[index] = 1;
        termination_codes[index] = 1;
    } else if (step >= max_steps) {
        rewards[index] = -0.01f;
        terminals[index] = 1;
        termination_codes[index] = 2;
    } else {
        rewards[index] = -0.01f;
        terminals[index] = 0;
        termination_codes[index] = 0;
    }
}

extern "C" __global__ void rlsw_ppo_objective_kernel(
    const float *old_log_probabilities,
    const float *new_log_probabilities,
    const float *advantages,
    const float *returns,
    const float *value_estimates,
    const float *entropies,
    int sample_count,
    float clip_range,
    float *partials
) {
    extern __shared__ float shared[];
    float *policy_sums = shared;
    float *value_sums = policy_sums + blockDim.x;
    float *entropy_sums = value_sums + blockDim.x;
    float *kl_sums = entropy_sums + blockDim.x;
    float *clip_sums = kl_sums + blockDim.x;
    float *count_sums = clip_sums + blockDim.x;

    int thread_index = threadIdx.x;
    int sample_index = blockIdx.x * blockDim.x + thread_index;
    float policy = 0.0f;
    float value = 0.0f;
    float entropy = 0.0f;
    float kl = 0.0f;
    float clipped = 0.0f;
    float counted = 0.0f;

    if (sample_index < sample_count) {
        float old_log = old_log_probabilities[sample_index];
        float new_log = new_log_probabilities[sample_index];
        float advantage = advantages[sample_index];
        float ratio = __expf(new_log - old_log);
        float lower = 1.0f - clip_range;
        float upper = 1.0f + clip_range;
        float clipped_ratio = ratio < lower ? lower : (ratio > upper ? upper : ratio);
        float value_error = returns[sample_index] - value_estimates[sample_index];
        float unclipped_policy = ratio * advantage;
        float clipped_policy = clipped_ratio * advantage;
        float ratio_delta = ratio - 1.0f;

        policy = unclipped_policy < clipped_policy ? unclipped_policy : clipped_policy;
        value = 0.5f * value_error * value_error;
        entropy = entropies[sample_index];
        kl = old_log - new_log;
        clipped = (ratio_delta < 0.0f ? -ratio_delta : ratio_delta) > clip_range ? 1.0f : 0.0f;
        counted = 1.0f;
    }

    policy_sums[thread_index] = policy;
    value_sums[thread_index] = value;
    entropy_sums[thread_index] = entropy;
    kl_sums[thread_index] = kl;
    clip_sums[thread_index] = clipped;
    count_sums[thread_index] = counted;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            policy_sums[thread_index] += policy_sums[thread_index + stride];
            value_sums[thread_index] += value_sums[thread_index + stride];
            entropy_sums[thread_index] += entropy_sums[thread_index + stride];
            kl_sums[thread_index] += kl_sums[thread_index + stride];
            clip_sums[thread_index] += clip_sums[thread_index + stride];
            count_sums[thread_index] += count_sums[thread_index + stride];
        }
        __syncthreads();
    }

    if (thread_index == 0) {
        int base = blockIdx.x * 6;
        partials[base] = policy_sums[0];
        partials[base + 1] = value_sums[0];
        partials[base + 2] = entropy_sums[0];
        partials[base + 3] = kl_sums[0];
        partials[base + 4] = clip_sums[0];
        partials[base + 5] = count_sums[0];
    }
}
)CUDA";
}

struct DriverAPI {
    int (*cuGetErrorName)(CUresult, const char **) = nullptr;
    int (*cuGetErrorString)(CUresult, const char **) = nullptr;
    int (*cuInit)(unsigned int) = nullptr;
    int (*cuDeviceGet)(CUdevice *, int) = nullptr;
    int (*cuDeviceGetAttribute)(int *, int, CUdevice) = nullptr;
    int (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice) = nullptr;
    int (*cuCtxSetCurrent)(CUcontext) = nullptr;
    int (*cuMemAlloc)(CUdeviceptr *, std::size_t) = nullptr;
    int (*cuMemFree)(CUdeviceptr) = nullptr;
    int (*cuMemcpyHtoD)(CUdeviceptr, const void *, std::size_t) = nullptr;
    int (*cuMemcpyDtoH)(void *, CUdeviceptr, std::size_t) = nullptr;
    int (*cuModuleLoadData)(CUmodule *, const void *) = nullptr;
    int (*cuModuleUnload)(CUmodule) = nullptr;
    int (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *) = nullptr;
    int (*cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void **, void **) = nullptr;
    int (*cuCtxSynchronize)() = nullptr;
};

struct NVRTCAPI {
    nvrtcResult (*nvrtcCreateProgram)(nvrtcProgram *, const char *, const char *, int, const char *const *, const char *const *) = nullptr;
    nvrtcResult (*nvrtcCompileProgram)(nvrtcProgram, int, const char *const *) = nullptr;
    nvrtcResult (*nvrtcGetPTXSize)(nvrtcProgram, std::size_t *) = nullptr;
    nvrtcResult (*nvrtcGetPTX)(nvrtcProgram, char *) = nullptr;
    nvrtcResult (*nvrtcGetCUBINSize)(nvrtcProgram, std::size_t *) = nullptr;
    nvrtcResult (*nvrtcGetCUBIN)(nvrtcProgram, char *) = nullptr;
    nvrtcResult (*nvrtcGetProgramLogSize)(nvrtcProgram, std::size_t *) = nullptr;
    nvrtcResult (*nvrtcGetProgramLog)(nvrtcProgram, char *) = nullptr;
    nvrtcResult (*nvrtcDestroyProgram)(nvrtcProgram *) = nullptr;
    const char *(*nvrtcGetErrorString)(nvrtcResult) = nullptr;
};

class CUDARuntime {
public:
    static CUDARuntime &shared() {
        static CUDARuntime runtime;
        return runtime;
    }

    bool ensure() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) {
            if (available_ && context_ != nullptr && driver_.cuCtxSetCurrent != nullptr) {
                CUresult status = driver_.cuCtxSetCurrent(context_);
                if (status != CUDA_SUCCESS_VALUE) {
                    last_error = "Failed to bind cached CUDA primary context: " + driverError(status);
                    return false;
                }
            }
            return available_;
        }
        initialized_ = true;

        if (!loadLibraries() || !loadDriverSymbols() || !loadNVRTCSymbols()) {
            available_ = false;
            return false;
        }
        if (driver_.cuInit(0) != CUDA_SUCCESS_VALUE) {
            last_error = "cuInit failed.";
            available_ = false;
            return false;
        }
        CUdevice device = 0;
        if (driver_.cuDeviceGet(&device, 0) != CUDA_SUCCESS_VALUE) {
            last_error = "No CUDA device 0 is visible.";
            available_ = false;
            return false;
        }
        if (driver_.cuDevicePrimaryCtxRetain(&context_, device) != CUDA_SUCCESS_VALUE ||
            driver_.cuCtxSetCurrent(context_) != CUDA_SUCCESS_VALUE) {
            last_error = "Failed to retain or bind the CUDA primary context.";
            available_ = false;
            return false;
        }

        int major = 0;
        int minor = 0;
        if (driver_.cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) != CUDA_SUCCESS_VALUE ||
            driver_.cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) != CUDA_SUCCESS_VALUE) {
            major = 5;
            minor = 2;
        }
        available_ = compileKernels(major, minor);
        return available_;
    }

    DriverAPI &driver() { return driver_; }
    CUfunction lineworld() const { return lineworld_; }
    CUfunction ppoObjective() const { return ppo_objective_; }
    std::string describe(CUresult status) { return driverError(status); }

private:
    bool loadLibraries() {
        cuda_library_ = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (cuda_library_ == nullptr) {
            cuda_library_ = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
        }
        if (cuda_library_ == nullptr) {
            last_error = "Unable to load libcuda.";
            return false;
        }

        nvrtc_library_ = dlopen("libnvrtc.so", RTLD_NOW | RTLD_LOCAL);
        if (nvrtc_library_ == nullptr) {
            nvrtc_library_ = dlopen("libnvrtc.so.13", RTLD_NOW | RTLD_LOCAL);
        }
        if (nvrtc_library_ == nullptr) {
            nvrtc_library_ = dlopen("libnvrtc.so.12", RTLD_NOW | RTLD_LOCAL);
        }
        if (nvrtc_library_ == nullptr) {
            last_error = "Unable to load libnvrtc.";
            return false;
        }
        return true;
    }

    bool loadDriverSymbols() {
        return resolve(cuda_library_, "cuGetErrorName", &driver_.cuGetErrorName) &&
            resolve(cuda_library_, "cuGetErrorString", &driver_.cuGetErrorString) &&
            resolve(cuda_library_, "cuInit", &driver_.cuInit) &&
            resolve(cuda_library_, "cuDeviceGet", &driver_.cuDeviceGet) &&
            resolve(cuda_library_, "cuDeviceGetAttribute", &driver_.cuDeviceGetAttribute) &&
            resolve(cuda_library_, "cuDevicePrimaryCtxRetain", &driver_.cuDevicePrimaryCtxRetain) &&
            resolve(cuda_library_, "cuCtxSetCurrent", &driver_.cuCtxSetCurrent) &&
            resolve(cuda_library_, "cuMemAlloc_v2", &driver_.cuMemAlloc) &&
            resolve(cuda_library_, "cuMemFree_v2", &driver_.cuMemFree) &&
            resolve(cuda_library_, "cuMemcpyHtoD_v2", &driver_.cuMemcpyHtoD) &&
            resolve(cuda_library_, "cuMemcpyDtoH_v2", &driver_.cuMemcpyDtoH) &&
            resolve(cuda_library_, "cuModuleLoadData", &driver_.cuModuleLoadData) &&
            resolve(cuda_library_, "cuModuleUnload", &driver_.cuModuleUnload) &&
            resolve(cuda_library_, "cuModuleGetFunction", &driver_.cuModuleGetFunction) &&
            resolve(cuda_library_, "cuLaunchKernel", &driver_.cuLaunchKernel) &&
            resolve(cuda_library_, "cuCtxSynchronize", &driver_.cuCtxSynchronize);
    }

    bool loadNVRTCSymbols() {
        bool required = resolve(nvrtc_library_, "nvrtcCreateProgram", &nvrtc_.nvrtcCreateProgram) &&
            resolve(nvrtc_library_, "nvrtcCompileProgram", &nvrtc_.nvrtcCompileProgram) &&
            resolve(nvrtc_library_, "nvrtcGetPTXSize", &nvrtc_.nvrtcGetPTXSize) &&
            resolve(nvrtc_library_, "nvrtcGetPTX", &nvrtc_.nvrtcGetPTX) &&
            resolve(nvrtc_library_, "nvrtcGetProgramLogSize", &nvrtc_.nvrtcGetProgramLogSize) &&
            resolve(nvrtc_library_, "nvrtcGetProgramLog", &nvrtc_.nvrtcGetProgramLog) &&
            resolve(nvrtc_library_, "nvrtcDestroyProgram", &nvrtc_.nvrtcDestroyProgram) &&
            resolve(nvrtc_library_, "nvrtcGetErrorString", &nvrtc_.nvrtcGetErrorString);
        nvrtc_.nvrtcGetCUBINSize = reinterpret_cast<decltype(nvrtc_.nvrtcGetCUBINSize)>(dlsym(nvrtc_library_, "nvrtcGetCUBINSize"));
        nvrtc_.nvrtcGetCUBIN = reinterpret_cast<decltype(nvrtc_.nvrtcGetCUBIN)>(dlsym(nvrtc_library_, "nvrtcGetCUBIN"));
        return required;
    }

    bool compileKernels(int major, int minor) {
        nvrtcProgram program = nullptr;
        nvrtcResult created = nvrtc_.nvrtcCreateProgram(&program, kernel_source(), "rlswift_kernels.cu", 0, nullptr, nullptr);
        if (created != NVRTC_SUCCESS_VALUE) {
            last_error = std::string("nvrtcCreateProgram failed: ") + nvrtc_.nvrtcGetErrorString(created);
            return false;
        }

        std::string architecture = "--gpu-architecture=sm_" + std::to_string(major) + std::to_string(minor);
        const char *options[] = {"--std=c++11", architecture.c_str()};
        nvrtcResult compiled = nvrtc_.nvrtcCompileProgram(program, 2, options);
        if (compiled != NVRTC_SUCCESS_VALUE) {
            last_error = std::string("NVRTC compilation failed: ") + nvrtc_.nvrtcGetErrorString(compiled) + "\n" + programLog(program);
            nvrtc_.nvrtcDestroyProgram(&program);
            return false;
        }

        std::vector<char> module_image;
        if (nvrtc_.nvrtcGetCUBINSize != nullptr && nvrtc_.nvrtcGetCUBIN != nullptr) {
            std::size_t cubin_size = 0;
            if (nvrtc_.nvrtcGetCUBINSize(program, &cubin_size) == NVRTC_SUCCESS_VALUE && cubin_size > 0) {
                module_image.resize(cubin_size);
                if (nvrtc_.nvrtcGetCUBIN(program, module_image.data()) != NVRTC_SUCCESS_VALUE) {
                    last_error = "Failed to get NVRTC CUBIN.";
                    nvrtc_.nvrtcDestroyProgram(&program);
                    return false;
                }
            }
        }
        if (module_image.empty()) {
            std::size_t ptx_size = 0;
            if (nvrtc_.nvrtcGetPTXSize(program, &ptx_size) != NVRTC_SUCCESS_VALUE || ptx_size == 0) {
                last_error = "Failed to get NVRTC PTX size.";
                nvrtc_.nvrtcDestroyProgram(&program);
                return false;
            }
            module_image.resize(ptx_size);
            if (nvrtc_.nvrtcGetPTX(program, module_image.data()) != NVRTC_SUCCESS_VALUE) {
                last_error = "Failed to get NVRTC PTX.";
                nvrtc_.nvrtcDestroyProgram(&program);
                return false;
            }
        }
        nvrtc_.nvrtcDestroyProgram(&program);

        CUresult module_status = driver_.cuModuleLoadData(&module_, module_image.data());
        if (module_status != CUDA_SUCCESS_VALUE) {
            last_error = "cuModuleLoadData failed for compiled RLSwift kernels: " + driverError(module_status);
            return false;
        }
        if (driver_.cuModuleGetFunction(&lineworld_, module_, "rlsw_lineworld_step_kernel") != CUDA_SUCCESS_VALUE ||
            driver_.cuModuleGetFunction(&ppo_objective_, module_, "rlsw_ppo_objective_kernel") != CUDA_SUCCESS_VALUE) {
            last_error = "Failed to resolve compiled RLSwift kernel functions.";
            return false;
        }
        last_error = "CUDA kernels are available.";
        return true;
    }

    std::string programLog(nvrtcProgram program) {
        std::size_t log_size = 0;
        if (nvrtc_.nvrtcGetProgramLogSize(program, &log_size) != NVRTC_SUCCESS_VALUE || log_size == 0) {
            return "";
        }
        std::vector<char> log(log_size);
        if (nvrtc_.nvrtcGetProgramLog(program, log.data()) != NVRTC_SUCCESS_VALUE) {
            return "";
        }
        return std::string(log.data());
    }

    std::string driverError(CUresult status) {
        const char *name = nullptr;
        const char *message = nullptr;
        std::string result = "status " + std::to_string(status);
        if (driver_.cuGetErrorName != nullptr && driver_.cuGetErrorName(status, &name) == CUDA_SUCCESS_VALUE && name != nullptr) {
            result += " ";
            result += name;
        }
        if (driver_.cuGetErrorString != nullptr && driver_.cuGetErrorString(status, &message) == CUDA_SUCCESS_VALUE && message != nullptr) {
            result += " (";
            result += message;
            result += ")";
        }
        return result;
    }

    std::mutex mutex_;
    bool initialized_ = false;
    bool available_ = false;
    void *cuda_library_ = nullptr;
    void *nvrtc_library_ = nullptr;
    DriverAPI driver_;
    NVRTCAPI nvrtc_;
    CUcontext context_ = nullptr;
    CUmodule module_ = nullptr;
    CUfunction lineworld_ = nullptr;
    CUfunction ppo_objective_ = nullptr;
};

bool allocateDevice(std::size_t bytes, CUdeviceptr *pointer) {
    if (bytes == 0 || pointer == nullptr) {
        last_error = "Invalid CUDA allocation request.";
        return false;
    }
    CUresult status = CUDARuntime::shared().driver().cuMemAlloc(pointer, bytes);
    if (status != CUDA_SUCCESS_VALUE) {
        last_error = "CUDA allocation failed: " + CUDARuntime::shared().describe(status);
        return false;
    }
    return true;
}

void freeDevice(CUdeviceptr pointer) {
    if (pointer != 0) {
        CUDARuntime::shared().driver().cuMemFree(pointer);
    }
}

template <typename T>
bool copyHostToDevice(CUdeviceptr device, const T *host, int count) {
    CUresult status = CUDARuntime::shared().driver().cuMemcpyHtoD(device, host, sizeof(T) * static_cast<std::size_t>(count));
    if (status != CUDA_SUCCESS_VALUE) {
        last_error = "CUDA host-to-device copy failed: " + CUDARuntime::shared().describe(status);
        return false;
    }
    return true;
}

template <typename T>
bool copyDeviceToHost(T *host, CUdeviceptr device, int count) {
    CUresult status = CUDARuntime::shared().driver().cuMemcpyDtoH(host, device, sizeof(T) * static_cast<std::size_t>(count));
    if (status != CUDA_SUCCESS_VALUE) {
        last_error = "CUDA device-to-host copy failed: " + CUDARuntime::shared().describe(status);
        return false;
    }
    return true;
}

bool validateCommonCount(int count) {
    if (count <= 0) {
        last_error = "Kernel sample count must be positive.";
        return false;
    }
    return true;
}

} // namespace

int rlsw_cuda_is_available(void) {
    return CUDARuntime::shared().ensure() ? 1 : 0;
}

const char *rlsw_cuda_last_error_message(void) {
    return last_error.c_str();
}

int rlsw_cuda_lineworld_step(
    const int *actions,
    int *positions,
    int *step_indices,
    float *rewards,
    unsigned char *terminals,
    int *termination_codes,
    int environment_count,
    int length,
    int max_steps
) {
    if (actions == nullptr || positions == nullptr || step_indices == nullptr || rewards == nullptr ||
        terminals == nullptr || termination_codes == nullptr || !validateCommonCount(environment_count) ||
        length < 2 || max_steps <= 0) {
        last_error = "Invalid LineWorld CUDA kernel arguments.";
        return 1;
    }
    if (!CUDARuntime::shared().ensure()) {
        return 2;
    }

    CUdeviceptr d_actions = 0;
    CUdeviceptr d_positions = 0;
    CUdeviceptr d_steps = 0;
    CUdeviceptr d_rewards = 0;
    CUdeviceptr d_terminals = 0;
    CUdeviceptr d_codes = 0;
    bool ok = allocateDevice(sizeof(int) * environment_count, &d_actions) &&
        allocateDevice(sizeof(int) * environment_count, &d_positions) &&
        allocateDevice(sizeof(int) * environment_count, &d_steps) &&
        allocateDevice(sizeof(float) * environment_count, &d_rewards) &&
        allocateDevice(sizeof(unsigned char) * environment_count, &d_terminals) &&
        allocateDevice(sizeof(int) * environment_count, &d_codes) &&
        copyHostToDevice(d_actions, actions, environment_count) &&
        copyHostToDevice(d_positions, positions, environment_count) &&
        copyHostToDevice(d_steps, step_indices, environment_count);

    if (ok) {
        int block = 256;
        int grid = (environment_count + block - 1) / block;
        void *arguments[] = {
            &d_actions,
            &d_positions,
            &d_steps,
            &d_rewards,
            &d_terminals,
            &d_codes,
            &environment_count,
            &length,
            &max_steps,
        };
        CUresult launch_status = CUDARuntime::shared().driver().cuLaunchKernel(
            CUDARuntime::shared().lineworld(),
            static_cast<unsigned int>(grid),
            1,
            1,
            static_cast<unsigned int>(block),
            1,
            1,
            0,
            nullptr,
            arguments,
            nullptr
        );
        if (launch_status != CUDA_SUCCESS_VALUE) {
            last_error = "LineWorld CUDA kernel launch failed: " + CUDARuntime::shared().describe(launch_status);
            ok = false;
        }
        if (ok) {
            CUresult sync_status = CUDARuntime::shared().driver().cuCtxSynchronize();
            if (sync_status != CUDA_SUCCESS_VALUE) {
                last_error = "LineWorld CUDA kernel synchronize failed: " + CUDARuntime::shared().describe(sync_status);
                ok = false;
            }
        }
        if (ok && !copyDeviceToHost(positions, d_positions, environment_count)) {
            last_error = "LineWorld CUDA positions copy failed.";
            ok = false;
        }
        if (ok && !copyDeviceToHost(step_indices, d_steps, environment_count)) {
            last_error = "LineWorld CUDA step copy failed.";
            ok = false;
        }
        if (ok && !copyDeviceToHost(rewards, d_rewards, environment_count)) {
            last_error = "LineWorld CUDA reward copy failed.";
            ok = false;
        }
        if (ok && !copyDeviceToHost(terminals, d_terminals, environment_count)) {
            last_error = "LineWorld CUDA terminal copy failed.";
            ok = false;
        }
        if (ok && !copyDeviceToHost(termination_codes, d_codes, environment_count)) {
            last_error = "LineWorld CUDA termination copy failed.";
            ok = false;
        }
    }

    freeDevice(d_actions);
    freeDevice(d_positions);
    freeDevice(d_steps);
    freeDevice(d_rewards);
    freeDevice(d_terminals);
    freeDevice(d_codes);

    if (!ok) {
        if (last_error.empty()) {
            last_error = "LineWorld CUDA kernel launch or copy failed.";
        }
        return 3;
    }
    last_error = "LineWorld CUDA kernel completed.";
    return 0;
}

int rlsw_cuda_ppo_objective(
    const float *old_log_probabilities,
    const float *new_log_probabilities,
    const float *advantages,
    const float *returns,
    const float *value_estimates,
    const float *entropies,
    int sample_count,
    float clip_range,
    float value_loss_coefficient,
    float entropy_coefficient,
    rlsw_cuda_ppo_objective_result *result
) {
    if (old_log_probabilities == nullptr || new_log_probabilities == nullptr || advantages == nullptr ||
        returns == nullptr || value_estimates == nullptr || entropies == nullptr || result == nullptr ||
        !validateCommonCount(sample_count) || clip_range < 0.0f || value_loss_coefficient < 0.0f ||
        entropy_coefficient < 0.0f) {
        last_error = "Invalid PPO CUDA kernel arguments.";
        return 1;
    }
    if (!CUDARuntime::shared().ensure()) {
        return 2;
    }

    constexpr int block = 256;
    int grid = (sample_count + block - 1) / block;
    int partial_count = grid * 6;
    CUdeviceptr d_old = 0;
    CUdeviceptr d_new = 0;
    CUdeviceptr d_advantages = 0;
    CUdeviceptr d_returns = 0;
    CUdeviceptr d_values = 0;
    CUdeviceptr d_entropies = 0;
    CUdeviceptr d_partials = 0;

    bool ok = allocateDevice(sizeof(float) * sample_count, &d_old) &&
        allocateDevice(sizeof(float) * sample_count, &d_new) &&
        allocateDevice(sizeof(float) * sample_count, &d_advantages) &&
        allocateDevice(sizeof(float) * sample_count, &d_returns) &&
        allocateDevice(sizeof(float) * sample_count, &d_values) &&
        allocateDevice(sizeof(float) * sample_count, &d_entropies) &&
        allocateDevice(sizeof(float) * partial_count, &d_partials) &&
        copyHostToDevice(d_old, old_log_probabilities, sample_count) &&
        copyHostToDevice(d_new, new_log_probabilities, sample_count) &&
        copyHostToDevice(d_advantages, advantages, sample_count) &&
        copyHostToDevice(d_returns, returns, sample_count) &&
        copyHostToDevice(d_values, value_estimates, sample_count) &&
        copyHostToDevice(d_entropies, entropies, sample_count);

    std::vector<float> partials(static_cast<std::size_t>(partial_count), 0.0f);
    if (ok) {
        void *arguments[] = {
            &d_old,
            &d_new,
            &d_advantages,
            &d_returns,
            &d_values,
            &d_entropies,
            &sample_count,
            &clip_range,
            &d_partials,
        };
        unsigned int shared_bytes = static_cast<unsigned int>(6 * block * sizeof(float));
        ok = CUDARuntime::shared().driver().cuLaunchKernel(
                 CUDARuntime::shared().ppoObjective(),
                 static_cast<unsigned int>(grid),
                 1,
                 1,
                 static_cast<unsigned int>(block),
                 1,
                 1,
                 shared_bytes,
                 nullptr,
                 arguments,
                 nullptr
             ) == CUDA_SUCCESS_VALUE &&
             CUDARuntime::shared().driver().cuCtxSynchronize() == CUDA_SUCCESS_VALUE &&
             copyDeviceToHost(partials.data(), d_partials, partial_count);
    }

    freeDevice(d_old);
    freeDevice(d_new);
    freeDevice(d_advantages);
    freeDevice(d_returns);
    freeDevice(d_values);
    freeDevice(d_entropies);
    freeDevice(d_partials);

    if (!ok) {
        last_error = "PPO CUDA kernel launch or copy failed.";
        return 3;
    }

    double policy_sum = 0.0;
    double value_sum = 0.0;
    double entropy_sum = 0.0;
    double kl_sum = 0.0;
    double clipped_sum = 0.0;
    double count_sum = 0.0;
    for (int block_index = 0; block_index < grid; ++block_index) {
        int base = block_index * 6;
        policy_sum += partials[base];
        value_sum += partials[base + 1];
        entropy_sum += partials[base + 2];
        kl_sum += partials[base + 3];
        clipped_sum += partials[base + 4];
        count_sum += partials[base + 5];
    }

    if (count_sum <= 0.0) {
        last_error = "PPO CUDA kernel returned an empty reduction.";
        return 3;
    }

    result->sample_count = static_cast<int>(count_sum);
    result->policy_loss = -policy_sum / count_sum;
    result->value_loss = value_sum / count_sum;
    result->entropy_bonus = entropy_sum / count_sum;
    result->total_loss = result->policy_loss +
        static_cast<double>(value_loss_coefficient) * result->value_loss -
        static_cast<double>(entropy_coefficient) * result->entropy_bonus;
    result->mean_approximate_kl = kl_sum / count_sum;
    result->clipped_fraction = clipped_sum / count_sum;
    last_error = "PPO CUDA kernel completed.";
    return 0;
}
