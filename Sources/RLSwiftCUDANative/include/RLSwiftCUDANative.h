#ifndef RLSWIFT_CUDA_NATIVE_H
#define RLSWIFT_CUDA_NATIVE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rlsw_cuda_ppo_objective_result {
    double policy_loss;
    double value_loss;
    double entropy_bonus;
    double total_loss;
    double mean_approximate_kl;
    double clipped_fraction;
    int sample_count;
} rlsw_cuda_ppo_objective_result;

int rlsw_cuda_is_available(void);
const char *rlsw_cuda_last_error_message(void);

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
);

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
);

#ifdef __cplusplus
}
#endif

#endif
