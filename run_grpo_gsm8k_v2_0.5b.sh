xi#!/usr/bin/env bash
set -xeuo pipefail

# ============================================================================
# GRPO Training: Qwen2.5-0.5B-Instruct on DeepScaler Algebra (FSDP)
# ============================================================================
# Hardware: 4× A800-SXM4-80GB (80GB VRAM each)
# Model:    Qwen2.5-0.5B-Instruct (HuggingFace format)
# Dataset:  DeepScaler Algebra (512 train / 20 test)
# Backend:  VERL + FSDP + vLLM rollout
# ============================================================================

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

########################### Project Paths ###########################

PROJECT_ROOT="/lixiang/project/huxing/Project_test/AgentSDKMetricSkill"
VERL_HOME="${PROJECT_ROOT}/verl_latest"

HF_MODEL_PATH="${PROJECT_ROOT}/Qwen2.5-0.5B-Instruct"
TRAIN_DATA="${PROJECT_ROOT}/data/gsm8k_256/train.parquet"
TEST_DATA="${PROJECT_ROOT}/data/gsm8k_256/test.parquet"
OUTPUT_DIR="${PROJECT_ROOT}/verl_test_outputs"

########################### Quick Config ###########################

rollout_mode=${rollout_mode:-async}
return_raw_chat=${return_raw_chat:-True}

# Rollout TP: set to 1 for FSDP (vLLM uses single GPU per worker)
GEN_TP=${GEN_TP:-1}

train_files=${train_files:-"['${TRAIN_DATA}']"}
test_files=${test_files:-"['${TEST_DATA}']"}

########################### Data Config ###########################

temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

DATA=(
    "data.train_files=${train_files}"
    "data.val_files=${test_files}"
    "data.return_raw_chat=${return_raw_chat}"
    data.train_batch_size=32
    data.max_prompt_length=512
    data.max_response_length=512
    data.filter_overlong_prompts=True
    data.truncation='error'
)

########################### Model Config ###########################

MODEL=(
    "actor_rollout_ref.model.path=${HF_MODEL_PATH}"
)

########################### Actor Config (FSDP) ###########################

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1
)

########################### Rollout Config (vLLM) ###########################

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.name=vllm
    "actor_rollout_ref.rollout.mode=${rollout_mode}"
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.n=8
    
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
)

########################### Reference Model Config (FSDP) ###########################

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.fsdp_config.fsdp_size=-1
)

########################### Algorithm Config ###########################

########################### Reward Config ###########################

REWARD=(
    "custom_reward_function.path=${PROJECT_ROOT}/custom_rewards/deepscaler_reward.py"
    custom_reward_function.name=compute_score
)

########################### Algorithm Config ###########################

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

########################### Trainer Config ###########################

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","file","tensorboard"]'
    trainer.project_name='deepscaler_algebra_grpo'
    trainer.experiment_name='qwen2.5-0.5b-instruct_fsdp'
    trainer.n_gpus_per_node=4
    trainer.nnodes=1
    trainer.save_freq=20
    trainer.test_freq=5
    trainer.total_epochs=15
    "trainer.default_local_dir=${OUTPUT_DIR}"
)

########################### Launch Training ###########################

cd "${VERL_HOME}"

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    "${DATA[@]}" \
    "${REWARD[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"
