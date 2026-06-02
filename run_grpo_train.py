#!/usr/bin/env python3
"""
Qwen2.5-7B + GSM8K + GRPO via VERL native API
"""
import os

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

VERL_CONFIG_DIR = os.path.join(
    os.path.dirname(__import__("verl").__file__), "trainer", "config"
)

with initialize_config_dir(config_dir=VERL_CONFIG_DIR, version_base=None):
    cfg = compose(config_name="ppo_trainer")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen2.5-7B")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/gsm8k")

cfg_dict = OmegaConf.to_container(cfg, resolve=True)

# Data
cfg_dict["data"]["train_files"] = f"{DATA_DIR}/train.parquet"
cfg_dict["data"]["val_files"] = f"{DATA_DIR}/test.parquet"
cfg_dict["data"]["train_batch_size"] = 128
cfg_dict["data"]["max_prompt_length"] = 512
cfg_dict["data"]["max_response_length"] = 1024

# Model
cfg_dict["actor_rollout_ref"]["model"]["path"] = MODEL_PATH
cfg_dict["actor_rollout_ref"]["model"]["override_config"]["attn_implementation"] = "eager"
cfg_dict["actor_rollout_ref"]["hybrid_engine"] = True

# Actor
cfg_dict["actor_rollout_ref"]["actor"]["strategy"] = "fsdp"
cfg_dict["actor_rollout_ref"]["actor"]["optim"]["lr"] = 1e-6
cfg_dict["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 32
cfg_dict["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] = 2
cfg_dict["actor_rollout_ref"]["actor"]["use_kl_loss"] = False
cfg_dict["actor_rollout_ref"]["actor"]["use_torch_compile"] = False

# Rollout (vllm)
cfg_dict["actor_rollout_ref"]["rollout"]["name"] = "vllm"
cfg_dict["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] = 1
cfg_dict["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.5
cfg_dict["actor_rollout_ref"]["rollout"]["n"] = 4
cfg_dict["actor_rollout_ref"]["rollout"]["max_model_len"] = 1536
cfg_dict["actor_rollout_ref"]["rollout"]["max_num_seqs"] = 16
cfg_dict["actor_rollout_ref"]["rollout"]["enforce_eager"] = True
cfg_dict["actor_rollout_ref"]["rollout"]["load_format"] = "auto"
cfg_dict["actor_rollout_ref"]["rollout"]["enable_chunked_prefill"] = False
cfg_dict["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] = 2
cfg_dict["actor_rollout_ref"]["rollout"]["checkpoint_engine"] = {
    "_target_": "verl.workers.config.CheckpointEngineConfig",
    "backend": "naive",
    "update_weights_bucket_megabytes": 4096,
    "engine_kwargs": {},
}

# Critic (disabled by GRPO, but path needed)
cfg_dict["critic"]["model"]["path"] = MODEL_PATH
cfg_dict["critic"]["optim"]["lr"] = 1e-6
cfg_dict["critic"]["ppo_mini_batch_size"] = 32
cfg_dict["critic"]["ppo_micro_batch_size_per_gpu"] = 2

# Algorithm
cfg_dict["algorithm"]["adv_estimator"] = "grpo"
cfg_dict["algorithm"]["use_kl_in_reward"] = False

# Trainer
TOTAL_STEPS = 100
cfg_dict["trainer"]["project_name"] = "verl_grpo_gsm8k_gpu"
cfg_dict["trainer"]["experiment_name"] = "qwen2_7b_math"
cfg_dict["trainer"]["total_training_steps"] = TOTAL_STEPS
cfg_dict["trainer"]["total_epochs"] = 1
cfg_dict["trainer"]["n_gpus_per_node"] = 4
cfg_dict["trainer"]["save_freq"] = 50
cfg_dict["trainer"]["test_freq"] = 10
cfg_dict["trainer"]["val_before_train"] = True
cfg_dict["trainer"]["logger"] = ["console"]
cfg_dict["trainer"]["device"] = "cuda"

# Reward
cfg_dict["reward_model"]["enable"] = False

config = OmegaConf.create(cfg_dict)

print("=" * 60)
print("VERL GRPO Training: Qwen2.5-7B on GSM8K")
print(f"  Model:   {MODEL_PATH}")
print(f"  Data:    {DATA_DIR}")
print(f"  Steps:   {TOTAL_STEPS}")
print(f"  Batch:   128, n=8, lr=1e-6")
print(f"  GPUs:    4")
print("=" * 60)

import ray
ray.init(include_dashboard=False)

from verl.utils import hf_tokenizer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker, CriticWorker
from verl.single_controller.ray import RayWorkerGroup

tokenizer = hf_tokenizer(MODEL_PATH)

role_worker_mapping = {
    Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
    Role.Critic: ray.remote(CriticWorker),
}

resource_pool_manager = ResourcePoolManager(
    resource_pool_spec={"global_pool": [config.trainer.n_gpus_per_node]},
    mapping={
        Role.ActorRollout: "global_pool",
        Role.Critic: "global_pool",
    },
)

trainer = RayPPOTrainer(
    config=config,
    tokenizer=tokenizer,
    role_worker_mapping=role_worker_mapping,
    resource_pool_manager=resource_pool_manager,
    ray_worker_group_cls=RayWorkerGroup,
)

print("Initializing workers...")
trainer.init_workers()
print("Workers initialized. Starting training...")
trainer.fit()
print("Training completed!")

ray.shutdown()
