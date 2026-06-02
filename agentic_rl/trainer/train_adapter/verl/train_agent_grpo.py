# -*- coding: utf-8 -*-
# ruff: noqa: E402
from agentic_rl.trainer.train_adapter.verl import patch

patch.apply_patch()
import ray
from omegaconf import OmegaConf, DictConfig

from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.trainer.train_adapter.verl.configs.parse_verl_config import (
    VerlConfigParser,
)

logger = Loggers(__name__)


def _create_tokenizer(config):
    trust_remote_code = config.data.get("trust_remote_code", False)
    try:
        from verl.utils import hf_tokenizer
        tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path, trust_remote_code=trust_remote_code)
    except ValueError as e:
        logger.error(f"Failed to create tokenizer: {e}")
        raise e
    return tokenizer


def _define_worker_classes(config):
    if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
        logger.error(f"actor strategy {config.actor_rollout_ref.actor.strategy} is not supported")
        raise ValueError(f"actor strategy {config.actor_rollout_ref.actor.strategy} is not supported")
    if config.critic.strategy not in {"fsdp", "fsdp2"}:
        logger.error(f"critic strategy {config.critic.strategy} is not supported")
        raise ValueError(f"critic strategy {config.critic.strategy} is not supported")

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
    }
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
    return role_worker_mapping, RayWorkerGroup


def _create_resource_pool_manager(config):
    global_pool_id = "global_pool"
    resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }
    return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)


@ray.remote
def train(config):
    try:
        agentic_rl_config, input_config, verl_config, gen_config = VerlConfigParser(config).process_config()
        logger.info("Config parsed successfully")
        tokenizer = _create_tokenizer(verl_config)
        role_worker_mapping, ray_worker_group_cls = _define_worker_classes(verl_config)
        resource_pool_manager = _create_resource_pool_manager(verl_config)

        from verl.trainer.ppo.reward import load_reward_manager
        reward_fn = load_reward_manager(
            verl_config, tokenizer, **verl_config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            verl_config, tokenizer, **verl_config.reward_model.get("reward_kwargs", {})
        )
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Configuration or initialization error: {e}")
        raise RuntimeError(f"Configuration or initialization error: {e}") from e
    except OSError as e:
        logger.error(f"OS error: {e}")
        raise RuntimeError(f"OS error: {e}") from e
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Unexpected error: {e}")
        raise RuntimeError(f"Unexpected error: {e}") from e
    trainer = None
    try:
        from agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer import AgentGRPOTrainer
        trainer = AgentGRPOTrainer(
            config=verl_config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            ray_worker_group_cls=ray_worker_group_cls,
            resource_pool_manager=resource_pool_manager,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            tokenizer_path=input_config.get("tokenizer_name_or_path"),
            dataset_additional_keys=input_config.get("dataset_additional_keys"),
            generate_config=gen_config,
            agentic_rl_config=agentic_rl_config,
            use_tensorboard=input_config.get("use_tensorboard", False),
            tensorboard_flush_interval=input_config.get("tensorboard_flush_interval", 20),
        )
        trainer.init_workers()
        trainer.fit()
        logger.info("Training completed successfully")
    except (AttributeError, ValueError, TypeError) as e:
        logger.error(f"Trainer initialization error: {e}")
        raise RuntimeError(f"Trainer initialization error: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during trainer fit: {e}")
        raise RuntimeError(f"Unexpected error during trainer fit: {e}") from e
    finally:
        if trainer is not None:
            trainer.shutdown()
