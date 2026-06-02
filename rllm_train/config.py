"""
Training configuration and natural language launcher.
"""

import json
import os
import shlex
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _detect_gpus() -> int | None:
    """Auto-detect number of available GPUs via nvidia-smi or torch.cuda."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpus = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            if gpus:
                return len(gpus)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return None

_DATASET_PATHS = {
    "deepscaler": {"train": "data/deepscaler_verl/train.parquet", "test": "data/deepscaler_verl/test.parquet"},
    "deepscaler_algebra": {"train": "data/deepscaler_algebra/train.parquet", "test": "data/deepscaler_algebra/test.parquet"},
    "deepscaler_algebra_verl": {"train": "data/deepscaler_algebra_verl/train.parquet", "test": "data/deepscaler_algebra_verl/test.parquet"},
    "deepscaler_algebra_verl_fmt": {"train": "data/deepscaler_algebra_verl_fmt/train.parquet", "test": "data/deepscaler_algebra_verl_fmt/test.parquet"},
}


def _resolve_model_path(model_name: str) -> str:
    if os.path.isdir(model_name):
        return model_name
    basename = model_name.split("/")[-1]
    for candidate in [
        os.path.join(_PROJECT_ROOT, basename),
        os.path.join(_PROJECT_ROOT, "models", basename),
        os.path.join(_PROJECT_ROOT, "model", basename),
    ]:
        if os.path.isdir(candidate):
            return candidate
    return model_name


def _resolve_dataset_paths(dataset: str, dataset_path: str) -> dict[str, str]:
    if dataset in _DATASET_PATHS:
        return _DATASET_PATHS[dataset]
    if dataset_path:
        p = Path(dataset_path)
        if p.suffix == ".parquet":
            return {"train": str(p), "test": str(p)}
        return {"train": str(p / "train.parquet"), "test": str(p / "test.parquet")}
    raise ValueError(f"Cannot resolve VERL dataset paths. dataset={dataset!r}, dataset_path={dataset_path!r}")


@dataclass
class TrainingConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Dataset
    num_problems: int = 256       # R9实证: 256 problems + batch32 → reward 0.800
    seed: int = 48                # R9实证有效种子
    difficulty: str = "mixed"
    dataset: str = ""             # "deepscaler" or "" (default: synthetic)
    dataset_path: str = ""        # Path to external dataset directory
    gradient_checkpointing: bool = False

    # Agent / Environment
    task_type: str = "math"
    max_agent_steps: int = 4      # R9: 4 agent steps
    max_response_length: int = 1024   # R9: 1024 (足够, 1536无增益)
    max_prompt_length: int = 512
    temperature: float = 0.7
    top_p: float = 1.0            # R9: top_p=1.0

    # Training
    num_epochs: int = 10          # R9: 10 epochs, 目标≥0.9用15-20
    batch_size: int = 2           # per-GPU batch, train_batch=2×4×4=32
    num_generations: int = 8      # GRPO group size, 0.5B推荐8-12
    max_completion_length: int = 1024
    learning_rate: float = 2e-6   # R9: 2e-6 effective, 3e-6可能更好
    gradient_accumulation_steps: int = 4

    # VERL-specific (R9实证, also used by rllm-config skill)
    entropy_coeff: float = 0.003     # R9: 0.003有效, 目标≥0.9用0.005
    use_kl_loss: bool = True         # R9关键: KL loss对0.5B有益, 不要关!
    kl_loss_coef: float = 0.01       # R9: low_var_kl + coef=0.01
    kl_loss_type: str = "low_var_kl"
    train_batch_size: int = 32       # R9: 32 for smooth gradients
    ppo_mini_batch_size: int = 16    # R9: train_batch/2
    ppo_micro_batch_size_per_gpu: int = 2
    ppo_epochs: int = 1
    test_freq: int = 5               # Validation frequency
    save_freq: int = 20              # Checkpoint save frequency
    gpu_memory_utilization: float = 0.5  # R9: 0.5 stable
    rollout_mode: str = "async"
    return_raw_chat: bool = True
    val_top_p: float = 0.7
    adv_estimator: str = "grpo"
    use_kl_in_reward: bool = False

    # Output
    run_id: str = ""
    output_dir: str = ""
    save_model: bool = True

    # Logging
    logging_steps: int = 1
    verbose: bool = True

    def __post_init__(self):
        if not self.run_id:
            self.run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        if not self.output_dir:
            base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "runs")
            self.output_dir = os.path.join(base, self.run_id)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Training Configuration",
            "=" * 60,
            f"  Run ID:           {self.run_id}",
            f"  Task type:        {self.task_type}",
            f"  Model:            {self.model_name}",
            f"  Problems:         {self.num_problems} (seed={self.seed})",
            f"  Agent steps:      {self.max_agent_steps}",
            f"  Temperature:      {self.temperature}",
            f"  Epochs:           {self.num_epochs}",
            f"  Batch size:       {self.batch_size} (train: {self.train_batch_size})",
            f"  Generations/prompt: {self.num_generations}",
            f"  Learning rate:    {self.learning_rate}",
            f"  Grad accum steps: {self.gradient_accumulation_steps}",
            f"  Entropy coeff:    {self.entropy_coeff}",
            f"  KL loss:          {self.use_kl_loss} (coef={self.kl_loss_coef})",
            f"  Max response len: {self.max_response_length}",
            f"  Output:           {self.output_dir}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_json(self, path: str | None = None) -> str:
        data = asdict(self)
        text = json.dumps(data, indent=2, ensure_ascii=False)
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        return text

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            data = json.load(f)
        known_fields = {f.name for f in __import__("dataclasses").fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    # ── VERL CLI generation (for backend=verl) ──

    def _verl_safe_ppo_mini_batch_size(self) -> int:
        """Compute ppo_mini_batch_size that satisfies VERL constraint.

        VERL requires: train_batch_size % ppo_mini_batch_size == 0
        (derived from rollout_samples % mini_batch == 0 in protocol.py:816)
        Returns the largest divisor of train_batch_size that is <= 16.
        """
        max_val = 16
        for candidate in range(min(self.batch_size, max_val), 0, -1):
            if self.batch_size % candidate == 0:
                return candidate
        return 1

    def to_verl_args(self) -> list[str]:
        """Generate VERL Hydra CLI args."""
        args = []
        args.append("algorithm.adv_estimator=grpo")
        args.append("algorithm.use_kl_in_reward=False")

        paths = _resolve_dataset_paths(self.dataset, self.dataset_path)
        args.append(f"data.train_files=['{os.path.abspath(paths['train'])}']")
        args.append(f"data.val_files=['{os.path.abspath(paths['test'])}']")
        args.append(f"data.train_batch_size={self.batch_size}")
        args.append(f"data.max_prompt_length={self.max_prompt_length}")
        args.append(f"data.max_response_length={self.max_response_length}")
        args.append("data.filter_overlong_prompts=True")
        args.append("data.truncation='error'")
        # Limit training samples to num_problems (VERL native max_samples)
        if self.num_problems > 0:
            args.append(f"data.train_max_samples={self.num_problems}")

        model_path = _resolve_model_path(self.model_name)
        args.append(f"actor_rollout_ref.model.path={model_path}")
        args.append("actor_rollout_ref.model.use_remove_padding=True")
        gc = getattr(self, "gradient_checkpointing", False)
        args.append(f"actor_rollout_ref.model.enable_gradient_checkpointing={'True' if gc else 'False'}")

        args.append(f"actor_rollout_ref.actor.optim.lr={self.learning_rate}")
        args.append(f"actor_rollout_ref.actor.ppo_mini_batch_size={self._verl_safe_ppo_mini_batch_size()}")
        args.append("actor_rollout_ref.actor.use_dynamic_bsz=True")
        args.append("actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384")
        args.append("actor_rollout_ref.actor.use_kl_loss=" + ("True" if self.use_kl_loss else "False"))
        args.append(f"actor_rollout_ref.actor.kl_loss_coef={self.kl_loss_coef}")
        args.append(f"actor_rollout_ref.actor.kl_loss_type={self.kl_loss_type}")
        args.append(f"actor_rollout_ref.actor.entropy_coeff={self.entropy_coeff}")
        fsdp_offload = getattr(self, "fsdp_param_offload", False)
        args.append(f"actor_rollout_ref.actor.fsdp_config.param_offload={'True' if fsdp_offload else 'False'}")
        args.append(f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={'True' if fsdp_offload else 'False'}")

        args.append("actor_rollout_ref.rollout.name=vllm")
        args.append("actor_rollout_ref.rollout.tensor_model_parallel_size=1")
        gpu_mem = getattr(self, "gpu_memory_utilization", None) or self.gpu_memory_utilization
        args.append(f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_mem}")
        args.append(f"actor_rollout_ref.rollout.n={self.num_generations}")
        args.append(f"actor_rollout_ref.rollout.temperature={self.temperature}")
        args.append("actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True")
        args.append("actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384")

        args.append("actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True")
        args.append("actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384")
        args.append(f"actor_rollout_ref.ref.fsdp_config.param_offload={'True' if fsdp_offload else 'False'}")

        args.append("trainer.balance_batch=True")
        args.append('trainer.logger=["console","file","tensorboard"]')
        args.append("trainer.project_name=verl_grpo_agent")
        args.append(f"trainer.experiment_name={self.run_id}")
        # GPU detection priority: explicit ngpus_per_node > resolved_num_gpus (runtime probe) > auto-detect > default 4
        ngpus = getattr(self, "ngpus_per_node", None)
        if ngpus is None:
            ngpus = getattr(self, "resolved_num_gpus", None)
        if ngpus is None:
            ngpus = _detect_gpus()
        if ngpus is None:
            ngpus = 4
        args.append(f"trainer.n_gpus_per_node={ngpus}")
        args.append("trainer.nnodes=1")
        args.append(f"trainer.save_freq={self.save_freq}")
        args.append(f"trainer.test_freq={self.test_freq}")
        args.append(f"trainer.total_epochs={self.num_epochs}")
        args.append("trainer.val_before_train=False")

        abs_output = os.path.abspath(self.output_dir)
        args.append(f"trainer.rollout_data_dir={os.path.join(abs_output, 'trajectories')}")
        args.append(f"trainer.validation_data_dir={os.path.join(abs_output, 'validation')}")

        reward_path = os.path.abspath("custom_rewards/deepscaler_reward.py")
        if os.path.exists(reward_path):
            args.append(f"reward.custom_reward_function.path={reward_path}")
            args.append("reward.custom_reward_function.name=compute_score")

        return args

    def to_verl_script(self) -> str:
        """Generate VERL launch shell script, return file path."""
        args = self.to_verl_args()
        os.makedirs(self.output_dir, exist_ok=True)
        ngpus = getattr(self, "ngpus_per_node", None)
        if ngpus is None:
            ngpus = getattr(self, "resolved_num_gpus", None)
        if ngpus is None:
            ngpus = _detect_gpus()
        if ngpus is None:
            ngpus = 4

        lines = ["#!/usr/bin/env bash", "set -xeuo pipefail", ""]
        lines.append(f"# VERL GRPO training: {self.model_name}")
        lines.append(f"# GPUs: {ngpus}")
        lines.append("")
        lines.append(f"NNODES=1")
        lines.append(f"NGPUS_PER_NODE={ngpus}")
        lines.append(f"export VERL_FILE_LOGGER_PATH={self.output_dir}/verl_metrics.jsonl")
        lines.append("")
        lines.append("python3 -m verl.trainer.main_ppo \\")
        for i, arg in enumerate(args):
            suffix = " \\" if i < len(args) - 1 else ""
            lines.append(f"    {shlex.quote(arg)}{suffix}")
        lines.append("")

        script_path = os.path.join(self.output_dir, "run_verl.sh")
        with open(script_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.chmod(script_path, 0o755)
        return script_path

    def verl_summary(self) -> dict:
        """Return VERL config summary for display."""
        paths = _resolve_dataset_paths(self.dataset, self.dataset_path)
        return {
            "model": self.model_name,
            "lr": self.learning_rate,
            "num_epochs": self.num_epochs,
            "num_generations": self.num_generations,
            "train_batch_size": self.batch_size,
            "max_response_length": self.max_response_length,
            "temperature": self.temperature,
            "data_train": paths["train"],
            "data_val": paths["test"],
        }


# Keyword mappings for natural language parsing
MODEL_ALIASES = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
}


def parse_natural_language(description: str) -> TrainingConfig:
    """Parse a natural language training description into a TrainingConfig.

    Examples:
        "用 qwen-0.5b 训练数学 agent，100 个问题，3 个 epoch"
        "train math agent with qwen-1.5b, 200 problems, lr=5e-6"
        "快速测试，16 个问题，1 epoch"
    """
    import re
    config = TrainingConfig()
    desc = description.lower()

    # Model
    for alias, full_name in MODEL_ALIASES.items():
        if alias in desc:
            config.model_name = full_name
            break

    # Number of problems
    m = re.search(r'(\d+)\s*(?:个|道)?(?:问题|题目|problems?|samples?|examples?)', desc)
    if m:
        config.num_problems = int(m.group(1))

    # Epochs
    m = re.search(r'(\d+)\s*(?:个)?\s*(?:epoch|轮)', desc)
    if m:
        config.num_epochs = int(m.group(1))

    # Learning rate
    m = re.search(r'lr\s*=?\s*([\d.e-]+)', desc)
    if m:
        config.learning_rate = float(m.group(1))

    # Batch size
    m = re.search(r'batch\s*(?:_?size)?\s*=?\s*(\d+)', desc)
    if m:
        config.batch_size = int(m.group(1))

    # Temperature
    m = re.search(r'(?:temp|temperature)\s*=?\s*([\d.]+)', desc)
    if m:
        config.temperature = float(m.group(1))

    # Generations
    m = re.search(r'(\d+)\s*(?:个)?(?:generations?|生成)', desc)
    if m:
        config.num_generations = int(m.group(1))

    # Max steps
    m = re.search(r'(\d+)\s*(?:个)?(?:steps?|步)', desc)
    if m:
        config.max_agent_steps = int(m.group(1))

    # Quick test mode
    if any(kw in desc for kw in ("快速测试", "quick test", "fast test", "测试一下")):
        config.num_problems = 16
        config.num_epochs = 1
        config.batch_size = 2
        config.num_generations = 2

    # Task type
    if any(kw in desc for kw in ("数学", "math", "计算", "calc")):
        config.task_type = "math"
    elif any(kw in desc for kw in ("代码", "code", "coding")):
        config.task_type = "code"
    elif any(kw in desc for kw in ("搜索", "search")):
        config.task_type = "search"

    return config
