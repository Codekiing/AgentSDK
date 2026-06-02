"""
Agent RL Demo: rllm + TRL integration on Mac Air

Usage:
    # Natural language:
    python -m rllm_train.train "用 qwen-0.5b 训练数学 agent，64 个问题，2 个 epoch"
    python -m rllm_train.train "quick test with 16 problems"

    # Default config:
    python -m rllm_train.train
"""

import os
import re
import sys
import warnings

os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*attention mask is not set.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import torch
from datasets import Dataset
import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from rllm_train.config import TrainingConfig, parse_natural_language
from rllm_train.logger import TrainingLogger
from rllm_train.math_env import (
    CalculateTool,
    FinishTool,
    MathCalcEnv,
    eval_answer_to_float,
    generate_math_problems,
    is_symbolic_calculator_error,
    score_math_trajectory,
)
from rllm_train.parsers import QwenToolParser
from rllm_train.rollout import make_rllm_rollout_func
from rllm_train.trajectory_writer import TrajectoryWriter
from rllm_train.tool_agent import ToolAgent
from transformers import TrainerCallback

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

USER_INSTRUCTION = """You are a tool-using math agent. Return tool calls in <tool_call> XML blocks.
The final answer must be submitted with finish.
Every tool call must include both <tool_call> and </tool_call>; never omit the closing tag.
After the final </tool_call>, stop generating.
calculate is optional and only for concrete numeric arithmetic after all values are substituted.
Only call calculate when the expression contains numbers, operators, parentheses, and allowed numeric functions.
Never call calculate if the expression contains letters such as x, y, n, a, b, equations, function definitions, solve, simplify, expand, factor, or symbolic algebra.
For algebra/function/equation problems, reason in text internally, derive the final numeric answer, then call finish directly.
If you are not sure the calculate input is pure numeric, do not call calculate.

Valid calculate example:
<tool_call>
{"name": "calculate", "arguments": {"expression": "sqrt(16) + 3"}}
</tool_call>

Final answer example:
<tool_call>
{"name": "finish", "arguments": {"response": "42"}}
</tool_call>
"""

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


def build_dataset(problems):
    records = []
    for p in problems:
        records.append({
            "prompt": [
                {
                    "role": "user",
                    "content": USER_INSTRUCTION + "\n\nProblem:\n" + p["question"],
                }
            ],
            "answer": p["answer"],
        })
    return Dataset.from_list(records)


_SYMBOLIC_DATASET_PATTERN = re.compile(
    r"\\(?:prove|sum|prod|int|lim)|"
    r"\b(prove|show that|solve for|find all|polynomial|equation|integer|positive integer|"
    r"real number|function|sequence|geometry)\b|"
    r"[a-z]\s*[=^]",
    re.I,
)


def _passes_dataset_filter(item, filter_name):
    if not filter_name:
        return True
    if filter_name != "numeric_calculator":
        raise ValueError(f"Unsupported dataset_filter: {filter_name}")
    question = str(item.get("problem", item.get("question", "")))
    answer = item.get("answer")
    if _SYMBOLIC_DATASET_PATTERN.search(question):
        return False
    return eval_answer_to_float(answer) is not None


def load_external_dataset(config):
    from datasets import load_from_disk
    ds = load_from_disk(config.dataset_path)
    original_size = len(ds)
    filter_name = getattr(config, "dataset_filter", "")
    if filter_name:
        indices = [i for i, item in enumerate(ds) if _passes_dataset_filter(item, filter_name)]
        if not indices:
            raise ValueError(f"dataset_filter={filter_name} removed all {original_size} examples")
        ds = ds.select(indices)
    filtered_size = len(ds)
    if config.num_problems < len(ds):
        ds = ds.shuffle(seed=config.seed).select(range(config.num_problems))
    records = []
    for item in ds:
        question = item.get("problem", item.get("question", ""))
        records.append({
            "prompt": [
                {
                    "role": "user",
                    "content": USER_INSTRUCTION + "\n\nProblem:\n" + question,
                },
            ],
            "answer": item["answer"],
        })
    dataset = Dataset.from_list(records)
    filter_summary = {
        "filter": filter_name or "none",
        "original_size": original_size,
        "filtered_size": filtered_size,
        "selected_size": len(dataset),
    }
    return dataset, filter_summary


def _completion_text(completion):
    if isinstance(completion, list):
        return " ".join(
            msg.get("content", "") for msg in completion if isinstance(msg, dict)
        )
    return str(completion)


def _completion_reward(text, answer):
    parser = QwenToolParser()
    successful_calculates = 0
    calculator_errors = 0
    symbolic_calculator_errors = 0
    unknown_tools = 0
    finished = False
    final_response = ""
    env = MathCalcEnv({"answer": answer})
    try:
        tool_call_dicts, diagnostics = parser.parse_with_diagnostics(text)
    except Exception:
        tool_call_dicts = []
        diagnostics = {"parse_error_type": "parser_exception", "malformed_tool_call": True}

    for tool_call in tool_call_dicts:
        name = tool_call.get("name")
        args = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
        if name == "calculate":
            result = env._safe_eval(args.get("expression", ""))
            result_text = env._format_tool_result(result)
            if result_text.startswith("Error:"):
                calculator_errors += 1
                if is_symbolic_calculator_error(result_text):
                    symbolic_calculator_errors += 1
            else:
                successful_calculates += 1
        elif name == "finish":
            finished = True
            final_response = args.get("response", "")
        else:
            unknown_tools += 1

    if not tool_call_dicts:
        final_response = text
        finished = True

    parser_errors = 1 if diagnostics.get("parse_error_type") else 0
    malformed_tool_calls = 1 if diagnostics.get("malformed_tool_call") else 0
    reward, _ = score_math_trajectory(
        final_response,
        answer,
        parsed_tool_call=bool(tool_call_dicts),
        synthetic_finish=not bool(tool_call_dicts),
        finished=finished,
        steps=1 if finished else 3,
        max_steps=3,
        successful_calculates=successful_calculates,
        calculator_errors=calculator_errors,
        unknown_tools=unknown_tools,
        symbolic_calculator_errors=symbolic_calculator_errors,
        parser_errors=parser_errors,
        malformed_tool_calls=malformed_tool_calls,
    )
    return reward


def math_reward_fn(prompts, completions, **kwargs):
    answers = kwargs.get("answer", [])
    return [
        _completion_reward(_completion_text(completion), answers[i] if i < len(answers) else None)
        for i, completion in enumerate(completions)
    ]


def deepscaler_reward_fn(prompts, completions, **kwargs):
    return math_reward_fn(prompts, completions, **kwargs)


def _compute_eval_metrics(eval_results, eval_dataset):
    """Compute evaluation metrics from rollout results on test set."""
    rewards = []
    num_finish = 0
    num_calculate = 0
    num_numeric = 0
    for i, result in enumerate(eval_results):
        r = result.get("reward", 0.0)
        rewards.append(r)
        chat_completions = result.get("chat_completions", [])
        has_finish = any("finish" in str(c.get("content", "")).lower() for c in chat_completions)
        has_calculate = any("calculate" in str(c.get("content", "")).lower() for c in chat_completions)
        if has_finish:
            num_finish += 1
        if has_calculate:
            num_calculate += 1
        response_text = result.get("response_text", "")
        if re.search(r'-?\d+\.?\d*', response_text):
            num_numeric += 1
    n = max(len(rewards), 1)
    avg = sum(rewards) / n
    variance = sum((r - avg) ** 2 for r in rewards) / n
    return {
        "avg_reward": avg,
        "reward_std": variance ** 0.5,
        "rewards": rewards,
        "finish_rate": num_finish / n,
        "tool_usage_rate": num_calculate / n,
        "answer_coverage": num_numeric / n,
        "num_samples": n,
    }


def main(config: TrainingConfig | None = None):
    if config is None:
        config = TrainingConfig()

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    config.to_json(os.path.join(output_dir, "config.json"))

    log_file = os.path.join(output_dir, "training_log.txt")
    log = TrainingLogger(verbose=config.verbose, log_file=log_file)
    log.log_training_start(config)

    class RllmCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                log.update_training_metrics(logs)

    model_path = _resolve_model_path(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True,
    )
    model.enable_input_require_grads()
    param_count = sum(p.numel() for p in model.parameters())
    log.log_model_loaded(model_path, device, param_count)

    if not config.dataset_path:
        raise ValueError(
            "dataset_path is required. "
            "Provide a path to a HuggingFace dataset directory with 'problem'/'question' and 'answer' fields."
        )
    dataset, filter_summary = load_external_dataset(config)
    if config.dataset == "deepscaler":
        reward_fn = deepscaler_reward_fn
    else:
        reward_fn = math_reward_fn
    log.log_dataset_ready(len(dataset), filter_summary)

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        logging_strategy="steps",
        log_level="info",
        save_strategy="no",
        bf16=(dtype == torch.bfloat16),
        fp16=False,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=getattr(config, 'gradient_checkpointing', False),
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=True,
        temperature=config.temperature,
        dataloader_num_workers=0,
    )

    log.log_trainer_ready(config.num_epochs)

    trajectory_writer = TrajectoryWriter(output_dir, enabled=True)

    answer_map = {}
    for item in dataset:
        for msg in item["prompt"]:
            if msg["role"] == "user":
                answer_map[msg["content"]] = item["answer"]
                break

    rollout_func = make_rllm_rollout_func(
        agent_class=ToolAgent,
        agent_args={
            "system_prompt":"",
            "tool_map": {"calculate": CalculateTool, "finish": FinishTool},
        },
        env_class=MathCalcEnv,
        max_steps=config.max_agent_steps,
        max_response_length=config.max_response_length,
        max_prompt_length=config.max_prompt_length,
        sampling_params={"temperature": config.temperature, "top_p": config.top_p},
        training_logger=log,
        trajectory_writer=trajectory_writer,
        answer_map=answer_map,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        rollout_func=rollout_func,
        peft_config=LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        callbacks=[RllmCallback()],
    )

    trainer.train()

    # ── Post-training evaluation on user-provided test set ──
    eval_metrics = None
    eval_dataset_dir = getattr(config, 'eval_dataset_path', '') or ''
    if eval_dataset_dir and os.path.isdir(eval_dataset_dir):
        log.write("\n========== Evaluation on Test Set ==========\n")
        from datasets import load_from_disk
        raw_eval = load_from_disk(eval_dataset_dir)
        eval_records = []
        for item in raw_eval:
            question = item.get("problem", item.get("question", ""))
            eval_records.append({
                "prompt": [
                    {
                        "role": "user",
                        "content": USER_INSTRUCTION + "\n\nProblem:\n" + question,
                    },
                ],
                "answer": item["answer"],
            })
        eval_dataset = Dataset.from_list(eval_records)
        eval_results = rollout_func(trainer.model, eval_dataset)
        eval_metrics = _compute_eval_metrics(eval_results, eval_dataset)
        log.write(f"  Test reward: {eval_metrics['avg_reward']:.4f} +- {eval_metrics['reward_std']:.4f}\n")
        log.write(f"  Test finish_rate: {eval_metrics['finish_rate']*100:.0f}%\n")
        log.write(f"  Test answer_coverage: {eval_metrics['answer_coverage']*100:.0f}%\n")
        log.write(f"  Test size: {eval_metrics['num_samples']}\n")
        eval_out_path = os.path.join(output_dir, "eval_metrics.json")
        import json as _json
        with open(eval_out_path, "w") as f:
            _json.dump(eval_metrics, f, indent=2)
        log.write(f"  Saved: {eval_out_path}\n")

    log.print_training_report(config, eval_metrics, output_dir)

    if config.save_model:
        save_path = os.path.join(output_dir, "final_model")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        log.log_model_saved(save_path)

    log.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = " ".join(sys.argv[1:])
        if arg.endswith(".json") and os.path.isfile(arg):
            cfg = TrainingConfig.from_json(arg)
        else:
            cfg = parse_natural_language(arg)
    else:
        cfg = TrainingConfig()
    main(cfg)
