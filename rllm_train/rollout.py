import asyncio
import json
import logging
import math
import re
import time

import torch

from rllm_train.hf_engine import HFAgentExecutionEngine
from rllm_train.math_env import MathCalcEnv

logger = logging.getLogger(__name__)


def make_rllm_rollout_func(
    agent_class,
    agent_args=None,
    env_class=None,
    env_args=None,
    max_steps=3,
    max_response_length=256,
    max_prompt_length=512,
    sampling_params=None,
    training_logger=None,
    trajectory_writer=None,
    perf_tracker=None,
    answer_map=None,
):
    if agent_args is None:
        agent_args = {}
    if env_args is None:
        env_args = {}
    if env_class is None:
        env_class = MathCalcEnv
    if sampling_params is None:
        sampling_params = {"temperature": 0.7, "top_p": 0.9}
    if answer_map is None:
        answer_map = {}

    _step_counter = [0]

    def rollout_func(prompts, trainer):
        _step_counter[0] += 1
        step = _step_counter[0]
        model = trainer.model
        tokenizer = trainer.processing_class
        device = trainer.accelerator.device

        num_gens = getattr(trainer.args, "num_generations", len(prompts))
        if training_logger:
            training_logger.log_rollout_start(step, len(prompts), num_gens)
        if perf_tracker:
            perf_tracker.start_rollout(step)

        model.eval()

        engine = HFAgentExecutionEngine(
            model=model,
            tokenizer=tokenizer,
            n_parallel_agents=len(prompts),
            max_steps=max_steps,
            max_response_length=max_response_length,
            max_prompt_length=max_prompt_length,
            agent_class=agent_class,
            agent_args=agent_args,
            env_class=env_class,
            env_args=env_args,
            sampling_params=sampling_params,
            on_trajectory_done=lambda idx, total, reward: (
                training_logger.log_trajectory_done(step, idx, total, reward)
                if training_logger else None
            ),
        )

        envs = []
        agents = []
        for prompt in prompts:
            task = _extract_task_from_prompt(prompt)
            if task["question"] in answer_map:
                task["answer"] = answer_map[task["question"]]
            task.update(env_args)
            env = env_class.from_dict(task)
            agent = agent_class(**agent_args)
            envs.append(env)
            agents.append(agent)

        engine.update_envs_and_agents(envs, agents)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    results = pool.submit(
                        lambda: asyncio.run(engine.run_trajectories(mode="Token"))
                    ).result()
            else:
                results = loop.run_until_complete(engine.run_trajectories(mode="Token"))
        except RuntimeError:
            results = asyncio.run(engine.run_trajectories(mode="Token"))

        model.train()

        # Perf tracking: end rollout phase
        rollout_stats = None
        if perf_tracker:
            rollout_stats = perf_tracker.end_rollout(results)

        prompt_ids_list = []
        completion_ids_list = []
        env_mask_list = []

        for result in results:
            prompt_ids_list.append(result["prompt_tokens"].tolist())
            completion_ids_list.append(result["response_tokens"].tolist())
            env_mask_list.append(result["response_masks"].tolist())

        rewards = [r.get("trajectory_reward", 0.0) if isinstance(r, dict) else 0.0 for r in results]
        behavior_metrics = _compute_behavior_metrics(agents, results, max_steps, num_gens)
        if training_logger:
            training_logger.log_rollout_done(
                step,
                len(results),
                rewards,
                rollout_stats=rollout_stats,
                behavior_metrics=behavior_metrics,
            )

        # Compute logprobs with timing
        if training_logger:
            training_logger.log_logprob_start(step)
        if perf_tracker:
            perf_tracker.start_logprob_compute()

        logprobs_list = _compute_logprobs(
            model, tokenizer, prompt_ids_list, completion_ids_list, device
        )

        if perf_tracker:
            perf_tracker.end_logprob_compute()

        # Write trajectories to file (after logprobs are computed)
        if trajectory_writer:
            trajectory_writer.write_rollout(step, agents, envs, results, tokenizer, logprobs_list=logprobs_list)

        if training_logger:
            training_logger.log_training_update_start(step)

        return {
            "prompt_ids": prompt_ids_list,
            "completion_ids": completion_ids_list,
            "logprobs": logprobs_list,
            "env_mask": env_mask_list,
        }

    return rollout_func


def _compute_behavior_metrics(agents, results, max_steps, num_generations=8):
    total = len(results)
    if total == 0:
        return {}

    rewards = [float(r.get("trajectory_reward", 0.0)) for r in results if isinstance(r, dict)]
    reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
    reward_variance = math.sqrt(sum((r - reward_mean) ** 2 for r in rewards) / len(rewards)) if rewards else 0.0

    # pass@1 and pass@K: group rewards by prompt (num_generations per prompt)
    pass1 = 0.0
    passK = 0.0
    if num_generations > 0 and total >= num_generations:
        num_prompts = total // num_generations
        pass1_count = 0
        passK_count = 0
        for p in range(num_prompts):
            group = rewards[p * num_generations:(p + 1) * num_generations]
            if any(r >= 1.0 for r in group):
                pass1_count += 1
            if all(r >= 1.0 for r in group):
                passK_count += 1
        pass1 = pass1_count / num_prompts
        passK = passK_count / num_prompts

    finish_count = 0
    finish_numeric_count = 0
    calculate_count = 0
    answer_count = 0
    truncation_count = 0
    parser_error_count = 0
    malformed_tool_count = 0
    synthetic_finish_count = 0
    calculator_error_count = 0
    symbolic_calculator_error_count = 0
    successful_calculate_count = 0
    no_finish_count = 0
    valid_finish_count = 0
    response_lengths = []
    agent_steps = []

    for agent, result in zip(agents, results):
        calls = _extract_tool_calls(agent)
        finish_calls = [c for c in calls if c.get("name") == "finish"]
        calculate_calls = [c for c in calls if c.get("name") == "calculate"]
        has_finish = bool(finish_calls)
        has_calculate = bool(calculate_calls)
        finish_values = [_finish_value(c) for c in finish_calls]
        has_numeric_finish = any(_is_numeric_answer(v) for v in finish_values)
        has_answer = any(_has_answer_text(v) for v in finish_values)

        if has_finish:
            finish_count += 1
        if has_numeric_finish:
            finish_numeric_count += 1
        if has_calculate:
            calculate_count += 1
        if has_answer:
            answer_count += 1

        step_infos = [getattr(step, "info", {}) for step in getattr(agent.trajectory, "steps", [])]
        reward_components = [info.get("reward_components") for info in step_infos if info.get("reward_components")]
        if any(info.get("parse_error_type") for info in step_infos):
            parser_error_count += 1
        if any(info.get("malformed_tool_call") for info in step_infos):
            malformed_tool_count += 1
        if any(info.get("synthetic_finish") for info in step_infos):
            synthetic_finish_count += 1
        if any((comp.get("calculator_errors") or 0) > 0 for comp in reward_components):
            calculator_error_count += 1
        if any((comp.get("symbolic_calculator_errors") or 0) > 0 for comp in reward_components):
            symbolic_calculator_error_count += 1
        if any((comp.get("successful_calculates") or 0) > 0 for comp in reward_components):
            successful_calculate_count += 1
        if any("no_finish" in (comp.get("events") or []) for comp in reward_components):
            no_finish_count += 1
        if any("valid_finish" in (comp.get("events") or []) for comp in reward_components):
            valid_finish_count += 1

        if isinstance(result, dict):
            steps = int(result.get("metrics", {}).get("steps", 0) or 0)
            reward = float(result.get("trajectory_reward", 0.0) or 0.0)
            if steps >= max_steps and reward == 0.0:
                truncation_count += 1
            agent_steps.append(steps)
            response_lengths.append(len(result.get("response_tokens", [])))

    return {
        "finish_rate": finish_count / total,
        "finish_format_rate": finish_numeric_count / finish_count if finish_count else 0.0,
        "tool_usage_rate": calculate_count / total,
        "truncation_rate": truncation_count / total,
        "answer_coverage": answer_count / total,
        "reward_variance": reward_variance,
        "parser_error_rate": parser_error_count / total,
        "malformed_tool_rate": malformed_tool_count / total,
        "synthetic_finish_rate": synthetic_finish_count / total,
        "calculator_error_rate": calculator_error_count / total,
        "symbolic_calculator_error_rate": symbolic_calculator_error_count / total,
        "successful_calculate_rate": successful_calculate_count / total,
        "no_finish_rate": no_finish_count / total,
        "valid_finish_rate": valid_finish_count / total,
        "avg_agent_steps": sum(agent_steps) / len(agent_steps) if agent_steps else 0.0,
        "completion_length_mean": sum(response_lengths) / len(response_lengths) if response_lengths else 0.0,
        "pass1": pass1,
        "passK": passK,
    }


def _extract_tool_calls(agent):
    calls = []
    for msg in getattr(agent, "chat_completions", []) or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        for match in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, flags=re.S):
            try:
                call = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            name = str(call.get("name", "")).lower()
            args = call.get("arguments") or {}
            calls.append({"name": name, "arguments": args})
    return calls


def _finish_value(call):
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        return ""
    if "response" in args:
        return args["response"]
    if "answer" in args:
        return args["answer"]
    if len(args) == 1:
        return next(iter(args.values()))
    return ""


def _is_numeric_answer(value):
    text = str(value).strip()
    return bool(
        re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?", text)
        or re.fullmatch(r"\\frac\{[-+]?\d+\}\{[-+]?\d+\}", text)
    )


def _has_answer_text(value):
    return bool(re.search(r"[-+]?\d+(?:\.\d+)?|\\frac\{[-+]?\d+\}\{[-+]?\d+\}", str(value)))


def _extract_task_from_prompt(prompt):
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return {"question": msg["content"]}
        return {"question": str(prompt)}
    return {"question": str(prompt)}


def _compute_logprobs(model, tokenizer, prompt_ids_list, completion_ids_list, device):
    logprobs_list = []
    for prompt_ids, completion_ids in zip(prompt_ids_list, completion_ids_list):
        if not completion_ids:
            logprobs_list.append([])
            continue
        input_ids = torch.tensor([prompt_ids + completion_ids], device=device)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
        prompt_len = len(prompt_ids)
        completion_logits = logits[0, prompt_len - 1:-1, :]
        completion_tokens = torch.tensor(completion_ids, device=device)
        log_probs = torch.log_softmax(completion_logits, dim=-1)
        token_logprobs = log_probs.gather(1, completion_tokens.unsqueeze(1)).squeeze(1)
        logprobs_list.append(token_logprobs.cpu().tolist())
    return logprobs_list
