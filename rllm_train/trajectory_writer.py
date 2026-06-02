"""
Trajectory file writer — saves rollout trajectories to JSONL for inspection.
Each line is one trajectory with full conversation, metrics, and reward.
"""

import json
import os
import time
from pathlib import Path


class TrajectoryWriter:
    def __init__(self, output_dir, enabled=True):
        self.enabled = enabled
        if not enabled:
            return
        self.traj_dir = os.path.join(output_dir, "trajectories")
        os.makedirs(self.traj_dir, exist_ok=True)
        self._global_traj_id = 0

    def write_rollout(self, step, agents, envs, results, tokenizer=None, logprobs_list=None):
        if not self.enabled:
            return
        filepath = os.path.join(self.traj_dir, f"step_{step:04d}.jsonl")
        with open(filepath, "w") as f:
            for i, (agent, env, result) in enumerate(zip(agents, envs, results)):
                self._global_traj_id += 1
                logprobs = logprobs_list[i] if logprobs_list and i < len(logprobs_list) else None
                record = self._build_record(
                    traj_id=self._global_traj_id,
                    step=step,
                    index=i,
                    agent=agent,
                    env=env,
                    result=result,
                    tokenizer=tokenizer,
                    logprobs=logprobs,
                )
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _build_record(self, traj_id, step, index, agent, env, result, tokenizer, logprobs=None):
        record = {
            "trajectory_id": traj_id,
            "step": step,
            "index": index,
            "question": getattr(env, "question", ""),
            "expected_answer": getattr(env, "answer", ""),
            "chat_completions": agent.chat_completions if hasattr(agent, "chat_completions") else [],
        }

        if hasattr(agent, "trajectory"):
            step_infos = [getattr(step, "info", {}) for step in agent.trajectory.steps]
            parser_diagnostics = [
                {
                    "parsed_tool_call": bool(info.get("parsed_tool_call")),
                    "synthetic_finish": bool(info.get("synthetic_finish")),
                    "parse_error_type": info.get("parse_error_type"),
                    "malformed_tool_call": bool(info.get("malformed_tool_call")),
                    "empty_tool_call": bool(info.get("empty_tool_call")),
                    "has_tool_tag": bool(info.get("has_tool_tag")),
                    "invalid_tool_call_count": int(info.get("invalid_tool_call_count", 0) or 0),
                    "recovered_missing_end_tag_count": int(info.get("recovered_missing_end_tag_count", 0) or 0),
                }
                for info in step_infos
            ]
            reward_breakdown = [info.get("reward_components") for info in step_infos if info.get("reward_components")]
            if parser_diagnostics:
                record["parser_diagnostics_by_step"] = parser_diagnostics
                record["parser_diagnostics"] = parser_diagnostics[-1]
            if reward_breakdown:
                record["reward_breakdown_by_step"] = reward_breakdown
                record["reward_components"] = reward_breakdown[-1]

        if isinstance(result, dict):
            metrics = result.get("metrics", {})
            record["reward"] = result.get("trajectory_reward", 0.0)
            record["num_steps"] = metrics.get("steps", 0)
            record["prompt_tokens"] = len(result.get("prompt_tokens", []))
            record["response_tokens"] = len(result.get("response_tokens", []))
            record["response_masks_sum"] = int(sum(
                result.get("response_masks", [])
                if not hasattr(result.get("response_masks", []), "sum")
                else [result["response_masks"].sum().item()]
            ))
            record["timing"] = {
                "llm_time": metrics.get("llm_time", 0),
                "env_time": metrics.get("env_time", 0),
                "total_time": metrics.get("total_time", 0),
                "reward_time": metrics.get("reward_time"),
            }

            if tokenizer and "response_tokens" in result:
                tokens = result["response_tokens"]
                if hasattr(tokens, "tolist"):
                    tokens = tokens.tolist()
                record["response_text"] = tokenizer.decode(tokens, skip_special_tokens=True)

            # Token-level data for RL export
            if "prompt_tokens" in result:
                pt = result["prompt_tokens"]
                record["prompt_ids"] = pt.tolist() if hasattr(pt, "tolist") else pt
            if "response_tokens" in result:
                rt = result["response_tokens"]
                record["completion_ids"] = rt.tolist() if hasattr(rt, "tolist") else rt
            if "response_masks" in result:
                rm = result["response_masks"]
                record["env_mask"] = rm.tolist() if hasattr(rm, "tolist") else rm
            if logprobs is not None:
                record["logprobs"] = logprobs

        return record

    def write_summary(self, step, summary_stats):
        if not self.enabled:
            return
        filepath = os.path.join(self.traj_dir, f"step_{step:04d}_summary.json")
        with open(filepath, "w") as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
