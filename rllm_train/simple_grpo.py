"""Simple GRPO training loop bypassing TRL's GRPOTrainer."""

import os
import re
import sys
import time
import json
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from rllm_train.config import TrainingConfig
from rllm_train.math_env import generate_math_problems


def compute_reward(completion_text, ground_truth):
    numbers = re.findall(r'-?\d+\.?\d*', completion_text)
    if numbers and ground_truth is not None:
        try:
            return 1.0 if abs(float(numbers[-1]) - float(ground_truth)) < 1e-6 else 0.0
        except (ValueError, TypeError):
            return 0.0
    return 0.0


def main(config):
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config.to_json(os.path.join(output_dir, "config.json"))

    log_file = os.path.join(output_dir, "training_log.txt")

    def log(msg):
        print(msg, flush=True)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    log("=" * 60)
    log("Simple GRPO Training")
    log("=" * 60)
    log(f"  Model:      {config.model_name}")
    log(f"  Problems:   {config.num_problems}")
    log(f"  Epochs:     {config.num_epochs}")
    log(f"  Batch:      {config.batch_size} x {config.gradient_accumulation_steps} accum")
    log(f"  Generations: {config.num_generations}")
    log(f"  LR:         {config.learning_rate}")
    log(f"  Output:     {output_dir}")
    log("=" * 60)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda()
    model.enable_input_require_grads()

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"  LoRA:       {trainable:,} trainable / {total:,} total ({trainable/total*100:.2f}%)")
    log("")

    # Generate problems
    problems = generate_math_problems(n=config.num_problems, seed=config.seed, difficulty=config.difficulty)
    prompts_text = []
    answers = []
    for p in problems:
        messages = [
            {"role": "system", "content": "Solve the math problem. Show your reasoning, then write the final numeric answer."},
            {"role": "user", "content": p["question"]},
        ]
        prompts_text.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        answers.append(p["answer"])

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate)

    batch_size = config.batch_size
    num_gens = config.num_generations
    grad_accum = config.gradient_accumulation_steps
    total_steps = len(problems) * config.num_epochs // (batch_size * grad_accum)

    log(f"Training: {len(problems)} problems, {total_steps} steps")
    log("")

    start_time = time.time()
    step_times = []
    reward_history = []
    global_step = 0
    problem_idx = 0

    for epoch in range(config.num_epochs):
        # Shuffle
        import random
        indices = list(range(len(problems)))
        random.seed(config.seed + epoch)
        random.shuffle(indices)

        for step_start in range(0, len(indices), batch_size * grad_accum):
            global_step += 1
            step_t0 = time.time()

            optimizer.zero_grad()

            batch_rewards = []
            accum_count = 0

            for micro in range(grad_accum):
                idx_offset = step_start + micro * batch_size
                if idx_offset >= len(indices):
                    break

                batch_indices = indices[idx_offset:idx_offset + batch_size]
                if not batch_indices:
                    break

                accum_count += 1

                # Generate completions for each prompt in micro-batch
                all_prompt_ids = []
                all_completion_ids = []
                all_rewards = []
                all_answers_flat = []

                for bi in batch_indices:
                    prompt_text = prompts_text[bi]
                    answer = answers[bi]
                    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").cuda()

                    # Generate num_gens completions
                    gen_outputs = []
                    for _ in range(num_gens):
                        with torch.no_grad():
                            out = model.generate(
                                prompt_ids,
                                max_new_tokens=config.max_completion_length,
                                do_sample=True,
                                temperature=config.temperature,
                                top_p=config.top_p,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                        gen_outputs.append(out[0, prompt_ids.shape[1]:])

                    # Compute rewards
                    for comp_ids in gen_outputs:
                        comp_text = tokenizer.decode(comp_ids, skip_special_tokens=True)
                        reward = compute_reward(comp_text, answer)
                        all_prompt_ids.append(prompt_ids[0].tolist())
                        all_completion_ids.append(comp_ids.tolist())
                        all_rewards.append(reward)
                        all_answers_flat.append(answer)

                batch_rewards.extend(all_rewards)

                if not all_rewards:
                    continue

                # GRPO advantage computation
                rewards_t = torch.tensor(all_rewards, dtype=torch.float32)
                # Group by prompt: each group has num_gens entries
                num_prompts = len(batch_indices)
                rewards_grouped = rewards_t.view(num_prompts, num_gens)
                mean_r = rewards_grouped.mean(dim=1, keepdim=True)
                std_r = rewards_grouped.std(dim=1, keepdim=True)
                advantages = (rewards_grouped - mean_r) / (std_r + 1e-4)
                advantages_flat = advantages.view(-1)

                # Compute policy gradient loss
                loss_accum = torch.tensor(0.0, device="cuda")

                for i, (p_ids, c_ids, adv) in enumerate(zip(all_prompt_ids, all_completion_ids, advantages_flat)):
                    if not c_ids:
                        continue

                    input_ids = torch.tensor([p_ids + c_ids], device="cuda")
                    prompt_len = len(p_ids)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(input_ids).logits

                    # Logprobs for completion tokens only
                    comp_logits = logits[0, prompt_len - 1:-1, :]
                    comp_tokens = torch.tensor(c_ids, device="cuda")
                    log_probs = F.log_softmax(comp_logits, dim=-1)
                    token_logps = log_probs.gather(1, comp_tokens.unsqueeze(1)).squeeze(1)

                    # Policy gradient: -advantage * log_prob
                    loss_accum = loss_accum - (adv * token_logps.mean())

                if loss_accum.requires_grad:
                    (loss_accum / grad_accum).backward()

            if accum_count > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()

            step_time = time.time() - step_t0
            step_times.append(step_time)

            avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
            reward_history.append(avg_reward)

            elapsed = time.time() - start_time
            remaining = sum(step_times) / len(step_times) * (total_steps - global_step) if step_times else 0

            log(f"  {global_step}/{total_steps}  "
                f"reward={avg_reward:.3f}  "
                f"loss_accum={loss_accum.item():.4f}  "
                f"step={step_time:.1f}s  "
                f"elapsed={elapsed:.0f}s  "
                f"eta={remaining:.0f}s")

    # Training complete
    elapsed = time.time() - start_time
    avg_reward_final = sum(reward_history[-5:]) / min(5, len(reward_history)) if reward_history else 0

    log("")
    log("=" * 60)
    log("Training Report")
    log("=" * 60)
    log(f"  Total steps:    {global_step}")
    log(f"  Final reward:   {reward_history[-1]:.3f}" if reward_history else "  No reward data")
    log(f"  Avg last 5:     {avg_reward_final:.3f}")
    log(f"  Reward range:   {min(reward_history):.3f} - {max(reward_history):.3f}" if reward_history else "")
    log(f"  Total time:     {elapsed:.1f}s")
    log(f"  Avg step time:  {sum(step_times)/len(step_times):.1f}s" if step_times else "")
    log("=" * 60)

    # Save model
    if config.save_model:
        save_path = os.path.join(output_dir, "final_model")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        log(f"  Model saved:    {save_path}")

    # Save summary
    summary = {
        "total_steps": global_step,
        "final_reward": reward_history[-1] if reward_history else 0,
        "avg_reward_last5": avg_reward_final,
        "reward_history": reward_history,
        "total_time": elapsed,
        "completed": True,
    }
    with open(os.path.join(output_dir, "analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log.close = lambda: None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = " ".join(sys.argv[1:])
        if arg.endswith(".json") and os.path.isfile(arg):
            cfg = TrainingConfig.from_json(arg)
        else:
            from rllm_train.config import parse_natural_language
            cfg = parse_natural_language(arg)
    else:
        cfg = TrainingConfig()
    main(cfg)
