"""Minimal GRPO training loop for Qwen2.5-7B with LoRA."""
import os, re, time, json, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from rllm_train.config import TrainingConfig
from rllm_train.math_env import generate_math_problems

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    if config_path and config_path.endswith(".json"):
        with open(config_path) as f:
            cfg_d = json.load(f)
        config = TrainingConfig(**{k: v for k, v in cfg_d.items()
                                    if k in TrainingConfig.__dataclass_fields__})
    else:
        config = TrainingConfig()

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {config.model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0"
    )
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.train()

    # Collect trainable params before any gradient changes
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params for optimizer: {sum(p.numel() for p in trainable_params)}", flush=True)

    problems = generate_math_problems(n=config.num_problems, seed=config.seed, difficulty=config.difficulty)
    print(f"Dataset: {len(problems)} problems, difficulty={config.difficulty}", flush=True)

    optimizer = AdamW(trainable_params, lr=config.learning_rate)

    system_msg = "Solve the math problem. Show your reasoning, then write the final numeric answer."
    pad_id = tokenizer.pad_token_id

    total_steps = config.num_problems * config.num_epochs // (config.batch_size * config.gradient_accumulation_steps)
    beta = 0.04  # KL penalty

    log_lines = []
    step = 0
    t_start = time.time()
    rewards_history = []

    for epoch in range(config.num_epochs):
        indices = list(range(len(problems)))
        # Simple shuffle with seed per epoch
        import random
        rng = random.Random(config.seed + epoch)
        rng.shuffle(indices)

        for i in range(0, len(indices), config.batch_size):
            batch_indices = indices[i:i + config.batch_size]
            batch_prompts = []
            batch_answers = []
            for idx in batch_indices:
                p = problems[idx]
                batch_prompts.append([
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": p["question"]},
                ])
                batch_answers.append(p["answer"])

            all_rewards = []
            all_logprobs = []
            all_ref_logprobs = []
            all_masks = []

            for micro in range(config.gradient_accumulation_steps):
                prompt = [batch_prompts[micro] if micro < len(batch_prompts) else batch_prompts[0]]
                answer = [batch_answers[micro] if micro < len(batch_answers) else batch_answers[0]]

                text = tokenizer.apply_chat_template(prompt[0], tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
                prompt_len = inputs["input_ids"].shape[1]

                completions = []
                comp_texts = []
                for _ in range(config.num_generations):
                    with torch.no_grad():
                        out = model.generate(
                            **inputs, max_new_tokens=config.max_completion_length,
                            do_sample=True, temperature=config.temperature, top_p=config.top_p,
                            pad_token_id=pad_id,
                        )
                    comp_ids = out[0, prompt_len:]
                    comp_text = tokenizer.decode(comp_ids, skip_special_tokens=True)
                    completions.append(comp_ids)
                    comp_texts.append(comp_text)

                rewards = []
                for ci, ct in enumerate(comp_texts):
                    nums = re.findall(r'-?\d+\.?\d*', ct)
                    if nums and answer[0]:
                        rewards.append(1.0 if abs(float(nums[-1]) - float(answer[0])) < 1e-6 else 0.0)
                    else:
                        rewards.append(0.0)
                all_rewards.extend(rewards)

                # Compute logprobs
                for ci, comp_ids in enumerate(completions):
                    full_ids = torch.cat([inputs["input_ids"][0], comp_ids]).unsqueeze(0).to("cuda:0")
                    comp_len = comp_ids.shape[0]

                    # Reference logprobs (base model without LoRA)
                    with torch.no_grad():
                        with model.disable_adapter():
                            ref_out = model(full_ids)
                            ref_logits = ref_out.logits[0, prompt_len - 1:-1, :]
                            ref_log_p = torch.log_softmax(ref_logits, dim=-1)
                            ref_token_lp = ref_log_p.gather(1, comp_ids.unsqueeze(1).to("cuda:0")).squeeze(1)

                    # Training logprobs (with LoRA)
                    model_out = model(full_ids)
                    logits = model_out.logits[0, prompt_len - 1:-1, :]
                    log_p = torch.log_softmax(logits, dim=-1)
                    token_lp = log_p.gather(1, comp_ids.unsqueeze(1).to("cuda:0")).squeeze(1)

                    all_logprobs.append(token_lp)
                    all_ref_logprobs.append(ref_token_lp)

                    mask = torch.ones_like(comp_ids, dtype=torch.float32)
                    all_masks.append(mask)

            # GRPO advantages
            rewards_t = torch.tensor(all_rewards, dtype=torch.float32, device="cuda:0")
            n_gen = config.num_generations
            n_prompts = config.gradient_accumulation_steps
            mean_r = rewards_t.view(n_prompts, n_gen).mean(dim=1, keepdim=True).repeat(1, n_gen).view(-1)
            std_r = rewards_t.view(n_prompts, n_gen).std(dim=1, keepdim=True).repeat(1, n_gen).view(-1)
            advantages = (rewards_t - mean_r) / (std_r + 1e-4)

            # Compute loss
            loss = torch.tensor(0.0, device="cuda:0", requires_grad=True)
            for k in range(len(all_logprobs)):
                logp = all_logprobs[k]
                ref_logp = all_ref_logprobs[k].detach()
                mask = all_masks[k].to("cuda:0")
                adv = advantages[k]

                kl = (torch.exp(ref_logp - logp) - (ref_logp - logp) - 1).detach()
                token_loss = -(torch.exp(logp - logp.detach()) * adv - beta * kl) * mask
                loss = loss + token_loss.sum() / mask.sum().clamp(min=1)

            loss = loss / len(all_logprobs)
            loss.backward()

            if (i // config.batch_size + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()

                step += 1
                avg_reward = rewards_t.mean().item()
                rewards_history.append(avg_reward)
                elapsed = time.time() - t_start
                log_line = f"Step {step}/{total_steps} | reward={avg_reward:.3f} | loss={loss.item():.4f} | time={elapsed:.0f}s"
                print(log_line, flush=True)
                log_lines.append({"step": step, "reward": avg_reward, "loss": loss.item(), "time": elapsed})

    # Save results
    final_reward = sum(rewards_history) / len(rewards_history) if rewards_history else 0
    print(f"\nTraining complete! {step} steps, avg_reward={final_reward:.3f}, time={time.time()-t_start:.0f}s", flush=True)

    # Save model
    if config.save_model:
        save_path = os.path.join(output_dir, "final_model")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}", flush=True)

    # Save training report
    report = {
        "total_steps": step,
        "avg_reward": final_reward,
        "rewards_history": rewards_history,
        "total_time": time.time() - t_start,
        "config": config.__dict__,
    }
    report_path = os.path.join(output_dir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {report_path}", flush=True)

if __name__ == "__main__":
    main()
