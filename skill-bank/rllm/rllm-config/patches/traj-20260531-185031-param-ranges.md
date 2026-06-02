---
id: traj-20260531-185031-param-ranges
target_section: param-ranges
action: append
description: 0.5B模型GSM8K参数安全范围：基于4轮GRPO训练的实证参数范围收紧（max_completion_length、lr、num_generations、temperature）
status: proposed
source: trajectory-analysis
source_sessions: ["run_1780247306", "run_1780248991", "run_1780249675", "run_1780250271"]
---

### 0.5B 模型 GSM8K 参数安全范围（4轮实证）

基于 Qwen2.5-0.5B-Instruct 在 GSM8K 上 4 轮 GRPO 训练的实证参数范围:

| 参数 | 通用范围 | **0.5B GSM8K 推荐** | 实证依据 |
|---|---|---|---|
| learning_rate | 1e-7 ~ 1e-3 | **5e-7 ~ 2e-6** | R1 lr=5e-6 -> Step 17后崩溃; R2-4 lr=1e-6 -> 稳定训练 |
| max_completion_length | 128 ~ 2048 | **1024 ~ 2048** | R1 len=256 -> 88.3%截断; R2 len=1024 -> 截断降至45%; R3-4 len=1536 -> 截断<14% |
| num_generations | 2 ~ 8 | **6 ~ 8** | R1-2 gen=4 -> 有时advantage=0; R3-4 gen=8 -> advantage稳定(+/-0.935) |
| temperature | 0.3 ~ 1.5 | **0.8 ~ 1.1** | R1 temp=0.7 -> entropy快速下降; R3 temp=1.0 -> 探索充分; R4 temp=0.85 -> epoch2退化,需配合低lr |
| num_epochs | 1 ~ 20 | **1 ~ 3** | R4 epochs=2 -> epoch1上升, epoch2退化; 多epoch需降低lr或early stopping |
| num_problems | 8 ~ 512 | **32 ~ 128** | R1 64题 ok; R2 32题 49.2%; R3-4 48题 37-52%; 0.5B不需要海量数据 |

### 截断-崩溃预防规则

```python
# 0.5B GSM8K 配置安全检查
if task == 'gsm8k':
    assert config.max_completion_length >= 1024, "GSM8K需要长推理链, len<1024将导致截断崩溃(R1证据:88%)"
    assert config.num_generations >= 4, "gen<4可能导致GRPO advantage=0(R1-2证据)"
    if config.num_epochs > 1:
        assert config.learning_rate <= 1e-6, "多epoch需极低lr防止退化(R4证据: lr=1e-6+epochs=2仍有退化)"
```

### 0.5B 模型容量上限

15层诊断 L15 结论: 0.5B 在 GSM8K 单轮(无tool)场景下验证精度上限约 **49-52%**。
若高于此精度需求，必须: 换大模型(1.5B/7B) 或 启用多轮 tool-calling。

