---
id: traj-20260602-092046-evidence-defaults
target_section: initial-config
action: replace
description: [P0] 用 R9 实证证据替换初始配置生成的通用规则 — 修复 "patch编译但不传播到配置生成" 的问题
status: accepted
source: trajectory-analysis
source_sessions: ["743765dc-6160-4b38-9ebc-fb46ff27a8ef"]
priority: P0
---

### 初始配置生成 (证据驱动)

**R9 实证基线** (0.5B + GSM8K 256, session=743765dc-6160-4b38-9ebc-fb46ff27a8ef):
- 配置: lr=2e-6, entropy_coeff=0.003, batch_size=32(train)/2(per_gpu), epochs=10,
  num_generations=8, use_kl_loss=True, kl_loss_coef=0.01, temp=0.7, max_resp=1024
- 结果: reward=0.8003, 全局avg=0.707, last_5=0.806

**生成规则** (按模型大小):

| 模型 | num_problems | lr | entropy_coeff | use_kl_loss | batch_size | epochs | num_gen | temp |
|------|-------------|-----|---------------|-------------|-----------|--------|---------|------|
| 0.5B | 256 | 2e-6 | 0.003-0.005 | True | 32(train) | 10-20 | 8-12 | 0.7-0.85 |
| 7B | 256 | 1e-6~3e-6 | 0.001-0.003 | True | 16-32 | 5-10 | 4-8 | 0.5-0.7 |

**关键规则**:
1. use_kl_loss 默认 True (R9 证明对 0.5B 有效, 不要再关闭)
2. max_response_length 默认 1024 (R9 证明足够, 1536 无收益)
3. 当 target_reward >= 0.8: epochs 至少 15, entropy_coeff 至少 0.004
4. 当 target_reward >= 0.9: epochs 至少 20, num_generations 至少 10

**反回归检查**: 如果新配置的 use_kl_loss=False 或 max_response_length>1024,
必须在配置注释中标注理由（默认不允许回退已验证有效的参数）。
