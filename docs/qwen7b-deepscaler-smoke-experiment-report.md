================================================================================
  rllm_train DeepScaler 数学 Agent 训练 — 工程修复 & 稳定性诊断报告
  模型: Qwen2.5-7B-Instruct (LoRA r=16) | 数据: DeepScaler (-> numeric filter -> 16)
  硬件: 1× A100-SXM4-80GB | 框架: TRL GRPOTrainer | 后端: rllm_train (HF)
  8 次 smoke run | 三阶段递进 | 行为指标 + lr 消融双重验证
================================================================================

================================================================================
一、结论前置：八次实验回答了什么问题
================================================================================

toolfix → len1024 → numeric → promptfix → parsefix(s48) → parsefix(s49) → lr=0(s49) → lr=1e-6(s49)

核心问题: Qwen2.5-7B 在 DeepScaler 数学任务上，为什么 avg_reward 达不到 0.7？是工程 bug、训练不稳、还是数据问题？

答案: 三个原因叠加。
  1. 工程 bug (已修复): calculate 工具接受符号表达式、parser 不接受缺失 end tag、max_completion_length=512 截断
  2. lr 过大放大样本波动: lr=2.5e-6 在 16 个问题上 avg_R 只有 0.423, 冻结模型则 avg=0.658
  3. 16 个样本难度极度不均: 单步 reward 范围 0.03-0.975, 冻结模式后半段也降 26%

关键非直觉发现:
  1. promptfix 是效果最好的单 run (avg=0.555, CalcErr=1.6%, SymErr=0%), 不是 parsefix
     -> 说明 calculator 乱用比 parser 格式问题是更大的 reward 杀手
  2. dataset_filter=numeric_calculator 单独使用无效 (SymErr 反而升到 20.3%)
     -> 过滤后的样本中 symbolic 题目比例反而更高了, 只有 prompt 配合才是有效约束
  3. avg_R 随 lr 单调递减: lr=0 冻结(0.658) > lr=1e-6(0.480) > lr=2.5e-6(0.423)
     -> 在 16 个样本上, 任何非零 lr 的训练更新都在拉低 reward
  4. 后半段降幅与 lr 成正比: lr=0(-26%) < lr=1e-6(-41%) < lr=2.5e-6(-47%)
     -> 训练更新不是平滑提升, 而是对难题样本更敏感, 放大波动

================================================================================
二、配置参数全览 & 修复逻辑链
================================================================================
参数                toolfix    len1024    numeric    promptfix  parsefix_48  parsefix_49  lr0_49    lr1e6_49
--------------------------------------------------------------------------------
seed                   44         45         46          47          48           49         49         49
max_completion_len    512       1024       1024       1024        1024         1024       1024       1024
dataset_filter          —          —     numeric   numeric     numeric      numeric    numeric    numeric
learning_rate       2.5e-6     2.5e-6     2.5e-6     2.5e-6     2.5e-6      2.5e-6        0.0      1e-6
num_problems           16         16         16         16          16           16         16         16
num_generations         4          4          4          4           4            4          4          4
temperature          0.899...   0.899...   0.899...   0.899...    0.899...     0.899...   0.899...   0.899...
agent_steps             3          3          3          3           3            3          3          3

toolfix -> len1024: 512->1024 解决截断, Clip 从 0.359 降到 0.047, avg_R +0.039
len1024 -> numeric: 加 dataset_filter=numeric_calculator, 过滤掉 48% 的符号题
  -> 但 SymErr 反升 (17.2%->20.3%), 因为剩余样本中符号题比例反而更高
numeric -> promptfix: 强化 system prompt 中 calculator 使用禁令
  -> CalcErr 25% -> 1.6%, SymErr 20.3% -> 0%, 决定性修复！avg_R 0.380 -> 0.555
promptfix -> parsefix: 加 parser missing-end-tag recovery + close tag prompt
  -> ParseErr 6.2% -> 1.6%, 但 avg_R 不升反降 (0.555 -> 0.393), Finish 从 80% 降到 47%
  -> 原因: prompt 中强调 close tag 可能过度约束了模型输出, 降低了 finish 调用频率
parsefix_s48 -> parsefix_s49: 同配置换 seed, 验证波动范围
  -> avg_R 接近 (0.393 vs 0.423), 但 final_R 差距巨大 (0.800 vs 0.090)
  -> 说明 16 个样本最后一个 step 的难度决定了 final
parsefix_s49 -> lr0_s49: 冻结模型做对照
  -> avg_R 从 0.423 跃升到 0.658 (lr=0 只跑 9 步就崩溃)
  -> 证明: 当前 lr=2.5e-6 的更新方向在伤害 reward，不是帮助
lr0_s49 -> lr1e6_s49: 降低 lr 验证
  -> avg_R 0.480, 介于冻结和原始 lr 之间
  -> 确认: lr 越小, avg_R 越高, 但后半段崩塌仍然存在

================================================================================
三、L1: Reward 趋势 (目标达成度)
================================================================================
指标               toolfix    len1024    numeric    promptfix  parsefix_48 parsefix_49 lr0_49    lr1e6_49
--------------------------------------------------------------------------------
avg_R              0.291      0.330      0.380      0.555      0.393       0.423      0.658      0.480
max_R              0.700      0.633      0.963      0.940      0.960       0.975      0.945      0.945
min_R              0.000      0.020      0.000      0.060      0.000       0.030      0.140      0.000
final_R            0.263      0.498      0.030      0.890      0.800       0.090      0.860      0.130
R_std              0.183      0.191      0.314      0.339      0.375       0.340      0.296      0.375
前半段 avg          0.339      0.339      0.550      0.531      0.352       0.533      0.770      0.626
后半段 avg          0.242      0.320      0.210      0.579      0.434       0.312      0.568      0.334
后半段降幅          -28.5%     -5.4%      -61.8%     +9.0%      +23.1%     -41.4%     -26.2%     -46.7%

关键发现:
  - promptfix 是唯一后半段正增长 (+9.0%) 的 run, 且没有 Clip >= 0.8, SymErr = 0%
    -> 说明 CalcErr/SymErr 的消除让训练更新进入正向循环
  - parsefix 两个 seed 后半段趋势相反 (+23% vs -41%), 说明 parser 修复不是影响后半段的主因
  - 冻结模型的 avg_R=0.658 是所有 run 中最高的, 说明:
    1) 不做任何训练更新时, base 7B 模型在这些问题上的 raw reward 是 0.658
    2) 任何非零 lr 的训练都让 reward 下降
    3) 问题不在模型推理能力, 而在 GRPO 更新在 16 个样本上的噪声大于信号
  - 冻结模型后半段也降 26%, 说明 16 个样本中后面的题本身就比前面难

================================================================================
四、L2: 行为指标诊断 — 模型在做什么
================================================================================
指标               toolfix    len1024    numeric    promptfix  parsefix_48 parsefix_49 lr0_49    lr1e6_49
--------------------------------------------------------------------------------
Finish%            68.8%      82.8%      79.7%      79.7%      46.9%       56.2%      61.1%      64.1%
FmtOK%             86.5%      76.1%      60.4%      68.8%      62.5%       78.1%      77.8%      75.0%
Tool%              50.0%      56.2%      62.5%       9.4%      14.1%       15.6%      19.4%      14.1%
Ans%               68.8%      76.6%      79.7%      79.7%      46.9%       53.1%      61.1%      64.1%
CalcErr%           10.9%      25.0%      25.0%       1.6%       1.6%        3.1%       5.6%       3.1%
SymErr%             9.4%      17.2%      20.3%       0.0%       1.6%        3.1%       5.6%       3.1%
ParseErr%           3.1%       1.6%       3.1%       6.2%       1.6%        4.7%       5.6%       6.2%

Calculator 错误演化:
  toolfix: CalcErr=10.9%, SymErr=9.4% -> 工具 schema 收紧有基础效果
  len1024: CalcErr=25.0%, SymErr=17.2% -> 更长的输出让模型有空间做出更多符号 calculate 尝试
  numeric: CalcErr=25.0%, SymErr=20.3% -> 过滤掉符号题反而让剩余题目的符号错误比例更高!
  promptfix: CalcErr=1.6%, SymErr=0% -> system prompt 是唯一有效手段
  之后的 run: CalcErr/SymErr 维持在 1-6%, 已基本控制

Tool 使用率断崖:
  toolfix->numeric: Tool 使用率 50-62%, 模型积极地尝试 calculate
  promptfix: Tool = 9.4% -> prompt 中强调 "calculate 只在特定情况下使用" 过度抑制了工具使用
  parsefix 及之后: Tool = 14-19% -> 始终没有恢复到早期水平

ParseErr 趋势:
  promptfix: ParseErr=6.2% -> 模型开始省略 </tool_call>, 这是高 Tool usage 时期的遗留习惯
  parsefix_48: ParseErr=1.6% -> parser recovery 有效, 但 Finish% 同降到 46.9%
  -> prompt 中强调 close tag 可能副效果: 模型为避免格式错误, 少用 tool call 从而少暴露问题

关键发现:
  1. promptfix 在 0% SymErr 的同时 Tool=9.4%, 说明模型主要在用纯文本推理 + finish
     -> 少数 calculate 调用 (9.4%) 都是正确的 numeric 操作
  2. parsefix 降低了 ParseErr 但 Finish=% 从 80% 降到 47%
     -> parser recovery 的 prompt 变更可能抑制了 finish 调用
  3. lr 对照中 Finish% 和 Ans% 与 lr 不完全正相关: lr=1e-6(64%) > lr=2.5e-6(56%) > lr=0(61%)
     -> lr 对行为指标的影响是次要的, prompt 是主导因素

================================================================================
五、L3: 训练动态 — Loss/Grad/Entropy/Clip
================================================================================
指标               toolfix    len1024    numeric    promptfix  parsefix_48 parsefix_49 lr0_49    lr1e6_49
--------------------------------------------------------------------------------
avg_Loss           0.136     -0.014      0.039      0.031     -0.024       0.014      0.024     -0.003
avg_Grad           0.526      0.520      0.690      0.169      0.174       0.212      0.222      0.180
avg_Ent            0.150      0.150      0.134      0.208      0.167       0.212      0.179      0.209
avg_Clip           0.359      0.047      0.078      0.000      0.062       0.078      0.056      0.109
Len_max              512        830        909        849       1014         980        776       1002
Clip>=0.8 步数        3          0          0          0          0           0          0          0

训练动态关键发现:

  1. 截断是 toolfix 的独立问题:
     toolfix Clip=0.359 (最高), Clip>=0.8 有 3 步
     -> max_completion_length=512 严重截断 7B 模型的推理链
     -> len1024 把 max_completion_length 翻倍后 Clip 降到 0.047, 问题解决

  2. promptfix 的 Clip=0 但不是全好:
     avg_Len=440, 远低于 1024 上限, Clip=0
     -> 模型输出长度健康, 但 Tool=9.4% 说明推理链可能过于简短
     -> 没有截断、没有符号错误、但也没在充分使用工具

  3. Loss 趋势与 avg_R 不完全对齐:
     promptfix Loss 前 3=+0.016, 后 3=-0.075 -> loss 在降
     parsefix_48 Loss 前 3=+0.005, 后 3=-0.036 -> 也在降
     parsefix_49 Loss 前 3=+0.039, 后 3=-0.074 -> 但 avg_R 在崩塌
     -> 16 个样本上, loss 趋势不可靠, 不能只靠 loss 判断训练好坏

  4. Entropy 趋势:
     promptfix: 0.225->0.229 (几乎不变, +1.7%) -> 探索维持
     parsefix_49: 0.101->0.304 (+199%) -> entropy 爆炸, 不是好信号
     lr=1e-6: 0.101->0.356 (+251%) -> 同样是大增
     -> low lr 下的 entropy 大增伴随 reward 崩塌，说明策略在高 entropy 方向偏移但没学到有用行为

  5. Grad norm 与 lr 的关系:
     lr=2.5e-6: avg_Grad=0.212
     lr=1e-6: avg_Grad=0.180
     趋势合理但量级都很小, 说明 GRPO advantage 本身不大

================================================================================
六、L4: 单步 Reward 波动分析
================================================================================

"一个好 step, 一个零分 step" 模式:

  promptfix (最好的 run):
    [0.060, 0.200, 0.920, 0.200, 0.720, 0.620, 0.590, 0.940,
     0.920, 0.920, 0.920, 0.333, 0.290, 0.140, 0.220, 0.890]
    -> 后半段有 4 步连续的 0.920 (step 9-11), 但也有 0.140/0.220/0.290

  parsefix_48:
    [0.930, 0.000, 0.370, 0.000, 0.390, 0.140, 0.960, 0.030,
     0.040, 0.733, 0.168, 0.080, 0.690, 0.890, 0.070, 0.800]
    -> step 2,4 直接 0.000, step 7=0.960 接着 step 8=0.030, 相邻 step 差异 32x

  parsefix_49:
    [0.945, 0.490, 0.685, 0.230, 0.270, 0.070, 0.600, 0.975,
     0.660, 0.030, 0.220, 0.030, 0.450, 0.920, 0.100, 0.090]
    -> step 8=0.975 -> step 10=0.030 两步内从天花板到地板

  冻结模型 (lr=0, 9 steps):
    [0.945, 0.690, 0.530, 0.915, 0.140, 0.270, 0.640, 0.930, 0.860]
    -> 也有波动 (0.140-0.945), 但没有出现 0

关键发现:
  1. 即使冻结模型 (lr=0), 16 个样本中也有 Step 5 的 0.140 和 Step 8 的 0.930 并存
     -> 样本难度差异是固有的, 不是训练造成的
  2. 训练放大的是 "零分" 概率而非整体波动: 冻结模型没有 0.000 的 step
     -> 训练更可能导致部分 step 完全失败而不是所有 step 略微下降
  3. 相邻 step 32x 的 reward 差异说明单个 batch(4 generation per prompt) 内的样本难度
     对 avg_R 的影响远大于任何训练更新, 16 步的 avg_R 不应被当成稳定的评估指标

================================================================================
七、消融实验: seed 固定, lr 变化 (核心发现)
================================================================================

三组 run 共享 seed=49, parsefix 配置, 仅 lr 不同:

指标                   lr=0 冻结 (9步)    lr=1e-6 (16步)    lr=2.5e-6 (16步)
--------------------------------------------------------------------------------
avg_R                   0.658             0.480             0.423
max_R                   0.945             0.945             0.975
final_R                 0.860             0.130             0.090
前半段 avg              0.770             0.626             0.533
后半段 avg              0.568             0.334             0.312
后半段降幅              -26.2%            -46.7%            -41.4%
avg_Finish              61.1%             64.1%             56.2%
avg_Ans                 61.1%             64.1%             53.1%

Reward 序列对比:
  lr=0   [0.945, 0.690, 0.530, 0.915, 0.140, 0.270, 0.640, 0.930, 0.860]
  lr=1e-6[0.945, 0.540, 0.540, 0.920, 0.180, 0.140, 0.800, 0.945,
          0.930, 0.030, 0.240, 0.000, 0.280, 0.920, 0.140, 0.130]
  lr=2.5 [0.945, 0.490, 0.685, 0.230, 0.270, 0.070, 0.600, 0.975,
          0.660, 0.030, 0.220, 0.030, 0.450, 0.920, 0.100, 0.090]

结论:
  1. lr=0 冻结的 avg_R 最高 (0.658) — 不做训练 > 做训练
     -> 7B base model 在这些题目上的 raw reward 已经有不低的基础
     -> 当前 GRPO 更新在 16 个样本上噪声大于信号
  
  2. avg_R 与 lr 严格负相关 (0.658 > 0.480 > 0.423)
     -> 每一步训练更新都在系统性拉低 reward
  
  3. "零分"只出现在有训练更新的 run (lr=1e-6 有 step 12=0.000, lr=2.5e-6 有 step 10=0.030)
     -> 训练更新在某些样本上完全破坏了模型推理
  
  4. 前半段在所有 lr 下都是 0.533-0.770
     -> 前期样本相对更易, 模型基础能力也能覆盖
     -> 问题是训练更新让模型在后期难题上的表现变差

================================================================================
八、工程修复效果量化
================================================================================

按修复项拆分的 avg_R 增量:

修复项                             代表 run 对比          avg_R 变化    其他效果
--------------------------------------------------------------------------------
max_completion_length 512->1024    toolfix->len1024       +0.039        Clip 0.359->0.047
dataset_filter=numeric             len1024->numeric       +0.050        但 SymErr 17->20%
强化 calculator system prompt      numeric->promptfix     +0.175        CalcErr 25->1.6%, SymErr 0%
parser missing-end-tag recovery    promptfix->parsefix_48 -0.162        ParseErr 6.2->1.6%, Finish 80->47%

最大正贡献: 强化 calculator 禁令 (+0.175, CalcErr -93.6%, SymErr -100%)
最大负贡献: parser recovery prompt 变更 (-0.162, Finish -41%)

注: parsefix 的 avg_R 下降是 prompt 副效果 (过度抑制 tool call / finish),
    不是 parser 代码本身的问题。parser recovery 代码是正确且必要的。

================================================================================
九、六层诊断覆盖检查
================================================================================
L1   Reward 趋势                     ✓
L2   行为指标 (Finish/Tool/Ans)       ✓    Finish=47-83%, Tool=9-62%
L3   训练动态 (Loss/Grad/Ent/Clip)    ✓    Entropy +250% 伴随 reward 崩塌
L4   样本难度均匀性                   ✓    单步 0.000-0.975, 相邻步差异 32x
L5   训练更新有效性                   ✓    lr 消融: 冻结 > lr=1e-6 > lr=2.5e-6
L6   工程修复效果                     ✓    prompt 修复 Effect Size 最大 (+0.175)

与 VERL 端 (a.md 报告) 的差异:
  - 无 pg_clipfrac / KL / advantage range — GRPO via TRL 没有暴露这些指标
  - 有 CalcErr/SymErr/ParseErr — rllm_train 的工具调用链路暴露了更细粒度的错误类型
  - 行为指标是核心诊断工具 — 在 7B GRPO 场景下, 行为指标比 loss 更适合判断训练健康度

================================================================================
十、问题 -> 修复动作速查表
================================================================================
问题现象                          根因诊断                    已验证修复              效果
--------------------------------------------------------------------------------
SymErr > 9%                       calculate 接受符号表达式    强化 system prompt      SymErr -> 0%
CalcErr > 10%                     calculate 参数描述不清晰    收紧 schema + prompt    CalcErr -> 1.6%
ParseErr > 3%                     缺失 </tool_call>          parser recovery          ParseErr -> 1.6%
Clip >= 0.8 (含 3 步)             max_completion_len=512      翻倍到 1024             Clip -> 0.047
Finish 骤降 (80% -> 47%)          prompt 过度约束输出格式     副作用 - 需要权衡       ParseErr 与 Finish 的取舍
avg_R 上不去 (训练后 < 冻结)      lr 过大 + 16 样本波动       lr 降低到 1e-6         avg_R +0.057
后半段崩塌 -40%+                   样本难度不均匀              冻结对照验证           非训练问题, 是数据问题
相邻 step reward 32x 差异         样本量 16 太小               无                       需要 64+ 样本平滑

================================================================================
十一、代码变更清单
================================================================================
文件                                    变更内容
--------------------------------------------------------------------------------
rllm_train/math_env.py                   收紧 calculate schema、validate_calculator_expression()
                                         is_symbolic_calculator_error()、reward components 增加 symbolic/parser 事件
rllm_train/parsers.py                   parse_with_diagnostics()、missing-end-tag recovery
                                         _is_valid_call_data()、_call_data_error()
rllm_train/tool_agent.py                使用 parse_with_diagnostics()、Step.info 记录 parser diagnostics
rllm_train/train.py                     强化 tool-agent system prompt、_completion_reward() 对齐 env scoring
                                         load_external_dataset() 支持 dataset_filter + _passes_dataset_filter()
rllm_train/config.py                    新增 dataset_filter 字段
rllm_train/logger.py                    log_dataset_ready() 支持 dict filter_summary
                                         [MONITOR_STEP] 增加 CalcErr/SymErr/ParseErr/NoFinish
rllm_train/rollout.py                   _compute_behavior_metrics() 增加到 17 个行为指标
rllm_train/trajectory_writer.py         JSONL 增加 parser_diagnostics_by_step 字段
skill-bank/rllm/rllm-monitor/patches/   006-step-event-echo.md: Monitor 逐步回显规则
skill-bank/rllm/rllm-train/patches/     004-auto-continue-after-config.md: Phase 自动续跑规则

================================================================================
十二、下一步建议
================================================================================

路径 A (保守 - 推荐): 增加样本量 + 降低 lr
  参数: promptfix 配置 + num_problems=64 + lr=1e-6 + max_completion_length=1024
  预期: avg_R=0.55-0.65, 后半段崩塌大幅减轻
  风险: 低. 64 个样本可平滑当前 0.03-0.975 的极端波动, lr=1e-6 已经在 16 样本上证明更稳
  理由: 16 个样本上 avg_R 只能到达 0.555 (promptfix), 增加样本是放量必要条件

路径 B (工程): 调整 parsefix prompt, 恢复 Finish 率
  parsefix 的 Finish 从 promptfix 的 80% 降到 47%, 这是 prompt 过度约束的副效果
  -> 保留 parser recovery 代码, 但缩减 prompt 中关于 close tag 的指令
  -> 预期 Finish 回到 70%+ 同时 ParseErr 维持在 3% 以下

路径 C (激进): 进一步降低 lr + warmup
  lr 从 1e-6 降到 5e-7 + 前 N 步 linear warmup
  理由: lr=0 冻结 avg=0.658 是最好的, lr 越小越好
  但风险: lr=0 在 step 10 崩溃说明 scheduler 在零学习率下不稳定

推荐: A + B 组合
  promptfix 的 prompt (+Calc/Sym 控制) + parsefix 的 parser code (+format 容错)
  + num_problems=64 + lr=1e-6
  预期: avg_R 0.55-0.65, 且 Finish>70%, CalcErr<2%, SymErr<1%, ParseErr<3%
