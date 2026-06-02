# 系统总览

> Agent RL 训练 + 自动 Skill 优化系统。通过轨迹分析自动发现训练问题，生成 skill-bank patch，形成持续优化闭环。

> 命名说明：文档中 `rllm_train` 指代训练后端（代码目录 `rllm_train/`），`traj_opt` 指代优化后端（代码目录 `traj_opt/`）。

## 1. 四层架构

| 层 | 模块 | 代码目录 | 职责 |
|----|------|---------|------|
| 训练后端 | rllm_train | `rllm_train/` | Agent RL 训练 pipeline：模型加载、rollout、GRPO 训练、reward 计算 |
| 优化后端 | traj_opt | `traj_opt/` | 轨迹捕获、存储、分割、分析基础设施、patch 生成 |
| Skill 管理 | skill-bank | `skill-bank/` | base + patch + compile 架构，管理 skill 的版本和优化 |
| Skills | rllm-xx / traj-xx | `skill-bank/rllm/` / `skill-bank/traj/` | 两组 Claude Code skill，是系统的使用入口 |

各层关系：
- **rllm_train** 是独立可运行的训练代码，不依赖其他层
- **traj_opt** 是 traj-xx skills 的 Python 后端，提供捕获/存储/分割/分析/优化的基础设施
- **skill-bank** 是通用的 skill 管理系统，不绑定具体 skill
- **rllm-xx skills** 编排 rllm_train 执行训练；**traj-xx skills** 编排 traj_opt 分析轨迹并优化 rllm-xx skills

## 2. 数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                        Skills (使用入口)                          │
│                                                                  │
│  skill-bank/rllm/                    skill-bank/traj/            │
│  ┌──────────────────────┐            ┌──────────────────────┐    │
│  │ rllm-train (编排)     │            │ traj-train-optimize   │    │
│  │  ├─ rllm-clarify     │            │  ├─ traj-segment      │    │
│  │  ├─ rllm-config      │            │  ├─ traj-analyze-rllm │    │
│  │  ├─ rllm-run         │            │  └─ traj-optimize     │    │
│  │  ├─ rllm-monitor     │            │                       │    │
│  │  └─ rllm-analyze     │            │ traj-launch-training  │    │
│  └──────────┬───────────┘            └──────────┬────────────┘    │
│             │ 调用                               │ 调用            │
└─────────────┼───────────────────────────────────┼────────────────┘
              ▼                                   ▼
┌──────────────────────┐            ┌──────────────────────────────┐
│  rllm_train          │            │  traj_opt                     │
│  (rllm_train/)         │            │  (traj_opt/)                │
│                      │            │                               │
│  训练 pipeline:       │  hooks     │  捕获 → 存储 → 分割 →         │
│  config → rollout    │ ────────→  │  分析基础设施 → patch 生成     │
│  → GRPO → reward     │  写入轨迹   │                               │
└──────────────────────┘            └──────────────────────────────┘
                                                  │
                                                  │ 生成 patch
                                                  ▼
                                    ┌──────────────────────────────┐
                                    │  skill-bank                   │
                                    │  base.md + patches/ +         │
                                    │  manifest.yaml → compile →    │
                                    │  .claude/skills/*/SKILL.md    │
                                    └──────────────────────────────┘
```

**闭环**：rllm-xx skills 驱动 rllm_train 训练 → hooks 捕获轨迹到 traj_opt → traj-xx skills 分析轨迹生成 patch → skill-bank 编译更新 rllm-xx skills → 下一轮训练使用优化后的 skills。

### 三层自演进

- **Layer 1 (rllm)**: 训练 agent，轨迹存储在 `traj_opt/output/rllm/`
- **Layer 2 (traj)**: 优化 agent，分析 Layer 1 轨迹，生成 `skill-bank/rllm/` 的 patch
- **Layer 3 (meta, 可选)**: Meta 优化，分析 Layer 2 轨迹，生成 `skill-bank/traj/` 的 patch

## 3. 双 CLI 架构

训练和优化在两个独立的 Claude Code 进程中执行，通过文件系统协调。

| 模块 | CLI | 职责 |
|------|-----|------|
| 训练 Agent | CLI-1 (Terminal 1) | 执行 rllm-xx skills，产出训练结果和轨迹 |
| 优化 Agent | CLI-2 (Terminal 2) | 执行 traj-xx skills，分析轨迹，优化训练 skills |
| 轨迹存储 | 共享文件系统 | CLI-1 通过 hooks 写入，CLI-2 读取分析 |
| Skill Bank | 共享文件系统 | CLI-2 写入 patch 并编译，CLI-1 读取编译后的 SKILL.md |

```
┌─────────────────────────────┐     ┌─────────────────────────────┐
│  CLI-1: 训练 Agent           │     │  CLI-2: 优化 Agent           │
│  (Terminal 1)                │     │  (Terminal 2)                │
│                              │     │                              │
│  /rllm-train "round=1|..."  │     │  /traj-train-optimize round=1│
│    ├─ rllm-clarify           │     │    ├─ traj-segment           │
│    ├─ rllm-config            │     │    ├─ traj-analyze-rllm      │
│    ├─ rllm-run               │     │    ├─ traj-optimize          │
│    ├─ rllm-monitor           │     │    └─ compile                │
│    ├─ rllm-analyze           │     │                              │
│    └─ 写入 round status      │     │  更新 round status           │
└──────────┬───────────────────┘     └──────────┬────────────────────┘
           │  writes (hooks)                     │  reads
           ▼                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                    轨迹存储 (traj_opt/output/)                   │
│  rllm/raw/{session}/events.jsonl    CLI-1 hooks 写入              │
│  rllm/trajectories/                 CLI-2 traj-segment 写入       │
│  rllm/reports/                      CLI-2 traj-analyze 写入       │
│  rounds/round_{n}/status.json       CLI-1 & CLI-2 协调文件        │
└──────────────────────────────────────────────────────────────────┘
```

### 设计动机

单 CLI + Agent 子 agent 方案（已废弃）无法同时满足上下文隔离和进度可观测性 — Agent 工具同步阻塞，父对话无法在子 agent 运行期间做任何事情。双 CLI 用进程边界实现物理隔离，两个终端独立可调试。

## 4. 轮次协调协议

路径：`traj_opt/output/rounds/round_{n}/status.json`

```json
{
  "round": 1,
  "status": "training_complete",
  "training": {
    "run_id": "run_1777723566",
    "run_ids": ["run_1777723566"],
    "session_id": "d77ca2b0-...",
    "reward": 0.773,
    "success": true,
    "completed_at": "2026-05-02T20:16:49Z"
  },
  "optimization": null
}
```

状态流转：

```
CLI-1 创建 → "training_complete"
                    ↓
CLI-2 读取，执行优化
CLI-2 更新 → "optimization_complete"
                    ↓
CLI-1 读取，开始下一轮
```

CLI-1 每次执行训练必须新建进程，保证一个 session_id 对应一次训练任务。

## 5. 隔离设计

观察者（traj-xx）与被观察者（rllm-xx）之间保持严格隔离，确保分析器从轨迹数据推断而非预设知识。

### 五个隔离维度

| 维度 | 机制 | 说明 |
|------|------|------|
| 上下文隔离 | 双 CLI 独立进程 | CLI-2 物理上无法看到 CLI-1 的执行上下文 |
| 数据流隔离 | `traj_opt/output/` 单一通道 | traj-xx 只从 traj_opt/output/ 读取，rllm-xx 通过 hooks 写入 |
| 文件目录隔离 | skill 指令中的 data-boundary 规则 | `rllm_train/` 属于 rllm-xx，`traj_opt/` 属于 traj-xx |
| 领域知识隔离 | 模式识别替代硬编码表 | traj-analyze-rllm 的领域知识来自轨迹模式推断 |
| Layer 数据隔离 | 按 layer 隔离存储 | rllm 轨迹在 `rllm/`，traj 轨迹在 `traj/`，防止交叉读取 |

### 目录归属

| 目录 | 归属 | traj-xx 可访问 | rllm-xx 可访问 |
|------|------|---------------|---------------|
| `rllm_train/` | rllm-xx | 禁止 | 完全访问 |
| `skill-bank/rllm/` | rllm-xx | 禁止 | 完全访问 |
| `traj_opt/output/rllm/` | traj-xx | 完全访问 | 不感知 |
| `skill-bank/traj/` | traj-xx | 完全访问 | 不感知 |

## 6. 优化目标边界

skill-bank 的 group 是优化目标的天然边界：

- `skill-bank/rllm/` — 训练 Agent (CLI-1) 使用的 skill，**是优化目标**
- `skill-bank/traj/` — 优化 Agent (CLI-2) 使用的 skill，**不是优化目标**

三层强制执行：
1. **traj-analyze-rllm** section:intro 明确禁止优化 traj/ group
2. **traj-optimize** Step 1 group 校验，跳过非 rllm/ group 的建议
3. **PatchGenerator** `ALLOWED_TARGET_GROUPS = {"rllm"}`，代码级硬性阻止

## 7. 设计准则

### 7.1 数据完整性

被观察的 skill（rllm-xx）必须用 Read/Bash 工具将关键数据（reward 趋势、config、perf_stats、错误日志）带入对话。hooks 只捕获工具调用的 input/response，如果 skill 不读取数据，轨迹中就没有数据。

如果轨迹中缺少关键信息，应优先修改 rllm-xx skill 使其将信息带入对话，而非在 traj_opt 中额外采集。

### 7.2 Hook 轻量性

Hook 脚本必须在 1 秒内完成。只做格式转换和文件追加，不做分割、不做分析。失败时静默。

### 7.3 分析 Skill 可插拔

分析层使用 LLM 而非规则引擎。不同场景构建不同的分析 skill（traj-analyze-rllm、traj-analyze-devops...），每个分析 skill 通过 skill-bank 管理和优化。

### 7.4 自动生成、人工确认

traj-optimize 自动生成 skill-bank patch，但需要人工确认后才编译生效。

## 8. Skill Package Layer（2026-05-25）

`skill-bank` 仍是源码和构建系统，package layer 是发行与归档索引层，用于解决新窗口索引、跨任务复用、垂类沉淀和历史复现问题。

```text
skill-bank/
├── registry.json
├── packages/
│   ├── stable/
│   ├── experimental/
│   ├── vertical/
│   ├── task-packages/
│   └── lineage-archive/
├── rllm/
├── traj/
└── compiled/
```

### package 类型

| 类型 | 作用 | 默认是否作为新任务入口 |
|---|---|---|
| `stable` | 通用 rllm/traj skill 基座包 | 是 |
| `experimental` | traj 针对新领域迭代修改的工作包 | 仅当前领域实验使用 |
| `vertical` | 通过验证后的垂类 skill 包 | 同领域新任务优先使用 |
| `task-package` | 某次具体 agent 训练的可复现交付包 | 否，除非用户要求复现 |
| `lineage-archive` | experimental 多轮演化过程归档 | 否，仅审计/复盘/恢复使用 |

### 生命周期

```text
stable
  ↓ copy
experimental/<domain>
  ↓ traj 多轮修改 + rllm 多轮训练验证
vertical/<domain>
  ↓ 被后续同领域任务复用

task-package      = 具体训练成功后的交付/复现包
lineage-archive   = traj 演化过程归档
```

### 新窗口选择协议

1. 读取 `skill-bank/registry.json`。
2. 从用户请求识别 `domain` / `agent_type` / `dataset_type` / `reward_type`。
3. 如果是新训练任务，优先使用 matching `vertical.current`；没有则用 `stable.current`。
4. 如果是新领域适配，从 `stable.current` 或已有 vertical 复制到 `experimental/<domain>`。
5. 如果是复现历史训练，必须按 task id 读取 `task-packages/`，不要用 vertical 反推。
6. 训练完成后可用 `python skill-bank/compile.py --package task-package --name <task-id> --run-id <run-id>` 归档真实 run 的 config、日志、analysis、perf stats、trajectories 和 skill package snapshot。
7. traj 多轮优化完成后可用 `python skill-bank/compile.py --package lineage-archive --domain <domain> --name <lineage-id> --round-range 1-3` 归档 round status、分析报告、raw events、segmented trajectories 和 skill package snapshot。
8. 如果是追溯 skill 演化，读取 `lineage-archive/`。

### 路径兼容约束

package layer 不改变现有运行路径。以下路径仍是兼容契约：

```text
skill-bank/compile.py
skill-bank/bank.yaml
skill-bank/rllm/
skill-bank/traj/
skill-bank/compiled/
.claude/skills/
traj_opt/output/
rllm_train/output/
```

路径写入通过 `skill_bank_paths.py` 做 containment 校验，防止 package/lifecycle 命令把文件写出预期目录。

## 9. 实测修复补充（2026-05-03）

基于 Round 1 和 Round 2 端到端实测：

- **session_id 快照差分法**: `os.environ.get('CLAUDE_SESSION_ID')` 在 hooks 中返回 unknown。改用快照差分：Phase 0 记录 `traj_opt/output/rllm/raw/` 目录快照，Phase 6.5 取差集得到本次 session_id
- **session 过滤**: `get_rllm_trajectories()` 支持 session_id 参数，避免历史数据污染当前轮次分析
- **PatchGenerator 增强**: 自动激活 patch（`_activate_patch`）、section 校验（`_validate_target_section`）、group 校验（`_validate_target_group`）
- **traj-launch-training**: 在 CLI-2 中一键启动新 CLI-1。交互式用 osascript 打开新 Terminal.app 窗口，非交互式用 `claude -p --permission-mode auto`
- **round_state 多 run_id**: `write_training_complete()` 支持 `run_ids: List[str]`，记录训练循环中所有 run_id
