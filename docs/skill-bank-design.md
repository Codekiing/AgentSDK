# Skill Bank 设计规范

> 通用的 Claude Code skill 管理系统，采用 base + patch + compile 架构。支持持续优化、版本追溯、可回退、可组合的 skill 演进。

## 核心概念

Skill Bank 将每个 skill 拆分为：
- **base.md** — 带 section 锚点的基础版本
- **patches/** — 独立的优化 patch 文件
- **manifest.yaml** — 控制哪些 patch 激活

编译器读取 base + active patches，输出最终的 SKILL.md。

## 目录结构

```
skill-bank/
├── compile.py                       # 编译器
├── bank.yaml                        # 全局配置（group 注册表 + 全局设置）
│
├── <group>/                         # 按领域分组
│   ├── <skill-name>/                # 每个 skill 一个目录
│   │   ├── base.md                  # 基础版本（带 section 锚点）
│   │   ├── patches/                 # 所有 patch 文件
│   │   │   ├── 001-xxx.md
│   │   │   └── 002-yyy.md
│   │   └── manifest.yaml            # 激活状态 + profile 定义
│   └── ...
│
└── compiled/                        # 编译历史快照
    ├── v001/
    └── v002/
```

示例：
```
skill-bank/
├── rllm/                            # rllm_train 训练相关
│   ├── rllm-config/
│   ├── rllm-analyze/
│   ├── rllm-monitor/
│   ├── rllm-train/
│   ├── rllm-clarify/
│   └── rllm-run/
│
├── devops/                          # 未来：部署和 CI 相关
│   ├── deploy-helper/
│   └── ci-monitor/
```

## 全局配置：bank.yaml

```yaml
version: "1.0"

groups:
  rllm:
    description: "rllm_train agent RL 训练相关"
    skills:
      rllm-config:
        output: .claude/skills/rllm-config/SKILL.md
      rllm-analyze:
        output: .claude/skills/rllm-analyze/SKILL.md
      # ...

settings:
  compiled_dir: compiled
  output_base: ../          # output 路径的相对基准（相对于 skill-bank/）
```

output 路径相对于 `settings.output_base`，这样 skill-bank 可以放在项目的任意位置。

## base.md — 带锚点的基础 skill

在每个功能块前后加 section 锚点（HTML 注释，不影响渲染）：

```markdown
---
name: rllm-config
description: ...
metadata:
  version: "1.0.0"
---

# rllm-config — 训练配置生成与调参

<!-- section:intro -->
你是 rllm_train 训练配置专家。...
<!-- /section:intro -->

<!-- section:param-ranges -->
### 参数安全范围
...
<!-- /section:param-ranges -->
```

锚点规则：
- 开始标记：`<!-- section:<name> -->`
- 结束标记：`<!-- /section:<name> -->`
- name 只允许 `[a-z0-9-]`
- section 不能嵌套
- 锚点之间的内容（不属于任何 section）称为"间隙文本"，编译时原样保留

## Patch 文件格式

```markdown
---
id: "001-model-safety"
target_section: "param-ranges"
action: replace
description: "按模型大小区分参数安全上限"
source: "2026-04-30 训练实验, run_1777516933"
created: "2026-04-30"

depends_on: []                       # 本 skill 内: ["001-xxx"]
                                     # 跨 skill: ["rllm-monitor:001-early-stopping"]
conflicts_with: []                   # 互斥 patch

status: active                       # active | deprecated | archived
superseded_by: ""                    # deprecated 时填写替代 patch id
---

(patch 正文)
```

**action 类型**：
- `replace` — 替换 target_section 的全部内容
- `append` — 追加到 target_section 末尾
- `prepend` — 插入到 target_section 开头
- `insert_after` — 在 target_section 之后插入新 section（section name = patch id）

**status 生命周期**：
- `active` — 正常可用，可被 manifest 激活
- `deprecated` — 已被更好的 patch 取代，`superseded_by` 指向替代者。编译器遇到 deprecated patch 在 active 列表中时发出警告
- `archived` — 已合并进 base.md，patch 文件保留但不再可激活。编译器遇到 archived patch 时报错

## manifest.yaml

```yaml
base: base.md

active:
  - 001-model-safety
  - 002-param-constraints

disabled:
  - 005-experimental-lr

profiles:
  small-model:
    description: "0.5B 模型的保守配置"
    active:
      - 001-model-safety
      - 002-param-constraints
    disabled:
      - 005-experimental-lr

  large-model:
    description: "3B 模型的宽松配置"
    active:
      - 002-param-constraints
    disabled:
      - 001-model-safety
```

## compile.py 核心逻辑

```
输入: skill-bank/<group>/<skill>/manifest.yaml  (+ 可选 --profile)
输出: bank.yaml 中该 skill 对应的 output 路径

流程:
1. 读取 bank.yaml 获取 skill 注册信息和 output 路径
2. 读取 manifest.yaml，确定 active patches
3. 读取 base.md，解析 section 锚点 → OrderedDict
4. 加载 active patches:
   a. 检查 status: deprecated → 警告, archived → 报错
   b. 按 depends_on 拓扑排序
   c. 检查冲突: 任意两个 active patch 不能互相在 conflicts_with 中
   d. 检查跨 skill 依赖
5. 逐个应用 patch (按拓扑序)
6. 拼接输出（干净 markdown，不保留锚点）
7. 保存快照到 compiled/vNNN/
```

**CLI 接口**：
```bash
python skill-bank/compile.py rllm-config                # 单个 skill
python skill-bank/compile.py rllm-config -p small-model  # 指定 profile
python skill-bank/compile.py --group rllm                # 编译整个 group
python skill-bank/compile.py --all                       # 全量编译
python skill-bank/compile.py --dry-run rllm-config       # 预览
python skill-bank/compile.py --diff rllm-config          # 显示差异
python skill-bank/compile.py --status                    # patch 状态摘要
python skill-bank/compile.py --squash rllm-config        # 合并 patches 进 base
```

## Patch 压缩（squash）

当 patch 积累过多（>10 个），squash 将稳定的 patch 合并回 base：

1. 按当前 active patches 编译出完整内容
2. 用编译结果覆盖 base.md
3. 将被合并的 patch 标记为 `status: archived`
4. 清空 manifest.yaml 的 active 列表
5. 提交编译快照作为 squash 前的备份

## 跨 Skill Patch 处理

跨 skill 的联动优化拆成多个 patch，用 `depends_on` 关联：

```
rllm/rllm-monitor/patches/001-early-stopping.md
  → target_section: "anomaly-detection"

rllm/rllm-train/patches/001-phase4-abort.md
  → depends_on: ["rllm-monitor:001-early-stopping"]
```

编译 rllm-train 时，编译器会检查 rllm-monitor 的 manifest，确认 001-early-stopping 是 active 的。不满足则报错。

## 添加新 Skill / 新 Group

### 添加新 Group

1. 在 bank.yaml 的 groups 中注册
2. 创建目录：`skill-bank/<group>/`

### 添加新 Skill

1. 在 bank.yaml 对应 group 的 skills 中注册（含 output 路径）
2. 创建目录和文件：
   ```
   skill-bank/<group>/<skill>/
   ├── base.md          # 带 section 锚点
   ├── patches/         # 空目录
   └── manifest.yaml    # active: []
   ```
3. 编译：`python skill-bank/compile.py <skill>`

## Package Layer（stable / experimental / vertical）

`skill-bank` 仍然是 skill 源码仓库和构建系统；package layer 是叠加在现有 `skill-bank/<group>/<skill>` 源布局之上的发行/归档层。初始重构阶段不得移动旧路径：

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

新增 package 目录：

```text
skill-bank/packages/
├── stable/             # 通用 skill 基座包
├── experimental/       # traj 正在演化的领域实验包
├── vertical/           # 通过验证后的垂类 skill 包
├── task-packages/      # 具体训练任务的可复现交付包
└── lineage-archive/    # experimental 多轮演化过程归档
```

新增索引文件：

```text
skill-bank/registry.json
```

新窗口或新任务选择 skill 时，先读 `skill-bank/registry.json`，不要靠模糊搜索目录名。

### Package 生命周期

```text
stable:rllm-stable-v1
  ↓ copy
experimental:<domain>:<name>
  ↓ traj 多轮修改与训练验证
vertical:<domain>:<name>
```

- `stable`：类似模型训练中的基座模型，只包含跨任务通用训练编排能力。
- `experimental`：从 stable 或 vertical 复制出的工作包，供 traj 针对新领域迭代修改。
- `vertical`：经过验证后沉淀的垂类经验包，例如 finance/math/code。
- `task-package`：某次具体 agent 训练成功后的完整可复现包，不作为新任务默认入口。
- `lineage-archive`：保存 traj 多轮演化过程，用于审计、复盘和恢复中间版本。

### Package CLI

旧命令保持兼容：

```bash
python skill-bank/compile.py rllm-config
python skill-bank/compile.py --group rllm
python skill-bank/compile.py --group traj
python skill-bank/compile.py --all
python skill-bank/compile.py --diff rllm-config
python skill-bank/compile.py --status
```

新增 package 命令：

```bash
python skill-bank/compile.py --validate
python skill-bank/compile.py --list-packages
python skill-bank/compile.py --smoke-test
python skill-bank/compile.py --smoke-test --package-id stable:rllm-stable-v3
python skill-bank/compile.py --package stable --name rllm-stable-v1
python skill-bank/compile.py --package experimental --domain finance --name rllm-finance-exp-001
python skill-bank/compile.py --package vertical --domain finance --name rllm-finance-v1 --review
python skill-bank/compile.py --package vertical --domain finance --name rllm-finance-v1 --dry-run
python skill-bank/compile.py --package task-package --domain finance --name finance-agent-qwen7b-001 --run-id run_1779240533
python skill-bank/compile.py --package task-package --domain finance --name finance-agent-qwen7b-001 --run-dir rllm_train/output/runs/run_1779240533 --dry-run
python skill-bank/compile.py --package lineage-archive --domain finance --name finance-lineage-001 --round 1 --dry-run
python skill-bank/compile.py --package lineage-archive --domain finance --name finance-lineage-001 --round-range 1-3 --dry-run
```

### 选择规则

- 新训练任务：优先使用 matching `vertical.current`，没有则用 `stable.current`。
- 新领域适配：从 `stable.current` 复制到 `experimental/<domain>/...`。
- traj 优化：只修改 experimental 工作包或现有 rllm skill 源，不直接污染 stable/vertical。
- 复现历史训练：使用 `task-packages/<task_id>`，不要用 vertical 包猜测历史配置。
- 追溯 skill 演化：使用 `lineage-archive/<lineage_id>`。

### 路径校验

`skill_bank_paths.py` 是共享路径解析入口。新增写入必须通过 containment 校验，保证：

- compiled skill 输出仍在 `.claude/skills/`。
- snapshot 仍在 `skill-bank/compiled/`。
- package 文件只写入 `skill-bank/packages/`。
- traj patch 只写入目标 skill 的 `patches/` 目录。

每次 package/lifecycle 变更后至少运行：

```bash
python skill-bank/compile.py --validate
python skill-bank/compile.py --list-packages
python skill-bank/compile.py --diff rllm-config
python skill-bank/compile.py --group rllm --dry-run
python skill-bank/compile.py --group traj --dry-run
```

## 验证方式

1. 无 patch 时: 编译产物与原 SKILL.md 一致
2. 有 patch 时: review 编译后的 SKILL.md，确认 patch 正确合并
3. 禁用 patch → 重编译 → 内容消失
4. 两个 replace 同一 section → 报错（conflicts_with）
5. 依赖缺失 → 报错
6. deprecated patch 在 active 中 → 警告
7. archived patch 在 active 中 → 报错
8. squash → base 更新 + patch archived + 编译结果不变
