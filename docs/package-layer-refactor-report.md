# Skill-bank Package Layer 重构检查报告

## 检查范围

本次检查聚焦 skill-bank package layer 重构后的逻辑一致性和兼容性，覆盖以下模块：

- `skill_bank_paths.py`：统一路径解析与 containment 校验。
- `skill-bank/compile.py`：编译兼容、registry 校验、stable/experimental/vertical/task-package/lineage-archive/smoke-test 命令。
- `skill-bank/registry.json`：package 索引与当前 package 状态。
- `rllm_train/config.py`：训练配置中的 task/package 元数据记录。
- `traj_opt/round_state.py`：训练轮次状态中的 task/package 元数据记录。
- `skill-bank/packages/stable/rllm-stable-v3/manifest.json` 与 `skill-bank/packages/experimental/finance/rllm-finance-exp-003/manifest.json`：当前主要包 manifest。
- 兼容路径是否仍保持不变。

本报告不包含任何本地环境变量、认证 token 或敏感配置内容。

## 总体结论

重构后的主线逻辑是可用的，且没有发现会立即破坏现有 rllm/traj 训练流程的兼容性问题。

当前实现已经把 package layer 作为 overlay 叠加在原有 `skill-bank/<group>/<skill>` 源码布局之上，而不是替换原有路径。这符合此前确定的安全重构方向：先保留旧路径和旧命令，再增加 stable、experimental、vertical、task-package、lineage-archive 的索引和归档能力。

当前风险等级：中低。

- 低风险部分：路径兼容、registry 校验、stable/experimental 包索引、基础 smoke test、运行时 metadata 写入。
- 中风险部分：自然语言配置的 package 解析时机、registry 中 stale source-gap 标记、lineage CLI 输入校验、promotion review 粒度、task-package 可复现完整度。

## 已完成能力

### 1. 兼容路径保留

以下路径仍作为兼容契约保留：

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

package layer 没有移动这些路径，而是新增：

```text
skill-bank/registry.json
skill-bank/packages/stable/
skill-bank/packages/experimental/
skill-bank/packages/vertical/
skill-bank/packages/task-packages/
skill-bank/packages/lineage-archive/
```

### 2. 路径解析与 containment 校验

`skill_bank_paths.py` 已集中定义：

- repo root
- skill-bank root
- `.claude/skills` 输出根目录
- `skill-bank/compiled` snapshot 根目录
- `traj_opt/output` 根目录
- `rllm_train/output` 根目录
- `skill-bank/packages` package 根目录

并提供：

- `validate_compile_output()`
- `validate_snapshot_path()`
- `validate_package_path()`
- `validate_patch_path()`
- `ensure_within()`

这能防止 package、snapshot、compiled skill、patch 写入逃逸到预期目录之外。

### 3. Registry-first package 索引

当前 registry 状态：

```text
stable.current = stable:rllm-stable-v3
experimental.finance.current = experimental:finance:rllm-finance-exp-003
vertical = empty
task_packages = empty
lineage_archive = empty
```

这满足“新窗口先读 registry，不靠模糊搜索目录名”的目标。

### 4. Stable / Experimental 生命周期

已具备：

```bash
python skill-bank/compile.py --package stable --name <name>
python skill-bank/compile.py --package experimental --domain <domain> --name <name>
python skill-bank/compile.py --list-packages
```

当前已创建：

- `stable:rllm-stable-v1`
- `stable:rllm-stable-v2`
- `stable:rllm-stable-v3`
- `experimental:finance:rllm-finance-exp-001`
- `experimental:finance:rllm-finance-exp-002`
- `experimental:finance:rllm-finance-exp-003`

### 5. Vertical promotion review

已具备：

```bash
python skill-bank/compile.py --package vertical --domain finance --name <name> --review
```

当前 review 逻辑会把：

- rllm skill 归入 vertical candidates。
- traj skill 归入 excluded_from_vertical。
- 单次训练产物归入 task-package-only。
- traj 过程证据归入 lineage-archive-only。

这符合“traj 层可复用，rllm 层沉淀垂类经验”的边界设计。

### 6. Task package 归档

已具备 run-backed task package 命令：

```bash
python skill-bank/compile.py --package task-package --name <task-id> --run-id <run_id>
python skill-bank/compile.py --package task-package --name <task-id> --run-dir <run_dir>
```

当前逻辑会复制或记录：

- run config
- analysis
- perf stats
- training log
- trajectories
- source skill package snapshot
- final model 路径引用
- provenance

设计上没有直接复制大模型目录，而是将 `final_model` 作为引用记录，避免 package 目录暴涨。

### 7. Lineage archive 归档

已具备 round-backed lineage archive 命令：

```bash
python skill-bank/compile.py --package lineage-archive --domain <domain> --name <lineage-id> --round <n>
python skill-bank/compile.py --package lineage-archive --domain <domain> --name <lineage-id> --round-range 1-3
```

当前逻辑会从 `traj_opt/output/rounds/round_<n>/status.json` 追溯：

- round 状态
- session id
- raw hook events
- segmented trajectories
- optimization report
- rllm reports
- source package snapshot

这满足“复盘 traj 多轮优化过程”和“恢复中间版本”的基础需要。

### 8. 运行时 metadata 记录

`rllm_train.config.TrainingConfig` 已新增：

```text
task_id
skill_package_id
skill_package_manifest
```

`traj_opt.round_state.RoundState.write_training_complete()` 已写入：

```text
training.task_id
training.skill_package_id
training.skill_package_manifest
```

这使训练 run、round status、task package、lineage archive 能串起来，不再只靠 run_id 猜测来源。

## 验证结果

本次检查运行并通过：

```bash
python -m py_compile skill_bank_paths.py rllm_train/config.py traj_opt/round_state.py skill-bank/compile.py
python skill-bank/compile.py --validate
python skill-bank/compile.py --list-packages
python skill-bank/compile.py --smoke-test --package-id stable:rllm-stable-v3
```

结果摘要：

```text
py_compile: passed
--validate: Validation passed: 0 warning(s)
--list-packages: stable.current = stable:rllm-stable-v3; finance experimental.current = experimental:finance:rllm-finance-exp-003
--smoke-test --package-id stable:rllm-stable-v3: Smoke test passed: 1 package(s)
```

说明：`--smoke-test` 会 dry-run 编译核心 rllm skills，因此当前输出较长，但结果通过。

## 发现的问题和风险

### 1. `registry.json` 中 `known_source_gaps` 已过期

当前 registry 仍记录：

```text
rllm-clarify: compiled_only
rllm-run: compiled_only
```

但实际源码目录已经存在：

```text
skill-bank/rllm/rllm-clarify/base.md
skill-bank/rllm/rllm-clarify/manifest.yaml
skill-bank/rllm/rllm-run/base.md
skill-bank/rllm/rllm-run/manifest.yaml
```

影响：

- 不影响当前 `--validate`，因为 validate 按实际目录检查并已通过。
- 但会误导未来读 registry 的新窗口或自动选择逻辑，让系统以为这两个 skill 仍是 compiled-only。

建议：

- 从 `registry.json.known_source_gaps` 删除这两个条目，或改为历史 note，不作为当前状态字段。
- 增强 `--validate`：如果 `known_source_gaps` 标记的 source 已恢复，应给出 warning。

优先级：高。

### 2. 自然语言配置中 package 解析早于 task_type 最终解析

`parse_natural_language()` 当前先执行：

```python
config = TrainingConfig()
```

而 `TrainingConfig.__post_init__()` 会立即根据默认 `task_type="math"` 解析 `skill_package_id`。之后自然语言解析才可能把 task_type 改成：

```text
finance / code / search
```

影响：

- 当前 registry 没有 `vertical.finance.current`，默认会回落到 stable，因此问题暂时不明显。
- 未来如果存在 `vertical.finance.current`，输入“训练金融 agent”时，可能仍拿到初始化时基于 math/default 解析出的 stable package，而不是 finance vertical package。

建议：

- 在 `parse_natural_language()` 解析完 task_type 后，如果用户没有显式设置 `RLLM_SKILL_PACKAGE_ID`，重新根据最终 `config.task_type` 解析 package。
- 或延迟 `skill_package_id` 的自动解析，放到自然语言解析完成之后统一 finalize。

优先级：高。

### 3. `find_package_entry()` 不搜索 task_packages 和 lineage_archive

`skill_bank_paths.package_manifest_path()` 已搜索：

```text
stable
experimental
vertical
task_packages
lineage_archive
```

但 `compile.py.find_package_entry()` 目前只搜索：

```text
stable
experimental
vertical
```

影响：

- 当前 stable/experimental/vertical 创建与 smoke test 不受影响。
- 如果未来某些命令要用 task package 或 lineage archive 作为 `--from-package`，会找不到。
- 这也会造成两个 package lookup 函数语义不一致。

建议：

- 统一 package lookup 行为，让 `find_package_entry()` 也覆盖 task_packages 和 lineage_archive。
- 或明确把它重命名为 `find_runtime_skill_package_entry()`，只用于 stable/experimental/vertical。

优先级：中。

### 4. `parse_rounds()` 对非法输入会抛出原始异常

当前 `parse_rounds()` 对以下输入缺少友好错误处理：

```bash
--round abc
--round-range 3
--round-range 5-2
--round-range a-b
```

影响：

- 用户输入错误时可能直接看到 Python traceback。
- 不影响合法输入。

建议：

- 捕获 `ValueError`，打印清晰 CLI 错误并返回失败。
- 校验 range 必须是 `start-end` 且 `start <= end`。

优先级：中。

### 5. Promotion review 目前是包级粗粒度分类

当前 `build_promotion_review()` 会把 source package 中所有 rllm skills 都列为 vertical candidates。

影响：

- 作为人工 review 清单是安全的，因为不会自动写入 vertical。
- 但它还不能区分“真正 finance 相关的变更”和“只是包里已有的通用 rllm skill”。
- 如果后续直接依赖这个结果自动 promotion，可能会把过宽的内容迁入 vertical。

建议：

- promotion 前引入 diff/provenance 检查：比较 source experimental 与 base stable 的差异。
- review 输出中区分：
  - changed rllm skills
  - unchanged inherited rllm skills
  - domain-specific patches
  - generic reusable patches
  - task-private artifacts

优先级：中。

### 6. Task package 还不是完全自包含的训练交付包

当前 task package 会复制主要 run 文件和 skill package snapshot，但以下内容还没有完整归档：

- 数据集快照或 dataset manifest。
- reward 代码快照。
- agent 代码快照。
- 依赖版本、git commit、环境信息。
- 模型 checkpoint 实体文件。

其中模型只记录 `final_model` 路径引用，这是合理的空间保护策略，但意味着 package 本身不是完全离线可用。

影响：

- 可以支持“索引和追溯一次训练经验”。
- 对“完全离线复现”还不够。

建议：

- 明确 task package 分两种模式：
  - lightweight：记录引用，不复制大模型和大数据。
  - full-repro：复制或登记可下载/可校验的 artifact manifest。
- 增加 `artifact_manifest.json`，记录文件 hash、大小、来源路径、是否内嵌。

优先级：中。

### 7. Smoke test 输出过长

`--smoke-test` 当前调用 `compile_skill(..., dry_run=True)`，会打印每个核心 skill 的编译内容摘要。

影响：

- 功能正确，当前 smoke test 通过。
- 但作为 CI 或日常检查会产生较多输出，不利于快速定位失败原因。

建议：

- 给 `compile_skill()` 增加 quiet dry-run 模式。
- `--smoke-test` 默认 quiet，只在 `--verbose` 时打印编译预览。

优先级：低。

## 设计一致性检查

### rllm/traj 分层

当前重构仍保持：

- `rllm-*`：训练具体 agent。
- `traj-*`：分析轨迹并优化 rllm skill。
- vertical package 应沉淀 rllm 领域经验，不把 traj 过程证据混入运行时 skill。
- task-package 保存一次具体训练成果。
- lineage-archive 保存 traj 多轮演化证据。

结论：分层方向正确。

### stable / experimental / vertical 生命周期

当前实现支持：

```text
stable.current
  ↓ copy
experimental.<domain>.current
  ↓ review / promote
vertical.<domain>.current
```

结论：生命周期主干正确。

注意：当前还没有真实 vertical 包，因为 finance 只是 experimental 包，还没经过真实训练、task package、lineage archive 和 promotion review 的完整闭环验证。

### 历史复现路径

当前设计把历史复现放在：

```text
task-packages/<task_id>
lineage-archive/<lineage_id>
```

而不是让 vertical 保存所有历史细节。

结论：方向正确，能避免 vertical 目录无限膨胀。

### 新窗口索引

当前 `registry.json` 已提供入口，但需要清理 stale `known_source_gaps`，否则新窗口可能读到误导信息。

结论：可用，但 registry 需要一次小修正。

## 建议修复顺序

### P0：立即修复

1. 清理 `registry.json.known_source_gaps` 中已经恢复源码的 `rllm-clarify` 和 `rllm-run`。
2. 修复 `parse_natural_language()` 中 package 解析早于 task_type 最终解析的问题。

### P1：近期修复

3. 统一 `find_package_entry()` 与 `package_manifest_path()` 的查找范围，或明确拆分命名。
4. 给 `parse_rounds()` 增加友好 CLI 错误处理。
5. 让 `--smoke-test` 支持 quiet 模式。

### P2：后续增强

6. promotion review 增加 diff/provenance 粒度。
7. task package 增加 `artifact_manifest.json` 和 full-repro 模式。
8. lineage archive 增加更严格的 report path 来源校验和 manifest 完整性校验。

## 后续闭环建议

在修复 P0/P1 后，建议用真实 finance 任务跑一次完整闭环：

1. 从 `stable:rllm-stable-v3` 创建或继续使用 `experimental:finance:rllm-finance-exp-003`。
2. 执行一次 finance agent 训练。
3. 将成功 run 冻结为 task package：

```bash
python skill-bank/compile.py --package task-package --domain finance --name <task-id> --run-id <run_id>
```

4. 将 traj 优化过程归档为 lineage archive：

```bash
python skill-bank/compile.py --package lineage-archive --domain finance --name <lineage-id> --round-range 1-3
```

5. 执行 promotion review：

```bash
python skill-bank/compile.py --package vertical --domain finance --name <vertical-name> --review
```

6. smoke test stable 和候选 vertical。
7. 人工确认后再创建真实 vertical 包。

## 最终判断

本次 skill-bank package layer 重构的主体设计和实现方向正确，当前基础验证通过，兼容路径没有被破坏。它已经解决了以下核心问题：

- stable 作为 skill 基座包。
- experimental 作为领域演化工作包。
- vertical 作为可复用垂类经验包。
- task-package 保存具体训练成果。
- lineage-archive 保存 traj 优化过程。
- registry-first 支持新窗口索引。
- containment 校验降低路径漂移风险。

但在进入真实多领域训练之前，建议先修复 P0 两项：registry stale source-gap 和自然语言 task_type/package 解析时机。否则新窗口索引和未来 vertical 自动选择可能出现误导或选错包。