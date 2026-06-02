---
description: Launch rllm_train training via config.json. Starts the training process
  as a background task, confirms successful startup, and returns task ID and log path
  for rllm-monitor to consume.
metadata:
  categories:
  - machine-learning
  - training
  version: 1.1.0
name: rllm-run
---


# rllm-run — 启动训练

你负责启动 rllm_train 训练进程并确保它正常运行。

## 职责边界

你只负责**启动 + 确认**。不负责：
- 生成配置（那是 rllm-config 的职责）
- 持续监控（那是 rllm-monitor 的职责）
- 分析结果（那是 rllm-analyze 的职责）

## 输入

编排者传入 run_id，你从中定位配置文件：

```
run_id: <run_id>
config 路径: rllm_train/output/runs/<run_id>/config.json
```

## 执行步骤

### 1. 验证配置

用程序化方式验证配置文件参数合理：

```bash
python -c "
from rllm_train.config import TrainingConfig
config = TrainingConfig.from_json('rllm_train/output/runs/<run_id>/config.json')
print(config.summary())
"
```

确认必要字段存在且合法：model_name, num_problems, learning_rate, run_id。

同时确认 package/task 元数据：
- `task_id` 存在；未提供时 `TrainingConfig` 会使用 run_id
- `skill_package_id` 存在；未提供时 `TrainingConfig` 会从 `skill-bank/registry.json` 推断
- `skill_package_manifest` 可为空；加载配置后会自动补齐

### 2. 确认输出目录

确保输出目录存在：

```bash
mkdir -p rllm_train/output/runs/<run_id>
```

### 3. 启动训练

使用 Bash 工具以后台模式启动训练：

```bash
RLLM_TASK_ID="<task_id>" RLLM_SKILL_PACKAGE_ID="<skill_package_id>" python -m rllm_train.run_training rllm_train/output/runs/<run_id>/config.json > rllm_train/output/runs/<run_id>/training_log.txt 2>&1
```

`<task_id>` 和 `<skill_package_id>` 从 config.json 读取；如果为空，可省略对应环境变量，让 `TrainingConfig` 使用默认值。

**必须使用 `run_in_background: true`**，使训练在后台运行。

### 4. 确认启动成功

等待 10 秒后检查：

```bash
sleep 10 && head -5 rllm_train/output/runs/<run_id>/training_log.txt
```

确认日志文件非空且无即时错误。

如果日志中出现以下任一内容，说明启动失败：
- `Traceback`
- `Error`
- `ModuleNotFoundError`
- `FileNotFoundError`

启动失败时：报告错误内容，不重试，将控制权交回编排者。

### 5. 返回结果

输出格式：

```
训练已启动
===========
run_id:     <run_id>
task_id:    <background_task_id>
日志文件:    rllm_train/output/runs/<run_id>/training_log.txt
配置文件:    rllm_train/output/runs/<run_id>/config.json
```

编排者将 task_id 和日志路径传递给 rllm-monitor。

## 错误处理

| 错误类型 | 处理方式 |
|----------|---------|
| config.json 不存在 | 报告错误，提示先运行 rllm-config |
| 配置参数不合法 | TrainingConfig 校验失败，报告具体字段和原因 |
| ModuleNotFoundError | 报告缺失模块，建议 pip install 安装依赖 |
| CUDA/MPS 错误 | 建议设置 CUDA_VISIBLE_DEVICES="" 使用 CPU，或检查 GPU 驱动 |
| OOM (Out of Memory) | 建议减小 batch_size、num_generations 或 max_completion_length |
| 启动后立即崩溃（日志有 Traceback） | 报告完整错误信息，不重试 |
