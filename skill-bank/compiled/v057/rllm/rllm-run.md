---
name: rllm-run
description: Launch VERL GRPO training via run_verl.sh. Validates config, probes GPU, starts training as background task.
metadata:
  version: "2.0.0"
---

# rllm-run — 启动 VERL 训练

你负责验证配置、探测 GPU、启动 VERL 训练进程并确认启动成功。

## 职责边界

只负责 **启动 + 确认**。不负责生成配置（rllm-config）、监控训练（rllm-monitor）、分析结果（rllm-analyze-deep）。

## 输入

编排者传入 `run_id`，定位配置文件：

```
config 路径: rllm_train/output/runs/<run_id>/config.json
run_verl.sh: rllm_train/output/runs/<run_id>/run_verl.sh
```

## 执行步骤

### 1. 验证配置

```bash
python3 -c "
from rllm_train.config import TrainingConfig
config = TrainingConfig.from_json('rllm_train/output/runs/<run_id>/config.json')
print(config.summary())
"
```

### 2. 确认输出目录

```bash
mkdir -p rllm_train/output/runs/<run_id>
```

### 3. 环境检查 + GPU 探测

```bash
python3 -c "
import sys, subprocess, json
from pathlib import Path

try:
    import verl; print(f'VERL: {verl.__version__}')
    import ray; print(f'Ray: {ray.__version__}')
except ImportError as e:
    print(f'ERROR: {e}'); sys.exit(1)

out = subprocess.check_output(['nvidia-smi', '-L'], text=True)
gpu_lines = [l for l in out.splitlines() if 'GPU' in l]
print(f'GPUs: {len(gpu_lines)}')
for l in gpu_lines: print(f'  {l.strip()}')

# Write resolved hardware
config_path = Path('rllm_train/output/runs/<run_id>/config.json')
data = json.loads(config_path.read_text())
data['resolved_num_gpus'] = len(gpu_lines)
data['resolved_gpu_type'] = gpu_lines[0] if gpu_lines else ''
config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
"
```

### 4. 启动训练

```bash
bash rllm_train/output/runs/<run_id>/run_verl.sh > rllm_train/output/runs/<run_id>/training_log.txt 2>&1
```

**必须使用 `run_in_background: true`**。

### 5. 确认启动

等待 30s 后检查：

```bash
sleep 30 && tail -30 rllm_train/output/runs/<run_id>/training_log.txt
```

确认出现 `Started a local Ray instance` 且无 `Traceback`。

如果 60s 内无训练内容：报告警告但不中止（VERL 模型加载 + vLLM 初始化需 2-5 分钟）。

### 6. 返回结果

```
训练已启动
===========
run_id:     <run_id>
task_id:    <background_task_id>
日志文件:    rllm_train/output/runs/<run_id>/training_log.txt
配置文件:    rllm_train/output/runs/<run_id>/config.json
```

## 错误处理

| 错误 | 处理 |
|------|------|
| config.json 不存在 | 报告，提示先运行 rllm-config |
| VERL/Ray 未安装 | 报告缺失模块 |
| OOM | 报告，建议减小 batch_size 或 num_generations |
| 启动后立即崩溃 | 报告完整 Traceback，不重试 |
