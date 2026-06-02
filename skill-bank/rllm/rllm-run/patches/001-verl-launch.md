---
id: "001-verl-launch"
target_section: "launch"
action: append
description: "When config.json has backend=verl, launch via run_verl.sh instead of torchrun, use Ray for GPU management"
source: "2026-05-28 VERL backend integration"
created: "2026-05-28"

depends_on: []
conflicts_with: []
status: active
superseded_by: ""
---

### VERL 后端启动 (backend=verl)

当 config.json 中包含 `"backend": "verl"` 时，跳过标准 torchrun 启动流程，使用以下 VERL 专用启动流程：

#### 环境检查

启动前检查 VERL 和 Ray 可用性：

```bash
python -c "
import sys
try:
    import verl
    print(f'VERL version: {verl.__version__}')
except ImportError:
    print('ERROR: verl not installed. Run: pip install -e verl_latest/')
    sys.exit(1)

try:
    import ray
    print(f'Ray version: {ray.__version__}')
except ImportError:
    print('ERROR: ray not installed. Run: pip install ray')
    sys.exit(1)

# Check GPU via nvidia-smi
import subprocess
try:
    out = subprocess.check_output(['nvidia-smi', '-L'], text=True)
    gpu_count = len([l for l in out.splitlines() if 'GPU' in l])
    print(f'GPUs available: {gpu_count}')
except Exception:
    print('WARN: nvidia-smi failed, Ray will auto-detect')
"
```

#### VERL 训练启动命令

```bash
bash rllm_train/output/runs/<run_id>/run_verl.sh > rllm_train/output/runs/<run_id>/training_log.txt 2>&1
```

**必须使用 `run_in_background: true`**。

注意：
- VERL 通过 Ray 自动管理 GPU 分配，不需要 torchrun 包装
- run_verl.sh 已包含完整的 `python -m verl.trainer.main_ppo` 命令和所有 Hydra CLI 参数
- 日志重定向到统一的 training_log.txt 路径，与 TRL 模式保持一致
- Ray 集群由 VERL 训练脚本自动初始化和管理

#### VERL 启动确认

等待 30 秒后检查（VERL 需要更长的 Ray 集群初始化时间）：

```bash
sleep 30 && tail -30 rllm_train/output/runs/<run_id>/training_log.txt
```

确认日志中出现以下内容表示启动成功：
- Ray 集群初始化消息（如 `Started a local Ray instance` 或 `Ray cluster is ready`）
- 任务配置打印（`TaskRunner` 相关日志）
- 无 Traceback / Error

如果 60 秒内日志仍为空或只含 Ray 启动信息无训练内容，报告警告但不中止（VERL 模型加载 + vLLM 初始化可能需 2-5 分钟）。
