# Skill: VERL + A100(40G) + Qwen2.5-0.5B 一键 PPO 训练（单文档版）

## 目标
在一台全新机器上，用 **单机单卡 A100 40G** 跑通 `verl` + `Qwen/Qwen2.5-0.5B-Instruct` 的 PPO 训练；并且让 Claude Code 能根据你指定的 agent 领域（如医疗/法律/数学）自行生成数据处理脚本。

> 本 Skill **只保留一个文档**。不预置任何 `scripts/*.py` 文件。

---

## 前置条件
- OS: Ubuntu/Debian
- GPU: NVIDIA A100 40G
- 已安装 NVIDIA 驱动 + CUDA 运行环境（`nvidia-smi` 可用）
- 具备 root 权限（执行 apt）

---

## 一键流程（任意路径可运行）

1) 新建目录并进入（示例）
```bash
mkdir -p ~/verl_qwen25_05b && cd ~/verl_qwen25_05b
```

2) 创建 `setup.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail

# 允许在任意路径执行：以脚本所在目录为根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$SCRIPT_DIR}"
VENV_DIR="${VENV_DIR:-$WORKDIR/.venv}"
VERL_DIR="${VERL_DIR:-$WORKDIR/verl}"
DATA_DIR="${DATA_DIR:-$WORKDIR/data/domain_data}"
LOG_DIR="${LOG_DIR:-$WORKDIR/logs}"

mkdir -p "$WORKDIR" "$DATA_DIR" "$LOG_DIR"

# 你指定必须包含
apt update
apt install python3.10-venv -y
apt install -y git curl build-essential python3-pip

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -U pip setuptools wheel

if [ ! -d "$VERL_DIR" ]; then
  git clone https://github.com/verl-project/verl.git "$VERL_DIR"
fi
cd "$VERL_DIR"

USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .

# 固定兼容依赖（防止 trl/transformers 漂移）
pip uninstall -y trl transformers huggingface_hub numpy tokenizers || true
pip install -U \
  "numpy==1.26.4" \
  "transformers==4.56.1" \
  "huggingface_hub==0.36.2" \
  "trl==0.11.4" \
  "datasets>=2.20.0" \
  "pyarrow>=16.0.0"

# 可选：flash-attn，失败不阻断
pip install -U packaging ninja || true
pip install -U flash-attn --no-build-isolation || echo "[WARN] flash-attn install failed, continue"

echo "[OK] setup finished"
echo "[INFO] WORKDIR=$WORKDIR"
echo "[INFO] DATA_DIR=$DATA_DIR"
```

3) 创建 `start_train.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$SCRIPT_DIR}"
VENV_DIR="${VENV_DIR:-$WORKDIR/.venv}"
VERL_DIR="${VERL_DIR:-$WORKDIR/verl}"
DATA_DIR="${DATA_DIR:-$WORKDIR/data/domain_data}"
LOG_DIR="${LOG_DIR:-$WORKDIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/verl_qwen25_05b_a10040g.log}"

mkdir -p "$LOG_DIR"
source "$VENV_DIR/bin/activate"
cd "$VERL_DIR"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

python3 -m verl.trainer.main_ppo \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size=256 \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  critic.optim.lr=1e-5 \
  critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  critic.ppo_micro_batch_size_per_gpu=2 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=console \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.total_epochs=15 2>&1 | tee "$LOG_FILE"
```

4) 执行
```bash
chmod +x setup.sh start_train.sh
bash setup.sh
# 先让 Claude Code 生成数据处理脚本并产出 train/test parquet
bash start_train.sh
```

---

## 给 Claude Code 的“数据处理思路链条”（核心）

> 你在对话里只需要说：
> “我要训练一个医疗 agent（或法律 agent / 数学 agent），请按 Skill 文档生成数据处理脚本并执行。”

Claude Code 应按以下链条工作并生成脚本：

1. **识别 agent 领域与任务类型**  
   - 输入：`agent_type`（medical/legal/math/...）  
   - 推导：主要任务是 QA、分类、摘要、对话，还是多任务混合。  

2. **数据源检索策略（联网）**  
   - 优先检索 Hugging Face Datasets；必要时补充公开基准。  
   - 搜索词模板：`<agent_type> + qa / instruction / reasoning / dataset`。  
   - 输出候选集：数据集 id、license、样本量、字段结构、语言、更新时间。  

3. **可用性筛选**  
   - 过滤不可商用或 license 不清晰的数据集。  
   - 过滤字段缺失严重、噪声过高、重复比例过高的数据集。  
   - 至少保留 1-3 个候选并说明取舍理由。  

4. **字段映射到统一训练格式**  
   - 统一目标字段：`prompt`、`response`。  
   - 常见映射：  
     - prompt: `question/prompt/instruction/input/query/problem`  
     - response: `answer/output/response/solution/label`  
   - 若无直接字段，允许模板拼接（例如多列合并成 prompt）。  

5. **质量清洗**  
   - 去空值、去重、截断超长样本。  
   - 过滤乱码和异常 token 比例过高样本。  
   - 医疗/法律场景可加规则：过滤缺失关键上下文的样本。  

6. **切分与采样**  
   - 输出 `train.parquet` 与 `test.parquet`。  
   - 默认建议：train 1万~20万（视数据规模），test 1千~5千。  
   - 固定随机种子，保证复现。  

7. **落盘路径约定**  
   - 必须输出到：`$DATA_DIR/train.parquet` 与 `$DATA_DIR/test.parquet`。  
   - 与 `start_train.sh` 参数保持一致。  

8. **自检与报告**  
   - 打印：样本数、字段统计、平均长度、去重比例。  
   - 抽样展示 3-5 条 `prompt/response`，确认可训练。  

9. **失败回退策略**  
   - 若主数据集不可用，自动切到次优候选。  
   - 若字段无法自动识别，要求用户指定 prompt/response 字段名后重跑。  

---

## Claude Code 生成脚本时的约束
- 生成脚本要支持参数：`--agent-type`、`--dataset-id`（可选）、`--out-dir`。
- 默认输出 parquet，且列名必须是 `prompt` 与 `response`。
- 代码应包含异常处理与日志打印（下载失败、字段缺失、空数据集）。
- 若用户未指定数据集 id，先执行“检索+筛选”再下载处理。

---

## 常见问题

1) `please install trl to make it valid`  
- 重新执行 `setup.sh` 的 pinned 依赖段。

2) `flash_attn` 相关报错  
- 可先忽略 flash-attn，优先保证训练主流程可跑。

3) OOM  
- 依次降低：
  - `data.max_response_length: 256 -> 128`
  - `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu: 2 -> 1`
  - `critic.ppo_micro_batch_size_per_gpu: 2 -> 1`
  - `actor_rollout_ref.rollout.gpu_memory_utilization: 0.35 -> 0.30`

---

## 成功标志
- `start_train.sh` 日志持续出现 step 输出。
- 周期性有验证与 checkpoint 保存记录。
