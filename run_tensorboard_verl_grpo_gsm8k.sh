#!/usr/bin/env bash
set -xeuo pipefail

# ============================================================================
# TensorBoard Launch: VERL GRPO GSM8K Training Logs
# ============================================================================
# Log directory: tensorboard_log/verl_grpo_gsm8k
# Runs:         qwen2.5-0.5b-instruct_base, qwen2.5-0.5b-instruct_fsdp,
#               run_verl_gsm8k_0.5b_r3 ~ r9
# ============================================================================

########################### Project Paths ###########################

PROJECT_ROOT="/lixiang/project/huxing/Project_test/AgentSDKMetricSkill"
LOG_DIR="${PROJECT_ROOT}/tensorboard_log/verl_grpo_gsm8k"

########################### TensorBoard Config ###########################

# Default port (override with: PORT=6007 ./run_tensorboard_verl_grpo_gsm8k.sh)
PORT=${PORT:-6006}

# Bind address (override with: BIND_ADDR=0.0.0.0 ...)
BIND_ADDR=${BIND_ADDR:-0.0.0.0}

# Extra args (e.g. --reload_interval 30, --samples_per_plugin, etc.)
EXTRA_ARGS=${EXTRA_ARGS:-}

########################### Launch TensorBoard ###########################

echo "=========================================="
echo " TensorBoard: VERL GRPO GSM8K"
echo "=========================================="
echo " Log dir:   ${LOG_DIR}"
echo " URL:       http://${BIND_ADDR}:${PORT}"
echo "=========================================="

cd "${PROJECT_ROOT}"

tensorboard \
    --logdir="${LOG_DIR}" \
    --host="${BIND_ADDR}" \
    --port="${PORT}" \
    ${EXTRA_ARGS} \
    "$@"
