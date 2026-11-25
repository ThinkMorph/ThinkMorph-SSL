#!/bin/bash
# Launch training for one config.
#
# Single node:
#   NGPUS=8 bash scripts/train.sh configs/train/stage1_pretext.yaml
#
# Multiple nodes (run the same command on every node):
#   NNODES=4 NODE_RANK=<0..3> MASTER_ADDR=<node 0 address> NGPUS=8 \
#     bash scripts/train.sh configs/train/stage1_pretext.yaml
#
# ${VAR} references in the config are expanded from the environment before
# launch. Extra key=value arguments are passed through as config overrides,
# for example: trainer_args.learning_rate=2e-5

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash $0 <config.yaml> [key=value ...]" >&2
    exit 1
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$(cd -- "$(dirname -- "$1")" && pwd)/$(basename -- "$1")"
shift

NGPUS="${NGPUS:?Set NGPUS to the number of GPUs allocated to this process}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29601}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: config not found: ${CONFIG_FILE}" >&2
    exit 1
fi
if [[ ! "${NGPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NGPUS must be a positive integer." >&2
    exit 1
fi
if [ "${DRY_RUN:-0}" != 1 ] && ! command -v bagel-train >/dev/null 2>&1; then
    echo "ERROR: bagel-train is not installed. Run 'uv sync' or 'pip install -e .' first." >&2
    exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

RUN_NAME="$(basename "${CONFIG_FILE}" .yaml)"
RENDERED_DIR="${REPO_ROOT}/outputs/_rendered_configs"
mkdir -p "${RENDERED_DIR}"
RENDERED="${RENDERED_DIR}/${RUN_NAME}_node${NODE_RANK}.yaml"

# Expand ${VAR} and fail on any variable that is not set.
"${PYTHON_BIN}" - "${CONFIG_FILE}" "${RENDERED}" <<'PYEOF'
import os, re, sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
missing = sorted({m for m in re.findall(r"\$\{(\w+)\}", text) if m not in os.environ})
missing += [m for m in re.findall(r"\$\{(\w+)\}", text) if m in os.environ and not os.environ[m]]
if missing:
    sys.exit("ERROR: set these environment variables first: " + ", ".join(missing))
open(dst, "w").write(os.path.expandvars(text))
PYEOF

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

LOG_DIR="${REPO_ROOT}/outputs/logs"
mkdir -p "${LOG_DIR}"
export TRAIN_LOG_FILE="${LOG_DIR}/${RUN_NAME}_node${NODE_RANK}_$(date '+%Y%m%d_%H%M%S').log"

if [ $# -gt 0 ]; then
    OVERRIDES="$(IFS=,; echo "$*")"
    export BAGEL_OVERRIDES="${BAGEL_OVERRIDES:+${BAGEL_OVERRIDES},}${OVERRIDES}"
fi

echo "Config:   ${CONFIG_FILE} (rendered to ${RENDERED})"
echo "Topology: ${NNODES} node(s) x ${NGPUS} GPU(s), node rank ${NODE_RANK}"
echo "Log:      ${TRAIN_LOG_FILE}"

if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "Dry run: config rendered; training was not started."
    exit 0
fi

torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NGPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "$(command -v bagel-train)" \
    "config_yaml=${RENDERED}"
