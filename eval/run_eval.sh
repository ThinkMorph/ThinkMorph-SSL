#!/bin/bash
# Evaluate one ThinkMorph-SSL checkpoint with the bundled VLMEvalKit fork.
#
#   bash eval/run_eval.sh configs/eval/ssl.yaml
#
# See configs/eval/ssl.yaml for the fields. Each benchmark runs as its own
# torchrun job with one model replica per GPU. A failed benchmark is logged and
# the next one continues; rerunning the same config skips finished benchmarks.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash $0 <eval_config.yaml>" >&2
    exit 1
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VLMEVALKIT_DIR="${REPO_ROOT}/eval/VLMEvalKit"
CONFIG="$(cd -- "$(dirname -- "$1")" && pwd)/$(basename -- "$1")"

if [ "${DRY_RUN:-0}" != 1 ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: set OPENAI_API_KEY (and OPENAI_API_BASE for a compatible endpoint) for the LLM judge." >&2
    exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Parse the yaml into shell variables.
set +e
PYOUT="$(python3 - "${CONFIG}" <<'PYEOF'
import os, re, shlex, sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1])) or {}

def get(key, required=False, default=None):
    value = cfg.get(key, default)
    if isinstance(value, str):
        value = os.path.expandvars(value)
        if re.search(r"\$\{\w+\}", value):
            sys.exit(f"ERROR: unresolved environment variable in '{key}': {value}")
    if required and not value:
        sys.exit(f"ERROR: '{key}' is required")
    return value

model_path = get("model_path") if cfg.get("model_path") else ""
checkpoint = get("checkpoint") if cfg.get("checkpoint") else ""
if bool(model_path) == bool(checkpoint):
    sys.exit("ERROR: set exactly one of 'model_path' or 'checkpoint'")
merged_dir = get("merged_dir", required=bool(checkpoint)) if checkpoint else ""
checkpoint_type = get("checkpoint_type", default="regular")
if checkpoint_type not in ("regular", "ema"):
    sys.exit("ERROR: checkpoint_type must be 'regular' or 'ema'")

benchmarks = cfg.get("benchmarks") or []
if not benchmarks:
    sys.exit("ERROR: 'benchmarks' must be a non-empty list")
inference = cfg.get("inference") or {}

out = {
    "MODEL_NAME": get("model_name", required=True),
    "MODEL_PATH": model_path or merged_dir,
    "CHECKPOINT": checkpoint,
    "CHECKPOINT_TYPE": checkpoint_type,
    "BASE_MODEL": get("base_model", required=bool(checkpoint)) if checkpoint else get("base_model", default="") or "",
    "TRAINING_CONFIG": get("training_config", default="") or "",
    "OUTPUT_DIR": get("output_dir", required=True),
    "JUDGE": get("judge", default="gpt-5"),
    "MASTER_PORT": str(int(cfg.get("master_port", 22712))),
    "NGPUS": str(int(os.environ.get("EVAL_NGPUS", inference.get("ngpus", 1)))),
    "EVAL_DO_SAMPLE": str(inference.get("do_sample", False)).lower(),
    "EVAL_TEXT_TEMPERATURE": str(inference.get("text_temperature", 0.3)),
    "EVAL_NUM_TIMESTEPS": str(int(inference.get("num_timesteps", 50))),
    "EVAL_MAX_THINK_TOKEN_N": str(int(inference.get("max_think_token_n", 4096))),
    "BENCHMARKS": " ".join(benchmarks),
}
if int(out["NGPUS"]) < 1:
    sys.exit("ERROR: EVAL_NGPUS/inference.ngpus must be positive")
for key, value in out.items():
    print(f"{key}={shlex.quote(value)}")
PYEOF
)"
RC=$?
set -e
if [ "${RC}" -ne 0 ]; then
    echo "${PYOUT}" >&2
    exit 1
fi
eval "${PYOUT}"

if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "Dry run: ${MODEL_NAME}, ${NGPUS} allocated GPU(s), benchmarks: ${BENCHMARKS}"
    echo "No checkpoint merge, model loading, or judge request was made."
    exit 0
fi

# Merge a raw FSDP2 checkpoint if needed.
if [ -n "${CHECKPOINT}" ]; then
    if [ -s "${MODEL_PATH}/model.safetensors" ]; then
        echo "=== Merged checkpoint found at ${MODEL_PATH}, skipping merge ==="
    else
        echo "=== Merging ${CHECKPOINT} (${CHECKPOINT_TYPE}) into ${MODEL_PATH} ==="
        python "${REPO_ROOT}/scripts/merge_checkpoint.py" \
            "${CHECKPOINT}" "${MODEL_PATH}" "${BASE_MODEL}" \
            --checkpoint-type "${CHECKPOINT_TYPE}"
    fi
fi
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: model directory not found: ${MODEL_PATH}" >&2
    exit 1
fi

# Use the training system prompt for evaluation.
PROMPT_FILE=""
cleanup() { [ -n "${PROMPT_FILE}" ] && rm -f "${PROMPT_FILE}"; }
trap cleanup EXIT
if [ -n "${TRAINING_CONFIG}" ]; then
    PROMPT_FILE="$(python3 - "${TRAINING_CONFIG}" <<'PYEOF'
import os, sys, tempfile
import yaml
cfg = yaml.safe_load(open(sys.argv[1]))
prompt = (cfg.get("dataset_config") or {}).get("action_control_system_prompt")
if prompt:
    fd, path = tempfile.mkstemp(prefix="ssl_prompt_", suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(path)
PYEOF
)"
    [ -n "${PROMPT_FILE}" ] && export SSL_SYSTEM_PROMPT_FILE="${PROMPT_FILE}"
fi

# Runtime registration of the model in eval/VLMEvalKit/vlmeval/config.py.
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd -- "${OUTPUT_DIR}" && pwd)"
cp "${CONFIG}" "${OUTPUT_DIR}/eval_config.yaml"
export SSL_MODEL_NAME="${MODEL_NAME}"
export SSL_MODEL_PATH="$(cd -- "${MODEL_PATH}" && pwd)"
export SSL_SAVE_DIR="${OUTPUT_DIR}/${MODEL_NAME}/images"
export EVAL_DO_SAMPLE EVAL_TEXT_TEMPERATURE EVAL_NUM_TIMESTEPS EVAL_MAX_THINK_TOKEN_N
mkdir -p "${SSL_SAVE_DIR}"

FAILED=()
cd "${VLMEVALKIT_DIR}"
for BENCH in ${BENCHMARKS}; do
    echo ""
    echo "=== [${MODEL_NAME}] ${BENCH} on ${NGPUS} GPU(s) ==="
    set +e
    torchrun \
        --nnodes 1 --node_rank 0 \
        --nproc_per_node "${NGPUS}" \
        --master_port "${MASTER_PORT}" \
        run.py \
        --data "${BENCH}" \
        --model "${MODEL_NAME}" \
        --judge "${JUDGE}" \
        --work-dir "${OUTPUT_DIR}" \
        --reuse
    RC=$?
    set -e
    if [ "${RC}" -ne 0 ]; then
        echo "WARNING: ${BENCH} failed with exit code ${RC}; continuing."
        FAILED+=("${BENCH}")
    fi
done

echo ""
echo "=== Done. Results are in ${OUTPUT_DIR}/${MODEL_NAME}/ ==="
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Failed benchmarks: ${FAILED[*]}"
    echo "Rerun the same command to retry them; finished benchmarks are skipped."
    exit 1
fi
