#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

# Accept --verbose anywhere among the positional evaluation arguments.
EVAL_VERBOSE=0
_EVAL_ARGS=()
for _arg in "$@"; do
    if [ "$_arg" = "--verbose" ]; then
        EVAL_VERBOSE=1
    else
        _EVAL_ARGS+=("$_arg")
    fi
done
set -- "${_EVAL_ARGS[@]}"
export EVAL_VERBOSE

source "$(dirname "${BASH_SOURCE[0]}")/scripts/local_env.sh"
export HF_ENDPOINT=https://hf-mirror.com
export SETUPTOOLS_USE_DISTUTILS=local
export DIST_TIMEOUT=43200
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

if [ -z "${DP:-}" ] && [ -z "${NPROC:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        export DP=$(python3 - <<'PY'
import os

visible = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
print(len(visible) or 1)
PY
)
    else
        export DP=$(python3 - <<'PY'
try:
    import torch
    print(torch.cuda.device_count() or 1)
except Exception:
    print(1)
PY
)
    fi
fi

# Usage:
#   sh eval.sh config.json [--verbose]                                # Mode 1: pass a config file directly
#   sh eval.sh [model] [benchmark] <scheme> [redi/uniform] [--verbose]  # Mode 2: generate config on-the-fly
#
#   model     : optional; 'qwen3_5', 'qwen3_5_debug',
#               'qwen3_5_moe', or 'qwen3_6' (default: qwen3_5)
#   benchmark : longvideobench | videomme | mlvu | rtvbench (default: longvideobench if omitted)
#   scheme    : e.g. d2f512, d1f128 (required in Mode 2)
#   sampling  : redi or uniform (default: redi)
#               redi is supported by qwen3_5 / qwen3_5_debug / qwen3_5_moe / qwen3_6
#
# Examples:
#   sh eval.sh d2f512                                    # qwen3_5, longvideobench, d2f512, redi
#   sh eval.sh longvideobench d2f512 redi                 # qwen3_5, longvideobench, d2f512, redi
#   sh eval.sh videomme d1f128 uniform                   # qwen3_5, videomme, d1f128, uniform
#   sh eval.sh qwen3_5_moe mlvu d2f1024 uniform          # Qwen3.5-35B-A3B FramePack, MLVU, 1024 frames
#   sh eval.sh qwen3_6 mlvu d2f1024 uniform              # Qwen3.6-27B FramePack, MLVU, 1024 frames
#   sh eval.sh qwen3_5 mlvu d1f128d2f256 redi            # Qwen3.5-4B FramePack, MLVU, MMR (lambda=0.7)
#   REDI_LAMBDA=0.5 sh eval.sh qwen3_5 videomme d1f128 redi
#
# Env-var overrides:
#   MODEL_SIZE=9b sh eval.sh d2f512 redi
#   REDI_LAMBDA=0.5 sh eval.sh qwen3_5 videomme d1f128 redi          # (redi) MMR weight on relevance
#   FRAME_FEATURES=/path sh eval.sh qwen3_5 mlvu d1f128 redi          # (redi) override feature root

# Known benchmark names (lower-case)
_BENCHMARKS="longvideobench videomme videomme-short videomme-medium videomme-long mlvu rtvbench rtv-bench streamingbench streaming-bench streamingbench-rtvu streamingbench-sqa streamingbench-context streamingbench-omni streamingbench-proactive"

# Known model family names (lower-case)
_MODELS="qwen3_5 qwen3_5_debug qwen3_5_moe qwen3_6"

# Detect mode: if $1 ends with .json, it's a config file; otherwise generate
if [[ "$1" == *.json ]]; then
    CONFIG_FILE=$1
else
    # Shift off model family if present as first arg
    _ARG1_LOWER=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    if echo "$_MODELS" | grep -qw "$_ARG1_LOWER"; then
        MODEL_FAMILY=$_ARG1_LOWER
        shift
    else
        MODEL_FAMILY=qwen3_5
    fi

    # Determine if remaining $1 is a benchmark name or a scheme string
    _ARG1_LOWER=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    if echo "$_BENCHMARKS" | grep -qw "$_ARG1_LOWER"; then
        BENCHMARK=$1
        SCHEME=${2:?Usage: sh eval.sh [model] [benchmark] <scheme> [redi/uniform]}
        SAMPLING=${3:-redi}
    else
        BENCHMARK=longvideobench
        SCHEME=$1
        SAMPLING=${2:-redi}
    fi

    CONFIG_FILE=$(python "$FRAMEPACK_ROOT/configs/gen_config.py" "$BENCHMARK" "$SCHEME" "$SAMPLING" --model "$MODEL_FAMILY") || exit $?
    echo "Generated config: $CONFIG_FILE"
fi

export USE_VLLM=${USE_VLLM:-0}
python "$FRAMEPACK_ROOT/scripts/check_local_paths.py" "$CONFIG_FILE" || exit $?
export VLMEVAL_RUN_SCRIPT="${VLMEVAL_RUN_SCRIPT:-$FRAMEPACK_ROOT/run.py}"

if grep -q '"use_vllm"[[:space:]]*:[[:space:]]*true' "$CONFIG_FILE"; then
    export USE_VLLM=1
fi

CONFIG_NAME=$(basename "$CONFIG_FILE")
CONFIG_STEM=${CONFIG_NAME%.*}
WORK_DIR="./outputs/${CONFIG_STEM#config_}"

if [ "$EVAL_VERBOSE" = "1" ]; then
    bash "$FRAMEPACK_ROOT/watchdog.sh" "$CONFIG_FILE" "$WORK_DIR"
else
    # Hide routine watchdog progress while retaining failure and restart messages.
    bash "$FRAMEPACK_ROOT/watchdog.sh" "$CONFIG_FILE" "$WORK_DIR" | awk '
        /^\[watchdog\]/ && $0 !~ /(ERROR|STUCK|EVAL_STUCK|failed|Interrupted|Terminated|Restarting)/ { next }
        { print; fflush() }
    '
    exit "${PIPESTATUS[0]}"
fi
