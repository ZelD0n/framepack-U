#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/local_env.sh"

CCCL_INC=$(python - <<'PY'
import importlib.util
import os

p = importlib.util.find_spec('nvidia.cuda_cccl')
if p is not None and p.origin:
    print(os.path.join(os.path.dirname(p.origin), 'include'))
PY
)
if [ -n "$CCCL_INC" ]; then
    export NVCC_PREPEND_FLAGS="-I$CCCL_INC"
fi
unset CPATH    # 避免和上一步冲突
export HF_ENDPOINT=https://hf-mirror.com

export SETUPTOOLS_USE_DISTUTILS=local
export DIST_TIMEOUT=43200
export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# These launchers use local model paths; ignore inherited ModelScope settings.
export VLLM_USE_MODELSCOPE=False
export VLLM_QWEN3_NEXT_GDN_BACKEND="${VLLM_QWEN3_NEXT_GDN_BACKEND:-auto}"

# ============================================================
# Distributed / vLLM / NCCL settings
# ============================================================

if [ -z "${HOST_IP:-}" ]; then
    HOST_IP=$(python3 - <<'PY'
import socket
import subprocess


def candidates():
    try:
        out = subprocess.check_output(
            ["hostname", "-I"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for ip in out.split():
            yield ip
    except Exception:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM):
            yield info[4][0]
    except Exception:
        pass

    # Last-resort route lookup. UDP connect does not send packets.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        yield sock.getsockname()[0]
    except Exception:
        pass
    finally:
        sock.close()


seen = set()
for ip in candidates():
    if ip in seen or ip.startswith("127.") or ":" in ip:
        continue
    seen.add(ip)
    sock = socket.socket(socket.AF_INET)
    try:
        sock.bind((ip, 0))
    except OSError:
        continue
    finally:
        sock.close()
    print(ip)
    break
else:
    print("127.0.0.1")
PY
)
else
    python3 - <<PY
import socket
ip = "${HOST_IP}"
sock = socket.socket(socket.AF_INET)
try:
    sock.bind((ip, 0))
except OSError as exc:
    raise SystemExit(f"[eval_vllm] ERROR: HOST_IP={ip} is not bindable on this node: {exc}")
finally:
    sock.close()
PY
fi

export HOST_IP

export MASTER_ADDR="$HOST_IP"
export VLLM_HOST_IP="$HOST_IP"
export VLLM_LOOPBACK_IP="$HOST_IP"
export RAY_NODE_IP_ADDRESS="$HOST_IP"

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_IBEXT_DISABLE=1
export NCCL_UCX_DISABLE=1
export NCCL_SHARP_DISABLE=1
export NCCL_NET=Socket
export NCCL_NET_PLUGIN=none

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

unset NCCL_SHM_DISABLE
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_OOB_NET_IFNAME="${NCCL_OOB_NET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_FAMILY=AF_INET

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ============================================================
# Args
# ============================================================

_BENCHMARKS="longvideobench videomme videomme-short videomme-medium videomme-long mlvu"
_MODELS="qwen3_5 qwen3_5_debug qwen3_5_moe qwen3_6"

if [[ "${1:-}" == *.json ]]; then
    CONFIG_FILE="$1"
else
    _ARG1_LOWER=$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')

    if echo "$_MODELS" | grep -qw "$_ARG1_LOWER"; then
        MODEL_FAMILY="$_ARG1_LOWER"
        shift
    else
        MODEL_FAMILY=qwen3_5
    fi

    _ARG1_LOWER=$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')

    if echo "$_BENCHMARKS" | grep -qw "$_ARG1_LOWER"; then
        BENCHMARK="$1"
        SCHEME="${2:?Usage: bash eval_vllm.sh [model] [benchmark] <scheme> [redi/uniform]}"
        SAMPLING="${3:-redi}"
    else
        BENCHMARK=longvideobench
        SCHEME="${1:?Usage: bash eval_vllm.sh [model] [benchmark] <scheme> [redi/uniform]}"
        SAMPLING="${2:-redi}"
    fi

    # 只取 gen_config.py 输出的最后一行，避免多行日志污染 CONFIG_FILE.
    # gen_config.py 会校验已有 config 是否和本次参数一致，不一致时自动覆盖。
    CONFIG_FILE=$(python "$FRAMEPACK_ROOT/configs/gen_config.py" "$BENCHMARK" "$SCHEME" "$SAMPLING" --model "$MODEL_FAMILY" | tail -n 1)
    echo "Generated config: $CONFIG_FILE"
fi

# ============================================================
# Validate config
# ============================================================

if [ -z "${CONFIG_FILE:-}" ]; then
    echo "[eval_vllm] ERROR: CONFIG_FILE is empty"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[eval_vllm] ERROR: CONFIG_FILE does not exist: $CONFIG_FILE"
    exit 1
fi

python "$FRAMEPACK_ROOT/scripts/check_local_paths.py" "$CONFIG_FILE"
export VLMEVAL_RUN_SCRIPT="${VLMEVAL_RUN_SCRIPT:-$FRAMEPACK_ROOT/run.py}"

if [[ "$CONFIG_FILE" != *.json ]]; then
    echo "[eval_vllm] ERROR: CONFIG_FILE should be a .json file, got: $CONFIG_FILE"
    exit 1
fi

case "${USE_VLLM:-1}" in
    1|true|TRUE|yes|YES|on|ON)
        USE_VLLM=1
        ;;
    *)
        USE_VLLM=0
        ;;
esac

if [ "$USE_VLLM" = "1" ]; then
    case "$VLLM_QWEN3_NEXT_GDN_BACKEND" in
        native|fla|0|false|FALSE|no|NO|off|OFF)
            if grep -q '"Qwen3.5-' "$CONFIG_FILE"; then
                python "$FRAMEPACK_ROOT/scripts/patch_vllm_qwen3_next_gdn.py"
            fi
            ;;
    esac
fi

CONFIG_NAME=$(basename "$CONFIG_FILE")
CONFIG_STEM=${CONFIG_NAME%.*}
WORK_DIR="./outputs/${CONFIG_STEM#config_}"
RUN_SCRIPT="${VLMEVAL_RUN_SCRIPT:-run.py}"

# ============================================================
# Debug logs
# ============================================================

RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="${EVAL_VLLM_LOG_ROOT:-$PWD/log}"
RUN_LOG_DIR="${EVAL_VLLM_LOG_DIR:-$LOG_ROOT/eval_vllm_${CONFIG_STEM}_${RUN_TS}_pid$$}"
mkdir -p "$RUN_LOG_DIR"
export EVAL_VLLM_LOG_DIR="$RUN_LOG_DIR"
export EVAL_VLLM_DEBUG="${EVAL_VLLM_DEBUG:-1}"
export DEBUG_VIDEO_PIPELINE="${DEBUG_VIDEO_PIPELINE:-1}"
export EVAL_VLLM_CONFIG_FILE="$CONFIG_FILE"
export EVAL_VLLM_WORK_DIR="$WORK_DIR"
export EVAL_VLLM_FAIL_FAST="${EVAL_VLLM_FAIL_FAST:-1}"
export VIDEO_INFER_FAIL_FAST="${VIDEO_INFER_FAIL_FAST:-$EVAL_VLLM_FAIL_FAST}"

# ============================================================
# GPU / DP / TP
# ============================================================

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    TOTAL_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
else
    TOTAL_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
fi

DP=${DP:-1}

if [ $((TOTAL_GPUS % DP)) -ne 0 ]; then
    echo "[eval_vllm] ERROR: TOTAL_GPUS=$TOTAL_GPUS not divisible by DP=$DP"
    exit 1
fi

TP_PER_RANK=$((TOTAL_GPUS / DP))
export EVAL_VLLM_TOTAL_GPUS="$TOTAL_GPUS"
export EVAL_VLLM_DP="$DP"
export EVAL_VLLM_TP_PER_RANK="$TP_PER_RANK"

# 端口绑定到真实 IPv4，不要 bind("", 0)
FREE_PORT=$(python3 - <<PY
import socket
ip = "${HOST_IP}"
s = socket.socket(socket.AF_INET)
s.bind((ip, 0))
print(s.getsockname()[1])
s.close()
PY
)

if [ -z "${VLLM_PORT_BASE:-}" ] && [ -z "${VLLM_PORT:-}" ]; then
    VLLM_PORT_BASE=$(python3 - <<PY
import random
import socket

ip = "${HOST_IP}"
dp = int("${DP}")


def can_bind(port):
    sock = socket.socket(socket.AF_INET)
    try:
        sock.bind((ip, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


for _ in range(512):
    base = random.randint(20000, 54000)
    # run.py assigns VLLM_PORT=base + local_rank * 100. Check each
    # outer DP rank's base port so concurrent eval jobs do not collide on
    # vLLM's TP worker rendezvous.
    if all(can_bind(base + rank * 100) for rank in range(dp)):
        print(base)
        break
else:
    raise SystemExit("[eval_vllm] ERROR: failed to find free VLLM_PORT_BASE")
PY
)
    export VLLM_PORT_BASE
fi

python3 - <<'PY'
import json
import os
from datetime import datetime
from pathlib import Path

log_dir = Path(os.environ['EVAL_VLLM_LOG_DIR'])
prefixes = (
    'CUDA', 'NCCL', 'VLLM', 'TORCH', 'MASTER', 'RANK', 'WORLD', 'LOCAL',
    'HOST', 'HF_ENDPOINT', 'LMUData', 'PYTORCH', 'EVAL_VLLM',
    'DEBUG_VIDEO_PIPELINE',
)
env = {
    k: v for k, v in sorted(os.environ.items())
    if k.startswith(prefixes) or k in {'USE_VLLM', 'DP'}
}
payload = {
    'time': datetime.now().astimezone().isoformat(timespec='seconds'),
    'cwd': os.getcwd(),
    'env': env,
}
(log_dir / 'eval_vllm.env.json').write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
    encoding='utf-8',
)
PY

echo "[eval_vllm] HOST_IP=$HOST_IP"
echo "[eval_vllm] use_vllm=$([ "$USE_VLLM" = "1" ] && echo True || echo False) | total_GPUs=$TOTAL_GPUS | DP=$DP | TP/rank=$TP_PER_RANK | port=$FREE_PORT | vllm_port_base=${VLLM_PORT_BASE:-${VLLM_PORT:-}} | gdn_backend=$VLLM_QWEN3_NEXT_GDN_BACKEND | fail_fast=$EVAL_VLLM_FAIL_FAST"
echo "[eval_vllm] config=$CONFIG_FILE"
echo "[eval_vllm] work_dir=$WORK_DIR"
echo "[eval_vllm] log_dir=$RUN_LOG_DIR"
echo "[eval_vllm] run_script=$RUN_SCRIPT"

# ============================================================
# Run
# ============================================================

USE_VLLM_ARGS=()
if [ "$USE_VLLM" = "1" ]; then
    USE_VLLM_ARGS+=(--use-vllm)
fi

case "${EVAL_VLLM_WATCHDOG:-1}" in
    1|true|TRUE|yes|YES|on|ON)
        export DP
        export USE_VLLM
        export STUCK_MINUTES="${STUCK_MINUTES:-30}"
        export CHECK_INTERVAL="${CHECK_INTERVAL:-120}"
        echo "[eval_vllm] watchdog=on | stuck_minutes=$STUCK_MINUTES | check_interval=$CHECK_INTERVAL"
        bash "$FRAMEPACK_ROOT/watchdog.sh" "$CONFIG_FILE" "$WORK_DIR" \
            2>&1 | tee -a "$RUN_LOG_DIR/watchdog.stdout.log"
        ;;
    *)
        echo "[eval_vllm] watchdog=off"
        torchrun \
            --nproc-per-node="$DP" \
            --master-addr="$HOST_IP" \
            --master-port="$FREE_PORT" \
            "$RUN_SCRIPT" \
            --config "$CONFIG_FILE" \
            --judge exact_matching \
            --reuse \
            "${USE_VLLM_ARGS[@]}" \
            --work-dir "$WORK_DIR" \
            2>&1 | tee -a "$RUN_LOG_DIR/torchrun.stdout.log"
        ;;
esac
