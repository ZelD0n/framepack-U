#!/usr/bin/env bash
# watchdog.sh: Monitor vLLM inference job and auto-restart if progress stalls.
# Usage: bash watchdog.sh <config_file> [work_dir]
#
# Completion signal: the expected number of result JSON files appear in the
# latest T* directory (one per model×dataset pair in the config).
# The watchdog only exits (allowing the next task to start) after the JSON is
# confirmed to exist.

set -uo pipefail   # no -e: we handle errors manually

CONFIG_FILE=${1:?Usage: bash watchdog.sh <config_file> [work_dir]}
CONFIG_NAME=$(basename "$CONFIG_FILE")
CONFIG_STEM=${CONFIG_NAME%.*}
WORK_DIR=${2:-"./outputs/${CONFIG_STEM#config_}"}
RUN_SCRIPT=${VLMEVAL_RUN_SCRIPT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run.py}

STUCK_MINUTES=${STUCK_MINUTES:-60}   # minutes without pkl update -> consider stuck
if [ "$STUCK_MINUTES" -lt 60 ]; then
    echo "[watchdog] STUCK_MINUTES=$STUCK_MINUTES is below 60; using 60 minutes."
    STUCK_MINUTES=60
fi
CHECK_INTERVAL=${CHECK_INTERVAL:-120} # seconds between checks
DP=${DP:-${NPROC:-1}}                # outer data-parallel workers
USE_VLLM=${USE_VLLM:-1}
WATCHDOG_RESTART_ON_FAILURE=${WATCHDOG_RESTART_ON_FAILURE:-1}

if grep -Eq '"frame_sampling_strategy"[[:space:]]*:[[:space:]]*"redi"' "$CONFIG_FILE"; then
    export EVAL_VLLM_FAIL_FAST=1
    export VIDEO_INFER_FAIL_FAST=1
    WATCHDOG_RESTART_ON_FAILURE=0
fi

case "${EVAL_VLLM_FAIL_FAST:-${VIDEO_INFER_FAIL_FAST:-0}}" in
    1|true|TRUE|yes|YES|on|ON)
        export EVAL_VLLM_FAIL_FAST=1
        export VIDEO_INFER_FAIL_FAST=1
        WATCHDOG_RESTART_ON_FAILURE=0
        ;;
esac

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export SETUPTOOLS_USE_DISTUTILS="${SETUPTOOLS_USE_DISTUTILS:-local}"
export DIST_TIMEOUT="${DIST_TIMEOUT:-43200}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
# These launchers use local model paths; ignore inherited ModelScope settings.
export VLLM_USE_MODELSCOPE=False
export VLLM_QWEN3_NEXT_GDN_BACKEND="${VLLM_QWEN3_NEXT_GDN_BACKEND:-auto}"

export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_IBEXT_DISABLE="${NCCL_IBEXT_DISABLE:-1}"
export NCCL_UCX_DISABLE="${NCCL_UCX_DISABLE:-1}"
export NCCL_SHARP_DISABLE="${NCCL_SHARP_DISABLE:-1}"
export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_OOB_NET_IFNAME="${NCCL_OOB_NET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"

if [ -z "${HOST_IP:-}" ]; then
    HOST_IP=$(python3 - <<'PY'
import socket
import subprocess

def candidates():
    try:
        out = subprocess.check_output(["hostname", "-I"], stderr=subprocess.DEVNULL, text=True)
        for ip in out.split():
            yield ip
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM):
            yield info[4][0]
    except Exception:
        pass

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
fi
export HOST_IP
export MASTER_ADDR="$HOST_IP"
export VLLM_HOST_IP="$HOST_IP"
export VLLM_LOOPBACK_IP="$HOST_IP"
export RAY_NODE_IP_ADDRESS="$HOST_IP"

if [ "$USE_VLLM" = "1" ]; then
    case "$VLLM_QWEN3_NEXT_GDN_BACKEND" in
        native|fla|0|false|FALSE|no|NO|off|OFF)
            if grep -q '"Qwen3.5-' "$CONFIG_FILE"; then
                python "$(dirname "${BASH_SOURCE[0]}")/scripts/patch_vllm_qwen3_next_gdn.py"
            fi
            ;;
    esac
fi

JOB_PID=""
JOB_PGID=""
JOB_START_TIME=0

start_job() {
    local free_port
    local vllm_port_base
    free_port=$(python3 - <<PY
import socket
ip = "${HOST_IP}"
s = socket.socket(socket.AF_INET)
s.bind((ip, 0))
print(s.getsockname()[1])
s.close()
PY
)
    vllm_port_base=$(python3 - <<PY
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
    if all(can_bind(base + rank * 100) for rank in range(dp)):
        print(base)
        break
else:
    raise SystemExit("[watchdog] ERROR: failed to find free VLLM_PORT_BASE")
PY
)
    export VLLM_PORT_BASE="$vllm_port_base"
    # 在启动前记录时间，避免 job 写 pkl 比 JOB_START_TIME 赋值更快的竞态
    JOB_START_TIME=$(date +%s)
    if [ "$USE_VLLM" = "1" ]; then
        echo "[watchdog] Starting vLLM job (port=$free_port, vllm_port_base=$VLLM_PORT_BASE, DP=$DP, stuck=${STUCK_MINUTES}m)..."
    else
        echo "[watchdog] Starting job (port=$free_port, DP=$DP, stuck=${STUCK_MINUTES}m, USE_VLLM=0)..."
    fi
    local use_vllm_args=()
    if [ "$USE_VLLM" = "1" ]; then
        use_vllm_args+=(--use-vllm)
    fi
    setsid torchrun \
        --nproc-per-node="$DP" \
        --master-addr="$HOST_IP" \
        --master-port="$free_port" \
        "$RUN_SCRIPT" \
        --config "$CONFIG_FILE" \
        --judge exact_matching \
        --reuse \
        "${use_vllm_args[@]}" \
        --work-dir "$WORK_DIR" &
    JOB_PID=$!
    JOB_PGID=$JOB_PID
    echo "[watchdog] Job PID=$JOB_PID PGID=$JOB_PGID started at $(date '+%H:%M:%S')"
}

kill_job() {
    [ -z "$JOB_PGID" ] && return 0
    echo "[watchdog] Killing job (PGID=$JOB_PGID)..."
    kill -TERM -- -"$JOB_PGID" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        if ! ps -o pid= -g "$JOB_PGID" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    if ps -o pid= -g "$JOB_PGID" >/dev/null 2>&1; then
        echo "[watchdog] Job still alive after TERM; sending KILL to PGID=$JOB_PGID..."
        kill -KILL -- -"$JOB_PGID" 2>/dev/null || true
    fi
    # Reap zombie
    wait "$JOB_PID" 2>/dev/null || true
    JOB_PID=""
    JOB_PGID=""
    echo "[watchdog] Job killed."
}

find_pkl_dir() {
    python3 - "$CONFIG_FILE" "$WORK_DIR" <<'PY'
import json
import os
import sys
from pathlib import Path

config_file = Path(sys.argv[1])
work_dir = Path(sys.argv[2])

try:
    with config_file.open() as f:
        cfg = json.load(f)
except Exception:
    cfg = {}

models = list(cfg.get("model", {}).keys())
if not work_dir.is_dir() or not models:
    sys.exit(0)

model_dirs = []
children = [p for p in work_dir.iterdir() if p.is_dir()]

for model in models:
    exact = work_dir / model
    if exact.is_dir():
        model_dirs.append(exact)
        continue

    # Compatibility for older interrupted runs that normalized model names differently.
    lower_model = model.lower()
    matches = [p for p in children if p.name.lower() == lower_model]
    model_dirs.extend(matches)

candidates = []
seen = set()
for model_dir in model_dirs:
    try:
        for t_dir in model_dir.iterdir():
            if not t_dir.is_dir() or not t_dir.name.startswith("T"):
                continue
            key = str(t_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(t_dir)
    except OSError:
        pass

if candidates:
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(str(latest) + os.sep)
PY
}

# How many result JSON files to expect: one per model×dataset pair in config.
# Excludes status.json; matches files produced by run.py (e.g. *_rating.json).
expected_json_count() {
    python3 - <<EOF 2>/dev/null
import json, sys
try:
    with open("$CONFIG_FILE") as f:
        cfg = json.load(f)
    models   = list(cfg.get("model", {}).keys())
    datasets = list(cfg.get("data",  {}).keys())
    print(len(models) * len(datasets))
except Exception:
    print(0)
EOF
}

is_done() {
    local pkl_dir
    pkl_dir=$(find_pkl_dir)
    [ -z "$pkl_dir" ] && return 1

    local expected
    expected=$(expected_json_count)
    [ "${expected:-0}" -eq 0 ] && return 1

    # Count result JSON files in the latest T* dir (skip status.json)
    local actual
    actual=$(find "$pkl_dir" -maxdepth 1 -name "*.json" ! -name "status.json" 2>/dev/null | wc -l)
    echo "[watchdog] JSON check: found=$actual expected=$expected (dir=$pkl_dir)"
    [ "$actual" -ge "$expected" ]
}

check_stuck() {
    local pkl_dir
    pkl_dir=$(find_pkl_dir)
    if [ -z "$pkl_dir" ]; then
        local now age
        now=$(date +%s)
        age=$((now - JOB_START_TIME))
        if [ "$age" -ge $((STUCK_MINUTES * 60)) ]; then
            echo "[watchdog] STUCK: no T* pkl dir after ${STUCK_MINUTES}m"
            return 1
        fi
        echo "[watchdog] No pkl dir found yet, waiting... ($((age / 60))m/${STUCK_MINUTES}m)"
        return 0
    fi

    python3 - "$pkl_dir" "$JOB_START_TIME" "$STUCK_MINUTES" <<'EOF'
import sys, os, pickle, time, glob

pkl_dir      = sys.argv[1]
job_start    = float(sys.argv[2])
stuck_min    = int(sys.argv[3])

dp = int(os.environ.get("DP", "1"))
all_pkls = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))

def parse_rank_nproc(path):
    prefix = os.path.basename(path).split("_")[0]
    if len(prefix) >= 2 and prefix[:-1].isdigit() and prefix[-1].isdigit():
        return int(prefix[:-1]), int(prefix[-1])
    return None, None

pkls = [p for p in all_pkls if parse_rank_nproc(p)[1] == dp]
if not pkls:
    pkls = all_pkls
if not pkls:
    now = time.time()
    job_age_min = (now - job_start) / 60
    if job_age_min > stuck_min:
        print(f"[watchdog] STUCK: no pkl files in {pkl_dir} after {job_age_min:.0f}m")
        sys.exit(1)
    print(f"[watchdog] WAIT:  no pkl files yet, job only {job_age_min:.1f}m old")
    sys.exit(0)

now = time.time()
job_age_min = (now - job_start) / 60

# Load every pkl and collect sample counts + max index
all_data = {}
for path in pkls:
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            all_data[path] = data
    except Exception:
        all_data[path] = {}

# Global max sample index (tells us approximate dataset size)
global_max_idx = -1
for data in all_data.values():
    keys = [k for k in data.keys() if isinstance(k, int)]
    if keys:
        global_max_idx = max(global_max_idx, max(keys))

def get_expected(fname):
    """返回该 rank 预期处理的样本数，无法解析时返回 -1。"""
    rank, nproc = parse_rank_nproc(fname)
    if nproc and nproc > 0 and global_max_idx >= 0:
        return len(range(rank, global_max_idx + 1, nproc))
    return -1

any_stuck = False
all_reused_and_done = True   # 用于判断是否所有 rank 都是真正完成的复用

for path in pkls:
    fname    = os.path.basename(path)
    mtime    = os.path.getmtime(path)
    age_min  = (now - mtime) / 60
    data     = all_data.get(path, {})
    count    = len(data)
    expected = get_expected(fname)
    exp_str  = str(expected) if expected > 0 else "?"

    # 判断该 rank 是否真正跑完了所有样本
    rank_complete = (expected > 0 and count >= expected)

    if mtime < job_start:
        if rank_complete:
            # 上次运行已完成，合法复用
            print(f"[watchdog] DONE:  {fname} completed in prior run (reused, {count}/{exp_str} samples)")
            continue
        else:
            # ★ 核心修复：pkl 是旧文件但未完成
            # → 当前 job 中该 rank 卡死（从未写入新数据）
            all_reused_and_done = False
            if job_age_min > stuck_min:
                print(f"[watchdog] STUCK: {fname} not updated since job start "
                      f"({age_min:.0f}m ago), {count}/{exp_str} samples — rank stuck at startup")
                any_stuck = True
            else:
                print(f"[watchdog] WAIT:  {fname} old pkl, job only {job_age_min:.1f}m old, waiting...")
            continue

    # mtime >= job_start：当前 job 写过该 pkl
    all_reused_and_done = False

    if age_min < stuck_min:
        print(f"[watchdog] OK:    {fname} last updated {age_min:.1f}m ago ({count}/{exp_str} samples done)")
        continue

    # 超时未更新——看是否真的跑完
    if rank_complete:
        print(f"[watchdog] DONE:  {fname} finished all {count}/{expected} samples")
        # 推理已完成但 evaluation 还未产出 JSON，若 job 已运行超时则视为 eval 卡死
        if job_age_min > stuck_min:
            print(f"[watchdog] EVAL_STUCK: inference done but no result JSON after {job_age_min:.0f}m — eval may be hung")
            any_stuck = True
    else:
        print(f"[watchdog] STUCK: {fname} last updated {age_min:.0f}m ago ({count}/{exp_str} samples done)")
        any_stuck = True

if all_reused_and_done:
    if job_age_min > stuck_min:
        print(f"[watchdog] EVAL_STUCK: all {len(pkls)} ranks done (reused) but no result JSON after {job_age_min:.0f}m — eval may be hung")
        any_stuck = True
    else:
        print(f"[watchdog] All {len(pkls)} ranks completed (reused). Waiting for evaluation JSON ({job_age_min:.1f}m elapsed)...")

sys.exit(1 if any_stuck else 0)
EOF
    return $?
}

report_progress() {
    local pkl_dir
    pkl_dir=$(find_pkl_dir)
    if [ -z "$pkl_dir" ]; then
        echo "[watchdog] No T* directory found yet, nothing to report."
        return
    fi
    echo "[watchdog] ── Progress report (dir: $pkl_dir) ──"
    python3 - "$pkl_dir" "$STUCK_MINUTES" <<'EOF'
import sys, os, pickle, time, glob

pkl_dir  = sys.argv[1]
stuck_min = int(sys.argv[2])
dp = int(os.environ.get("DP", "1"))
all_pkls = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))

def parse_rank_nproc(path):
    prefix = os.path.basename(path).split("_")[0]
    if len(prefix) >= 2 and prefix[:-1].isdigit() and prefix[-1].isdigit():
        return int(prefix[:-1]), int(prefix[-1])
    return -1, -1

pkls = [p for p in all_pkls if parse_rank_nproc(p)[1] == dp]
if not pkls:
    pkls = all_pkls
if not pkls:
    print("[watchdog] No pkl files found.")
    sys.exit(0)

now = time.time()
total_done = 0
for path in pkls:
    fname = os.path.basename(path)
    # filename: {rank}{nproc}_...pkl  e.g. 08_Model_Dataset.pkl
    rank, nproc = parse_rank_nproc(fname)

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        done = len(data) if isinstance(data, dict) else len(data)
    except Exception as e:
        print(f"  rank {rank:2d}: ERROR reading pkl ({e})")
        continue

    mtime = os.path.getmtime(path)
    age_min = (now - mtime) / 60
    total_done += done
    status = "STUCK" if age_min >= stuck_min else "ok   "
    print(f"  rank {rank:2d}: {done:4d} samples done  |  last update {age_min:5.1f}m ago  [{status}]")

print(f"  ── total done: {total_done} samples across {len(pkls)} ranks ──")
EOF
    echo "[watchdog] ────────────────────────────────────────────"
}

cleanup_on_exit() {
    local code=$?
    if [ -n "$JOB_PGID" ]; then
        echo "[watchdog] Exiting, cleaning up running job..."
        kill_job
    fi
    return "$code"
}

# Trap Ctrl-C / SIGTERM to clean up before exiting
trap 'echo "[watchdog] Interrupted, killing job..."; kill_job; exit 130' INT
trap 'echo "[watchdog] Terminated, killing job..."; kill_job; exit 143' TERM
trap cleanup_on_exit EXIT

# ── Main loop ────────────────────────────────────────────────────────────────
report_progress
start_job

while true; do
    sleep $CHECK_INTERVAL

    # ── Check if job process has exited ──────────────────────────────────────
    if [ -n "$JOB_PID" ] && ! kill -0 "$JOB_PID" 2>/dev/null; then
        local_exit_code=0
        wait "$JOB_PID" 2>/dev/null && local_exit_code=$? || local_exit_code=$?
        echo "[watchdog] Job exited (PID=$JOB_PID, exit_code=$local_exit_code)."
        JOB_PID=""
        JOB_PGID=""

        if [ "$local_exit_code" -eq 0 ]; then
            echo "[watchdog] Job exited cleanly (code 0). Task complete, exiting watchdog."
            exit 0
        fi

        if [ "${EVAL_VLLM_FAIL_FAST:-0}" = 1 ]; then
            echo "[watchdog] Job failed (exit_code=$local_exit_code). Exiting in fail-fast mode."
            exit "$local_exit_code"
        fi

        # Non-zero exit: check for result JSON before deciding to restart
        if is_done; then
            echo "[watchdog] Result JSON confirmed. Task complete, exiting watchdog."
            exit 0
        elif [ "$WATCHDOG_RESTART_ON_FAILURE" = "0" ]; then
            echo "[watchdog] Job failed (exit_code=$local_exit_code). restart_on_failure=0, exiting watchdog."
            exit "$local_exit_code"
        else
            echo "[watchdog] Job failed (exit_code=$local_exit_code) and result JSON not found. Restarting same task..."
            sleep 5
            start_job
            continue
        fi
    fi

    # ── Periodic checks while job is running ─────────────────────────────────

    # Safety: if JSON already appeared (e.g. job is wrapping up), wait for
    # the job to exit cleanly rather than killing it.
    if is_done; then
        echo "[watchdog] Result JSON found. Waiting for job to exit cleanly..."
        wait "$JOB_PID" 2>/dev/null || true
        JOB_PID=""
        JOB_PGID=""
        echo "[watchdog] Task complete, exiting watchdog."
        exit 0
    fi

    echo "[watchdog] $(date '+%H:%M:%S') Checking ranks..."
    if ! check_stuck; then
        echo "[watchdog] Stuck rank detected! Restarting same task (NOT advancing to next)..."
        report_progress
        kill_job
        sleep 5
        start_job
    fi
done
