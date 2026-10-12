from __future__ import annotations

import datetime as _datetime
import json
import os
from pathlib import Path
import threading
import time
import traceback

_START = time.monotonic()
_LOCK = threading.Lock()


def eval_vllm_debug_enabled() -> bool:
    flag = os.environ.get('EVAL_VLLM_DEBUG')
    if flag is None:
        flag = os.environ.get('DEBUG_VIDEO_PIPELINE')
    if flag is None:
        return bool(os.environ.get('EVAL_VLLM_LOG_DIR'))
    return flag.lower() not in ('0', 'false', 'no', 'off')


def eval_vllm_log_dir(create: bool = True) -> Path | None:
    log_dir = os.environ.get('EVAL_VLLM_LOG_DIR')
    if not log_dir:
        if not eval_vllm_debug_enabled():
            return None
        log_dir = str(Path(__file__).resolve().parents[1] / 'log' / 'eval_vllm_manual')
    path = Path(log_dir)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_value(value, depth: int = 0):
    if depth > 3:
        return repr(value)[:256]
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > 1000:
            return value[:1000] + '...<truncated>'
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        out = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= 32:
                out['...'] = f'{len(value) - 32} more keys'
                break
            out[str(k)] = _safe_value(v, depth + 1)
        return out
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        out = [_safe_value(v, depth + 1) for v in seq[:32]]
        if len(seq) > 32:
            out.append(f'... {len(seq) - 32} more')
        return out
    shape = getattr(value, 'shape', None)
    if shape is not None:
        return {
            'type': type(value).__name__,
            'shape': tuple(shape),
            'dtype': str(getattr(value, 'dtype', '')),
            'device': str(getattr(value, 'device', '')),
        }
    return repr(value)[:1000]


def _rank_fields() -> dict:
    return {
        'rank': os.environ.get('EVAL_VLLM_OUTER_RANK') or os.environ.get('RANK'),
        'local_rank': os.environ.get('EVAL_VLLM_OUTER_LOCAL_RANK') or os.environ.get('LOCAL_RANK'),
        'world_size': os.environ.get('EVAL_VLLM_OUTER_WORLD_SIZE') or os.environ.get('WORLD_SIZE'),
        'local_world_size': (
            os.environ.get('EVAL_VLLM_OUTER_LOCAL_WORLD_SIZE') or os.environ.get('LOCAL_WORLD_SIZE')
        ),
    }


def eval_vllm_log(event: str, **fields) -> None:
    if not eval_vllm_debug_enabled():
        return
    try:
        log_dir = eval_vllm_log_dir(create=True)
        if log_dir is None:
            return

        ranks = _rank_fields()
        rank = ranks.get('rank') or 'na'
        local_rank = ranks.get('local_rank') or 'na'
        log_file = log_dir / f'rank{rank}_local{local_rank}_pid{os.getpid()}.jsonl'
        record = {
            'time': _datetime.datetime.now().astimezone().isoformat(timespec='milliseconds'),
            'unix_time': time.time(),
            'elapsed_s': round(time.monotonic() - _START, 3),
            'event': event,
            'pid': os.getpid(),
            'ppid': os.getppid(),
            **ranks,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'vllm_port': os.environ.get('VLLM_PORT'),
            'fields': {k: _safe_value(v) for k, v in fields.items()},
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with _LOCK:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
                f.flush()
    except Exception:
        return


def eval_vllm_log_exception(event: str, exc: BaseException, **fields) -> None:
    eval_vllm_log(
        event,
        exception_type=type(exc).__name__,
        exception=str(exc),
        traceback=''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        **fields,
    )
