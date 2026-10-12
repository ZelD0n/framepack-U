#!/usr/bin/env python3
"""Check selected local links before launching an evaluation job."""

import json
import sys
from pathlib import Path


LOCAL = Path(__file__).resolve().parents[1] / 'local'
LOCAL_MODELS = (
    'Qwen3.5-35B-A3B', 'Qwen3.5-27B', 'Qwen3.5-9B', 'Qwen3.5-4B',
    'Qwen3.5-2B', 'Qwen3.6-27B', 'Qwen3-VL-4B-Instruct', 'Qwen3-VL-8B-Instruct',
)


def benchmark_name(name: str) -> str | None:
    if name.startswith('Video-MME'):
        return 'Video-MME'
    for prefix, canonical in (
        ('LongVideoBench', 'LongVideoBench'), ('MLVU', 'MLVU'),
        ('RTV-Bench', 'RTV-Bench'), ('StreamingBench', 'StreamingBench'),
        ('SVBench', 'SVBench'),
    ):
        if name.startswith(prefix):
            return canonical
    return None


def missing_resources(config: dict) -> list[Path]:
    missing = []
    for name, options in config.get('model', {}).items():
        for model in LOCAL_MODELS:
            if name == model or name.startswith(model + '-FramePack'):
                path = LOCAL / 'models' / model
                if not path.is_dir():
                    missing.append(path)
                break
        if options.get('frame_sampling_strategy') == 'redi':
            for key, default in (
                ('clip_scores_dir', LOCAL / 'clip_scores'),
                ('frame_features_dir', LOCAL / 'frame_feature'),
            ):
                path = Path(options.get(key) or default)
                if not path.is_dir():
                    missing.append(path)
    for name in config.get('data', {}):
        canonical = benchmark_name(name)
        if canonical:
            path = LOCAL / 'benchmarks' / canonical
            if not path.is_dir():
                missing.append(path)
    return list(dict.fromkeys(missing))


def main() -> None:
    with open(sys.argv[1], encoding='utf-8') as file:
        config = json.load(file)
    missing = missing_resources(config)
    if missing:
        message = '\n'.join(f'  {path}' for path in missing)
        raise SystemExit(
            f'Missing local resources:\n{message}\n'
            'Run scripts/setup_local_paths.py with the matching model, benchmark, '
            'CLIP score, and feature directories.'
        )


if __name__ == '__main__':
    main()
