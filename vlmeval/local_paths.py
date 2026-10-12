"""Repository-local entry points for externally stored models and data."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = REPO_ROOT / 'local'


def model_path(name: str) -> str:
    return str(LOCAL_ROOT / 'models' / name)


def benchmark_path(name: str) -> str:
    path = LOCAL_ROOT / 'benchmarks' / name
    if not path.is_dir():
        raise FileNotFoundError(
            f'Benchmark directory is missing: {path}. '
            f'Create the local link with scripts/setup_local_paths.py.'
        )
    return str(path)


def asset_path(name: str) -> str:
    return str(LOCAL_ROOT / name)
