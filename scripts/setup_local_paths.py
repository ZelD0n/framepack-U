#!/usr/bin/env python3
"""Create repository-local links to model weights and benchmark resources."""

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'local'
MODELS = {
    'Qwen3.5-2B', 'Qwen3.5-4B', 'Qwen3.5-9B', 'Qwen3.5-27B',
    'Qwen3.5-35B-A3B', 'Qwen3.6-27B',
    'Qwen3-VL-4B-Instruct', 'Qwen3-VL-8B-Instruct',
}
BENCHMARKS = {'LongVideoBench', 'Video-MME', 'MLVU', 'RTV-Bench', 'StreamingBench', 'SVBench'}


def parse_link(value: str, valid_names: set[str]) -> tuple[str, Path]:
    name, separator, target = value.partition('=')
    if not separator or name not in valid_names or not target:
        raise argparse.ArgumentTypeError(
            f'Expected NAME=/path; supported names: {", ".join(sorted(valid_names))}'
        )
    return name, Path(target).expanduser().resolve(strict=True)


def link_directory(link: Path, target: Path) -> None:
    if not target.is_dir():
        raise ValueError(f'Not a directory: {target}')
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target:
            print(f'Already linked: {link} -> {target}')
            return
        raise FileExistsError(f'{link} already points to {link.resolve()}; remove it explicitly to change the target')
    if link.exists():
        raise FileExistsError(f'{link} already exists; move it explicitly before creating a link')
    link.symlink_to(target, target_is_directory=True)
    print(f'Linked: {link} -> {target}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', default=[], metavar='NAME=DIR',
                        help='Model name and existing weights directory; may be repeated')
    parser.add_argument('--benchmark', action='append', default=[], metavar='NAME=DIR',
                        help='Benchmark name and existing dataset directory; may be repeated')
    parser.add_argument('--clip-scores', type=Path, metavar='DIR')
    parser.add_argument('--frame-features', type=Path, metavar='DIR')
    args = parser.parse_args()
    if not (args.model or args.benchmark or args.clip_scores or args.frame_features):
        parser.error('Provide at least one resource to link')

    (LOCAL / 'cache').mkdir(parents=True, exist_ok=True)
    for value in args.model:
        name, target = parse_link(value, MODELS)
        link_directory(LOCAL / 'models' / name, target)
    for value in args.benchmark:
        name, target = parse_link(value, BENCHMARKS)
        link_directory(LOCAL / 'benchmarks' / name, target)
    for target, name in ((args.clip_scores, 'clip_scores'), (args.frame_features, 'frame_feature')):
        if target is not None:
            link_directory(LOCAL / name, target.expanduser().resolve(strict=True))


if __name__ == '__main__':
    main()
