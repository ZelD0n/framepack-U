#!/usr/bin/env python3
"""Repair RTV-Bench offline CLIP/frame caches for videos that decoded to 0 frames.

This script is intentionally narrow:
1. Find RTV frame-feature manifest entries with ``num_frames == 0`` while the
   source container reports frames.
2. Transcode only those source videos if OpenCV cannot read the first frame.
3. Recompute only those videos' frame features.
4. Recompute only those videos' CLIP-score rows and replace the corresponding
   rows in the existing allframes JSON files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import torch

from scripts.transcode_rtvbench_av1 import find_ffmpeg, transcode_one

from tools.save_clip_score import (
    build_extraction_dataset,
    collect_video_roots,
    encode_video_frames as encode_clip_score_frames,
    get_frame_count,
    load_clip_components,
    normalize_clip_query,
    relative_video_path_from_roots,
    source_video_for_item,
)
from tools.save_frame_feature import (
    build_manifest,
    collect_data_roots,
    rank_manifest_dir,
    save_video_feature,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', default='RTV-Bench')
    parser.add_argument('--rtv_root', default='/gaoshuo-sh/benchmarks/RTV-Bench')
    parser.add_argument('--frame_save_root', default='/gaoshuo-sh/frame_feature')
    parser.add_argument('--clip_save_root', default='clip_scores')
    parser.add_argument('--clip_model', default='/gaoshuo-sh/models/clip-vit-base-patch32')
    parser.add_argument('--ffmpeg', default=None)
    parser.add_argument('--crf', type=int, default=18)
    parser.add_argument('--preset', default='medium')
    parser.add_argument('--feature_batch_size', type=int, default=128)
    parser.add_argument('--clip_batch_size', type=int, default=256)
    parser.add_argument('--preprocess_backend', choices=('fast', 'processor'), default='fast')
    parser.add_argument('--decode_backend', choices=('cpu', 'gpu', 'auto'), default='cpu')
    parser.add_argument('--save_dtype', choices=('float16', 'float32', 'bfloat16'), default='float16')
    parser.add_argument('--sample_fps', type=float, default=0.0)
    parser.add_argument('--device', default=None)
    parser.add_argument('--skip_transcode', action='store_true')
    parser.add_argument('--skip_features', action='store_true')
    parser.add_argument('--skip_clip_scores', action='store_true')
    parser.add_argument('--backup_json', action='store_true')
    parser.add_argument(
        '--targets',
        nargs='*',
        default=None,
        help='Optional RTV relative video paths. Defaults to zero-frame manifest entries.',
    )
    return parser.parse_args()


def is_cv2_readable(path: Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        ok, frame = cap.read()
        return bool(ok and frame is not None)
    finally:
        cap.release()


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def dump_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def backup_file(path: Path) -> None:
    if not path.exists():
        return
    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup = path.with_name(path.name + f'.bak_repair_{stamp}')
    shutil.copy2(path, backup)
    print(f'Backed up {path} -> {backup}', flush=True)


def find_target_entries(manifest: dict[str, Any], targets: list[str] | None) -> list[dict[str, Any]]:
    videos = manifest.get('videos') or []
    if targets:
        wanted = {str(t).replace('\\', '/') for t in targets}
        return [entry for entry in videos if str(entry.get('video_path', '')).replace('\\', '/') in wanted]
    return [
        entry
        for entry in videos
        if int(entry.get('num_frames') or 0) == 0 and int(entry.get('source_frame_count') or 0) > 0
    ]


def find_clip_empty_target_entries(
    *,
    manifest: dict[str, Any],
    clip_dir: Path,
) -> list[dict[str, Any]]:
    frames_path = clip_dir / 'frames.json'
    video_paths_path = clip_dir / 'video_paths.json'
    if not frames_path.exists() or not video_paths_path.exists():
        return []

    try:
        import ijson
    except Exception as exc:
        raise RuntimeError('ijson is required to scan large clip-score JSON files') from exc

    video_paths = load_json(video_paths_path)
    empty_video_paths: set[str] = set()
    with open(frames_path, 'rb') as f:
        for row_i, frames in enumerate(ijson.items(f, 'item')):
            if len(frames) == 0 and row_i < len(video_paths):
                empty_video_paths.add(str(video_paths[row_i]).replace('\\', '/'))

    if not empty_video_paths:
        return []

    videos = manifest.get('videos') or []
    entries = [
        entry
        for entry in videos
        if str(entry.get('video_path', '')).replace('\\', '/') in empty_video_paths
    ]
    return entries


def replace_feature_manifest_entries(
    *,
    frame_dir: Path,
    manifest: dict[str, Any],
    new_entries: list[dict[str, Any]],
) -> None:
    by_source = {str(entry['source_video_path']): entry for entry in new_entries}
    videos = [
        by_source.get(str(entry.get('source_video_path')), entry)
        for entry in manifest.get('videos', [])
    ]
    new_manifest = build_manifest(
        benchmark=manifest.get('benchmark', 'RTV-Bench'),
        save_dir=frame_dir,
        feature_subdir=manifest.get('feature_subdir', 'features'),
        clip_model=manifest.get('clip_model'),
        sample_fps=float(manifest.get('sample_fps', 0.0)),
        save_dtype=manifest.get('feature_dtype', 'float16'),
        batch_size=int(manifest.get('batch_size', 0) or 0),
        world_size=int(manifest.get('world_size', 1) or 1),
        videos=videos,
        failed_videos=manifest.get('failed_videos', []),
        data_roots=manifest.get('data_roots', []),
    )
    write_json_atomic(frame_dir / 'manifest.json', new_manifest)
    print(f'Updated {frame_dir / "manifest.json"}', flush=True)

    rank_dir = rank_manifest_dir(frame_dir)
    if not rank_dir.exists():
        return
    for rank_path in sorted(rank_dir.glob('rank_*.json')):
        rank_manifest = load_json(rank_path)
        rank_videos = rank_manifest.get('videos', [])
        touched = any(str(entry.get('source_video_path')) in by_source for entry in rank_videos)
        if not touched:
            continue
        rank_videos = [
            by_source.get(str(entry.get('source_video_path')), entry)
            for entry in rank_videos
        ]
        rebuilt = build_manifest(
            benchmark=rank_manifest.get('benchmark', manifest.get('benchmark', 'RTV-Bench')),
            save_dir=frame_dir,
            feature_subdir=rank_manifest.get('feature_subdir', manifest.get('feature_subdir', 'features')),
            clip_model=rank_manifest.get('clip_model', manifest.get('clip_model')),
            sample_fps=float(rank_manifest.get('sample_fps', manifest.get('sample_fps', 0.0))),
            save_dtype=rank_manifest.get('feature_dtype', manifest.get('feature_dtype', 'float16')),
            batch_size=int(rank_manifest.get('batch_size', manifest.get('batch_size', 0)) or 0),
            world_size=int(rank_manifest.get('world_size', manifest.get('world_size', 1)) or 1),
            videos=rank_videos,
            failed_videos=rank_manifest.get('failed_videos', []),
            data_roots=rank_manifest.get('data_roots', manifest.get('data_roots', [])),
        )
        rebuilt['rank'] = rank_manifest.get('rank')
        write_json_atomic(rank_path, rebuilt)
        print(f'Updated {rank_path}', flush=True)


def build_sample_groups(dataset, target_sources: set[str]) -> dict[str, list[tuple[int, str, int]]]:
    groups: dict[str, list[tuple[int, str, int]]] = {}
    for sample_idx in range(len(dataset)):
        raw_message = dataset.build_prompt(sample_idx, video_llm=True)
        video_items = [item for item in raw_message if item['type'] == 'video']
        if len(video_items) != 1:
            continue
        source = source_video_for_item(video_items[0])
        if source not in target_sources:
            continue
        query = normalize_clip_query(' '.join(str(item['value']) for item in raw_message if item['type'] == 'text') or '')
        dataset_index = int(dataset.data.iloc[sample_idx]['index'])
        groups.setdefault(source, []).append((sample_idx, query, dataset_index))
    return groups


def repair_clip_scores(
    *,
    args: argparse.Namespace,
    dataset,
    target_sources: set[str],
    clip_processor,
    clip_model,
    device: torch.device,
) -> None:
    clip_dir = Path(args.clip_save_root) / args.benchmark / 'allframes'
    required = ['scores.json', 'frames.json', 'queries.json', 'video_paths.json', 'sample_indices.json']
    missing = [name for name in required if not (clip_dir / name).exists()]
    if missing:
        print(f'Skipping clip-score repair; missing files under {clip_dir}: {missing}', flush=True)
        return

    groups = build_sample_groups(dataset, target_sources)
    if not groups:
        print('No matching samples found for clip-score repair.', flush=True)
        return

    print(f'Loading existing clip-score JSON from {clip_dir}', flush=True)
    data = {name: load_json(clip_dir / name) for name in required}
    index_to_pos = {int(idx): pos for pos, idx in enumerate(data['sample_indices.json'])}
    video_roots = collect_video_roots(dataset)

    if args.backup_json:
        for name in required:
            backup_file(clip_dir / name)

    for video_i, (source, group) in enumerate(groups.items(), start=1):
        print(f'[{video_i}/{len(groups)}] Recomputing clip scores for {source} ({len(group)} samples)', flush=True)
        img_features, frame_indices = encode_clip_score_frames(
            video_path=source,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=device,
            batch_size=args.clip_batch_size,
            sample_fps=args.sample_fps,
            preprocess_backend=args.preprocess_backend,
            decode_backend=args.decode_backend,
        )
        img_features_on_device = img_features.to(device)
        if device.type == 'cuda':
            img_features_on_device = img_features_on_device.half()
        queries = [query for _, query, _ in group]
        scores_by_query = []
        from tools.save_clip_score import score_queries

        scores_by_query = score_queries(img_features_on_device, queries, clip_model, clip_processor, device)
        rel_path = relative_video_path_from_roots(source, video_roots)
        for (_, query, dataset_index), scores in zip(group, scores_by_query):
            pos = index_to_pos.get(int(dataset_index))
            if pos is None:
                raise KeyError(f'dataset_index {dataset_index} not found in {clip_dir / "sample_indices.json"}')
            data['scores.json'][pos] = [float(score) for score in scores]
            data['frames.json'][pos] = frame_indices
            data['queries.json'][pos] = query
            data['video_paths.json'][pos] = rel_path

    for name in required:
        dump_json_atomic(clip_dir / name, data[name])
        print(f'Updated {clip_dir / name}', flush=True)

    metadata_path = clip_dir / 'metadata.json'
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        metadata['last_repair'] = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'target_sources': sorted(target_sources),
            'script': 'scripts/repair_rtv_offline_cache.py',
        }
        dump_json_atomic(metadata_path, metadata)
        print(f'Updated {metadata_path}', flush=True)


def main() -> None:
    args = parse_args()
    frame_dir = Path(args.frame_save_root) / args.benchmark
    manifest_path = frame_dir / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Missing frame-feature manifest: {manifest_path}')
    manifest = load_json(manifest_path)
    target_entries = find_target_entries(manifest, args.targets)
    if not target_entries:
        clip_dir = Path(args.clip_save_root) / args.benchmark / 'allframes'
        target_entries = find_clip_empty_target_entries(manifest=manifest, clip_dir=clip_dir)
    if not target_entries:
        print('No zero-frame RTV feature entries or empty clip-score rows found; nothing to repair.', flush=True)
        return

    target_sources = {str(entry['source_video_path']) for entry in target_entries}
    print('Repair targets:', flush=True)
    for entry in target_entries:
        print(f'  {entry["video_path"]} ({entry["source_video_path"]})', flush=True)

    if not args.skip_transcode:
        ffmpeg = find_ffmpeg(args.ffmpeg)
        for source in sorted(target_sources):
            src = Path(source)
            if is_cv2_readable(src):
                print(f'Already readable, skip transcode: {src}', flush=True)
                continue
            transcode_one(
                ffmpeg=ffmpeg,
                src=src,
                crf=args.crf,
                preset=args.preset,
                dry_run=False,
                force=True,
            )

    dataset_result = build_extraction_dataset(args.benchmark)
    dataset = dataset_result[0] if isinstance(dataset_result, tuple) else dataset_result
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    clip_processor, clip_model = load_clip_components(args.clip_model, local_files_only=True, rank=0)
    clip_model = clip_model.to(device).eval()
    if device.type == 'cuda':
        clip_model = clip_model.half()

    if not args.skip_features:
        data_roots = collect_data_roots(dataset)
        new_entries = []
        for entry_i, entry in enumerate(target_entries, start=1):
            source = str(entry['source_video_path'])
            print(f'[{entry_i}/{len(target_entries)}] Recomputing frame feature for {source}', flush=True)
            new_entry = save_video_feature(
                video_path=source,
                spec=entry,
                save_dir=frame_dir,
                benchmark=args.benchmark,
                clip_model=clip_model,
                clip_processor=clip_processor,
                clip_model_name=args.clip_model,
                device=device,
                batch_size=args.feature_batch_size,
                sample_fps=args.sample_fps,
                save_dtype=args.save_dtype,
                source_frame_count=get_frame_count(source),
                rank=0,
                log_batch_interval=50,
                preprocess_backend=args.preprocess_backend,
                decode_backend=args.decode_backend,
            )
            if int(new_entry.get('num_frames') or 0) == 0:
                raise RuntimeError(f'Recomputed feature is still empty: {source}')
            new_entries.append(new_entry)
        manifest['data_roots'] = manifest.get('data_roots') or data_roots
        replace_feature_manifest_entries(frame_dir=frame_dir, manifest=manifest, new_entries=new_entries)

    if not args.skip_clip_scores:
        repair_clip_scores(
            args=args,
            dataset=dataset,
            target_sources=target_sources,
            clip_processor=clip_processor,
            clip_model=clip_model,
            device=device,
        )

    print('Repair done.', flush=True)


if __name__ == '__main__':
    main()
