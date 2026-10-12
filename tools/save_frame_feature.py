"""
Save per-frame CLIP image features for every unique video in a benchmark.

This reuses the video decoding and CLIP image encoding path from
``tools/save_clip_score.py``, but saves reusable frame features instead
of query-specific CLIP scores.

Default output layout:

  <save_root>/<benchmark>/
    manifest.json
    features/
      <video_id>.pt
    _rank_manifests/
      rank_<rank>.json

Each video ``.pt`` file is a torch-saved dict with the keys most useful for
later frame sampling work:

  - features: Tensor[num_frames, feature_dim], L2-normalized CLIP image features
  - frame_indices: LongTensor[num_frames], original 0-based frame indices
  - video_path: normalized benchmark-relative video path
  - source_video_path: absolute/source path used by the dataset
  - clip_model, sample_fps, feature_type, feature_dtype

Single-GPU example:
    python tools/save_frame_feature.py --benchmark LongVideoBench

8-GPU example:
    for RANK in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=${RANK} python tools/save_frame_feature.py \
        --benchmark LongVideoBench --rank ${RANK} --world_size 8 &
    done
    wait
    python tools/save_frame_feature.py \
      --benchmark LongVideoBench --world_size 8 --merge_only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# Reuse the exact CLIP frame extraction helpers used by save_clip_score.py.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from save_clip_score import (  # noqa: E402
    build_extraction_dataset,
    get_frame_count,
    load_clip_components,
    normalize_video_rel_path,
    source_video_for_item,
)


SCHEMA_VERSION = 1
FEATURE_TYPE = 'clip_image_l2_normalized'


def env_flag(name: str) -> bool:
    return str(os.environ.get(name, '')).strip().upper() in {'1', 'ON', 'TRUE', 'YES'}


def _as_image_feature_tensor(output, clip_model):
    """Normalize CLIP image feature outputs across transformers variants."""
    import torch

    def maybe_project(pooler_output):
        if not torch.is_tensor(pooler_output):
            return None
        visual_projection = getattr(clip_model, 'visual_projection', None)
        if visual_projection is None:
            return pooler_output

        in_features = getattr(visual_projection, 'in_features', None)
        out_features = getattr(visual_projection, 'out_features', None)
        feature_dim = int(pooler_output.shape[-1])

        if in_features is not None and feature_dim == int(in_features):
            return visual_projection(pooler_output)
        if out_features is not None and feature_dim == int(out_features):
            return pooler_output

        raise ValueError(
            'Unexpected CLIP pooled image feature dimension: '
            f'got {feature_dim}, visual_projection expects {in_features}->{out_features}'
        )

    if torch.is_tensor(output):
        return output

    image_embeds = getattr(output, 'image_embeds', None)
    if torch.is_tensor(image_embeds):
        return image_embeds

    projected = maybe_project(getattr(output, 'pooler_output', None))
    if projected is not None:
        return projected

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 2:
                return item
            projected = maybe_project(getattr(item, 'pooler_output', None))
            if projected is not None:
                return projected

    raise TypeError(f'Unsupported CLIP image feature output type: {type(output)!r}')


def _get_clip_image_preprocess_config(clip_processor) -> dict[str, Any]:
    image_processor = getattr(clip_processor, 'image_processor', clip_processor)
    size = getattr(image_processor, 'size', {}) or {}
    crop_size = getattr(image_processor, 'crop_size', {}) or {}

    def read_size_value(value, keys: tuple[str, ...], default: int) -> int:
        if hasattr(value, 'get'):
            for key in keys:
                item = value.get(key)
                if item is not None:
                    return int(item)
            return int(default)
        if hasattr(value, keys[0]):
            return int(getattr(value, keys[0]))
        return int(value or default)

    resize_shortest_edge = read_size_value(size, ('shortest_edge', 'height', 'width'), 224)
    crop_height = read_size_value(crop_size, ('height', 'shortest_edge', 'width'), resize_shortest_edge)
    crop_width = read_size_value(crop_size, ('width', 'shortest_edge', 'height'), resize_shortest_edge)

    return {
        'resize_shortest_edge': resize_shortest_edge,
        'crop_height': crop_height,
        'crop_width': crop_width,
        'mean': list(getattr(image_processor, 'image_mean', [0.48145466, 0.4578275, 0.40821073])),
        'std': list(getattr(image_processor, 'image_std', [0.26862954, 0.26130258, 0.27577711])),
    }


def _model_dtype(clip_model):
    try:
        return next(clip_model.parameters()).dtype
    except Exception:
        return None


def _prepare_pixel_values_fast(batch_np, preprocess_config: dict[str, Any], device, dtype):
    """Vectorized CLIP image preprocessing for RGB uint8 NHWC batches."""
    import math
    import torch
    import torch.nn.functional as F

    pixel_values = torch.from_numpy(batch_np).to(device=device, non_blocking=True)
    pixel_values = pixel_values.permute(0, 3, 1, 2).contiguous()
    pixel_values = pixel_values.to(dtype=torch.float32).div_(255.0)

    _, _, height, width = pixel_values.shape
    target_short = int(preprocess_config['resize_shortest_edge'])
    if height <= width:
        resized_height = target_short
        resized_width = int(math.floor(width * target_short / height))
    else:
        resized_width = target_short
        resized_height = int(math.floor(height * target_short / width))

    try:
        pixel_values = F.interpolate(
            pixel_values,
            size=(resized_height, resized_width),
            mode='bicubic',
            align_corners=False,
            antialias=True,
        )
    except TypeError:
        pixel_values = F.interpolate(
            pixel_values,
            size=(resized_height, resized_width),
            mode='bicubic',
            align_corners=False,
        )

    crop_height = int(preprocess_config['crop_height'])
    crop_width = int(preprocess_config['crop_width'])
    top = max((resized_height - crop_height) // 2, 0)
    left = max((resized_width - crop_width) // 2, 0)
    pixel_values = pixel_values[:, :, top: top + crop_height, left: left + crop_width]

    mean = torch.tensor(preprocess_config['mean'], device=device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    std = torch.tensor(preprocess_config['std'], device=device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    pixel_values = (pixel_values - mean) / std
    if dtype is not None and dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        pixel_values = pixel_values.to(dtype=dtype)
    return pixel_values


def encode_video_frames(
    video_path: str,
    clip_model,
    clip_processor,
    device,
    batch_size: int = 128,
    sample_fps: float = 1.0,
    progress_callback=None,
    preprocess_backend: str = 'fast',
    decode_backend: str = 'cpu',
):
    """Decode video frames and encode L2-normalized CLIP image features."""
    import torch
    from PIL import Image

    preprocess_config = _get_clip_image_preprocess_config(clip_processor)
    model_dtype = _model_dtype(clip_model)

    def _encode_np_batch(batch_np):
        if preprocess_backend == 'processor':
            pil_frames = [Image.fromarray(frame) for frame in batch_np]
            img_inputs = clip_processor(images=pil_frames, return_tensors='pt', padding=True).to(device)
            if model_dtype is not None and 'pixel_values' in img_inputs:
                img_inputs['pixel_values'] = img_inputs['pixel_values'].to(dtype=model_dtype)
        elif preprocess_backend == 'fast':
            pixel_values = _prepare_pixel_values_fast(batch_np, preprocess_config, device, model_dtype)
            img_inputs = {'pixel_values': pixel_values}
        else:
            raise ValueError(f'Unsupported preprocess_backend: {preprocess_backend}')

        with torch.no_grad():
            output = clip_model.get_image_features(**img_inputs)
            feats = _as_image_feature_tensor(output, clip_model)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.cpu()

    all_features = []
    sampled_indices: list[int] = []

    try:
        import decord

        try:
            if decode_backend == 'gpu':
                gpu_id = device.index if hasattr(device, 'index') and device.index is not None else 0
                ctx = decord.gpu(gpu_id)
            elif decode_backend == 'auto' and device.type == 'cuda':
                gpu_id = device.index if hasattr(device, 'index') and device.index is not None else 0
                ctx = decord.gpu(gpu_id)
            else:
                ctx = decord.cpu(0)
        except Exception:
            ctx = decord.cpu(0)
        vr = decord.VideoReader(str(video_path), ctx=ctx)
        total = len(vr)
        native_fps = vr.get_avg_fps()
        step = max(1, round(native_fps / sample_fps)) if sample_fps > 0 and native_fps > 0 else 1
        indices = list(range(0, total, step))
        batch_count = (len(indices) + batch_size - 1) // batch_size
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start: start + batch_size]
            batch_np = vr.get_batch(batch_idx).asnumpy()
            all_features.append(_encode_np_batch(batch_np))
            sampled_indices.extend(batch_idx)
            if progress_callback is not None:
                progress_callback(
                    len(sampled_indices),
                    len(indices),
                    start // batch_size + 1,
                    batch_count,
                )
            del batch_np
    except Exception:
        # If decord fails after emitting some batches, discard the partial
        # result before falling back to cv2; otherwise frame indices can restart
        # at zero and produce duplicated feature rows.
        all_features = []
        sampled_indices = []
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(native_fps / sample_fps)) if sample_fps > 0 and native_fps > 0 else 1
        indices = list(range(0, total, step))
        idx_set = set(indices)
        batch_count = (len(indices) + batch_size - 1) // batch_size
        emitted_batches = 0
        buf: list = []
        buf_idx: list[int] = []
        frame_no = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_no in idx_set:
                buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                buf_idx.append(frame_no)
                if len(buf) >= batch_size:
                    import numpy as np

                    all_features.append(_encode_np_batch(np.stack(buf, axis=0)))
                    sampled_indices.extend(buf_idx)
                    emitted_batches += 1
                    if progress_callback is not None:
                        progress_callback(
                            len(sampled_indices),
                            len(indices),
                            emitted_batches,
                            batch_count,
                        )
                    buf, buf_idx = [], []
            frame_no += 1
        if buf:
            import numpy as np

            all_features.append(_encode_np_batch(np.stack(buf, axis=0)))
            sampled_indices.extend(buf_idx)
            emitted_batches += 1
            if progress_callback is not None:
                progress_callback(
                    len(sampled_indices),
                    len(indices),
                    emitted_batches,
                    batch_count,
                )
        cap.release()

    img_features = torch.cat(all_features, dim=0) if all_features else torch.zeros(0, 512)
    return img_features, sampled_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', required=True, help='Base benchmark name, e.g. LongVideoBench or Video-MME')
    parser.add_argument(
        '--save_root',
        default=os.environ.get('FRAME_FEATURES', 'frame_feature'),
        help='Root directory for saved frame features',
    )
    parser.add_argument('--feature_subdir', default='features', help='Subdirectory for per-video .pt files')
    parser.add_argument('--clip_model', default='openai/clip-vit-base-patch32', help='CLIP checkpoint name')
    parser.add_argument('--batch_size', type=int, default=64, help='CLIP image inference batch size')
    parser.add_argument('--sample_fps', type=float, default=0.0, help='Sample rate in fps; set <=0 for every frame')
    parser.add_argument('--device', default=None, help='Torch device for CLIP, e.g. cuda:0 or cpu')
    parser.add_argument('--stream_context', default=None,
                        help='Override stream_context for SVBench / StreamingBench. Leave unset to use the '
                             'dataset default the eval also uses.')
    parser.add_argument(
        '--preprocess_backend',
        choices=('fast', 'processor'),
        default='fast',
        help='Use vectorized tensor preprocessing or the slower CLIPProcessor image path',
    )
    parser.add_argument(
        '--decode_backend',
        choices=('cpu', 'gpu', 'auto'),
        default='cpu',
        help='Video decode backend for decord; cpu avoids GPU decode/asnumpy round-trips',
    )
    parser.add_argument(
        '--save_dtype',
        choices=('float16', 'float32', 'bfloat16'),
        default='float16',
        help='Tensor dtype used when saving features',
    )
    parser.add_argument('--start', type=int, default=None, help='Start sample index for --no_balance mode')
    parser.add_argument('--end', type=int, default=None, help='End sample index for --no_balance mode')
    parser.add_argument('--rank', type=int, default=0, help='Rank of this process (0-indexed)')
    parser.add_argument('--world_size', type=int, default=1, help='Total number of parallel processes')
    parser.add_argument('--merge_only', action='store_true', help='Skip inference; merge rank manifests only')
    parser.add_argument(
        '--no_balance',
        action='store_true',
        help='Disable video/frame-balanced sharding; fall back to contiguous sample chunks',
    )
    parser.add_argument('--checkpoint_interval', type=int, default=20, help='Rewrite rank manifest every N videos')
    parser.add_argument('--indices_file', default=None, help='JSON file containing explicit sample indices to process')
    parser.add_argument('--overwrite', action='store_true', help='Recompute existing .pt feature files')
    parser.add_argument('--stop_on_error', action='store_true', help='Stop on the first video extraction error')
    parser.add_argument(
        '--log_batch_interval',
        type=int,
        default=50,
        help='Print per-video progress every N CLIP image batches (0 to disable)',
    )
    parser.add_argument(
        '--local_files_only',
        action='store_true',
        default=env_flag('TRANSFORMERS_OFFLINE') or env_flag('HF_HUB_OFFLINE'),
        help='Load CLIP from local cache/path only; defaults to true when offline env vars are set',
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _as_unique_list(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    unique: list[Any] = []
    for value in values:
        key = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(_jsonable(value))
    return unique


def collect_data_roots(dataset) -> list[str]:
    roots: list[str] = []

    def add(root) -> None:
        if not root:
            return
        root_str = normalize_video_rel_path(str(root))
        if root_str not in roots:
            roots.append(root_str)

    # frame_root first: clip-based streaming benchmarks feed per-question clips
    # that live under frame_root, not data_root. Keying relative to it keeps the
    # stored path a clean suffix of the runtime clip path so feature lookup matches.
    add(getattr(dataset, 'frame_root', None))
    add(getattr(dataset, 'data_root', None))

    dataset_map = getattr(dataset, 'dataset_map', None)
    if dataset_map:
        for sub_dataset in dataset_map.values():
            add(getattr(sub_dataset, 'frame_root', None))
            add(getattr(sub_dataset, 'data_root', None))

    return sorted(roots, key=len, reverse=True)


def relative_video_path_from_roots(video_path: str, data_roots: list[str]) -> str:
    path = normalize_video_rel_path(str(video_path))
    if not os.path.isabs(path):
        return path

    for root in data_roots:
        root = normalize_video_rel_path(root)
        try:
            rel_path = os.path.relpath(path, root)
        except ValueError:
            continue
        rel_norm = normalize_video_rel_path(rel_path)
        if rel_norm == '.':
            return os.path.basename(path)
        if not rel_norm.startswith('../') and rel_norm != '..' and not os.path.isabs(rel_norm):
            return rel_norm

    return path


def safe_slug(text: str, max_len: int = 120) -> str:
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', text).strip('._-')
    slug = re.sub(r'_+', '_', slug)
    if not slug:
        slug = 'video'
    return slug[:max_len].strip('._-') or 'video'


def video_id_from_rel_path(rel_path: str) -> str:
    rel_path = normalize_video_rel_path(rel_path)
    stem = os.path.splitext(rel_path)[0]
    digest = hashlib.sha1(rel_path.encode('utf-8')).hexdigest()[:12]
    return f'{safe_slug(stem)}_{digest}'


def get_sample_indices(args: argparse.Namespace, total: int) -> list[int]:
    if args.indices_file is not None:
        with open(args.indices_file) as f:
            indices = json.load(f)
        return [int(i) for i in indices]

    if args.no_balance:
        if args.start is None:
            chunk = (total + args.world_size - 1) // args.world_size
            args.start = args.rank * chunk
        if args.end is None:
            chunk = (total + args.world_size - 1) // args.world_size
            args.end = min((args.rank + 1) * chunk, total)
        if not (0 <= args.start <= args.end <= total):
            raise ValueError(
                f'Invalid sample range: start={args.start}, end={args.end}, dataset_size={total}'
            )
        return list(range(args.start, args.end))

    return list(range(total))


def collect_video_groups(dataset, sample_indices: list[int]) -> OrderedDict[str, dict[str, Any]]:
    """Build a source video path -> samples mapping without decoding frames."""
    video_groups: OrderedDict[str, dict[str, Any]] = OrderedDict()

    sample_range = sample_indices
    if tqdm is not None:
        sample_range = tqdm(sample_indices, desc='Scanning video paths', total=len(sample_indices))

    for sample_idx in sample_range:
        # In VLMEvalKit the benchmark-specific video path is normalized by build_prompt.
        # We only read the video item here; no text feature or CLIP score is computed.
        raw_message = dataset.build_prompt(sample_idx, video_llm=True)
        if raw_message is None:
            raise ValueError(f'dataset.build_prompt returned None for sample {sample_idx}')

        video_items = [item for item in raw_message if item.get('type') == 'video']
        if not video_items:
            continue

        try:
            dataset_index = dataset.data.iloc[sample_idx].get('index', sample_idx)
        except Exception:
            dataset_index = sample_idx
        dataset_index = _jsonable(dataset_index)

        for video_item in video_items:
            video_path = source_video_for_item(video_item)
            if video_path not in video_groups:
                video_groups[video_path] = {
                    'source_video_path': video_path,
                    'sample_indices': [],
                    'dataset_indices': [],
                }
            video_groups[video_path]['sample_indices'].append(int(sample_idx))
            video_groups[video_path]['dataset_indices'].append(dataset_index)

    for group in video_groups.values():
        group['sample_indices'] = _as_unique_list(group['sample_indices'])
        group['dataset_indices'] = _as_unique_list(group['dataset_indices'])

    return video_groups


def build_video_specs(
    video_groups: OrderedDict[str, dict[str, Any]],
    data_roots: list[str],
    feature_subdir: str,
) -> OrderedDict[str, dict[str, Any]]:
    specs: OrderedDict[str, dict[str, Any]] = OrderedDict()
    used_ids: dict[str, str] = {}

    for video_path, group in video_groups.items():
        rel_path = relative_video_path_from_roots(video_path, data_roots)
        video_id = video_id_from_rel_path(rel_path)
        if video_id in used_ids and used_ids[video_id] != video_path:
            suffix = hashlib.sha1(video_path.encode('utf-8')).hexdigest()[:8]
            video_id = f'{video_id}_{suffix}'
        used_ids[video_id] = video_path

        feature_path = normalize_video_rel_path(str(Path(feature_subdir) / f'{video_id}.pt'))
        specs[video_path] = {
            'video_id': video_id,
            'video_path': rel_path,
            'source_video_path': video_path,
            'feature_path': feature_path,
            'sample_indices': group['sample_indices'],
            'dataset_indices': group['dataset_indices'],
        }

    return specs


def load_or_scan_video_frame_counts(
    save_dir: Path,
    video_paths: list[str],
    rank: int,
    world_size: int,
    wait_for_rank0: bool,
) -> dict[str, int]:
    frame_counts_path = save_dir / 'video_frame_counts.json'
    if frame_counts_path.exists():
        with open(frame_counts_path) as f:
            cached = json.load(f)
        if all(video_path in cached for video_path in video_paths):
            print(f'Loaded cached video frame counts from {frame_counts_path}', flush=True)
            return {video_path: int(cached[video_path]) for video_path in video_paths}

    if wait_for_rank0 and world_size > 1 and rank != 0:
        print(f'Waiting for rank 0 to create {frame_counts_path}...', flush=True)
        for _ in range(1800):
            if frame_counts_path.exists():
                with open(frame_counts_path) as f:
                    cached = json.load(f)
                if all(video_path in cached for video_path in video_paths):
                    print(f'Loaded cached video frame counts from {frame_counts_path}', flush=True)
                    return {video_path: int(cached[video_path]) for video_path in video_paths}
            time.sleep(1)
        print(f'Warning: timed out waiting for {frame_counts_path}; scanning on rank {rank}.', flush=True)

    print(f'Scanning frame counts for {len(video_paths)} unique videos...', flush=True)
    counts: dict[str, int] = {}
    scan_range = video_paths
    if tqdm is not None:
        scan_range = tqdm(video_paths, desc='Scanning frames', total=len(video_paths))
    for video_path in scan_range:
        counts[video_path] = get_frame_count(video_path)

    existing: dict[str, int] = {}
    if frame_counts_path.exists():
        try:
            with open(frame_counts_path) as f:
                existing = {k: int(v) for k, v in json.load(f).items()}
        except Exception:
            existing = {}
    existing.update(counts)

    tmp_path = frame_counts_path.with_suffix('.tmp')
    with open(tmp_path, 'w') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    tmp_path.replace(frame_counts_path)
    print(f'Saved video frame counts to {frame_counts_path}', flush=True)

    return counts


def balanced_shard_by_videos(
    video_paths: list[str],
    video_frame_counts: dict[str, int],
    world_size: int,
    rank: int,
) -> list[str]:
    indexed = sorted(video_paths, key=lambda path: -video_frame_counts.get(path, 1))
    rank_totals = [0] * world_size
    rank_video_paths: list[list[str]] = [[] for _ in range(world_size)]

    for video_path in indexed:
        min_rank = min(range(world_size), key=lambda r: rank_totals[r])
        rank_video_paths[min_rank].append(video_path)
        rank_totals[min_rank] += video_frame_counts.get(video_path, 1)

    selected = set(rank_video_paths[rank])
    selected_video_paths = [video_path for video_path in video_paths if video_path in selected]
    total_frames = sum(video_frame_counts.get(path, 1) for path in video_paths)
    print(
        f'Video-balanced sharding: rank {rank} gets {len(selected_video_paths)} videos, '
        f'~{rank_totals[rank]} frames '
        f'({100 * rank_totals[rank] / max(total_frames, 1):.1f}% of {total_frames} total)'
    )
    return selected_video_paths


def rank_manifest_dir(save_dir: Path) -> Path:
    return save_dir / '_rank_manifests'


def rank_manifest_path(save_dir: Path, rank: int) -> Path:
    return rank_manifest_dir(save_dir) / f'rank_{rank}.json'


def sort_video_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(entry: dict[str, Any]) -> tuple[int, str]:
        sample_indices = entry.get('sample_indices') or []
        first_sample = min(int(i) for i in sample_indices) if sample_indices else 10**18
        return first_sample, str(entry.get('video_path', ''))

    return sorted(entries, key=key)


def build_manifest(
    *,
    benchmark: str,
    save_dir: Path,
    feature_subdir: str,
    clip_model: str,
    sample_fps: float,
    save_dtype: str,
    batch_size: int,
    world_size: int,
    videos: list[dict[str, Any]],
    failed_videos: list[dict[str, Any]],
    data_roots: list[str] | None = None,
) -> dict[str, Any]:
    videos = sort_video_entries(videos)

    video_path_to_feature: dict[str, str] = {}
    source_video_path_to_feature: dict[str, str] = {}
    video_id_to_feature: dict[str, str] = {}
    sample_idx_to_video_ids: dict[str, list[str]] = {}
    dataset_index_to_video_ids: dict[str, list[str]] = {}

    for entry in videos:
        video_path = str(entry['video_path'])
        video_id = str(entry['video_id'])
        feature_path = str(entry['feature_path'])
        video_path_to_feature[video_path] = feature_path
        source_video_path = entry.get('source_video_path')
        if source_video_path:
            source_video_path_to_feature[str(source_video_path)] = feature_path
        video_id_to_feature[video_id] = feature_path
        for sample_idx in entry.get('sample_indices', []):
            sample_idx_to_video_ids.setdefault(str(sample_idx), []).append(video_id)
        for dataset_index in entry.get('dataset_indices', []):
            dataset_index_to_video_ids.setdefault(str(dataset_index), []).append(video_id)

    return {
        'schema_version': SCHEMA_VERSION,
        'benchmark': benchmark,
        'save_dir': str(save_dir),
        'feature_subdir': feature_subdir,
        'clip_model': clip_model,
        'feature_type': FEATURE_TYPE,
        'feature_dtype': save_dtype,
        'sample_fps': sample_fps,
        'frame_coverage': 'all_frames' if sample_fps <= 0 else f'{sample_fps:g}fps',
        'frame_index_base': 0,
        'batch_size': batch_size,
        'world_size': world_size,
        'num_videos': len(videos),
        'num_failed_videos': len(failed_videos),
        'data_roots': data_roots or [],
        'videos': videos,
        'failed_videos': failed_videos,
        'video_path_to_feature': video_path_to_feature,
        'source_video_path_to_feature': source_video_path_to_feature,
        'video_id_to_feature': video_id_to_feature,
        'sample_idx_to_video_ids': sample_idx_to_video_ids,
        'dataset_index_to_video_ids': dataset_index_to_video_ids,
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with open(tmp_path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def load_rank_manifest(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.exists():
        return [], []
    with open(path) as f:
        data = json.load(f)
    return data.get('videos', []), data.get('failed_videos', [])


def entry_matches_run(entry: dict[str, Any], args: argparse.Namespace) -> bool:
    try:
        sample_fps_matches = float(entry.get('sample_fps', -1)) == float(args.sample_fps)
    except Exception:
        sample_fps_matches = False
    return (
        str(entry.get('clip_model')) == str(args.clip_model)
        and sample_fps_matches
        and str(entry.get('feature_dtype')) == str(args.save_dtype)
        and str(entry.get('preprocess_backend')) == str(args.preprocess_backend)
    )


def cast_features(features, save_dtype: str):
    import torch

    if save_dtype == 'float32':
        return features.float().contiguous()
    if save_dtype == 'float16':
        return features.to(torch.float16).contiguous()
    if save_dtype == 'bfloat16':
        return features.to(torch.bfloat16).contiguous()
    raise ValueError(f'Unsupported save_dtype: {save_dtype}')


def save_video_feature(
    *,
    video_path: str,
    spec: dict[str, Any],
    save_dir: Path,
    benchmark: str,
    clip_model,
    clip_processor,
    clip_model_name: str,
    device,
    batch_size: int,
    sample_fps: float,
    save_dtype: str,
    source_frame_count: int | None,
    rank: int,
    log_batch_interval: int,
    preprocess_backend: str,
    decode_backend: str,
) -> dict[str, Any]:
    import torch

    def progress_callback(done_frames: int, total_frames: int, batch_i: int, batch_count: int) -> None:
        if log_batch_interval <= 0:
            return
        if batch_i != batch_count and batch_i % log_batch_interval != 0:
            return
        message = (
            f'[Rank {rank}] encoding {spec["video_id"]}: '
            f'{done_frames}/{total_frames} frames, batch {batch_i}/{batch_count}'
        )
        if tqdm is not None:
            tqdm.write(message)
        else:
            print(message)

    img_features, frame_indices = encode_video_frames(
        video_path=video_path,
        clip_model=clip_model,
        clip_processor=clip_processor,
        device=device,
        batch_size=batch_size,
        sample_fps=sample_fps,
        progress_callback=progress_callback,
        preprocess_backend=preprocess_backend,
        decode_backend=decode_backend,
    )
    img_features = cast_features(img_features, save_dtype)
    frame_indices_tensor = torch.tensor(frame_indices, dtype=torch.long)

    feature_path = save_dir / spec['feature_path']
    feature_path.parent.mkdir(parents=True, exist_ok=True)

    feature_dim = int(img_features.shape[-1]) if img_features.ndim == 2 else None
    payload = {
        'schema_version': SCHEMA_VERSION,
        'benchmark': benchmark,
        'video_id': spec['video_id'],
        'video_path': spec['video_path'],
        'source_video_path': spec['source_video_path'],
        'feature_path': spec['feature_path'],
        'sample_indices': spec['sample_indices'],
        'dataset_indices': spec['dataset_indices'],
        'features': img_features,
        'frame_indices': frame_indices_tensor,
        'frame_index_base': 0,
        'feature_type': FEATURE_TYPE,
        'features_are_normalized': True,
        'frame_indices_are_original_0based': True,
        'feature_dtype': save_dtype,
        'feature_dim': feature_dim,
        'clip_model': clip_model_name,
        'preprocess_backend': preprocess_backend,
        'decode_backend': decode_backend,
        'sample_fps': sample_fps,
        'frame_coverage': 'all_frames' if sample_fps <= 0 else f'{sample_fps:g}fps',
        'source_frame_count': source_frame_count,
        'num_frames': int(img_features.shape[0]),
    }

    tmp_path = feature_path.with_suffix(feature_path.suffix + '.tmp')
    torch.save(payload, tmp_path)
    tmp_path.replace(feature_path)

    return {
        **spec,
        'feature_path': normalize_video_rel_path(os.path.relpath(feature_path, save_dir)),
        'num_frames': int(img_features.shape[0]),
        'feature_dim': feature_dim,
        'feature_dtype': save_dtype,
        'clip_model': clip_model_name,
        'sample_fps': sample_fps,
        'preprocess_backend': preprocess_backend,
        'decode_backend': decode_backend,
        'source_frame_count': source_frame_count,
        'frame_index_start': int(frame_indices[0]) if frame_indices else None,
        'frame_index_end': int(frame_indices[-1]) if frame_indices else None,
    }


def make_existing_entry(
    *,
    spec: dict[str, Any],
    save_dir: Path,
    source_frame_count: int | None,
) -> dict[str, Any]:
    feature_path = save_dir / spec['feature_path']
    return {
        **spec,
        'feature_path': normalize_video_rel_path(os.path.relpath(feature_path, save_dir)),
        'num_frames': None,
        'feature_dim': None,
        'feature_dtype': None,
        'source_frame_count': source_frame_count,
        'frame_index_start': None,
        'frame_index_end': None,
    }


def merge_manifests(save_dir: Path, world_size: int) -> None:
    videos_by_id: dict[str, dict[str, Any]] = {}
    failed_by_path: dict[str, dict[str, Any]] = {}
    shard_meta: dict[str, Any] | None = None

    for rank in range(world_size):
        path = rank_manifest_path(save_dir, rank)
        if not path.exists():
            raise FileNotFoundError(f'Rank manifest not found: {path}. Run all ranks before merging.')
        with open(path) as f:
            shard = json.load(f)
        if shard_meta is None:
            shard_meta = shard
        for entry in shard.get('videos', []):
            videos_by_id[str(entry['video_id'])] = entry
        for entry in shard.get('failed_videos', []):
            failed_by_path[str(entry.get('source_video_path', entry.get('video_path')))] = entry

    if shard_meta is None:
        raise RuntimeError('No rank manifests found.')

    manifest = build_manifest(
        benchmark=shard_meta['benchmark'],
        save_dir=save_dir,
        feature_subdir=shard_meta.get('feature_subdir', 'features'),
        clip_model=shard_meta['clip_model'],
        sample_fps=float(shard_meta['sample_fps']),
        save_dtype=shard_meta['feature_dtype'],
        batch_size=int(shard_meta['batch_size']),
        world_size=world_size,
        videos=list(videos_by_id.values()),
        failed_videos=list(failed_by_path.values()),
        data_roots=shard_meta.get('data_roots', []),
    )
    write_json_atomic(save_dir / 'manifest.json', manifest)
    print(f'Wrote {save_dir / "manifest.json"} with {manifest["num_videos"]} videos')
    if manifest['num_failed_videos']:
        print(f'Warning: {manifest["num_failed_videos"]} videos failed; see failed_videos in manifest.json')


def main() -> None:
    args = parse_args()

    if not (0 <= args.rank < args.world_size):
        raise ValueError(f'Invalid rank={args.rank} for world_size={args.world_size}.')

    save_dir = Path(args.save_root) / args.benchmark
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / args.feature_subdir).mkdir(parents=True, exist_ok=True)
    rank_manifest_dir(save_dir).mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        merge_manifests(save_dir, args.world_size)
        return

    import torch

    if args.device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f'[Rank {args.rank}] Building dataset: {args.benchmark}', flush=True)
    dataset, build_kwargs = build_extraction_dataset(
        args.benchmark, stream_context=args.stream_context,
    )
    print(f'[Rank {args.rank}] [build] dataset={args.benchmark} build_kwargs={build_kwargs}', flush=True)

    total_samples = len(dataset.data)
    print(f'[Rank {args.rank}] Collecting sample indices / video paths...', flush=True)
    sample_indices = get_sample_indices(args, total_samples)
    video_groups = collect_video_groups(dataset, sample_indices)
    print(
        f'[Rank {args.rank}] Collected {len(video_groups)} unique videos from '
        f'{len(sample_indices)} samples',
        flush=True,
    )
    data_roots = collect_data_roots(dataset)
    video_specs = build_video_specs(video_groups, data_roots, args.feature_subdir)

    all_video_paths = list(video_groups.keys())
    if not all_video_paths:
        raise RuntimeError(f'No videos found for benchmark `{args.benchmark}`.')

    if args.no_balance or args.indices_file is not None:
        video_frame_counts = {}
        selected_video_paths = all_video_paths
    else:
        video_frame_counts = load_or_scan_video_frame_counts(
            save_dir=save_dir,
            video_paths=all_video_paths,
            rank=args.rank,
            world_size=args.world_size,
            wait_for_rank0=True,
        )
        selected_video_paths = balanced_shard_by_videos(
            all_video_paths,
            video_frame_counts,
            args.world_size,
            args.rank,
        )

    rank_path = rank_manifest_path(save_dir, args.rank)
    existing_videos, existing_failed = load_rank_manifest(rank_path)
    existing_by_source = {
        str(entry.get('source_video_path')): entry
        for entry in existing_videos
        if (
            entry.get('source_video_path')
            and (save_dir / str(entry.get('feature_path'))).exists()
            and entry_matches_run(entry, args)
        )
    }

    rank_videos: list[dict[str, Any]] = []
    rank_failed: list[dict[str, Any]] = []
    todo_video_paths: list[str] = []

    for video_path in selected_video_paths:
        spec = video_specs[video_path]
        feature_path = save_dir / spec['feature_path']
        if not args.overwrite and video_path in existing_by_source:
            rank_videos.append(existing_by_source[video_path])
            continue
        todo_video_paths.append(video_path)

    print(f'Rank        : {args.rank}/{args.world_size}', flush=True)
    print(f'Benchmark   : {args.benchmark}', flush=True)
    print(f'Samples     : {len(sample_indices)}/{total_samples}', flush=True)
    print(f'Videos      : {len(selected_video_paths)} selected / {len(all_video_paths)} unique', flush=True)
    print(f'Remaining   : {len(todo_video_paths)} videos', flush=True)
    print(f'Frame mode  : {"all frames" if args.sample_fps <= 0 else f"{args.sample_fps:g} fps"}', flush=True)
    print(f'CLIP model  : {args.clip_model}', flush=True)
    print(f'Decode      : {args.decode_backend}', flush=True)
    print(f'Preprocess  : {args.preprocess_backend}', flush=True)
    print(f'Save dtype  : {args.save_dtype}', flush=True)
    print(f'Device      : {device}', flush=True)
    print(f'Save dir    : {save_dir}', flush=True)

    print(f'[Rank {args.rank}] Loading CLIP model: {args.clip_model}', flush=True)
    print(f'[Rank {args.rank}] Local files only: {args.local_files_only}', flush=True)
    clip_processor, clip_model = load_clip_components(
        args.clip_model,
        local_files_only=args.local_files_only,
        rank=args.rank,
    )
    clip_model = clip_model.to(device).eval()
    if device.type == 'cuda':
        clip_model = clip_model.half()
    print(f'[Rank {args.rank}] CLIP model ready on {device}', flush=True)

    def write_rank_state() -> None:
        manifest = build_manifest(
            benchmark=args.benchmark,
            save_dir=save_dir,
            feature_subdir=args.feature_subdir,
            clip_model=args.clip_model,
            sample_fps=args.sample_fps,
            save_dtype=args.save_dtype,
            batch_size=args.batch_size,
            world_size=args.world_size,
            videos=rank_videos,
            failed_videos=rank_failed,
            data_roots=data_roots,
        )
        manifest['rank'] = args.rank
        write_json_atomic(rank_path, manifest)

    video_range = todo_video_paths
    if tqdm is not None:
        video_range = tqdm(todo_video_paths, desc=f'Rank {args.rank}', total=len(todo_video_paths))

    for video_i, video_path in enumerate(video_range):
        spec = video_specs[video_path]
        source_frame_count = video_frame_counts.get(video_path)
        if source_frame_count is None:
            source_frame_count = get_frame_count(video_path)

        try:
            entry = save_video_feature(
                video_path=video_path,
                spec=spec,
                save_dir=save_dir,
                benchmark=args.benchmark,
                clip_model=clip_model,
                clip_processor=clip_processor,
                clip_model_name=args.clip_model,
                device=device,
                batch_size=args.batch_size,
                sample_fps=args.sample_fps,
                save_dtype=args.save_dtype,
                source_frame_count=source_frame_count,
                rank=args.rank,
                log_batch_interval=args.log_batch_interval,
                preprocess_backend=args.preprocess_backend,
                decode_backend=args.decode_backend,
            )
            rank_videos.append(entry)
            message = (
                f'[Rank {args.rank}] saved {entry["video_id"]}: '
                f'{entry["num_frames"]} frames -> {entry["feature_path"]}'
            )
            if tqdm is not None:
                tqdm.write(message)
            else:
                print(message)
        except Exception as exc:
            failed = {
                **spec,
                'source_frame_count': source_frame_count,
                'error': repr(exc),
            }
            rank_failed.append(failed)
            message = f'[Rank {args.rank}] failed {spec["video_id"]}: {exc!r}'
            if tqdm is not None:
                tqdm.write(message)
            else:
                print(message)
            if args.stop_on_error:
                write_rank_state()
                raise

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        if args.checkpoint_interval > 0 and (video_i + 1) % args.checkpoint_interval == 0:
            write_rank_state()
            if tqdm is not None:
                tqdm.write(f'[Rank {args.rank}] rank manifest checkpoint saved')
            else:
                print(f'[Rank {args.rank}] rank manifest checkpoint saved')

    write_rank_state()
    print(f'[Rank {args.rank}] Wrote {rank_path}')

    if args.world_size == 1:
        merge_manifests(save_dir, world_size=1)


if __name__ == '__main__':
    main()
