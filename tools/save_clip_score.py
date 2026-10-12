"""
Save CLIP scores for ALL frames of each video.

Computes and saves per-frame CLIP similarity between each video frame (at native
FPS, all frames) and the sample query.  Because scores cover the full video, the
same file can be reused at inference time regardless of how many frames the model
samples — the runtime just indexes into the precomputed per-frame scores by the
original frame indices it actually loaded.

Outputs are written row-for-row:
  - scores.json       — list[list[float]], one float per original video frame
  - frames.json       — list[list[int]],  original frame indices (0-based), always [0,1,...,N-1]
  - queries.json      — list[str]
  - video_paths.json  — list[str]
  - metadata.json

Single-GPU example:
    python tools/save_clip_score.py \
        --benchmark LongVideoBench \
        --save_root clip_scores

8-GPU example:
    for RANK in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=${RANK} python tools/save_clip_score.py \
        --benchmark LongVideoBench --save_root clip_scores \
        --rank ${RANK} --world_size 8 &
    done
    wait
    python tools/save_clip_score.py \
        --benchmark LongVideoBench --save_root clip_scores \
        --world_size 8 --merge_only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def find_repo_root(start_path: Path) -> Path:
    """Find the nearest ancestor that looks like the VLMEvalKit repo root."""
    resolved = start_path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / 'vlmeval').is_dir():
            return candidate
    raise RuntimeError(
        f'Failed to locate repository root from {start_path}. '
        'Expected an ancestor directory containing `vlmeval/`.'
    )


ROOT = find_repo_root(Path(__file__))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


_MCQ_ANSWER_SUFFIXES = (
    "\nAnswer with the option's letter from the given choices directly.",
)


def normalize_clip_query(query: str) -> str:
    """Match the runtime CLIP query used in `model_wrapper.py`."""
    normalized = str(query).lstrip()

    for suffix in _MCQ_ANSWER_SUFFIXES:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].rstrip()
            break

    lines = normalized.splitlines()
    content_lines = []
    for line in lines:
        if re.match(r'^[A-Z][\.\)]\s', line.strip()):
            break
        content_lines.append(line.rstrip())

    question_only = '\n'.join(content_lines).strip()
    return question_only or normalized.strip()


def normalize_video_rel_path(video_path: str) -> str:
    path = str(video_path).replace('\\', '/')
    if path.startswith('file://'):
        path = path[len('file://'):]
    return os.path.normpath(path).replace('\\', '/')


def relative_video_path(video_path: str, data_root: str) -> str:
    try:
        rel_path = os.path.relpath(video_path, data_root)
    except ValueError:
        rel_path = video_path
    return normalize_video_rel_path(rel_path)


def source_video_for_item(item) -> str:
    """Return the video to encode/key by for a build_prompt video item.

    Clip-based streaming benchmarks (RTV-Bench / SVBench / StreamingBench) put a
    per-question clip in 'value' but expose the original video via
    '_source_video_path'. Offline scores/features must cover ALL native frames of
    the source video (fps-agnostic, reused across questions) and be keyed by it,
    so the runtime can score/decode against every frame of the segment. Full-video
    benchmarks have no '_source_video_path' and fall back to 'value' unchanged.
    """
    if isinstance(item, dict):
        src = item.get('_source_video_path')
        if src:
            return str(src)
        return str(item.get('value'))
    return str(item)


def collect_video_roots(dataset) -> list:
    """Roots used to relativise video paths for storage.

    ``frame_root`` is listed first because clip-based streaming benchmarks
    (RTV-Bench / SVBench / StreamingBench) feed per-question clips that live
    under ``frame_root`` (not ``data_root``); keying relative to it keeps the
    stored path a clean suffix of the absolute clip path so the runtime's
    suffix-match in ``_lookup_offline_clip`` / feature lookup succeeds.
    """
    roots: list[str] = []

    def add(root) -> None:
        if not root:
            return
        norm = normalize_video_rel_path(str(root))
        if norm not in roots:
            roots.append(norm)

    add(getattr(dataset, 'frame_root', None))
    add(getattr(dataset, 'data_root', None))
    dataset_map = getattr(dataset, 'dataset_map', None)
    if dataset_map:
        for sub in dataset_map.values():
            add(getattr(sub, 'frame_root', None))
            add(getattr(sub, 'data_root', None))
    return sorted(roots, key=len, reverse=True)


def relative_video_path_from_roots(video_path: str, roots: list) -> str:
    path = normalize_video_rel_path(str(video_path))
    if not os.path.isabs(path):
        return path
    for root in roots:
        root = normalize_video_rel_path(root)
        try:
            rel = os.path.relpath(path, root)
        except ValueError:
            continue
        rel = normalize_video_rel_path(rel)
        if rel == '.':
            return os.path.basename(path)
        if not rel.startswith('../') and rel != '..' and not os.path.isabs(rel):
            return rel
    return path


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


def _as_text_feature_tensor(output, clip_model):
    """Normalize CLIP text feature outputs across transformers variants."""
    import torch

    if torch.is_tensor(output):
        return output

    text_embeds = getattr(output, 'text_embeds', None)
    if torch.is_tensor(text_embeds):
        return text_embeds

    # Transformers 5 returns a model output whose pooler_output is already
    # projected. CLIP's text projection is 512 -> 512, so dimension checks
    # cannot distinguish projected from unprojected tensors here.
    pooled = getattr(output, 'pooler_output', None)
    if torch.is_tensor(pooled):
        return pooled

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 2:
                return item
            pooled = getattr(item, 'pooler_output', None)
            if torch.is_tensor(pooled):
                return pooled

    raise TypeError(f'Unsupported CLIP text feature output type: {type(output)!r}')


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


def build_extraction_dataset(benchmark: str, stream_context=None):
    """Build the dataset so offline extraction matches the eval-time inputs.

    Offline scores are computed over source-video frames and then reused by
    segment metadata at runtime. An internal lightweight fps value satisfies
    the release dataset constructors; ``build_prompt(video_llm=True)`` still
    returns the source path without sampling frames. ``stream_context`` is kept
    for datasets where it changes segment metadata.

    Returns ``(dataset, build_kwargs)``.
    """
    from vlmeval.dataset import build_dataset

    # The release dataset constructors require either fps or nframe even when
    # video_llm=True returns the source path without sampling any frames.
    base = {'fps': 1.0}

    attempts: list[dict] = []
    if stream_context is not None:
        attempts.append({**base, 'stream_context': stream_context})
    attempts.append(base)

    last_err: Exception | None = None
    for kwargs in attempts:
        try:
            dataset = build_dataset(benchmark, **kwargs)
        except Exception as err:  # noqa: BLE001 - try the next fallback config
            last_err = err
            continue
        if dataset is not None:
            return dataset, kwargs
    raise RuntimeError(f'Failed to build dataset `{benchmark}`: {last_err}')


def encode_video_frames(
    video_path: str,
    clip_model,
    clip_processor,
    device,
    batch_size: int = 128,
    sample_fps: float = 1.0,
    end_time: float | None = None,
    preprocess_backend: str = 'fast',
    decode_backend: str = 'cpu',
):
    """Decode and encode video frames into CLIP image features.

    Frames are sampled at ``sample_fps`` fps (set <=0 for every frame).
    When ``end_time`` is given (seconds), only frames up to that timestamp are
    encoded — used by clip-based streaming benchmarks to skip the tail of the
    source video beyond the latest question timestamp (the runtime never looks
    past a question's segment anyway). A small safety margin is kept so the
    runtime's segment end_frame is always covered.
    Returns normalised image features as a CPU tensor to save GPU memory,
    along with the original frame indices.

    Returns:
        img_features:  torch.Tensor of shape (N, D), float32, on CPU
        frame_indices: list[int], original 0-based frame indices
    """
    import torch
    from PIL import Image
    import numpy as np

    preprocess_config = _get_clip_image_preprocess_config(clip_processor)
    model_dtype = _model_dtype(clip_model)

    def _frame_upper_bound(total: int, native_fps) -> int:
        last = total - 1
        if end_time is not None and native_fps and native_fps > 0:
            capped = int(round(float(end_time) * float(native_fps))) + 2
            last = min(last, capped)
        return max(0, last)

    def _encode_np_batch(batch_np):
        if preprocess_backend == 'processor':
            pil_frames = [Image.fromarray(frame) for frame in batch_np]
            img_inputs = clip_processor(images=pil_frames, return_tensors='pt', padding=True).to(device)
            if model_dtype is not None and 'pixel_values' in img_inputs:
                img_inputs['pixel_values'] = img_inputs['pixel_values'].to(dtype=model_dtype)
        elif preprocess_backend == 'fast':
            img_inputs = {
                'pixel_values': _prepare_pixel_values_fast(batch_np, preprocess_config, device, model_dtype)
            }
        else:
            raise ValueError(f'Unsupported preprocess_backend: {preprocess_backend}')

        with torch.no_grad():
            feats = clip_model.get_image_features(**img_inputs)
            feats = _as_image_feature_tensor(feats, clip_model)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.cpu()

    all_features = []
    sampled_indices: list[int] = []

    try:
        import decord
        try:
            if decode_backend == 'gpu':
                _gpu_id = device.index if hasattr(device, 'index') and device.index is not None else 0
                _ctx = decord.gpu(_gpu_id)
            elif decode_backend == 'auto' and device.type == 'cuda':
                _gpu_id = device.index if hasattr(device, 'index') and device.index is not None else 0
                _ctx = decord.gpu(_gpu_id)
            else:
                _ctx = decord.cpu(0)
        except Exception:
            _ctx = decord.cpu(0)
        vr = decord.VideoReader(str(video_path), ctx=_ctx)
        total = len(vr)
        native_fps = vr.get_avg_fps()
        step = max(1, round(native_fps / sample_fps)) if sample_fps > 0 and native_fps > 0 else 1
        indices = list(range(0, _frame_upper_bound(total, native_fps) + 1, step))
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start: start + batch_size]
            batch_np = vr.get_batch(batch_idx).asnumpy()
            all_features.append(_encode_np_batch(batch_np))
            sampled_indices.extend(batch_idx)
            del batch_np
    except Exception:
        # If decord fails after emitting some batches, discard the partial
        # result before falling back to cv2; otherwise frame indices can restart
        # at zero and produce duplicated feature/score rows.
        all_features = []
        sampled_indices = []
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(native_fps / sample_fps)) if sample_fps > 0 and native_fps > 0 else 1
        indices = list(range(0, _frame_upper_bound(total, native_fps) + 1, step))
        idx_set = set(indices)
        last_needed = indices[-1] if indices else -1
        buf: list = []
        buf_idx: list = []
        frame_no = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_no > last_needed:
                break
            if frame_no in idx_set:
                buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                buf_idx.append(frame_no)
                if len(buf) >= batch_size:
                    all_features.append(_encode_np_batch(np.stack(buf, axis=0)))
                    sampled_indices.extend(buf_idx)
                    buf, buf_idx = [], []
            frame_no += 1
        if buf:
            all_features.append(_encode_np_batch(np.stack(buf, axis=0)))
            sampled_indices.extend(buf_idx)
        cap.release()

    import torch
    img_features = torch.cat(all_features, dim=0) if all_features else torch.zeros(0, 512)
    return img_features, sampled_indices


def score_query(
    img_features_on_device,
    query: str,
    clip_model,
    clip_processor,
    device,
) -> list[float]:
    """Compute per-frame CLIP scores for a single text query.

    ``img_features_on_device`` must already reside on ``device`` (moved once
    per video by the caller, not per query).  Only text encoding + one matmul
    is done here.
    """
    import torch

    text_inputs = clip_processor(
        text=[query], return_tensors='pt', padding=True, truncation=True, max_length=77
    ).to(device)
    with torch.no_grad():
        text_feat = clip_model.get_text_features(**text_inputs)
        text_feat = _as_text_feature_tensor(text_feat, clip_model)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # (1, D)
    scores = (img_features_on_device @ text_feat.T).squeeze(-1)       # (N,)
    return scores.cpu().tolist()


def score_queries(
    img_features_on_device,
    queries: list[str],
    clip_model,
    clip_processor,
    device,
) -> list[list[float]]:
    """Compute per-frame CLIP scores for multiple text queries at once."""
    import torch

    if not queries:
        return []

    text_inputs = clip_processor(
        text=queries, return_tensors='pt', padding=True, truncation=True, max_length=77
    ).to(device)
    with torch.no_grad():
        text_feat = clip_model.get_text_features(**text_inputs)
        text_feat = _as_text_feature_tensor(text_feat, clip_model)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scores = text_feat @ img_features_on_device.T
    return scores.cpu().tolist()


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


class ClipProcessorShim:
    """Small wrapper matching the CLIPProcessor calls this script needs."""

    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, *args, **kwargs):
        if 'images' in kwargs:
            return self.image_processor(*args, **kwargs)
        if 'text' in kwargs:
            return self.tokenizer(*args, **kwargs)
        raise ValueError('ClipProcessorShim expects either images=... or text=...')


def load_clip_components(model_name: str, local_files_only: bool, rank: int):
    """Load CLIP model plus the two processors needed by this script.

    Some local caches have the CLIP model/tokenizer files but miss
    ``preprocessor_config.json``. In that case, fall back to the standard CLIP
    image preprocessing defaults instead of failing before inference starts.
    """
    from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizerFast

    common = {'local_files_only': local_files_only}
    try:
        processor = CLIPProcessor.from_pretrained(model_name, **common)
    except OSError as err:
        print(
            f'[Rank {rank}] CLIPProcessor load failed ({err}); '
            'trying tokenizer + default CLIP image processor fallback.',
            flush=True,
        )
        try:
            image_processor = CLIPImageProcessor.from_pretrained(model_name, **common)
        except OSError:
            image_processor = CLIPImageProcessor()
        tokenizer = CLIPTokenizerFast.from_pretrained(model_name, **common)
        processor = ClipProcessorShim(image_processor=image_processor, tokenizer=tokenizer)

    model = CLIPModel.from_pretrained(model_name, **common)
    return processor, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', required=True, help='Base benchmark name, e.g. LongVideoBench or Video-MME')
    parser.add_argument('--save_root', default='clip_scores', help='Root directory for saved CLIP scores')
    parser.add_argument('--clip_model', default='openai/clip-vit-base-patch32', help='CLIP checkpoint name')
    parser.add_argument('--clip_local_files_only', action='store_true',
                        default=env_flag('CLIP_LOCAL_FILES_ONLY', True),
                        help='Load CLIP only from local HuggingFace cache. '
                             'Set CLIP_LOCAL_FILES_ONLY=0 to allow network access.')
    parser.add_argument('--batch_size', type=int, default=256, help='CLIP image inference batch size')
    parser.add_argument('--sample_fps', type=float, default=0.0, help='Sample rate in fps; set <=0 for all frames')
    parser.add_argument('--device', default=None, help='Torch device for CLIP, e.g. cuda:0 or cpu')
    parser.add_argument('--preprocess_backend', choices=('fast', 'processor'), default='processor',
                        help='Use CLIPProcessor image preprocessing (default, matches the checked cache) '
                             'or vectorized fast preprocessing')
    parser.add_argument('--decode_backend', choices=('cpu', 'gpu', 'auto'), default='cpu',
                        help='Video decode backend for decord; cpu is safest when decord lacks CUDA support')
    parser.add_argument('--stream_context', default=None,
                        help='Override stream_context for SVBench / StreamingBench (e.g. history / segment). '
                             'Leave unset to use the dataset default the eval also uses.')
    parser.add_argument('--start', type=int, default=None, help='Start sample index (inclusive); only used when --no_balance is set')
    parser.add_argument('--end', type=int, default=None, help='End sample index (exclusive); only used when --no_balance is set')
    parser.add_argument('--rank', type=int, default=0, help='Rank of this process (0-indexed)')
    parser.add_argument('--world_size', type=int, default=1, help='Total number of parallel processes')
    parser.add_argument('--merge_only', action='store_true', help='Skip inference; just merge rank shards into final output')
    parser.add_argument('--no_balance', action='store_true', help='Disable frame-balanced sharding; fall back to contiguous chunk split')
    parser.add_argument('--checkpoint_interval', type=int, default=50, help='Save checkpoint every N samples (0 to disable)')
    parser.add_argument('--per_sample_save', action='store_true',
                        help='Save each sample to an individual JSON file (sample_<idx>.json) immediately after processing. '
                             'Enables per-sample crash recovery; supersedes --checkpoint_interval when set.')
    parser.add_argument('--indices_file', default=None, help='JSON file containing explicit list of sample indices to process')
    parser.add_argument('--out_dir', default=None, help='Override output directory (instead of auto-computed shard_dir)')
    return parser.parse_args()


def get_frame_count(video_path: str) -> int:
    """Quickly get total frame count from container metadata (no decoding)."""
    try:
        import decord
        vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        return max(len(vr), 1)
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(count, 1)
    except Exception:
        return 1


def balanced_shard_by_frames(
    frame_counts: list, world_size: int, rank: int
) -> list:
    """Greedy LPT scheduling: assign videos to ranks to balance total frame count.

    Sorts videos longest-first, then assigns each to the rank with the smallest
    current total, minimising the makespan across all ranks.
    """
    indexed = sorted(enumerate(frame_counts), key=lambda x: -x[1])
    rank_totals = [0] * world_size
    rank_indices = [[] for _ in range(world_size)]
    for orig_idx, fc in indexed:
        min_rank = min(range(world_size), key=lambda r: rank_totals[r])
        rank_indices[min_rank].append(orig_idx)
        rank_totals[min_rank] += fc
    total_frames = sum(frame_counts)
    print(
        f'Frame-balanced sharding: rank {rank} gets {len(rank_indices[rank])} samples, '
        f'~{rank_totals[rank]} frames '
        f'({100 * rank_totals[rank] / max(total_frames, 1):.1f}% of {total_frames} total)'
    )
    return sorted(rank_indices[rank])


def shard_dir(save_dir: Path, rank: int) -> Path:
    return save_dir / f'rank_{rank}'


def merge_shards(save_dir: Path, world_size: int, benchmark: str, clip_model: str, batch_size: int) -> None:
    """Merge per-rank shard files into the final output files in save_dir."""
    list_keys = ('scores.json', 'frames.json', 'queries.json', 'video_paths.json', 'sample_indices.json')

    merged: dict[str, list] = {k: [] for k in list_keys}
    total_samples = 0
    shard_meta = {}

    for rank in range(world_size):
        d = shard_dir(save_dir, rank)
        meta_path = d / 'metadata.json'
        if not meta_path.exists():
            raise FileNotFoundError(f'Shard {rank} metadata not found: {meta_path}. Run all ranks before merging.')
        with open(meta_path) as f:
            shard_meta[rank] = json.load(f)

        for key in list_keys:
            with open(d / key) as f:
                data = json.load(f)
            merged[key].extend(data)
            if key == 'scores.json':
                total_samples += len(data)

    # Sort by sample_indices to restore original dataset order
    order = sorted(range(total_samples), key=lambda i: merged['sample_indices.json'][i])
    for key in list_keys:
        merged[key] = [merged[key][i] for i in order]

    sample_fps = shard_meta[0].get('sample_fps', 0.0)
    outputs = dict(merged)
    outputs['metadata.json'] = {
        'benchmark': benchmark,
        'frame_coverage': 'all_frames',
        'clip_model': clip_model,
        'batch_size': batch_size,
        'sample_fps': sample_fps,
        'num_samples': total_samples,
        'world_size': world_size,
        'query_mode': 'question_only',
        'clip_scoring_source': 'tools.save_clip_score.score_queries',
    }

    for filename, payload in outputs.items():
        output_path = save_dir / filename
        with open(output_path, 'w') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f'Wrote {output_path}')

    print(f'Merged {total_samples} samples from {world_size} shards.')


def main() -> None:
    args = parse_args()

    if not (0 <= args.rank < args.world_size):
        raise ValueError(f'Invalid rank={args.rank} for world_size={args.world_size}.')

    print(f'[Rank {args.rank}] startup: benchmark={args.benchmark}, world_size={args.world_size}', flush=True)

    save_dir = Path(args.save_root) / args.benchmark / 'allframes'
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        merge_shards(save_dir, args.world_size, args.benchmark, args.clip_model, args.batch_size)
        return

    print(f'[Rank {args.rank}] importing transformers CLIP classes...', flush=True)
    import transformers  # noqa: F401
    print(f'[Rank {args.rank}] transformers import done', flush=True)

    dataset, build_kwargs = build_extraction_dataset(
        args.benchmark, stream_context=args.stream_context,
    )
    print(f'[Rank {args.rank}] [build] dataset={args.benchmark} build_kwargs={build_kwargs}', flush=True)

    total = len(dataset.data)

    if args.indices_file is not None:
        # Explicit sample indices provided via file (used for sub-rank parallelism)
        with open(args.indices_file) as f:
            sample_indices = json.load(f)
        args.start = sample_indices[0] if sample_indices else 0
        args.end = sample_indices[-1] + 1 if sample_indices else 0
    elif args.no_balance:
        # Legacy contiguous chunk split
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
        sample_indices = list(range(args.start, args.end))
    else:
        # Frame-balanced sharding: scan frame counts then use LPT scheduling.
        # Results are cached in save_dir/frame_counts.json for reuse.
        frame_counts_path = save_dir / 'frame_counts.json'
        if frame_counts_path.exists():
            with open(frame_counts_path) as f:
                frame_counts = json.load(f)
            print(f'Loaded cached frame counts from {frame_counts_path}')
        else:
            print(f'Scanning frame counts for {total} samples...')
            frame_counts = []
            scan_range = range(total)
            if tqdm is not None:
                scan_range = tqdm(scan_range, desc='Scanning frames', total=total)
            for i in scan_range:
                try:
                    msg = dataset.build_prompt(i, video_llm=True)
                    video_items = [item for item in msg if item['type'] == 'video']
                    fc = get_frame_count(source_video_for_item(video_items[0])) if video_items else 1
                except Exception:
                    fc = 1
                frame_counts.append(fc)
            with open(frame_counts_path, 'w') as f:
                json.dump(frame_counts, f)
            print(f'Saved frame counts to {frame_counts_path}')
        sample_indices = balanced_shard_by_frames(frame_counts, args.world_size, args.rank)
        # Keep start/end for logging
        args.start = sample_indices[0] if sample_indices else 0
        args.end = sample_indices[-1] + 1 if sample_indices else 0

    # Decide output directory
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = shard_dir(save_dir, args.rank) if args.world_size > 1 else save_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    if args.device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(
        f'[Rank {args.rank}] loading CLIP processor: {args.clip_model} '
        f'(local_files_only={args.clip_local_files_only})',
        flush=True,
    )
    try:
        clip_processor, clip_model = load_clip_components(
            args.clip_model,
            local_files_only=args.clip_local_files_only,
            rank=args.rank,
        )
    except Exception as err:
        if args.clip_local_files_only:
            raise RuntimeError(
                f'Failed to load CLIP `{args.clip_model}` from local cache. '
                'Pre-download it once, pass a local --clip_model path, or set '
                'CLIP_LOCAL_FILES_ONLY=0 to allow HuggingFace network access.'
            ) from err
        raise
    clip_model = clip_model.to(device).eval()
    if device.type == 'cuda':
        clip_model = clip_model.half()
    print(f'[Rank {args.rank}] CLIP model ready on {device}', flush=True)

    # --- Load checkpoint / per-sample files if exists ---
    ckpt_path = out_dir / 'checkpoint.json'
    all_scores: list[list[float]] = []
    all_frames: list[list[int]] = []
    all_queries: list[str] = []
    all_video_paths: list[str] = []
    all_indices: list[int] = []

    if args.per_sample_save:
        # Resume by scanning already-written per-sample files
        done_set: set[int] = set()
        existing_sample_files = sorted(out_dir.glob('sample_*.json'))
        for sf in existing_sample_files:
            try:
                with open(sf) as f:
                    s = json.load(f)
                all_scores.append(s['scores'])
                all_frames.append(s['frames'])
                all_queries.append(s['query'])
                all_video_paths.append(s['video_path'])
                all_indices.append(s['dataset_index'])
                done_set.add(s['sample_idx'])
            except Exception as e:
                print(f'[Rank {args.rank}] Warning: could not load {sf}: {e}')
        if done_set:
            sample_indices = [i for i in sample_indices if i not in done_set]
            print(f'[Rank {args.rank}] Resumed from {len(done_set)} per-sample files, {len(sample_indices)} remaining')
    elif ckpt_path.exists():
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        all_scores     = ckpt['scores']
        all_frames     = ckpt['frames']
        all_queries    = ckpt['queries']
        all_video_paths = ckpt['video_paths']
        all_indices    = ckpt['sample_indices']
        done_set = set(ckpt['done_sample_idx'])
        sample_indices = [i for i in sample_indices if i not in done_set]
        print(f'[Rank {args.rank}] Resumed from checkpoint: {len(done_set)} done, {len(sample_indices)} remaining')
    else:
        done_set = set()

    # Pre-scan: group sample_indices by video path to avoid re-encoding same video
    print(f'[Rank {args.rank}] Scanning sample→video mapping...')
    # video_path -> [(sample_idx, query, dataset_index), ...]
    from collections import OrderedDict
    video_groups: dict = OrderedDict()
    # video_path -> max segment end_time (seconds); None means encode all frames
    # (full-video benchmark, or any sample lacking a segment).
    video_end_caps: dict = {}
    for sample_idx in sample_indices:
        raw_message = dataset.build_prompt(sample_idx, video_llm=True)
        if raw_message is None:
            raise ValueError(f'dataset.build_prompt returned None for sample {sample_idx}')
        video_items = [item for item in raw_message if item['type'] == 'video']
        if len(video_items) != 1:
            raise ValueError(
                f'Expected exactly one video item for sample {sample_idx}, got {len(video_items)}: {raw_message}'
            )
        video_path = source_video_for_item(video_items[0])
        query = normalize_clip_query(' '.join(str(item['value']) for item in raw_message if item['type'] == 'text') or '')
        dataset_index = int(dataset.data.iloc[sample_idx]['index'])
        if video_path not in video_groups:
            video_groups[video_path] = []
        video_groups[video_path].append((sample_idx, query, dataset_index))

        # Track the latest question timestamp per source video; frames past it are
        # never used by any QA, so encoding can stop there. A missing segment means
        # the whole video is needed (full-video benchmarks) -> sticky None.
        seg = video_items[0].get('_segment_times') if isinstance(video_items[0], dict) else None
        end_t = float(seg[1]) if seg else None
        if video_path not in video_end_caps:
            video_end_caps[video_path] = end_t
        else:
            prev = video_end_caps[video_path]
            video_end_caps[video_path] = None if (prev is None or end_t is None) else max(prev, end_t)

    # Roots used to relativise stored video paths. Includes frame_root so the
    # per-question clips of streaming benchmarks key cleanly (see
    # collect_video_roots). VideoBaseDataset has data_root directly;
    # ConcatVideoDataset (e.g. MLVU) is covered via its sub-datasets.
    _video_roots = collect_video_roots(dataset)

    n_videos = len(video_groups)
    n_samples = len(sample_indices)
    print(f'Rank        : {args.rank}/{args.world_size}')
    print(f'Benchmark   : {args.benchmark}')
    print(f'Frame mode  : all frames')
    print(f'Videos      : {n_videos} unique / {n_samples} samples remaining')
    print(f'Balanced    : {not args.no_balance}')
    print(f'Sample FPS  : {args.sample_fps if args.sample_fps > 0 else "all"}')
    print(f'Decode      : {args.decode_backend}')
    print(f'Preprocess  : {args.preprocess_backend}')
    print(f'CLIP model  : {args.clip_model}')
    print(f'Device      : {device}')
    print(f'Save dir    : {out_dir}')

    def save_checkpoint(done_sample_idx: list) -> None:
        tmp = ckpt_path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump({
                'scores': all_scores,
                'frames': all_frames,
                'queries': all_queries,
                'video_paths': all_video_paths,
                'sample_indices': all_indices,
                'done_sample_idx': done_sample_idx,
            }, f)
        tmp.replace(ckpt_path)

    done_sample_idx: list[int] = list(done_set)

    video_range = list(video_groups.items())
    if tqdm is not None:
        video_range = tqdm(video_range, desc=f'Rank {args.rank}', total=n_videos)

    samples_processed = 0
    for video_i, (video_path, group) in enumerate(video_range):
        # Encode this video's frames once, reuse for all questions
        img_features, frame_indices = encode_video_frames(
            video_path=video_path,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=device,
            batch_size=args.batch_size,
            sample_fps=args.sample_fps,
            end_time=video_end_caps.get(video_path),
            preprocess_backend=args.preprocess_backend,
            decode_backend=args.decode_backend,
        )
        # Move to GPU once per video; score_query reuses without re-uploading
        img_features_on_device = img_features.to(device)
        if device.type == 'cuda':
            img_features_on_device = img_features_on_device.half()

        rel_path = relative_video_path_from_roots(video_path, _video_roots)

        group_queries = [query for _, query, _ in group]
        group_scores = score_queries(img_features_on_device, group_queries, clip_model, clip_processor, device)

        for (sample_idx, query, dataset_index), scores in zip(group, group_scores):

            all_scores.append([float(s) for s in scores])
            all_frames.append(frame_indices)
            all_queries.append(query)
            all_video_paths.append(rel_path)
            all_indices.append(dataset_index)
            done_sample_idx.append(sample_idx)
            samples_processed += 1

            if args.per_sample_save:
                sample_file = out_dir / f'sample_{sample_idx}.json'
                tmp_sf = sample_file.with_suffix('.tmp')
                with open(tmp_sf, 'w') as f:
                    json.dump({
                        'sample_idx': sample_idx,
                        'dataset_index': dataset_index,
                        'scores': [float(v) for v in scores],
                        'frames': frame_indices,
                        'query': query,
                        'video_path': rel_path,
                    }, f)
                tmp_sf.replace(sample_file)

        if (video_i + 1) % 20 == 0 or video_i == n_videos - 1:
            message = (
                f'[Rank {args.rank}] {video_i + 1}/{n_videos} videos, '
                f'{samples_processed}/{n_samples} samples '
                f'(last: {len(frame_indices)} frames, {len(group)} queries)'
            )
            if tqdm is not None:
                tqdm.write(message)
            else:
                print(message)

        if args.checkpoint_interval > 0 and (video_i + 1) % args.checkpoint_interval == 0:
            save_checkpoint(done_sample_idx)
            if tqdm is not None:
                tqdm.write(f'[Rank {args.rank}] Checkpoint saved ({samples_processed} samples)')
            else:
                print(f'[Rank {args.rank}] Checkpoint saved ({samples_processed} samples)')

    outputs = {
        'scores.json': all_scores,
        'frames.json': all_frames,
        'queries.json': all_queries,
        'video_paths.json': all_video_paths,
        'sample_indices.json': all_indices,
        'metadata.json': {
            'benchmark': args.benchmark,
            'frame_coverage': 'all_frames',
            'clip_model': args.clip_model,
            'batch_size': args.batch_size,
            'sample_fps': args.sample_fps,
            'decode_backend': args.decode_backend,
            'preprocess_backend': args.preprocess_backend,
            'device': str(device),
            'num_samples': len(all_scores),
            'frame_balanced': not args.no_balance,
            'rank': args.rank,
            'world_size': args.world_size,
            'query_mode': 'question_only',
            'clip_scoring_source': 'tools.save_clip_score.score_queries',
        },
    }

    for filename, payload in outputs.items():
        output_path = out_dir / filename
        with open(output_path, 'w') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f'[Rank {args.rank}] Wrote {output_path}')

    # Remove checkpoint after successful completion
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f'[Rank {args.rank}] Checkpoint removed')


if __name__ == '__main__':
    main()
