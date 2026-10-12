from __future__ import annotations

import logging
import os
import re
import time
import warnings

import torch

from ..base import BaseModel
from ..framepack_sampling import require_redi_frames
from ...local_paths import asset_path
from ..qwen3_5_framepack.model_wrapper import (
    _restrict_offline_candidates_to_segment,
)
from ...debug_logger import eval_vllm_log, eval_vllm_log_exception
from ...smp import get_gpu_memory

FRAME_FACTOR = 2  # temporal_patch_size=2
VLLM_MAX_IMAGE_INPUT_NUM = 24


def is_moe_model(model_path: str) -> bool:
    """Check if the model path/name contains an active-parameter MoE suffix."""
    return re.search(r'-A\d+B', model_path) is not None


def read_max_position_embeddings(model_path: str) -> int | None:
    try:
        import json
        with open(os.path.join(model_path, 'config.json'), encoding='utf-8') as f:
            config = json.load(f)
        value = config.get('max_position_embeddings')
        if value is None and isinstance(config.get('text_config'), dict):
            value = config['text_config'].get('max_position_embeddings')
        return int(value) if value is not None else None
    except Exception:
        return None


def read_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value in (None, ''):
        return None
    try:
        return int(value)
    except ValueError as err:
        raise ValueError(f'{name} must be an integer, got {value!r}.') from err


def read_env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value in (None, ''):
        return None
    try:
        return float(value)
    except ValueError as err:
        raise ValueError(f'{name} must be a float, got {value!r}.') from err


def read_env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value in (None, ''):
        return None
    normalized = value.strip().lower()
    if normalized in ('1', 'true', 'yes', 'on'):
        return True
    if normalized in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'{name} must be a boolean value, got {value!r}.')


def patch_vllm_qwen_pixel_inputs() -> None:
    """Allow vLLM Qwen parsers to accept preprocessed pixel tensors.

    vLLM's Qwen2/Qwen3 data parser accepts raw media or precomputed embeddings
    by default. FramePack already runs a custom processor that produces
    pixel_values(_videos) with variable per-grid resolutions, so we pass those
    tensors through directly and let vLLM run the vision encoder.
    """
    os.environ.setdefault('SETUPTOOLS_USE_DISTUTILS', 'local')
    from vllm.model_executor.models import qwen2_vl
    from vllm.multimodal.parse import DictEmbeddingItems

    parser_cls = qwen2_vl.Qwen2VLMultiModalDataParser
    if getattr(parser_cls, '_vlmeval_pixel_inputs_patch', False):
        return

    original_parse_image_data = parser_cls._parse_image_data
    original_parse_video_data = parser_cls._parse_video_data

    def _parse_image_data(self, data):
        if isinstance(data, dict):
            if 'image_embeds' in data:
                required_fields = {'image_embeds', 'image_grid_thw'}
            else:
                required_fields = {'pixel_values', 'image_grid_thw'}
            return DictEmbeddingItems(
                data,
                modality='image',
                required_fields=required_fields,
                fields_factory=qwen2_vl._create_qwen2vl_field_factory(self._spatial_merge_size),
            )
        return original_parse_image_data(self, data)

    def _parse_video_data(self, data):
        if isinstance(data, dict):
            if 'video_embeds' in data:
                required_fields = {'video_embeds', 'video_grid_thw'}
            else:
                required_fields = {'pixel_values_videos', 'video_grid_thw'}
            return DictEmbeddingItems(
                data,
                modality='video',
                required_fields=required_fields,
                fields_factory=qwen2_vl._create_qwen2vl_field_factory(self._spatial_merge_size),
            )
        return original_parse_video_data(self, data)

    parser_cls._parse_image_data = _parse_image_data
    parser_cls._parse_video_data = _parse_video_data
    parser_cls._vlmeval_pixel_inputs_patch = True


def load_qwen3_5_moe_model(model_path: str, device: str):
    from .model.configuration_qwen3_5_moe import Qwen3_5MoeConfig
    from .model.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

    config = Qwen3_5MoeConfig.from_pretrained(model_path)
    return Qwen3_5MoeForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype='auto',
        device_map={'': device},
        attn_implementation='flash_attention_2',
    )


def debug_video_pipeline_enabled() -> bool:
    return os.environ.get('DEBUG_VIDEO_PIPELINE', '0') == '1'


def debug_video_pipeline(prefix: str, **kwargs) -> None:
    if not debug_video_pipeline_enabled():
        return
    eval_vllm_log(prefix, **kwargs)


def resolve_process_cuda_device() -> tuple[str, int]:
    """Resolve the CUDA device visible to the current worker process."""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        raise RuntimeError('No CUDA device is visible to the current process.')

    if device_count == 1:
        return 'cuda:0', 0

    if 0 <= local_rank < device_count:
        return f'cuda:{local_rank}', local_rank

    logging.warning(
        'LOCAL_RANK=%s exceeds visible CUDA device count=%s after remapping; '
        'falling back to cuda:0 for this worker.',
        local_rank,
        device_count,
    )
    return 'cuda:0', 0


def bind_process_cuda_device() -> str:
    device, device_index = resolve_process_cuda_device()
    torch.cuda.set_device(device_index)
    return device


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image path: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video path: {video}')


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> block produced by Qwen3.5 thinking mode."""
    if '</think>' in text:
        text = text.split('</think>', 1)[-1].lstrip()
    return text


def fill_missing_video_metadata_fps(video_metadatas, fallback_fps: float | None) -> None:
    if fallback_fps is None or video_metadatas is None:
        return
    for meta in video_metadatas:
        if isinstance(meta, dict):
            if meta.get('fps') is None:
                meta['fps'] = fallback_fps
        elif getattr(meta, 'fps', None) is None:
            meta.fps = fallback_fps


def build_frame_list_video_metadata(
    frames,
    frame_indices,
    fps: float | None,
    total_num_frames: int | None = None,
):
    height = None
    width = None
    if frames:
        first_frame = frames[0]
        if hasattr(first_frame, 'size'):
            width, height = first_frame.size
        elif hasattr(first_frame, 'shape') and len(first_frame.shape) >= 2:
            height, width = first_frame.shape[0], first_frame.shape[1]

    if total_num_frames is None:
        total_num_frames = len(frame_indices) if frame_indices is not None else len(frames)

    indices = list(frame_indices) if frame_indices is not None else list(range(len(frames)))
    return dict(
        total_num_frames=total_num_frames,
        fps=fps,
        duration=(float(total_num_frames) / float(fps)) if fps else None,
        frames_indices=indices,
        height=height,
        width=width,
    )


def resize_frames_long_edge(frames: list, target_long_edge: int | None, align_factor: int = 32) -> list:
    """Resize frames proportionally so their long edge does not exceed target_long_edge."""
    if not target_long_edge or target_long_edge <= 0:
        return frames

    from PIL import Image as PILImage

    resized = []
    for frame in frames:
        width, height = frame.size
        long_edge = max(width, height)
        if long_edge <= target_long_edge:
            resized.append(frame)
            continue

        scale = float(target_long_edge) / float(long_edge)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        if align_factor and align_factor > 1:
            new_width = max(
                align_factor,
                min(target_long_edge, (new_width // align_factor) * align_factor),
            )
            new_height = max(
                align_factor,
                min(target_long_edge, (new_height // align_factor) * align_factor),
            )
        resized.append(frame.resize((new_width, new_height), PILImage.BILINEAR))
    return resized


class Qwen3_5MoeChat(BaseModel):
    """VLMEvalKit wrapper for Qwen3.5-MoE vision-language models.

    Key difference from Qwen3VLChat: disables thinking output by default
    (enable_thinking=False in apply_chat_template) and strips any residual
    <think>...</think> blocks so benchmark scoring is not polluted.

    Requires vlmeval-qwen35 environment (transformers >= 5.5).
    """

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        nframe: int = 128,
        fps: float | None = None,
        max_new_tokens: int = 2048,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        enable_thinking: bool = False,
        system_prompt: str | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.nframe = nframe
        self.fps = fps
        self.enable_thinking = enable_thinking
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        from transformers import AutoProcessor

        gpu_mems = get_gpu_memory()
        assert gpu_mems and max(gpu_mems) > 0, 'No GPU found'
        device = bind_process_cuda_device()
        self.device = device

        self.model = load_qwen3_5_moe_model(model_path, device)
        self.model.eval()
        torch.cuda.empty_cache()

        self.processor = AutoProcessor.from_pretrained(model_path)

    def _prepare_content(self, inputs: list[dict], dataset: str | None = None) -> list[dict]:
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                for key in ['min_pixels', 'max_pixels', 'resized_height', 'resized_width']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
                content.append(item)

            elif s['type'] == 'video':
                value = s['value']
                if isinstance(value, list):
                    item = {
                        'type': 'video',
                        'video': [ensure_image_url(v) if isinstance(v, str) else v for v in value],
                    }
                else:
                    item = {'type': 'video', 'video': ensure_video_url(value)}
                    if self.fps is not None:
                        item['fps'] = self.fps
                    else:
                        target = self.nframe
                        try:
                            import cv2
                            cap = cv2.VideoCapture(value)
                            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            cap.release()
                            if 0 < total < target:
                                target = max(FRAME_FACTOR, total // FRAME_FACTOR * FRAME_FACTOR)
                                print(f'[Qwen3_5MoeChat] {value}: total={total}, clamped nframes={target}')
                        except Exception:
                            pass
                        item['nframes'] = target
                content.append(item)

            elif s['type'] == 'text':
                content.append({'type': 'text', 'text': s['value']})
            else:
                raise ValueError(f"Unsupported message type: {s['type']}")
        return content

    def generate_inner(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as err:
            logging.critical("Please install via 'pip install qwen-vl-utils'")
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})

        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        # Disable thinking by default so benchmark answers are clean
        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # Processor version doesn't support enable_thinking; fall back
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
            fill_missing_video_metadata_fps(video_metadatas, self.fps)

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            do_resize=False,
            return_mm_token_type_ids=True,
            return_tensors='pt',
            **(video_kwargs or {}),
        )

        try:
            inputs = inputs.to(self.model.device)
            if hasattr(self.model, 'dtype'):
                inputs = inputs.to(self.model.dtype)
        except Exception:
            inputs = inputs.to(self.device)
            if hasattr(self.model, 'dtype'):
                inputs = inputs.to(self.model.dtype)

        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)

        # Strip input prefix
        generated_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Safety net: strip residual thinking block even when enable_thinking=False
        response = strip_thinking(response)

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response


# ---------------------------------------------------------------------------
# FramePack helpers
# ---------------------------------------------------------------------------

def parse_compression_scheme(scheme: str) -> dict[int, int]:
    """Parse scheme string like 'd1f128d2f256' → {1: 128, 2: 256}.

    Each ``d{ratio}f{count}`` token means *count* frames will be rendered at
    1/ratio of the original resolution (in each spatial dimension).
    """
    matches = re.findall(r'd(\d+)f(\d+)', scheme)
    reconstructed = ''.join(f'd{d}f{f}' for d, f in matches)
    if reconstructed != scheme:
        raise ValueError(
            f"Compression scheme {scheme!r} contains unexpected characters. "
            f"Parsed portion: {reconstructed!r}."
        )
    parsed: dict[int, int] = {}
    for d_str, f_str in matches:
        d, f = int(d_str), int(f_str)
        if d <= 0 or f <= 0:
            raise ValueError(f"depth and frame count must be positive, got d={d}, f={f}.")
        if d in parsed:
            raise ValueError(f"Duplicate depth d={d} in scheme {scheme!r}.")
        parsed[d] = f
    return parsed


def binary_tree_ranks(num_frames: int) -> list[int]:
    """Assign priority ranks via breadth-first binary-tree ordering.

    The centre frame gets rank 0 (highest priority).  Then the midpoints of
    the left and right halves get ranks 1 and 2, and so on recursively.
    Lower rank = higher priority = assigned to the finest resolution tier.
    """
    if num_frames <= 0:
        return []
    ranks = [0] * num_frames

    def _assign(lo: int, hi: int, rank: int) -> None:
        if lo > hi:
            return
        mid = (lo + hi) // 2
        ranks[mid] = rank
        _assign(lo, mid - 1, rank + (hi - lo) // 2 + 1)
        _assign(mid + 1, hi, rank + (hi - lo) // 2 + 1)

    # Use iterative BFS to avoid recursion depth issues on large N
    from collections import deque
    q: deque[tuple[int, int, int]] = deque()
    q.append((0, num_frames - 1, 0))
    current_rank = 0
    level_nodes: list[tuple[int, int]] = []

    q2: deque[tuple[int, int]] = deque([(0, num_frames - 1)])
    rank_counter = 0
    while q2:
        lo, hi = q2.popleft()
        if lo > hi:
            continue
        mid = (lo + hi) // 2
        ranks[mid] = rank_counter
        rank_counter += 1
        q2.append((lo, mid - 1))
        q2.append((mid + 1, hi))
    return ranks


def _scale_pil_image(img, ratio: int):
    """Downscale a PIL Image by 1/ratio in each spatial dimension."""
    if ratio == 1:
        return img
    from PIL import Image as PILImage
    w, h = img.size
    new_w = max(1, w // ratio)
    new_h = max(1, h // ratio)
    return img.resize((new_w, new_h), PILImage.BILINEAR)


def _apply_scheme_to_frames(
    frames: list,
    scheme_dict: dict[int, int],
    priority_ranks: list[int],
) -> list:
    """Apply per-frame resolution scaling based on scheme and priority ranks.

    Frames with lower rank (higher priority) get finer resolution tiers (d1).
    Frames with higher rank get coarser tiers (d2, d4, …).

    Returns a new list of (possibly downscaled) PIL Images in the original
    temporal order.
    """
    num_frames = len(frames)
    # Build rank → ratio mapping: sort tiers by depth ascending so best-ranked
    # frames get the smallest depth (highest resolution).
    sorted_tiers = sorted(scheme_dict.items())  # [(depth, count), ...]

    # Adjust scheme total to actual frame count
    total_scheme = sum(scheme_dict.values())
    if num_frames < total_scheme:
        excess = total_scheme - num_frames
        adjusted = dict(sorted_tiers)
        for depth in sorted(adjusted.keys(), reverse=True):
            if excess <= 0:
                break
            reduction = min(adjusted[depth], excess)
            adjusted[depth] -= reduction
            excess -= reduction
        sorted_tiers = [(d, c) for d, c in sorted(adjusted.items()) if c > 0]

    cumulative = 0
    tier_boundaries: list[tuple[int, int]] = []  # (cumulative_end, depth)
    for depth, count in sorted_tiers:
        cumulative += count
        tier_boundaries.append((cumulative, depth))

    def _rank_to_depth(r: int) -> int:
        for boundary, depth in tier_boundaries:
            if r < boundary:
                return depth
        return tier_boundaries[-1][1]

    scaled = []
    for frame, rank in zip(frames, priority_ranks):
        depth = _rank_to_depth(rank)
        scaled.append(_scale_pil_image(frame, depth))
    return scaled


# ---------------------------------------------------------------------------
# Offline CLIP helpers for REDI relevance scores
# ---------------------------------------------------------------------------

def _normalize_clip_video_path(video_path: str) -> str:
    path = str(video_path)
    if path.startswith('file://'):
        path = path[len('file://'):]
    return os.path.normpath(path).replace('\\', '/')


_MCQ_ANSWER_SUFFIXES = (
    "\nAnswer with the option's letter from the given choices directly.",
)


def _normalize_clip_query(query: str) -> str:
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


def _has_offline_clip_files(clip_scores_dir: str) -> bool:
    required = ('scores.json', 'frames.json', 'video_paths.json', 'queries.json')
    return all(os.path.exists(os.path.join(clip_scores_dir, f)) for f in required)


def _has_frame_features_manifest(features_root: str) -> bool:
    return os.path.isfile(os.path.join(features_root, 'manifest.json'))


def _redi_dist_summary(t: torch.Tensor) -> str:
    """Compact one-line distribution summary: min/p25/p50/p75/max | mean±std."""
    if t.numel() == 0:
        return '(empty)'
    tf = t.detach().float().flatten()
    qs = torch.quantile(tf, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    mn, p25, p50, p75, mx = (float(x) for x in qs)
    mean = float(tf.mean())
    std = float(tf.std(unbiased=False))
    return (f'min={mn:.4f} p25={p25:.4f} p50={p50:.4f} '
            f'p75={p75:.4f} max={mx:.4f} | mean={mean:.4f}±{std:.4f}')


def _candidate_clip_dataset_names(dataset: str | None) -> list[str]:
    if not dataset:
        return []
    candidates = [dataset]
    if '_' in dataset:
        candidates.append(dataset.split('_', 1)[0])
    deduped: list[str] = []
    for c in candidates:
        if c and c not in deduped:
            deduped.append(c)
    return deduped


# ---------------------------------------------------------------------------
# Qwen3_5MoeFramePackChat
# ---------------------------------------------------------------------------

class Qwen3_5MoeFramePackChat(BaseModel):
    """Qwen3.5-MoE wrapper with REDI/uniform frame sampling and scheme-based
    per-frame resolution allocation.

    Scheme format: ``'d1f128d2f256'``  →  128 frames at full resolution,
    256 frames at 1/2 resolution (each spatial dimension).

    frame_sampling_strategy='uniform':
        Sample ``nframe`` (= scheme total) frames uniformly, then assign
        resolution tiers by binary-tree priority rank (temporal midpoints → d1).

    frame_sampling_strategy='redi':
        MMR-style selection that balances query-relevance and diversity.
        First frame = argmax CLIP-relevance. Subsequent frames maximize
        ``score(i) = lambda * r_i - (1 - lambda) * max_{j in S} sim(v_i, v_j)``,
        using offline CLIP scores for ``r_i`` and offline image features for
        ``sim``. Resolution tiers follow MMR-selection order: 1st-selected
        → d1, 2nd → next slot of the scheme, etc.
    """

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        nframe: int = 128,
        fps: float | None = None,
        max_new_tokens: int = 2048,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        enable_thinking: bool = False,
        system_prompt: str | None = None,
        verbose: bool = False,
        # FramePack params
        frame_sampling_strategy: str = 'redi',
        clip_nframe: int = 128,
        clip_scores_dir: str | None = asset_path('clip_scores'),
        frame_features_dir: str | None = asset_path('frame_feature'),
        redi_lambda: float = 0.7,
        compression_scheme: str | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        pre_resize_long_edge: int | None = None,
        mlvu_pre_resize_long_edge: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_vllm = bool(kwargs.pop('use_vllm', False))
        debug_video_pipeline(
            'Qwen3_5MoeFramePackChat.init_start',
            model_path=model_path,
            use_vllm=self.use_vllm,
            nframe=nframe,
            fps=fps,
            frame_sampling_strategy=frame_sampling_strategy,
            clip_nframe=clip_nframe,
            compression_scheme=compression_scheme,
            pre_resize_long_edge=pre_resize_long_edge,
            kwargs=kwargs,
        )
        self.model_path = model_path
        self.fps = fps
        self.enable_thinking = enable_thinking
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.pre_resize_long_edge = pre_resize_long_edge
        self.mlvu_pre_resize_long_edge = mlvu_pre_resize_long_edge
        self._hf_max_position_embeddings = read_max_position_embeddings(model_path)

        # Scheme parsing
        self.compression_scheme = compression_scheme
        self._scheme_dict: dict[int, int] = {}
        if compression_scheme is not None:
            self._scheme_dict = parse_compression_scheme(compression_scheme)
            scheme_total = sum(self._scheme_dict.values())
            self.nframe = scheme_total
            self.clip_nframe = scheme_total
            print(f'*** Compression scheme: {compression_scheme} → total {scheme_total} frames, '
                  f'tiers: {self._scheme_dict}')
        else:
            self.nframe = nframe
            self.clip_nframe = clip_nframe

        if frame_sampling_strategy not in ('redi', 'uniform'):
            raise ValueError(f"frame_sampling_strategy must be 'redi' or 'uniform', got {frame_sampling_strategy!r}")
        self.frame_sampling_strategy = frame_sampling_strategy
        self.redi_lambda = float(redi_lambda)
        if self.frame_sampling_strategy == 'redi':
            if frame_features_dir is None:
                raise ValueError("frame_sampling_strategy='redi' requires `frame_features_dir`.")
            print(f'*** Frame Sampling Strategy: redi '
                  f'(clip_nframe={self.clip_nframe}, lambda={self.redi_lambda})')
        else:
            print(f'*** Frame Sampling Strategy: uniform (nframe={self.nframe})')

        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        # Offline CLIP state
        self._clip_scores_dir_root = clip_scores_dir
        self._loaded_offline_clip_dir: str | None = None
        self._warned_missing_offline_clip_targets: set[tuple[str | None, int]] = set()
        self._offline_clip: dict | None = None
        self._offline_clip_by_query: dict | None = None
        self._offline_clip_by_query_content: dict | None = None
        if clip_scores_dir is not None and not os.path.isdir(clip_scores_dir):
            print(f'Warning: clip_scores_dir={clip_scores_dir} does not exist.')

        # Offline frame-feature state (for 'redi' strategy)
        self._frame_features_root = frame_features_dir
        # Per-features-root manifest: { features_root: { norm_path_suffix: feature_abs_path } }
        self._frame_features_manifest_cache: dict[str, dict[str, str]] = {}
        # Per-video tensor cache: { feature_abs_path: (features_tensor, frame_indices_list) }
        self._frame_features_tensor_cache: dict[str, tuple[torch.Tensor, list[int]]] = {}
        if frame_features_dir is not None and not os.path.isdir(frame_features_dir):
            print(f'Warning: frame_features_dir={frame_features_dir} does not exist.')

        from transformers import AutoImageProcessor, AutoTokenizer
        gpu_mems = get_gpu_memory()
        assert gpu_mems and max(gpu_mems) > 0, 'No GPU found'
        device = bind_process_cuda_device()
        self.device = device

        # Build a custom processor using the local variable-resolution video processor.
        # This replaces AutoProcessor so that PIL-list videos of mixed resolutions are
        # processed frame-group by frame-group (temporal_patch_size=2) with intra-group
        # padding, rather than trying to torch.stack all frames into a uniform 4-D tensor.
        from .model.processing_qwen3_5_moe import Qwen3_5Processor as Qwen3_5MoeProcessor
        from .model.video_processing_qwen3_5_moe import Qwen3_5VideoProcessor as Qwen3_5MoeVideoProcessor

        debug_video_pipeline('Qwen3_5MoeFramePackChat.processor_load_start', model_path=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        image_processor = AutoImageProcessor.from_pretrained(model_path)
        video_processor = Qwen3_5MoeVideoProcessor.from_pretrained(model_path)
        self.processor = Qwen3_5MoeProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=getattr(tokenizer, 'chat_template', None),
        )
        debug_video_pipeline('Qwen3_5MoeFramePackChat.processor_load_done', model_path=model_path)

        if self.use_vllm:
            os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
            os.environ.setdefault('VLLM_QWEN3_NEXT_GDN_BACKEND', 'auto')
            patch_vllm_qwen_pixel_inputs()
            from vllm import LLM

            gpu_count = torch.cuda.device_count()
            tp_size = gpu_count if gpu_count > 0 else 1
            limit_mm_per_prompt = kwargs.get(
                'limit_mm_per_prompt',
                {
                    'image': VLLM_MAX_IMAGE_INPUT_NUM,
                    # FramePack exposes one vLLM video item per temporal patch.
                    'video': max(1, (self.nframe + FRAME_FACTOR - 1) // FRAME_FACTOR),
                },
            )
            logging.info(
                'Using vLLM for %s FramePack inference with %s GPUs (available: %s), '
                'limit_mm_per_prompt=%s',
                self.model_path,
                tp_size,
                gpu_count,
                limit_mm_per_prompt,
            )
            env_max_model_len = read_env_int('VLLM_MAX_MODEL_LEN')
            if env_max_model_len:
                os.environ.setdefault('VLLM_ALLOW_LONG_MAX_MODEL_LEN', '1')
            env_enforce_eager = read_env_bool('VLLM_ENFORCE_EAGER')
            env_gpu_memory_utilization = read_env_float('VLLM_GPU_MEMORY_UTILIZATION')
            if env_gpu_memory_utilization is not None and not (0 < env_gpu_memory_utilization <= 1):
                raise ValueError(
                    'VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1], '
                    f'got {env_gpu_memory_utilization}.'
                )
            env_async_scheduling = read_env_bool('VLLM_ASYNC_SCHEDULING')

            llm_kwargs = dict(
                model=self.model_path,
                max_num_seqs=kwargs.get('max_num_seqs', 1),
                limit_mm_per_prompt=limit_mm_per_prompt,
                tensor_parallel_size=tp_size,
                enable_expert_parallel=is_moe_model(self.model_path),
                distributed_executor_backend=kwargs.get('distributed_executor_backend', 'mp'),
                seed=0,
                gpu_memory_utilization=(
                    env_gpu_memory_utilization
                    if env_gpu_memory_utilization is not None
                    else kwargs.get('gpu_utils', 0.9)
                ),
                trust_remote_code=True,
                enforce_eager=(
                    env_enforce_eager
                    if env_enforce_eager is not None
                    else kwargs.get('enforce_eager', False)
                ),
                disable_custom_all_reduce=kwargs.get('disable_custom_all_reduce', True),
                enable_mm_embeds=kwargs.get('enable_mm_embeds', True),
            )
            if env_async_scheduling is not None:
                llm_kwargs['async_scheduling'] = env_async_scheduling
            if kwargs.get('max_model_len') is not None:
                llm_kwargs['max_model_len'] = kwargs['max_model_len']
            elif env_max_model_len:
                llm_kwargs['max_model_len'] = env_max_model_len
            env_kv_cache_memory_bytes = read_env_int('VLLM_KV_CACHE_MEMORY_BYTES')
            if env_kv_cache_memory_bytes is not None:
                llm_kwargs['kv_cache_memory_bytes'] = env_kv_cache_memory_bytes
            env_max_num_batched_tokens = read_env_int('VLLM_MAX_NUM_BATCHED_TOKENS')
            if env_max_num_batched_tokens is not None:
                llm_kwargs['max_num_batched_tokens'] = env_max_num_batched_tokens
            env_enable_chunked_prefill = read_env_bool('VLLM_ENABLE_CHUNKED_PREFILL')
            if env_enable_chunked_prefill is not None:
                llm_kwargs['enable_chunked_prefill'] = env_enable_chunked_prefill
            env_disable_chunked_mm_input = read_env_bool('VLLM_DISABLE_CHUNKED_MM_INPUT')
            if env_disable_chunked_mm_input is not None:
                llm_kwargs['disable_chunked_mm_input'] = env_disable_chunked_mm_input
            env_disable_log_stats = read_env_bool('VLLM_DISABLE_LOG_STATS')
            if env_disable_log_stats is not None:
                llm_kwargs['disable_log_stats'] = env_disable_log_stats
            env_logging_iteration_details = read_env_bool('VLLM_ENABLE_LOGGING_ITERATION_DETAILS')
            if env_logging_iteration_details is not None:
                llm_kwargs['enable_logging_iteration_details'] = env_logging_iteration_details
            debug_video_pipeline('Qwen3_5MoeFramePackChat.vllm_init_start', llm_kwargs=llm_kwargs)
            try:
                self.llm = LLM(**llm_kwargs)
            except Exception as e:
                eval_vllm_log_exception(
                    'Qwen3_5MoeFramePackChat.vllm_init_failed',
                    e,
                    llm_kwargs=llm_kwargs,
                )
                raise
            debug_video_pipeline('Qwen3_5MoeFramePackChat.vllm_init_done')
        else:
            debug_video_pipeline('Qwen3_5MoeFramePackChat.transformers_init_start', model_path=model_path, device=device)
            self.model = load_qwen3_5_moe_model(model_path, device)
            self.model.eval()
            debug_video_pipeline('Qwen3_5MoeFramePackChat.transformers_init_done')
        torch.cuda.empty_cache()
        debug_video_pipeline('Qwen3_5MoeFramePackChat.init_done')

    # ------------------------------------------------------------------
    # Offline CLIP infrastructure
    # ------------------------------------------------------------------

    def _resolve_offline_clip_dir(self, dataset: str | None, frame_label: str | None = None) -> str | None:
        root_dir = self._clip_scores_dir_root
        if root_dir is None or not os.path.isdir(root_dir):
            return None
        if frame_label is None:
            frame_label = f'{self.nframe}frame'
        candidates: list[str] = []
        if dataset is not None:
            for ds_name in _candidate_clip_dataset_names(dataset):
                candidates.append(os.path.join(root_dir, ds_name, frame_label))
                candidates.append(os.path.join(root_dir, ds_name))
            candidates.append(os.path.join(root_dir, frame_label))
        candidates.append(root_dir)
        seen: list[str] = []
        for c in candidates:
            if c not in seen:
                seen.append(c)
        for c in seen:
            if os.path.isdir(c) and _has_offline_clip_files(c):
                return c
        return None

    def _load_offline_clip_from_dir(self, clip_scores_dir: str) -> None:
        import json
        with open(os.path.join(clip_scores_dir, 'scores.json')) as f:
            all_scores = json.load(f)
        with open(os.path.join(clip_scores_dir, 'frames.json')) as f:
            all_frame_indices = json.load(f)
        with open(os.path.join(clip_scores_dir, 'video_paths.json')) as f:
            video_paths = json.load(f)
        with open(os.path.join(clip_scores_dir, 'queries.json')) as f:
            all_queries = json.load(f)

        n = len(all_scores)
        if not (len(all_frame_indices) == n and len(video_paths) == n and len(all_queries) == n):
            raise ValueError('Offline CLIP metadata length mismatch.')

        offline_clip: dict = {}
        offline_clip_by_query: dict = {}
        offline_clip_by_query_content: dict = {}
        duplicates = 0
        for vpath, query, scores, frame_indices in zip(
            video_paths, all_queries, all_scores, all_frame_indices
        ):
            norm_path = _normalize_clip_video_path(vpath)
            norm_query = str(query)
            norm_query_content = _normalize_clip_query(norm_query)
            key = (norm_path, norm_query)
            payload = (scores, None if frame_indices is None else list(frame_indices))
            if key in offline_clip:
                duplicates += 1
                continue
            offline_clip[key] = payload
            offline_clip_by_query.setdefault(norm_query, []).append((norm_path, payload))
            offline_clip_by_query_content.setdefault(norm_query_content, []).append((norm_path, payload))

        if duplicates:
            print(f'Warning: {duplicates} duplicate offline CLIP keys (keeping first).')
        self._offline_clip = offline_clip
        self._offline_clip_by_query = offline_clip_by_query
        self._offline_clip_by_query_content = offline_clip_by_query_content
        self._loaded_offline_clip_dir = clip_scores_dir
        print(f'Loaded offline CLIP scores for {len(offline_clip)} samples from {clip_scores_dir}')

    def _clear_offline_clip(self) -> None:
        self._offline_clip = None
        self._offline_clip_by_query = None
        self._offline_clip_by_query_content = None
        self._loaded_offline_clip_dir = None

    def _ensure_offline_clip_loaded(self, dataset: str | None) -> None:
        root_dir = self._clip_scores_dir_root
        if root_dir is None:
            return
        if not os.path.isdir(root_dir):
            self._clear_offline_clip()
            return
        resolved = self._resolve_offline_clip_dir(dataset)
        if resolved is None:
            self._clear_offline_clip()
            key = (dataset, self.nframe)
            if key not in self._warned_missing_offline_clip_targets:
                self._warned_missing_offline_clip_targets.add(key)
                print(f'Warning: no offline CLIP dir matched clip_scores_dir={root_dir}, '
                      f'dataset={dataset}, nframe={self.nframe}.')
            return
        if resolved != self._loaded_offline_clip_dir:
            self._load_offline_clip_from_dir(resolved)

    def _lookup_offline_clip(self, video_path: str, query: str):
        if self._offline_clip is None:
            return None
        norm_path = _normalize_clip_video_path(video_path)
        key = (norm_path, query)
        if key in self._offline_clip:
            return self._offline_clip[key]

        def _suffix_match(candidates):
            return [(p, payload) for p, payload in candidates
                    if norm_path.endswith(p) or p.endswith(norm_path)]

        matches = _suffix_match(self._offline_clip_by_query.get(query, []))
        if len(matches) == 1:
            return matches[0][1]
        if len(matches) > 1:
            logging.warning('Offline CLIP lookup ambiguous for %s query %r', norm_path, query)
            return None

        canonical = _normalize_clip_query(query)
        matches = _suffix_match((self._offline_clip_by_query_content or {}).get(canonical, []))
        if len(matches) == 1:
            return matches[0][1]
        if len(matches) > 1:
            logging.warning('Offline CLIP canonical lookup ambiguous for %s', norm_path)
        return None

    # ------------------------------------------------------------------
    # Frame decoding helper
    # ------------------------------------------------------------------

    def _decode_frames(self, video_path: str, frame_indices: list[int]) -> list:
        """Decode specific frame indices from video_path → list of PIL Images."""
        from PIL import Image as PILImage
        debug_video_pipeline(
            'Qwen3_5MoeFramePackChat.decode_frames_start',
            video=os.path.basename(video_path),
            video_path=video_path,
            frame_count=len(frame_indices),
            first_indices=frame_indices[:8],
        )
        t0 = time.monotonic()
        try:
            import decord
            decord.bridge.set_bridge('native')
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            debug_video_pipeline(
                'Qwen3_5MoeFramePackChat.decode_frames_reader_opened',
                video=os.path.basename(video_path),
                elapsed_s=round(time.monotonic() - t0, 3),
            )
            batch_t0 = time.monotonic()
            debug_video_pipeline(
                'Qwen3_5MoeFramePackChat.decode_frames_get_batch_start',
                video=os.path.basename(video_path),
                frame_count=len(frame_indices),
            )
            batch = vr.get_batch(frame_indices).asnumpy()
            frames = [PILImage.fromarray(batch[i]) for i in range(len(batch))]
            debug_video_pipeline(
                'Qwen3_5MoeFramePackChat.decode_frames_done',
                video=os.path.basename(video_path),
                frame_count=len(frames),
                batch_elapsed_s=round(time.monotonic() - batch_t0, 3),
                elapsed_s=round(time.monotonic() - t0, 3),
            )
            return frames
        except Exception as e:
            eval_vllm_log_exception(
                'Qwen3_5MoeFramePackChat.decode_frames_decord_failed',
                e,
                video_path=video_path,
                frame_count=len(frame_indices),
            )
            import cv2
            frames = []
            debug_video_pipeline(
                'Qwen3_5MoeFramePackChat.decode_frames_cv2_start',
                video=os.path.basename(video_path),
                frame_count=len(frame_indices),
            )
            cap = cv2.VideoCapture(video_path)
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
                ret, frame = cap.read()
                if ret:
                    frames.append(PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                elif frames:
                    frames.append(frames[-1])
            cap.release()
            debug_video_pipeline(
                'Qwen3_5MoeFramePackChat.decode_frames_cv2_done',
                video=os.path.basename(video_path),
                frame_count=len(frames),
                elapsed_s=round(time.monotonic() - t0, 3),
            )
            return frames

    def _get_video_stream_metadata(self, video_path: str) -> tuple[int | None, float | None]:
        try:
            import decord
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            return len(vr), float(vr.get_avg_fps())
        except Exception:
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps and fps > 0:
                    return total, float(fps)
                return total, None
            except Exception:
                return None, None

    def _should_apply_mlvu_pre_resize(self, dataset: str | None) -> bool:
        if not self.mlvu_pre_resize_long_edge or self.mlvu_pre_resize_long_edge <= 0:
            return False
        if dataset is None:
            return False
        dataset_name = str(dataset).lower()
        return dataset_name == 'mlvu' or dataset_name.startswith('mlvu_')

    def _pre_resize_long_edge_for_dataset(self, dataset: str | None) -> int | None:
        if self.pre_resize_long_edge is not None and self.pre_resize_long_edge > 0:
            return self.pre_resize_long_edge
        if self._should_apply_mlvu_pre_resize(dataset):
            return self.mlvu_pre_resize_long_edge
        return None

    # ------------------------------------------------------------------
    # CLIP frame pre-selection
    # ------------------------------------------------------------------

    def _preselect_clip_frames(self, message: list[dict], dataset: str | None) -> list[dict]:
        """Pre-select nframe frames from all video frames using offline CLIP scores,
        then apply scheme-based per-frame resolution scaling.

        Flow:
          1. Load allframes offline CLIP scores for this dataset.
          2. Select top nframe (= scheme total) frames by CLIP relevance score.
          3. Decode only those frames (random access via decord).
          4. Assign resolution tiers by CLIP rank: rank 0..d1_count-1 → full res (d1),
             rank d1_count..d1_count+d2_count-1 → d2, etc.
          5. Return frames as a PIL Image list (temporal order, with downscaling applied).

        Falls back to the original video path (uniform decode in processor) when
        offline CLIP scores are unavailable.
        """
        allframes_dir = self._resolve_offline_clip_dir(dataset, frame_label='allframes')
        if allframes_dir is None:
            print(f'[CLIP preselect] No allframes dir for dataset={dataset}. Falling back to uniform.')
            return message
        if allframes_dir != self._loaded_offline_clip_dir:
            self._load_offline_clip_from_dir(allframes_dir)

        query = _normalize_clip_query(
            ' '.join(str(s['value']) for s in message if s['type'] == 'text') or ''
        )

        target = max(FRAME_FACTOR, self.clip_nframe // FRAME_FACTOR * FRAME_FACTOR)

        new_message = []
        vid_idx = 0
        for s in message:
            if s['type'] == 'video' and isinstance(s.get('value'), str):
                video_path = s['value']
                offline_clip = self._lookup_offline_clip(video_path, query)

                if offline_clip is not None:
                    offline_scores, offline_frame_indices = offline_clip
                    cand_scores, cand_orig_indices, _ = _restrict_offline_candidates_to_segment(
                        offline_scores,
                        offline_frame_indices,
                        None,
                    )
                    n_offline = len(cand_scores)
                    if n_offline == 0:
                        vid_idx += 1
                        new_message.append(s)
                        continue
                    actual_target = min(target, n_offline)
                    actual_target = max(FRAME_FACTOR, actual_target // FRAME_FACTOR * FRAME_FACTOR)
                    actual_target = min(actual_target, n_offline)

                    # Select top frames by CLIP score, restore temporal order
                    ranked_desc = sorted(range(n_offline), key=lambda i: cand_scores[i], reverse=True)
                    top_ranked = sorted(ranked_desc[:actual_target])  # temporal order

                    orig_indices = [cand_orig_indices[i] for i in top_ranked]

                    # CLIP rank of each selected frame (0 = highest CLIP score).
                    # All selected frames came from ranked_desc[:actual_target], so
                    # their positions there are exactly 0..actual_target-1.
                    clip_rank_in_all = {idx: rank for rank, idx in enumerate(ranked_desc)}
                    clip_rank_of_frame = [clip_rank_in_all[i] for i in top_ranked]

                    frames = self._decode_frames(video_path, orig_indices)
                    before_sizes = [frame.size for frame in frames[:4]]
                    resize_cap = self._pre_resize_long_edge_for_dataset(dataset)
                    if resize_cap is not None:
                        frames = resize_frames_long_edge(frames, resize_cap)
                    after_resize_sizes = [frame.size for frame in frames[:4]]
                    total_num_frames, video_fps = self._get_video_stream_metadata(video_path)

                    # Apply scheme-based resolution scaling using CLIP rank
                    if self._scheme_dict and frames:
                        frames = _apply_scheme_to_frames(frames, self._scheme_dict, clip_rank_of_frame)
                    after_scheme_sizes = [frame.size for frame in frames[:4]]
                    debug_video_pipeline(
                        'Qwen3_5MoeFramePackChat.clip_preselect',
                        dataset=dataset,
                        video=os.path.basename(video_path),
                        total_num_frames=total_num_frames,
                        source_fps=video_fps,
                        selected_frames=len(orig_indices),
                        first_indices=orig_indices[:8],
                        first_clip_ranks=clip_rank_of_frame[:8],
                        before_sizes=before_sizes,
                        after_resize_sizes=after_resize_sizes,
                        after_scheme_sizes=after_scheme_sizes,
                        max_long_edge=max((max(frame.size) for frame in frames), default=0),
                        resize_cap=resize_cap,
                        scheme=self.compression_scheme,
                    )

                    print(f'[CLIP preselect] video {vid_idx}: {n_offline} candidates → '
                          f'{len(frames)} frames (target={actual_target})')
                    new_s = dict(s)
                    new_s['value'] = frames
                    new_s['_video_metadata'] = build_frame_list_video_metadata(
                        frames=frames,
                        frame_indices=orig_indices,
                        fps=video_fps,
                        total_num_frames=total_num_frames or n_offline,
                    )
                    new_message.append(new_s)
                    vid_idx += 1
                    continue

                vid_idx += 1
            new_message.append(s)
        return new_message

    # ------------------------------------------------------------------
    # Offline frame-feature infrastructure (for 'redi' strategy)
    # ------------------------------------------------------------------

    def _resolve_frame_features_dir(self, dataset: str | None) -> str | None:
        root = self._frame_features_root
        if root is None or not os.path.isdir(root):
            return None
        candidates: list[str] = []
        if dataset is not None:
            for ds_name in _candidate_clip_dataset_names(dataset):
                candidates.append(os.path.join(root, ds_name))
        candidates.append(root)
        seen: list[str] = []
        for c in candidates:
            if c not in seen:
                seen.append(c)
        for c in seen:
            if os.path.isdir(c) and _has_frame_features_manifest(c):
                return c
        return None

    def _load_frame_features_manifest(self, features_root: str) -> dict[str, str]:
        cached = self._frame_features_manifest_cache.get(features_root)
        if cached is not None:
            return cached
        import json
        manifest_path = os.path.join(features_root, 'manifest.json')
        with open(manifest_path) as f:
            manifest = json.load(f)
        feature_subdir = manifest.get('feature_subdir', 'features')
        path_to_feature: dict[str, str] = {}
        for entry in manifest.get('videos', []):
            feature_path_rel = entry.get('feature_path')
            if feature_path_rel is None:
                continue
            feat_abs = (
                feature_path_rel
                if os.path.isabs(feature_path_rel)
                else os.path.join(features_root, feature_path_rel)
            )
            for raw in (entry.get('source_video_path'), entry.get('video_path')):
                if not raw:
                    continue
                norm = _normalize_clip_video_path(raw)
                path_to_feature.setdefault(norm, feat_abs)
        self._frame_features_manifest_cache[features_root] = path_to_feature
        print(f'Loaded frame-feature manifest: {len(path_to_feature)} videos from {features_root} '
              f'(feature_subdir={feature_subdir})')
        return path_to_feature

    def _resolve_feature_path_for_video(
        self,
        path_to_feature: dict[str, str],
        video_path: str,
    ) -> str | None:
        norm = _normalize_clip_video_path(video_path)
        if norm in path_to_feature:
            return path_to_feature[norm]
        matches = [feat for p, feat in path_to_feature.items()
                   if norm.endswith(p) or p.endswith(norm)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logging.warning('Frame-feature lookup ambiguous for %s', norm)
        return None

    def _load_frame_features_tensor(
        self,
        feature_abs_path: str,
    ) -> tuple[torch.Tensor, list[int]] | None:
        cached = self._frame_features_tensor_cache.get(feature_abs_path)
        if cached is not None:
            return cached
        if not os.path.isfile(feature_abs_path):
            return None
        try:
            blob = torch.load(feature_abs_path, map_location='cpu', weights_only=False)
        except Exception as e:
            logging.warning('Failed to load frame features %s: %s', feature_abs_path, e)
            return None
        feats = blob.get('features') if isinstance(blob, dict) else None
        idxs = blob.get('frame_indices') if isinstance(blob, dict) else None
        if feats is None or idxs is None:
            logging.warning('Malformed feature blob at %s', feature_abs_path)
            return None
        feats = feats.to(torch.float32)
        if not bool(blob.get('features_are_normalized', True)):
            feats = torch.nn.functional.normalize(feats, dim=-1)
        if isinstance(idxs, torch.Tensor):
            idxs_list = idxs.to(torch.int64).tolist()
        else:
            idxs_list = [int(x) for x in idxs]
        result = (feats, idxs_list)
        self._frame_features_tensor_cache[feature_abs_path] = result
        return result

    # ------------------------------------------------------------------
    # 'redi' frame sampling: MMR over offline CLIP scores + image features
    # ------------------------------------------------------------------

    def _preselect_redi_frames(self, message: list[dict], dataset: str | None) -> list[dict]:
        """MMR-style preselection.

        Greedy: pick frame 0 = argmax CLIP relevance; subsequent frames
        maximize ``lambda * r_i - (1 - lambda) * max_{j in S} sim(v_i, v_j)``.
        Resolution tiers are assigned by selection order (1st pick → d1, etc.).
        Missing offline data is rejected by the REDI selection validator.
        """
        allframes_dir = self._resolve_offline_clip_dir(dataset, frame_label='allframes')
        features_root = self._resolve_frame_features_dir(dataset)
        if allframes_dir is None or features_root is None:
            return message
        if allframes_dir != self._loaded_offline_clip_dir:
            self._load_offline_clip_from_dir(allframes_dir)
        path_to_feature = self._load_frame_features_manifest(features_root)

        query = _normalize_clip_query(
            ' '.join(str(s['value']) for s in message if s['type'] == 'text') or ''
        )
        target = max(FRAME_FACTOR, self.clip_nframe // FRAME_FACTOR * FRAME_FACTOR)
        lam = self.redi_lambda

        new_message = []
        vid_idx = 0
        for s in message:
            if s['type'] == 'video' and isinstance(s.get('value'), str):
                video_path = s['value']
                offline_clip = self._lookup_offline_clip(video_path, query)
                feat_path = self._resolve_feature_path_for_video(path_to_feature, video_path)
                feats_blob = self._load_frame_features_tensor(feat_path) if feat_path else None

                if offline_clip is not None and feats_blob is not None:
                    offline_scores, offline_frame_indices = offline_clip
                    feats_all, feat_frame_indices = feats_blob

                    cand_clip_scores, clip_frame_indices, _ = _restrict_offline_candidates_to_segment(
                        offline_scores,
                        offline_frame_indices,
                        None,
                    )

                    feat_idx_to_row = {int(idx): row for row, idx in enumerate(feat_frame_indices)}
                    cand_scores: list[float] = []
                    cand_rows: list[int] = []
                    cand_orig_indices: list[int] = []
                    for clip_pos, fidx in enumerate(clip_frame_indices):
                        row = feat_idx_to_row.get(int(fidx))
                        if row is None:
                            continue
                        cand_scores.append(float(cand_clip_scores[clip_pos]))
                        cand_rows.append(row)
                        cand_orig_indices.append(int(fidx))

                    n_cand = len(cand_scores)
                    if n_cand == 0:
                        vid_idx += 1
                        new_message.append(s)
                        continue

                    actual_target = min(target, n_cand)
                    actual_target = max(FRAME_FACTOR, actual_target // FRAME_FACTOR * FRAME_FACTOR)
                    actual_target = min(actual_target, n_cand)

                    F = feats_all[cand_rows].to(torch.float32)
                    r = torch.tensor(cand_scores, dtype=torch.float32)

                    debug_redi_verbose = os.environ.get('DEBUG_REDI', '0') == '1'

                    g = torch.Generator().manual_seed(0)
                    n_pairs = min(2048, n_cand * (n_cand - 1) // 2) if n_cand > 1 else 0
                    if n_pairs > 0:
                        ai = torch.randint(0, n_cand, (n_pairs,), generator=g)
                        bi = torch.randint(0, n_cand, (n_pairs,), generator=g)
                        keep = ai != bi
                        ai, bi = ai[keep], bi[keep]
                        pair_sim = (F[ai] * F[bi]).sum(dim=-1)

                    selected_local: list[int] = []
                    max_sim = torch.full((n_cand,), float('-inf'))
                    available = torch.ones(n_cand, dtype=torch.bool)
                    step_trace: list[tuple] = []

                    first = int(torch.argmax(r).item())
                    selected_local.append(first)
                    available[first] = False
                    step_trace.append((0, first, float(r[first]),
                                       float('nan'), lam * float(r[first]),
                                       float('nan'), float(r[first])))
                    max_sim = torch.maximum(max_sim, F @ F[first])

                    while len(selected_local) < actual_target:
                        score = lam * r - (1.0 - lam) * max_sim
                        score = torch.where(available, score, torch.full_like(score, float('-inf')))
                        nxt = int(torch.argmax(score).item())
                        if not bool(available[nxt].item()):
                            break
                        rank = len(selected_local)
                        r_i = float(r[nxt])
                        ms_i = float(max_sim[nxt])
                        rel_term = lam * r_i
                        div_term = (1.0 - lam) * ms_i
                        step_trace.append((rank, nxt, r_i, ms_i, rel_term, div_term, rel_term - div_term))
                        selected_local.append(nxt)
                        available[nxt] = False
                        max_sim = torch.maximum(max_sim, F @ F[nxt])

                    sel_idx_t = torch.tensor(selected_local, dtype=torch.long)
                    r_sel = r[sel_idx_t]
                    if len(selected_local) >= 2:
                        F_sel = F[sel_idx_t]
                        sim_sel = F_sel @ F_sel.T
                        iu = torch.triu_indices(len(selected_local), len(selected_local), offset=1)
                        sim_pairs = sim_sel[iu[0], iu[1]]

                    if debug_redi_verbose:
                        n_show = len(step_trace)
                        head = step_trace[:5]
                        tail = step_trace[-3:] if n_show > 8 else []
                        for rank, idx, r_i, ms_i, rt, dt, sc in head:
                            ms_str = f'{ms_i:.4f}' if ms_i == ms_i else 'n/a'
                        if tail:
                            for rank, idx, r_i, ms_i, rt, dt, sc in tail:
                                pass

                    rank_by_local = {local: rank for rank, local in enumerate(selected_local)}
                    selected_local_sorted = sorted(selected_local,
                                                   key=lambda i: cand_orig_indices[i])
                    orig_indices = [cand_orig_indices[i] for i in selected_local_sorted]
                    priority_ranks = [rank_by_local[i] for i in selected_local_sorted]

                    frames = self._decode_frames(video_path, orig_indices)
                    before_sizes = [frame.size for frame in frames[:4]]
                    resize_cap = self._pre_resize_long_edge_for_dataset(dataset)
                    if resize_cap is not None:
                        frames = resize_frames_long_edge(frames, resize_cap)
                    after_resize_sizes = [frame.size for frame in frames[:4]]
                    total_num_frames, video_fps = self._get_video_stream_metadata(video_path)

                    if self._scheme_dict and frames:
                        frames = _apply_scheme_to_frames(frames, self._scheme_dict, priority_ranks)
                    after_scheme_sizes = [frame.size for frame in frames[:4]]
                    debug_video_pipeline(
                        'Qwen3_5MoeFramePackChat.redi_preselect',
                        dataset=dataset,
                        video=os.path.basename(video_path),
                        total_num_frames=total_num_frames,
                        source_fps=video_fps,
                        n_candidates=n_cand,
                        selected_frames=len(orig_indices),
                        first_indices=orig_indices[:8],
                        first_priority_ranks=priority_ranks[:8],
                        before_sizes=before_sizes,
                        after_resize_sizes=after_resize_sizes,
                        after_scheme_sizes=after_scheme_sizes,
                        max_long_edge=max((max(frame.size) for frame in frames), default=0),
                        resize_cap=resize_cap,
                        scheme=self.compression_scheme,
                        redi_lambda=lam,
                    )

                    new_s = dict(s)
                    new_s['value'] = frames
                    new_s['_video_metadata'] = build_frame_list_video_metadata(
                        frames=frames,
                        frame_indices=orig_indices,
                        fps=video_fps,
                        total_num_frames=total_num_frames or n_cand,
                    )
                    new_message.append(new_s)
                    vid_idx += 1
                    continue

                vid_idx += 1
            new_message.append(s)
        return new_message

    # ------------------------------------------------------------------
    # Uniform frame sampling with scheme-based resolution scaling
    # ------------------------------------------------------------------

    def _preselect_uniform_frames(self, message: list[dict], dataset: str | None = None) -> list[dict]:
        """Sample nframe frames uniformly; apply scheme-based resolution scaling.

        Frames are assigned to resolution tiers by binary-tree priority rank:
        the most representative frames (temporal midpoints) get d1 (full res),
        remaining frames get d2, d4, … according to the scheme.
        """
        if not self._scheme_dict:
            return message

        new_message = []
        for s in message:
            if s['type'] == 'video' and isinstance(s.get('value'), str):
                video_path = s['value']
                try:
                    import decord
                    debug_video_pipeline(
                        'Qwen3_5MoeFramePackChat.uniform_reader_start',
                        dataset=dataset,
                        video=os.path.basename(video_path),
                        video_path=video_path,
                    )
                    reader_t0 = time.monotonic()
                    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
                    total = len(vr)
                    video_fps = float(vr.get_avg_fps())
                    debug_video_pipeline(
                        'Qwen3_5MoeFramePackChat.uniform_reader_done',
                        dataset=dataset,
                        video=os.path.basename(video_path),
                        total_num_frames=total,
                        source_fps=video_fps,
                        elapsed_s=round(time.monotonic() - reader_t0, 3),
                    )
                except Exception as e:
                    eval_vllm_log_exception(
                        'Qwen3_5MoeFramePackChat.uniform_reader_failed',
                        e,
                        dataset=dataset,
                        video_path=video_path,
                    )
                    new_message.append(s)
                    continue

                nf = min(self.nframe, total)
                nf = max(FRAME_FACTOR, nf // FRAME_FACTOR * FRAME_FACTOR)

                import numpy as np
                indices = list(map(int, np.linspace(0, total - 1, nf).round()))
                frames = self._decode_frames(video_path, indices)
                before_sizes = [frame.size for frame in frames[:4]]
                resize_cap = self._pre_resize_long_edge_for_dataset(dataset)
                if resize_cap is not None:
                    frames = resize_frames_long_edge(frames, resize_cap)
                after_resize_sizes = [frame.size for frame in frames[:4]]

                ranks = binary_tree_ranks(len(frames))
                frames = _apply_scheme_to_frames(frames, self._scheme_dict, ranks)
                after_scheme_sizes = [frame.size for frame in frames[:4]]
                debug_video_pipeline(
                    'Qwen3_5MoeFramePackChat.uniform_preselect',
                    dataset=dataset,
                    video=os.path.basename(video_path),
                    total_num_frames=total,
                    source_fps=video_fps,
                    selected_frames=len(indices),
                    first_indices=indices[:8],
                    first_ranks=ranks[:8],
                    before_sizes=before_sizes,
                    after_resize_sizes=after_resize_sizes,
                    after_scheme_sizes=after_scheme_sizes,
                    max_long_edge=max((max(frame.size) for frame in frames), default=0),
                    resize_cap=resize_cap,
                    scheme=self.compression_scheme,
                )

                new_s = dict(s)
                new_s['value'] = frames
                new_s['_video_metadata'] = build_frame_list_video_metadata(
                    frames=frames,
                    frame_indices=indices,
                    fps=video_fps,
                    total_num_frames=total,
                )
                new_message.append(new_s)
            else:
                new_message.append(s)
        return new_message

    # ------------------------------------------------------------------
    # Content preparation
    # ------------------------------------------------------------------

    def _prepare_content(self, inputs: list[dict], dataset: str | None = None) -> list[dict]:
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                content.append(item)
            elif s['type'] == 'video':
                value = s['value']
                if isinstance(value, list):
                    item = {
                        'type': 'video',
                        'video': [ensure_image_url(v) if isinstance(v, str) else v for v in value],
                    }
                else:
                    item = {'type': 'video', 'video': ensure_video_url(value)}
                    if self.fps is not None:
                        item['fps'] = self.fps
                    else:
                        target = self.nframe
                        try:
                            import cv2
                            cap = cv2.VideoCapture(value)
                            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            cap.release()
                            if 0 < total < target:
                                target = max(FRAME_FACTOR, total // FRAME_FACTOR * FRAME_FACTOR)
                                print(f'[Qwen3_5MoeFramePackChat] {value}: total={total}, clamped nframes={target}')
                        except Exception:
                            pass
                        item['nframes'] = target
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                content.append(item)
            elif s['type'] == 'text':
                content.append({'type': 'text', 'text': s['value']})
            else:
                raise ValueError(f"Unsupported message type: {s['type']}")
        return content

    # ------------------------------------------------------------------
    # vLLM helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_cpu_data(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return value

    def _vllm_video_timestamps_and_counts(self, video_grid_thw, video_metadatas):
        total_grids = int(len(video_grid_thw))
        timestamps = []
        grid_counts = []
        grid_index = 0
        temporal_patch_size = getattr(self.processor.video_processor, 'temporal_patch_size', FRAME_FACTOR)

        for metadata in video_metadatas or []:
            if grid_index >= total_grids:
                break
            if isinstance(metadata, dict):
                fps = metadata.get('fps') or self.fps or 24
                frame_indices = metadata.get('frames_indices')
                total_num_frames = metadata.get('total_num_frames')
            else:
                fps = getattr(metadata, 'fps', None) or self.fps or 24
                frame_indices = getattr(metadata, 'frames_indices', None)
                total_num_frames = getattr(metadata, 'total_num_frames', None)

            if frame_indices is None:
                if total_num_frames is None:
                    remaining_frames = (total_grids - grid_index) * temporal_patch_size
                    frame_indices = list(range(remaining_frames))
                else:
                    frame_indices = list(range(int(total_num_frames)))

            curr_timestamps = self.processor._calculate_timestamps(
                frame_indices,
                float(fps),
                temporal_patch_size,
            )
            available = total_grids - grid_index
            count = min(len(curr_timestamps), available)
            if count <= 0:
                continue
            timestamps.extend([[float(ts)] for ts in curr_timestamps[:count]])
            grid_counts.append(count)
            grid_index += count

        if grid_index < total_grids:
            fps = float(self.fps or 24)
            missing = total_grids - grid_index
            logging.warning(
                'Missing video metadata for %s vLLM FramePack grids; using synthetic timestamps.',
                missing,
            )
            timestamps.extend(
                [[float((grid_index + i) * temporal_patch_size) / fps] for i in range(missing)]
            )
            grid_counts.append(missing)

        return timestamps, grid_counts

    def _repeat_vllm_video_placeholders(self, text: str, grid_counts: list[int]) -> str:
        marker = f'{self.processor.vision_start_token}{self.processor.video_token}{self.processor.vision_end_token}'
        for count in grid_counts:
            replacement = marker * count
            if marker in text:
                text = text.replace(marker, replacement, 1)
            elif self.processor.video_token in text:
                text = text.replace(self.processor.video_token, replacement, 1)
            else:
                raise ValueError('Video placeholder not found while preparing vLLM FramePack prompt.')
        return text

    def _vision_spatial_merge_size(self) -> int:
        for processor in (
            getattr(self.processor, 'video_processor', None),
            getattr(self.processor, 'image_processor', None),
        ):
            if processor is None:
                continue
            for attr in ('spatial_merge_size', 'merge_size'):
                value = getattr(processor, attr, None)
                if value is not None:
                    return max(1, int(value))
        return 2

    @staticmethod
    def _grid_rows_for_mrope(grid_thw, repeat_temporal: bool) -> list[list[int]]:
        if grid_thw is None:
            return []
        grid_tensor = torch.as_tensor(grid_thw, dtype=torch.long).detach().cpu()
        if grid_tensor.numel() == 0:
            return []
        if grid_tensor.ndim == 1:
            grid_tensor = grid_tensor.view(1, -1)

        rows = []
        for row in grid_tensor.tolist():
            if len(row) < 3:
                continue
            t, h, w = (max(1, int(row[0])), max(1, int(row[1])), max(1, int(row[2])))
            if repeat_temporal:
                rows.extend([[1, h, w] for _ in range(t)])
            else:
                rows.append([t, h, w])
        return rows

    def _estimate_mrope_position_length(self, inputs) -> int | None:
        """Match HF Qwen3.5/3.6 M-RoPE position accounting for length guard."""
        if 'input_ids' not in inputs:
            return None

        input_ids = torch.as_tensor(inputs['input_ids']).detach().cpu()
        if input_ids.ndim == 1:
            input_ids = input_ids.view(1, -1)
        prompt_tokens = int(input_ids.shape[-1])

        mm_token_type_ids = inputs.get('mm_token_type_ids')
        has_multimodal = 'image_grid_thw' in inputs or 'video_grid_thw' in inputs
        if mm_token_type_ids is None or not has_multimodal:
            return prompt_tokens

        token_types = torch.as_tensor(mm_token_type_ids, dtype=torch.long).detach().cpu()
        if token_types.ndim == 1:
            token_types = token_types.view(1, -1)
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = torch.as_tensor(attention_mask).detach().cpu().bool()
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.view(1, -1)

        spatial_merge_size = self._vision_spatial_merge_size()
        grid_iters = {
            1: iter(self._grid_rows_for_mrope(inputs.get('image_grid_thw'), repeat_temporal=False)),
            2: iter(self._grid_rows_for_mrope(inputs.get('video_grid_thw'), repeat_temporal=True)),
        }

        max_position = -1
        for batch_idx in range(token_types.shape[0]):
            current_types = token_types[batch_idx]
            if attention_mask is not None:
                current_types = current_types[attention_mask[batch_idx]]

            current_pos = 0
            type_list = current_types.tolist()
            idx = 0
            while idx < len(type_list):
                modality_type = int(type_list[idx])
                end_idx = idx + 1
                while end_idx < len(type_list) and int(type_list[end_idx]) == modality_type:
                    end_idx += 1

                if modality_type == 0:
                    text_len = end_idx - idx
                    if text_len > 0:
                        max_position = max(max_position, current_pos + text_len - 1)
                        current_pos += text_len
                else:
                    grid_iter = grid_iters.get(modality_type)
                    grid = next(grid_iter, None) if grid_iter is not None else None
                    if grid is None:
                        fallback_len = end_idx - idx
                        max_position = max(max_position, current_pos + fallback_len - 1)
                        current_pos += fallback_len
                    else:
                        _, grid_h, grid_w = grid
                        pos_h = max(1, grid_h // spatial_merge_size)
                        pos_w = max(1, grid_w // spatial_merge_size)
                        pos_span = max(pos_h, pos_w)
                        max_position = max(max_position, current_pos + pos_span - 1)
                        current_pos += max(1, max(grid_h, grid_w) // spatial_merge_size)

                idx = end_idx

        return max_position + 1 if max_position >= 0 else prompt_tokens

    def _generate_inner_vllm_from_inputs(self, text: str, inputs, video_metadatas):
        from vllm import SamplingParams

        debug_video_pipeline(
            'Qwen3_5MoeFramePackChat.vllm_prepare_start',
            input_keys=list(inputs.keys()),
            text_chars=len(text),
            input_ids_shape=tuple(inputs['input_ids'].shape) if 'input_ids' in inputs else None,
            pixel_values_shape=tuple(inputs['pixel_values'].shape) if 'pixel_values' in inputs else None,
            pixel_values_videos_shape=(
                tuple(inputs['pixel_values_videos'].shape) if 'pixel_values_videos' in inputs else None
            ),
            image_grid_thw=inputs['image_grid_thw'].tolist() if 'image_grid_thw' in inputs else None,
            video_grid_thw=inputs['video_grid_thw'].tolist() if 'video_grid_thw' in inputs else None,
        )
        mm_data = {}
        if 'pixel_values' in inputs and 'image_grid_thw' in inputs:
            mm_data['image'] = {
                'pixel_values': self._to_cpu_data(inputs['pixel_values']),
                'image_grid_thw': self._to_cpu_data(inputs['image_grid_thw']),
            }

        if 'pixel_values_videos' in inputs and 'video_grid_thw' in inputs:
            video_grid_thw = self._to_cpu_data(inputs['video_grid_thw'])
            timestamps, grid_counts = self._vllm_video_timestamps_and_counts(
                video_grid_thw,
                video_metadatas,
            )
            text = self._repeat_vllm_video_placeholders(text, grid_counts)
            mm_data['video'] = {
                'pixel_values_videos': self._to_cpu_data(inputs['pixel_values_videos']),
                'video_grid_thw': video_grid_thw,
                'timestamps': timestamps,
            }

        debug_video_pipeline(
            'Qwen3_5MoeFramePackChat.vllm_prepare_done',
            mm_keys=list(mm_data.keys()),
            text_chars=len(text),
            has_video='video' in mm_data,
            has_image='image' in mm_data,
        )
        prompt_tokens = int(inputs['input_ids'].shape[-1]) if 'input_ids' in inputs else None
        mrope_position_length = self._estimate_mrope_position_length(inputs)
        max_prompt_tokens = self._hf_max_position_embeddings
        env_max_prompt_tokens = os.environ.get('VLLM_MAX_PROMPT_TOKENS')
        if env_max_prompt_tokens:
            max_prompt_tokens = int(env_max_prompt_tokens)
        strict_token_guard = os.environ.get('VLLM_STRICT_TOKEN_LEN_GUARD', '0') == '1'
        guard_tokens = prompt_tokens if strict_token_guard else mrope_position_length
        guard_name = 'input_ids' if strict_token_guard else 'mrope_position_length'
        if (
            os.environ.get('VLLM_DISABLE_PROMPT_LEN_GUARD', '0') != '1'
            and max_prompt_tokens is not None
            and guard_tokens is not None
            and guard_tokens > max_prompt_tokens
        ):
            msg = (
                f'vLLM prompt is too long for {self.model_path}: '
                f'{guard_name}={guard_tokens} > limit={max_prompt_tokens} '
                f'(raw input_ids={prompt_tokens}). '
                'Reduce nframe/scheme, set PRE_RESIZE_LONG_EDGE, or explicitly set '
                'VLLM_DISABLE_PROMPT_LEN_GUARD=1 to bypass this guard.'
            )
            debug_video_pipeline(
                'Qwen3_5MoeFramePackChat.vllm_prompt_too_long',
                prompt_tokens=prompt_tokens,
                mrope_position_length=mrope_position_length,
                limit=max_prompt_tokens,
                strict_token_guard=strict_token_guard,
            )
            raise ValueError(msg)
        debug_video_pipeline(
            'Qwen3_5MoeFramePackChat.vllm_prompt_len_guard',
            prompt_tokens=prompt_tokens,
            mrope_position_length=mrope_position_length,
            limit=max_prompt_tokens,
            strict_token_guard=strict_token_guard,
        )
        env_max_model_len = read_env_int('VLLM_MAX_MODEL_LEN')
        if env_max_model_len is not None and prompt_tokens is not None and prompt_tokens > env_max_model_len:
            debug_video_pipeline(
                'Qwen3_5MoeFramePackChat.vllm_raw_prompt_exceeds_max_model_len',
                prompt_tokens=prompt_tokens,
                max_model_len=env_max_model_len,
                mrope_position_length=mrope_position_length,
            )
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            stop_token_ids=None,
        )
        req = {'prompt': text}
        if mm_data:
            req['multi_modal_data'] = mm_data

        debug_video_pipeline(
            'Qwen3_5MoeFramePackChat.vllm_generate_start',
            text_chars=len(text),
            has_mm_data=bool(mm_data),
            video_grid_count=len(mm_data.get('video', {}).get('video_grid_thw', [])) if mm_data else 0,
            max_new_tokens=self.max_new_tokens,
        )
        try:
            outputs = self.llm.generate([req], sampling_params=sampling_params, use_tqdm=False)
        except Exception as e:
            eval_vllm_log_exception(
                'Qwen3_5MoeFramePackChat.vllm_generate_failed',
                e,
                text_chars=len(text),
                has_mm_data=bool(mm_data),
            )
            raise
        debug_video_pipeline('Qwen3_5MoeFramePackChat.vllm_generate_done')
        return strip_thinking(outputs[0].outputs[0].text)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate_inner(self, message, dataset=None):
        debug_video_pipeline(
            'Qwen3_5MoeFramePackChat.generate_inner_start',
            dataset=dataset,
            frame_sampling_strategy=self.frame_sampling_strategy,
            message_len=len(message),
            video_values=[
                (len(item.get('value')) if isinstance(item.get('value'), list) else item.get('value'))
                for item in message
                if isinstance(item, dict) and item.get('type') == 'video'
            ],
        )
        # Apply frame sampling strategy
        if self.frame_sampling_strategy == 'redi':
            message = require_redi_frames(message, self._preselect_redi_frames(message, dataset), dataset)
        elif self._scheme_dict:
            # uniform + scheme: pre-decode and scale frames
            message = self._preselect_uniform_frames(message, dataset)
        # uniform without scheme: pass video path + nframes to processor as-is
        video_items = [item for item in message if isinstance(item, dict) and item.get('type') == 'video']
        debug_video_pipeline(
            'Qwen3_5MoeFramePackChat.generate_inner_after_preselect',
            dataset=dataset,
            num_videos=len(video_items),
            video_modes=[('frames' if isinstance(item.get('value'), list) else 'path') for item in video_items],
            video_lengths=[len(item.get('value')) if isinstance(item.get('value'), list) else None for item in video_items],
        )
        debug_video_pipeline(
            'Qwen3_5MoeFramePackChat.request',
            dataset=dataset,
            frame_sampling_strategy=self.frame_sampling_strategy,
            configured_fps=self.fps,
            configured_nframe=self.nframe,
            compression_scheme=self.compression_scheme,
            pre_resize_long_edge=self.pre_resize_long_edge,
            mlvu_pre_resize_long_edge=self.mlvu_pre_resize_long_edge,
            num_videos=len(video_items),
            video_modes=[('frames' if isinstance(item.get('value'), list) else 'path') for item in video_items],
            video_lengths=[len(item.get('value')) if isinstance(item.get('value'), list) else None for item in video_items],
        )

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})

        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        # Check whether any video in the messages is a pre-decoded PIL list (i.e. variable
        # resolution produced by FramePack).  process_vision_info / fetch_video cannot handle
        # such inputs because it tries to torch.stack all frames into a uniform 4-D tensor.
        _has_pil_video = any(
            isinstance(item.get('video'), list)
            for msg in messages
            for item in (msg.get('content') or [])
            if isinstance(item, dict) and item.get('type') == 'video'
        )

        if _has_pil_video:
            # Extract images (URL strings) and PIL-list videos directly from the message
            # dict and feed them to the custom processor.  do_sample_frames=False tells the
            # video processor to skip re-sampling (frames were already selected above).
            imgs_for_proc: list = []
            vids_for_proc: list = []
            video_metadatas: list[dict] = []
            for msg in messages:
                for item in (msg.get('content') or []):
                    if not isinstance(item, dict):
                        continue
                    if item.get('type') == 'image':
                        imgs_for_proc.append(item['image'])
            for item in message:
                if item.get('type') == 'video' and isinstance(item.get('value'), list):
                    vids_for_proc.append(item['value'])
                    video_metadatas.append(
                        item.get('_video_metadata')
                        or build_frame_list_video_metadata(
                            frames=item['value'],
                            frame_indices=None,
                            fps=self.fps,
                        )
                    )

            fill_missing_video_metadata_fps(video_metadatas, self.fps)

            debug_video_pipeline(
                'Qwen3_5MoeFramePackChat.processor_start',
                mode='predecoded_frames',
                image_count=len(imgs_for_proc),
                video_count=len(vids_for_proc),
                video_frame_counts=[len(v) for v in vids_for_proc],
                text_chars=len(text),
            )
            inputs = self.processor(
                text=text,
                images=imgs_for_proc or None,
                videos=vids_for_proc or None,
                video_metadata=video_metadatas or None,
                do_resize=False,
                do_sample_frames=False,
                return_mm_token_type_ids=True,
                return_tensors='pt',
            )
            debug_video_pipeline(
                'Qwen3_5MoeFramePackChat.processor',
                mode='predecoded_frames',
                pixel_values_shape=tuple(inputs['pixel_values_videos'].shape) if 'pixel_values_videos' in inputs else None,
                video_grid_thw=inputs['video_grid_thw'].tolist() if 'video_grid_thw' in inputs else None,
                input_ids_shape=tuple(inputs['input_ids'].shape) if 'input_ids' in inputs else None,
            )
        else:
            # Uniform without scheme: let process_vision_info decode frames uniformly,
            # then pass the decoded tensors to the processor.
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError as err:
                logging.critical("Please install via 'pip install qwen-vl-utils'")
                raise err

            debug_video_pipeline(
                'Qwen3_5MoeFramePackChat.process_vision_info_start',
                dataset=dataset,
                text_chars=len(text),
            )
            images, videos, video_kwargs = process_vision_info(
                messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True,
            )
            debug_video_pipeline(
                'Qwen3_5MoeFramePackChat.process_vision_info_done',
                image_count=len(images) if images is not None else 0,
                video_count=len(videos) if videos is not None else 0,
                video_kwargs=video_kwargs,
            )

            video_metadatas = None
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
                fill_missing_video_metadata_fps(video_metadatas, self.fps)

            debug_video_pipeline(
                'Qwen3_5MoeFramePackChat.processor_start',
                mode='process_vision_info',
                image_count=len(images) if images is not None else 0,
                video_count=len(videos) if videos is not None else 0,
                sampled_video_frames=[len(video) for video in videos] if videos is not None else None,
                text_chars=len(text),
            )
            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                do_resize=False,
                do_sample_frames=False if videos is not None else None,
                return_mm_token_type_ids=True,
                return_tensors='pt',
                **(video_kwargs or {}),
            )
            sampled_video_frames = None
            if videos is not None:
                try:
                    sampled_video_frames = [len(video) for video in videos]
                except Exception:
                    sampled_video_frames = None
            debug_video_pipeline(
                'Qwen3_5MoeFramePackChat.processor',
                mode='process_vision_info',
                video_kwargs=video_kwargs,
                sampled_video_frames=sampled_video_frames,
                metadata_total_frames=[
                    getattr(meta, 'total_num_frames', None) if not isinstance(meta, dict) else meta.get('total_num_frames')
                    for meta in (video_metadatas or [])
                ],
                pixel_values_shape=tuple(inputs['pixel_values_videos'].shape) if 'pixel_values_videos' in inputs else None,
                video_grid_thw=inputs['video_grid_thw'].tolist() if 'video_grid_thw' in inputs else None,
                input_ids_shape=tuple(inputs['input_ids'].shape) if 'input_ids' in inputs else None,
            )

        if self.use_vllm:
            response = self._generate_inner_vllm_from_inputs(text, inputs, video_metadatas)
            if self.verbose:
                print(f'\033[32m{response}\033[0m')
            return response

        try:
            inputs = inputs.to(self.model.device)
            if hasattr(self.model, 'dtype'):
                inputs = inputs.to(self.model.dtype)
        except Exception:
            inputs = inputs.to(self.device)
            if hasattr(self.model, 'dtype'):
                inputs = inputs.to(self.model.dtype)

        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
        generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        response = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        response = strip_thinking(response)

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response


Qwen3_5Chat = Qwen3_5MoeChat
Qwen3_5FramePackChat = Qwen3_5MoeFramePackChat
