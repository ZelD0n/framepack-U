# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""video processor class for Qwen3-VL."""

import math
from collections.abc import Sequence
from os import PathLike
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ChannelDimension, PILImageResampling, SizeDict, infer_channel_dimension_format
from transformers.processing_utils import Unpack, VideosKwargs
from transformers.utils import TensorType, add_start_docstrings, logging
from transformers.video_processing_utils import BASE_VIDEO_PROCESSOR_DOCSTRING, BaseVideoProcessor
from transformers.video_utils import VideoMetadata


logger = logging.get_logger(__name__)


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = 128 * 128,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144,
):
    if num_frames < temporal_factor:
        raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


class Qwen3_5VideoProcessorInitKwargs(VideosKwargs):
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]
    min_frames: Optional[int]
    max_frames: Optional[int]
    score_temperature: Optional[float]
    total_token_budget: Optional[int]
    anchor_count: Optional[int]
    window_size: Optional[int]


@add_start_docstrings(
    "Constructs a fast Qwen3-VL image processor that dynamically resizes videos based on the original videos.",
    BASE_VIDEO_PROCESSOR_DOCSTRING,
    """
        patch_size (`int`, *optional*, defaults to 16):
            The spacial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
    """,
)
class Qwen3_5VideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 128 * 32 * 32, "longest_edge": 32 * 32 * 768}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    patch_size = 16
    temporal_patch_size = 2
    merge_size = 2
    fps = 2
    min_frames = 4
    max_frames = 768
    score_temperature = 0.1
    do_sample_frames = True
    valid_kwargs = Qwen3_5VideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[Qwen3_5VideoProcessorInitKwargs]):
        super().__init__(**kwargs)
        if self.size is not None and (
            self.size.get("shortest_edge", None) is None or self.size.get("longest_edge", None) is None
        ):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

    def _further_process_kwargs(
        self,
        size: Optional[SizeDict] = None,
        **kwargs,
    ) -> dict:
        """
        Update kwargs that need further processing before being validated
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        return super()._further_process_kwargs(size=size, **kwargs)

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: Optional[int] = None,
        fps: Optional[Union[int, float]] = None,
        **kwargs,
    ):
        """
        Default sampling function which uniformly samples the desired number of frames between 0 and total number of frames.
        If `fps` is passed along with metadata, `fps` frames per second are sampled uniformty. Arguments `num_frames`
        and `fps` are mutually exclusive.

        Args:
            video (`torch.Tensor`):
                Video that need to be sampled.
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            num_frames (`int`, *optional*):
                Maximum number of frames to sample. Defaults to `self.num_frames`.
            fps (`int` or `float`, *optional*):
                Target frames to sample per second. Defaults to `self.fps`.
        Returns:
            torch.Tensor:
                Sampled video frames.
        """
        if fps is not None and num_frames is not None:
            raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

        total_num_frames = metadata.total_num_frames
        fps = fps if fps is not None else self.fps

        # If num_frames is not given but fps is, calculate num_frames from fps
        if num_frames is None and fps is not None:
            if metadata.fps is None:
                metadata.fps = 24
                logger.warning_once(
                    "Asked to sample `fps` frames per second but no video metadata was provided which is required when sampling with `fps`. "
                    "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
                )
            num_frames = int(total_num_frames / metadata.fps * fps)
            num_frames = min(min(max(num_frames, self.min_frames), self.max_frames), total_num_frames)

        if num_frames is None:
            num_frames = min(max(total_num_frames, self.min_frames), self.max_frames)

        indices = np.linspace(0, total_num_frames - 1, num_frames).round().astype(int)

        return indices

    def _infer_frame_channel_dimension(
        self,
        frame: Union[np.ndarray, torch.Tensor],
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> ChannelDimension:
        if input_data_format is not None:
            return ChannelDimension(input_data_format)

        if isinstance(frame, np.ndarray):
            return infer_channel_dimension_format(frame)

        if not isinstance(frame, torch.Tensor) or frame.ndim != 3:
            raise ValueError("Expected a single video frame as a 3D numpy array or torch tensor.")

        first_is_channel = frame.shape[0] in {1, 3}
        last_is_channel = frame.shape[-1] in {1, 3}

        if first_is_channel and not last_is_channel:
            return ChannelDimension.FIRST
        if last_is_channel and not first_is_channel:
            return ChannelDimension.LAST

        raise ValueError(
            "Could not infer the channel dimension for the input frame. Please set `input_data_format` explicitly."
        )

    def _normalize_video_to_frame_list(
        self,
        video: Union[np.ndarray, torch.Tensor, list[np.ndarray], list[torch.Tensor], list[Image.Image]],
    ) -> list[Union[np.ndarray, torch.Tensor, Image.Image]]:
        if isinstance(video, (np.ndarray, torch.Tensor)):
            if video.ndim != 4:
                raise ValueError(f"Expected a 4D video tensor/array, but got shape {tuple(video.shape)}.")
            frames = [video[idx] for idx in range(video.shape[0])]
        elif isinstance(video, (list, tuple)):
            if len(video) == 0:
                raise ValueError("The input video must contain at least one frame.")
            frames = list(video)
        else:
            raise TypeError(f"Unsupported video type: {type(video)}")

        if not all(
            isinstance(frame, Image.Image) or (isinstance(frame, (np.ndarray, torch.Tensor)) and frame.ndim == 3)
            for frame in frames
        ):
            raise ValueError("Expected the video to be a 4D tensor/array or a list of 3D frames.")

        frame_types = {
            Image.Image if isinstance(frame, Image.Image) else (torch.Tensor if isinstance(frame, torch.Tensor) else np.ndarray)
            for frame in frames
        }
        if len(frame_types) != 1:
            raise TypeError("All frames in the input video must have the same type.")

        return frames

    def _is_frame_input(self, value: object) -> bool:
        return isinstance(value, Image.Image) or (
            isinstance(value, (np.ndarray, torch.Tensor)) and getattr(value, "ndim", None) == 3
        )

    def _is_video_array(self, value: object) -> bool:
        return isinstance(value, (np.ndarray, torch.Tensor)) and getattr(value, "ndim", None) == 4

    def _contains_frame_sequence(self, videos: object) -> bool:
        if self._is_frame_input(videos):
            return True
        if not isinstance(videos, (list, tuple)) or len(videos) == 0:
            return False
        if all(self._is_frame_input(frame) for frame in videos):
            return True
        return any(self._contains_frame_sequence(item) for item in videos if isinstance(item, (list, tuple)))

    def _make_batched_frame_videos(
        self,
        videos: object,
    ) -> list[Union[np.ndarray, torch.Tensor, list[Union[np.ndarray, torch.Tensor, Image.Image]], str, PathLike]]:
        if isinstance(videos, (str, PathLike)) or self._is_video_array(videos):
            return [videos]
        if self._is_frame_input(videos):
            return [[videos]]
        if not isinstance(videos, (list, tuple)):
            raise ValueError(
                f"Invalid video input. Expected either a list of video frames or an input of 4 or 5 dimensions, but got type {type(videos)}."
            )
        if len(videos) == 0:
            return []
        if all(self._is_frame_input(frame) for frame in videos):
            return [list(videos)]

        batched_videos = []
        for item in videos:
            if isinstance(item, (str, PathLike)) or self._is_video_array(item):
                batched_videos.append(item)
            elif self._is_frame_input(item):
                batched_videos.append([item])
            elif isinstance(item, (list, tuple)):
                batched_videos.extend(self._make_batched_frame_videos(item))
            else:
                raise ValueError(f"Unsupported video item type: {type(item)}")
        return batched_videos

    def _get_frame_size(self, frame: Union[np.ndarray, torch.Tensor, Image.Image]) -> tuple[int, int]:
        if isinstance(frame, Image.Image):
            width, height = frame.size
            return height, width
        if isinstance(frame, np.ndarray):
            channel_dim = infer_channel_dimension_format(frame)
            return (frame.shape[0], frame.shape[1]) if channel_dim == ChannelDimension.LAST else (frame.shape[1], frame.shape[2])
        if not isinstance(frame, torch.Tensor) or frame.ndim != 3:
            raise ValueError("Expected a single video frame as a 3D numpy array, torch tensor, or PIL image.")
        channel_dim = self._infer_frame_channel_dimension(frame)
        return (frame.shape[1], frame.shape[2]) if channel_dim == ChannelDimension.FIRST else (frame.shape[0], frame.shape[1])

    def _make_batched_metadata(
        self,
        videos: list[Union[np.ndarray, torch.Tensor, list[Union[np.ndarray, torch.Tensor, Image.Image]], str, PathLike]],
        video_metadata: Optional[Union[VideoMetadata, dict, list[VideoMetadata], list[dict]]] = None,
    ) -> list[VideoMetadata]:
        if video_metadata is None:
            metadata = []
            for video in videos:
                if isinstance(video, (str, PathLike)):
                    metadata.append(
                        VideoMetadata(
                            total_num_frames=0,
                            fps=None,
                            duration=None,
                            frames_indices=[],
                            height=None,
                            width=None,
                        )
                    )
                    continue

                frames = self._normalize_video_to_frame_list(video)
                frame_sizes = [self._get_frame_size(frame) for frame in frames]
                metadata.append(
                    VideoMetadata(
                        total_num_frames=len(frames),
                        fps=None,
                        duration=None,
                        frames_indices=list(range(len(frames))),
                        height=max(height for height, _ in frame_sizes),
                        width=max(width for _, width in frame_sizes),
                    )
                )
            return metadata

        if isinstance(video_metadata, list):
            if len(video_metadata) == 0:
                return []
            if isinstance(video_metadata[0], list):
                return [VideoMetadata(**metadata) for metadata_list in video_metadata for metadata in metadata_list]
            if isinstance(video_metadata[0], VideoMetadata):
                return video_metadata
            if isinstance(video_metadata[0], dict):
                return [VideoMetadata(**metadata) for metadata in video_metadata]
        if isinstance(video_metadata, VideoMetadata):
            return [video_metadata]
        return [VideoMetadata(**video_metadata)]

    def _decode_and_sample_videos(
        self,
        videos,
        video_metadata: Union[VideoMetadata, dict],
        do_sample_frames: Optional[bool] = None,
        sample_indices_fn=None,
    ):
        if not self._contains_frame_sequence(videos):
            return super()._decode_and_sample_videos(
                videos,
                video_metadata=video_metadata,
                do_sample_frames=do_sample_frames,
                sample_indices_fn=sample_indices_fn,
            )

        videos = self._make_batched_frame_videos(videos)
        video_metadata = self._make_batched_metadata(videos, video_metadata=video_metadata)

        if do_sample_frames:
            sampled_videos = []
            sampled_metadata = []
            for video, metadata in zip(videos, video_metadata):
                if isinstance(video, (str, PathLike)):
                    raise ValueError("Sampling frames from video paths is not supported in variable-resolution mode.")
                indices = sample_indices_fn(metadata=metadata)
                metadata.frames_indices = indices.tolist() if isinstance(indices, np.ndarray) else list(indices)
                frames = self._normalize_video_to_frame_list(video)
                sampled_videos.append([frames[int(idx)] for idx in metadata.frames_indices])
                sampled_metadata.append(metadata)
            videos = sampled_videos
            video_metadata = sampled_metadata

        return videos, video_metadata

    def _prepare_input_videos(
        self,
        videos,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        device: Optional[str] = None,
    ) -> list[list[torch.Tensor]]:
        processed_videos = []
        for video in videos:
            frames = self._normalize_video_to_frame_list(video)
            processed_frames = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    frame = np.array(frame)

                if isinstance(frame, np.ndarray):
                    channel_dim = self._infer_frame_channel_dimension(frame, input_data_format=input_data_format)
                    frame_tensor = torch.from_numpy(frame).contiguous()
                else:
                    channel_dim = self._infer_frame_channel_dimension(frame, input_data_format=input_data_format)
                    frame_tensor = frame.contiguous()

                if channel_dim == ChannelDimension.LAST:
                    frame_tensor = frame_tensor.permute(2, 0, 1).contiguous()
                elif channel_dim != ChannelDimension.FIRST:
                    raise ValueError(f"Unsupported channel dimension format: {channel_dim}")

                if device is not None:
                    frame_tensor = frame_tensor.to(device)

                processed_frames.append(frame_tensor)
            processed_videos.append(processed_frames)
        return processed_videos

    def _pad_frame_to_size(self, frame: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        pad_height = target_height - frame.shape[-2]
        pad_width = target_width - frame.shape[-1]
        if pad_height < 0 or pad_width < 0:
            raise ValueError(
                f"Cannot pad frame of shape {tuple(frame.shape)} to a smaller size ({target_height}, {target_width})."
            )
        if pad_height == 0 and pad_width == 0:
            return frame
        return F.pad(frame, (0, pad_width, 0, pad_height))

    def _frame_to_clip_image(
        self,
        frame: Union[np.ndarray, torch.Tensor, Image.Image],
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")

        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu()
            channel_dim = self._infer_frame_channel_dimension(frame, input_data_format=input_data_format)
            if channel_dim == ChannelDimension.FIRST:
                frame = frame.permute(1, 2, 0).contiguous()
            elif channel_dim != ChannelDimension.LAST:
                raise ValueError(f"Unsupported channel dimension format: {channel_dim}")
            frame = frame.numpy()
        elif isinstance(frame, np.ndarray):
            channel_dim = self._infer_frame_channel_dimension(frame, input_data_format=input_data_format)
            if channel_dim == ChannelDimension.FIRST:
                frame = np.transpose(frame, (1, 2, 0))
            elif channel_dim != ChannelDimension.LAST:
                raise ValueError(f"Unsupported channel dimension format: {channel_dim}")
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}")

        if frame.dtype != np.uint8:
            frame = frame.astype(np.float32)
            if frame.min() >= 0.0 and frame.max() <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)

        return Image.fromarray(frame).convert("RGB")

    def get_clip_frame_query_similarity(
        self,
        video: Union[np.ndarray, torch.Tensor, list[np.ndarray], list[torch.Tensor], list[Image.Image]],
        query: str,
        clip_model=None,
        clip_processor=None,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 32,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        offline_scores: Optional[list[float]] = None,
        offline_frame_indices: Optional[list[float]] = None,
        current_frame_indices: Optional[list[float]] = None,
    ) -> list[float]:
        """
        Computes CLIP cosine similarity scores between a text query and each frame in a video.

        Args:
            video:
                A 4D video tensor/array or a list of 3D frames.
            query:
                Text query used to score each frame.
            clip_model:
                Optional pre-loaded `transformers.CLIPModel`. If omitted, it will be loaded from `clip_model_name`.
            clip_processor:
                Optional pre-loaded `transformers.CLIPProcessor`. If omitted, it will be loaded from `clip_model_name`.
            clip_model_name:
                CLIP checkpoint name used when `clip_model` or `clip_processor` is not provided.
            device:
                Target device for CLIP inference. Defaults to the CLIP model device if available, otherwise `cuda`
                when possible and `cpu` as fallback.
            batch_size:
                Number of frames processed by CLIP per batch.
            input_data_format:
                Explicit channel format of the input video/frames. If omitted, it is inferred from each frame.
            offline_scores:
                Pre-computed CLIP similarity scores for a set of frames (e.g. loaded from disk). When provided,
                CLIP inference is skipped only when these scores exactly correspond to the currently extracted frames.
            offline_frame_indices:
                Absolute frame indices (in the original video) corresponding to each entry in `offline_scores`.
                When provided together with `current_frame_indices`, the two lists must match exactly for offline
                scores to be used.
            current_frame_indices:
                Absolute frame indices (in the original video) of the frames in `video`, in the same order.
                When provided together with `offline_frame_indices`, the two lists must match exactly for offline
                scores to be used.

        Returns:
            A list of cosine similarity scores, one per frame.

        Notes on offline loading:
            * **Strict index match**: when both ``offline_frame_indices`` and ``current_frame_indices`` are given,
              offline scores are used only if the two index lists match exactly.
            * **Strict length match**: when index lists are unavailable, offline scores are used only if their length
              matches the number of current frames exactly.
            * Otherwise the function falls back to online CLIP inference.
        """
        if not query:
            raise ValueError("`query` must be a non-empty string.")

        # --- Offline path: use only exact-match cached scores ---
        if offline_scores is not None:
            frames = self._normalize_video_to_frame_list(video)
            n_current = len(frames)
            m_offline = len(offline_scores)

            if offline_frame_indices is not None and current_frame_indices is not None:
                if len(offline_frame_indices) != m_offline:
                    logger.warning(
                        "Offline CLIP scores ignored: `offline_frame_indices` length (%s) does not match "
                        "`offline_scores` length (%s). Falling back to online CLIP.",
                        len(offline_frame_indices),
                        m_offline,
                    )
                elif len(current_frame_indices) != n_current:
                    logger.warning(
                        "Offline CLIP scores ignored: `current_frame_indices` length (%s) does not match "
                        "the number of frames in `video` (%s). Falling back to online CLIP.",
                        len(current_frame_indices),
                        n_current,
                    )
                elif list(offline_frame_indices) == list(current_frame_indices):
                    logger.info("Offline CLIP scores: exact frame-index match, returning cached scores directly.")
                    return list(offline_scores)
                else:
                    logger.warning(
                        "Offline CLIP scores ignored: frame indices do not match exactly. "
                        "Falling back to online CLIP."
                    )
            else:
                if n_current == m_offline:
                    logger.info("Offline CLIP scores: exact length match, returning cached scores directly.")
                    return list(offline_scores)
                logger.warning(
                    "Offline CLIP scores ignored: cached length (%s) does not match current frame count (%s). "
                    "Falling back to online CLIP.",
                    m_offline,
                    n_current,
                )

        # --- Online path: run CLIP ---
        if batch_size <= 0:
            raise ValueError("`batch_size` must be positive.")

        from transformers import CLIPModel, CLIPProcessor

        frames = self._normalize_video_to_frame_list(video)
        clip_images = [self._frame_to_clip_image(frame, input_data_format=input_data_format) for frame in frames]

        if clip_processor is None:
            clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        if clip_model is None:
            clip_model = CLIPModel.from_pretrained(clip_model_name)

        if device is None:
            try:
                device = next(clip_model.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        clip_model = clip_model.to(device)
        clip_model.eval()

        with torch.inference_mode():
            text_inputs = clip_processor(text=[query], return_tensors="pt", padding=True, truncation=True)
            text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
            text_features = clip_model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, dim=-1)

            similarity_scores = []
            for start in range(0, len(clip_images), batch_size):
                image_inputs = clip_processor(images=clip_images[start : start + batch_size], return_tensors="pt")
                image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
                image_features = clip_model.get_image_features(**image_inputs)
                image_features = F.normalize(image_features, dim=-1)
                batch_scores = torch.matmul(image_features, text_features.T).squeeze(-1)
                similarity_scores.extend(batch_scores.detach().cpu().tolist())

        return similarity_scores

    def _resolve_compression_ratios(
        self,
        num_frames: int,
        compression_ratios: Sequence[Sequence[Union[int, float]]],
    ) -> list[tuple[float, float]]:
        if len(compression_ratios) == 0:
            raise ValueError("`compression_ratios` must contain at least one `[height_ratio, width_ratio]` pair.")

        resolved_compression_ratios = []
        for ratio_pair in compression_ratios:
            if len(ratio_pair) != 2:
                raise ValueError(
                    f"Each entry in `compression_ratios` must contain exactly two values, but got {ratio_pair}."
                )
            height_ratio, width_ratio = float(ratio_pair[0]), float(ratio_pair[1])
            if height_ratio <= 0 or width_ratio <= 0:
                raise ValueError(f"Compression ratios must be positive, but got {ratio_pair}.")
            resolved_compression_ratios.append((height_ratio, width_ratio))

        if len(resolved_compression_ratios) == 1 and num_frames > 1:
            resolved_compression_ratios = resolved_compression_ratios * num_frames
        if len(resolved_compression_ratios) != num_frames:
            raise ValueError(
                f"`compression_ratios` length ({len(resolved_compression_ratios)}) must match the number of frames ({num_frames})."
            )
        return resolved_compression_ratios

    def build_compression_ratios(
        self,
        num_frames: int,
        video: Union[np.ndarray, torch.Tensor, list[np.ndarray], list[torch.Tensor], list[Image.Image]],
        query: str,
        clip_model=None,
        clip_processor=None,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 32,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        offline_scores: Optional[list[float]] = None,
        offline_frame_indices: Optional[list[float]] = None,
        current_frame_indices: Optional[list[float]] = None,
    ) -> list[tuple[float, float]]:
        if num_frames <= 0:
            raise ValueError("`num_frames` must be positive.")

        scores = self.get_clip_frame_query_similarity(
            video,
            query,
            clip_model=clip_model,
            clip_processor=clip_processor,
            clip_model_name=clip_model_name,
            device=device,
            batch_size=batch_size,
            input_data_format=input_data_format,
            offline_scores=offline_scores,
            offline_frame_indices=offline_frame_indices,
            current_frame_indices=current_frame_indices,
        )
        # Rank frames by similarity (descending): rank 0 = most similar
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranks = [0] * len(scores)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank

        # Tiered ratio assignment:
        #   1 frame  at ratio 1  (rank 0)
        #   4 frames at ratio 2  (ranks 1-4)
        #  16 frames at ratio 4  (ranks 5-20)
        #  64 frames at ratio 8  (ranks 21-84)
        #  ...  each tier k has 4^k frames at ratio 2^k
        def _rank_to_ratio(r: int) -> float:
            k = 0
            cumulative = 1  # 4^0
            while r >= cumulative:
                k += 1
                cumulative += 4 ** k
            return float(2 ** k)

        ratios = [(_rank_to_ratio(r), _rank_to_ratio(r)) for r in ranks]

        # Log ratio distribution summary
        from collections import Counter
        dist = Counter(r for r, _ in ratios)
        dist_str = ", ".join(f"1/{int(k)}:{v} frames" for k, v in sorted(dist.items()))
        logger.info(f"Compression ratio distribution: {dist_str}")

        return ratios

    def apply_frame_compression(
        self,
        video: Union[np.ndarray, torch.Tensor, list[np.ndarray], list[torch.Tensor]],
        query: str,
        compression_ratios: Optional[Sequence[Sequence[Union[int, float]]]] = None,
        clip_model=None,
        clip_processor=None,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        clip_device: Optional[Union[str, torch.device]] = None,
        clip_batch_size: int = 32,
        interpolation: PILImageResampling = PILImageResampling.BICUBIC,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        offline_scores: Optional[list[float]] = None,
        offline_frame_indices: Optional[list[float]] = None,
        current_frame_indices: Optional[list[float]] = None,
    ) -> tuple[list[Union[np.ndarray, torch.Tensor]], list[tuple[int, int]], list[tuple[float, float]]]:
        frames = self._normalize_video_to_frame_list(video)
        resolved_compression_ratios = (
            self._resolve_compression_ratios(len(frames), compression_ratios)
            if compression_ratios is not None
            else self.build_compression_ratios(
                len(frames),
                video=video,
                query=query,
                clip_model=clip_model,
                clip_processor=clip_processor,
                clip_model_name=clip_model_name,
                device=clip_device,
                batch_size=clip_batch_size,
                input_data_format=input_data_format,
                offline_scores=offline_scores,
                offline_frame_indices=offline_frame_indices,
                current_frame_indices=current_frame_indices,
            )
        )
        compressed_frames, compressed_sizes = self.make_variable_resolution_video(
            frames,
            compression_ratios=resolved_compression_ratios,
            interpolation=interpolation,
            input_data_format=input_data_format,
            data_format=data_format,
            return_tensors=return_tensors,
        )
        return compressed_frames, compressed_sizes, resolved_compression_ratios

    def make_variable_resolution_video(
        self,
        video: Union[np.ndarray, torch.Tensor, list[np.ndarray], list[torch.Tensor]],
        compression_ratios: Sequence[Sequence[Union[int, float]]],
        interpolation: PILImageResampling = PILImageResampling.BICUBIC,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> tuple[list[Union[np.ndarray, torch.Tensor]], list[tuple[int, int]]]:
        """
        Applies per-frame spatial compression ratios to a video and returns a variable-resolution frame list.

        Args:
            video:
                A 4D video tensor/array or a list of 3D frames.
            compression_ratios:
                A sequence of `[height_ratio, width_ratio]` values. The target size of frame `i` is
                `(round(height / height_ratio), round(width / width_ratio))`.
                If a single ratio pair is provided, it is broadcast to every frame.
            interpolation:
                Interpolation method used for resizing.
            input_data_format:
                Explicit channel format of the input video/frames. If omitted, it is inferred from the first frame.
            data_format:
                Target channel format of the returned frames. Defaults to the input format.
            return_tensors:
                Output backend. Only `"pt"` and `"np"` are supported. Defaults to the input frame type.

        Returns:
            A tuple of:
                - variable-resolution frame list
                - per-frame `(height, width)` sizes after resizing

        Notes:
            The returned frames are intentionally kept as a list because the current preprocessing path stacks video
            frames into a single 4D tensor and therefore still assumes uniform frame shapes.
        """
        frames = self._normalize_video_to_frame_list(video)
        num_frames = len(frames)
        compression_ratios = self._resolve_compression_ratios(num_frames, compression_ratios)

        if isinstance(return_tensors, TensorType):
            return_tensors = return_tensors.value
        if return_tensors is None:
            return_tensors = "pt" if isinstance(frames[0], torch.Tensor) else "np"
        if return_tensors not in {"pt", "np"}:
            raise ValueError("`return_tensors` only supports `'pt'` and `'np'` for variable-resolution videos.")

        input_channel_dim = self._infer_frame_channel_dimension(frames[0], input_data_format=input_data_format)
        output_channel_dim = input_channel_dim if data_format is None else ChannelDimension(data_format)

        resized_frames: list[Union[np.ndarray, torch.Tensor]] = []
        resized_sizes: list[tuple[int, int]] = []

        for frame_idx, (frame, ratio_pair) in enumerate(zip(frames, compression_ratios)):
            height_ratio, width_ratio = ratio_pair

            frame_tensor = frame if isinstance(frame, torch.Tensor) else torch.from_numpy(frame)
            if input_channel_dim == ChannelDimension.LAST:
                frame_tensor = frame_tensor.permute(2, 0, 1).contiguous()
            elif input_channel_dim != ChannelDimension.FIRST:
                raise ValueError(f"Unsupported channel dimension format: {input_channel_dim}")

            height, width = frame_tensor.shape[-2:]
            min_dim = getattr(self, 'patch_size', 16) * getattr(self, 'merge_size', 2)
            target_height = max(min_dim, int(round(height / height_ratio)))
            target_width = max(min_dim, int(round(width / width_ratio)))

            if target_height == height and target_width == width:
                resized_frame = frame_tensor
            else:
                # Use F.interpolate directly to avoid torchvision antialias
                # bicubic bug (_upsample_bicubic2d_aa) on extreme downscaling
                resized_frame = F.interpolate(
                    frame_tensor.unsqueeze(0).float(),
                    size=(target_height, target_width),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0).to(frame_tensor.dtype)

            resized_sizes.append((target_height, target_width))

            if output_channel_dim == ChannelDimension.LAST:
                resized_frame = resized_frame.permute(1, 2, 0).contiguous()
            elif output_channel_dim != ChannelDimension.FIRST:
                raise ValueError(f"Unsupported output channel dimension format: {output_channel_dim}")

            if return_tensors == "np":
                resized_frames.append(resized_frame.cpu().numpy())
            else:
                resized_frames.append(resized_frame)

        return resized_frames, resized_sizes

    def preprocess(self, videos, **kwargs):
        # Newer huggingface_hub builds a strict dataclass from valid_kwargs and fills
        # every missing field with _TYPED_DICT_DEFAULT_VALUE = object(), which then
        # fails type-checking against Optional[bool] etc.  We replicate the base-class
        # preprocess logic verbatim but omit the validate_typed_dict call.
        from functools import partial
        from transformers.image_utils import ChannelDimension

        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        input_data_format = kwargs.pop("input_data_format")
        do_sample_frames = kwargs.pop("do_sample_frames")
        device = kwargs.pop("device")
        video_metadata = kwargs.pop("video_metadata")

        sample_indices_fn = partial(self.sample_frames, **kwargs) if do_sample_frames else None
        videos, video_metadata = self._decode_and_sample_videos(
            videos,
            video_metadata=video_metadata,
            do_sample_frames=do_sample_frames,
            sample_indices_fn=sample_indices_fn,
        )
        videos = self._prepare_input_videos(videos=videos, input_data_format=input_data_format, device=device)

        kwargs = self._standardize_kwargs(**kwargs)
        self._validate_preprocess_kwargs(**kwargs)

        kwargs.pop("data_format")
        return_metadata = kwargs.pop("return_metadata")

        preprocessed_videos = self._preprocess(videos=videos, **kwargs)
        if return_metadata:
            preprocessed_videos["video_metadata"] = video_metadata
        return preprocessed_videos

    def _preprocess(
        self,
        videos: list[list[torch.Tensor]],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: Optional[SizeDict] = None,
        interpolation: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        factor = patch_size * merge_size
        processed_videos = []
        processed_grids = []

        for video in videos:
            if len(video) == 0:
                raise ValueError("The input video must contain at least one frame.")

            num_groups = math.ceil(len(video) / temporal_patch_size)
            min_pixels_per_group = max(size.shortest_edge // max(num_groups, 1), factor * factor)
            max_pixels_per_group = max(size.longest_edge // max(num_groups, 1), factor * factor * temporal_patch_size)

            for start in range(0, len(video), temporal_patch_size):
                frame_group = list(video[start : start + temporal_patch_size])
                if len(frame_group) < temporal_patch_size:
                    frame_group.extend(frame_group[-1].clone() for _ in range(temporal_patch_size - len(frame_group)))

                resized_frames = []
                group_heights = []
                group_widths = []
                for frame in frame_group:
                    _, height, width = frame.shape
                    if do_resize:
                        resized_height, resized_width = smart_resize(
                            num_frames=temporal_patch_size,
                            height=height,
                            width=width,
                            temporal_factor=temporal_patch_size,
                            factor=factor,
                            min_pixels=min_pixels_per_group,
                            max_pixels=max_pixels_per_group,
                        )
                        if resized_height != height or resized_width != width:
                            frame = self.resize(
                                frame,
                                size=SizeDict(height=resized_height, width=resized_width),
                                interpolation=interpolation,
                            )
                    else:
                        resized_height = math.ceil(height / factor) * factor
                        resized_width = math.ceil(width / factor) * factor

                    frame = self.rescale_and_normalize(
                        frame.unsqueeze(0),
                        do_rescale,
                        rescale_factor,
                        do_normalize,
                        image_mean,
                        image_std,
                    ).squeeze(0)
                    resized_frames.append(frame)
                    group_heights.append(resized_height)
                    group_widths.append(resized_width)

                target_height = max(group_heights)
                target_width = max(group_widths)
                padded_frames = [self._pad_frame_to_size(frame, target_height, target_width) for frame in resized_frames]

                patches = torch.stack(padded_frames, dim=0).unsqueeze(0)
                batch_size, _, channel = patches.shape[:3]
                grid_h, grid_w = target_height // patch_size, target_width // patch_size

                patches = patches.view(
                    batch_size,
                    1,
                    temporal_patch_size,
                    channel,
                    grid_h // merge_size,
                    merge_size,
                    patch_size,
                    grid_w // merge_size,
                    merge_size,
                    patch_size,
                )
                patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
                flatten_patches = patches.reshape(
                    batch_size,
                    grid_h * grid_w,
                    channel * temporal_patch_size * patch_size * patch_size,
                )

                processed_videos.append(flatten_patches.squeeze(0))
                processed_grids.append([1, grid_h, grid_w])

        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)
        data = {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Qwen3_5VideoProcessor"]
