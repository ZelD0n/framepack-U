from __future__ import annotations

import logging
import os
import warnings

import torch

from ..base import BaseModel
from .prompt import Qwen3VLPromptMixin
from ...smp import get_gpu_memory, listinstr


VLLM_MAX_IMAGE_INPUT_NUM = 24


def resize_size_long_edge(
    width: int,
    height: int,
    target_long_edge: int | None,
    align_factor: int = 32,
) -> tuple[int, int] | None:
    """Return framepack-style (width, height) resize target, or None if no resize is needed."""
    if not target_long_edge or target_long_edge <= 0:
        return None
    long_edge = max(width, height)
    if long_edge <= target_long_edge:
        return None

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
    return new_width, new_height


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


def is_moe_model(model_path: str) -> bool:
    """Check if the model is a Mixture of Experts model by looking for active-param suffixes like A3B, A17B."""
    import re
    if re.search(r'-A\d+B', model_path):
        return True
    return False


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


class Qwen3VLChat(Qwen3VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        mlvu_pre_resize_long_edge: int | None = None,
        max_new_tokens: int = 32768,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
        use_audio_in_video: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.mlvu_pre_resize_long_edge = mlvu_pre_resize_long_edge
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        if self.total_pixels and self.total_pixels > 24576 * 32 * 32:
            print('The total number of video tokens might too large, resulting in an overly long input sequence.')
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        self.max_frames = kwargs.pop('max_frames', None)
        self.FRAME_FACTOR = 2
        self.use_audio_in_video = use_audio_in_video

        assert model_path is not None
        self.model_path = model_path
        from transformers import AutoProcessor, AutoModelForImageTextToText
        # Use official Qwen3-Omni classes when model_path indicates omni
        if listinstr(['omni'], model_path.lower()):
            try:
                from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
            except Exception as err:
                logging.critical("pip install git+https://github.com/huggingface/transformers")
                raise err
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        else:
            self.processor = AutoProcessor.from_pretrained(model_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0
        self.device = bind_process_cuda_device()

        self.use_vllm = kwargs.get('use_vllm', False)
        self.use_lmdeploy = kwargs.get('use_lmdeploy', False)
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        assert self.use_vllm + self.use_lmdeploy <= 1, "You can only set one flag `use_vllm` to True"
        if self.use_vllm:
            if listinstr(['omni'], self.model_path.lower()):
                os.environ['VLLM_USE_V1'] = '0'
            from vllm import LLM
            gpu_count = torch.cuda.device_count()
            tp_size = gpu_count if gpu_count > 0 else 1
            logging.info(
                f'Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})'
            )
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    "VLLM_WORKER_MULTIPROC_METHOD is not set to spawn. Use 'export VLLM_WORKER_MULTIPROC_METHOD=spawn'"
                )
            enable_expert_parallel = is_moe_model(self.model_path)
            # For Qwen3-Omni, vLLM engine v1 is not supported yet
            if listinstr(['omni'], self.model_path.lower()):
                limit_mm = {"image": 3, "video": 3, "audio": 3}
            else:
                limit_mm = {"image": self.limit_mm_per_prompt}
            self.llm = LLM(
                model=self.model_path,
                max_num_seqs=kwargs.get("max_num_seqs", 8),
                limit_mm_per_prompt=limit_mm,
                tensor_parallel_size=tp_size,
                enable_expert_parallel=enable_expert_parallel,
                distributed_executor_backend=kwargs.get("distributed_executor_backend", "mp"),
                seed=0,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
                trust_remote_code=True,
                enforce_eager=kwargs.get("enforce_eager", False),
                disable_custom_all_reduce=kwargs.get("disable_custom_all_reduce", True),
            )
        else:
            if listinstr(['omni'], model_path.lower()):
                self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    model_path, dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
                )
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path, torch_dtype='auto', device_map={'': self.device}, attn_implementation='flash_attention_2'
                )
                # self.model = AutoModelForImageTextToText.from_pretrained(
                #     model_path, torch_dtype='auto', device_map="auto", attn_implementation='flash_attention_2'
                # )
            self.model.eval()

        torch.cuda.empty_cache()

    def _should_apply_mlvu_pre_resize(self, dataset: str | None) -> bool:
        if not self.mlvu_pre_resize_long_edge or self.mlvu_pre_resize_long_edge <= 0:
            return False
        if dataset is None:
            return False
        dataset_name = str(dataset).lower()
        return dataset_name == 'mlvu' or dataset_name.startswith('mlvu_')

    @staticmethod
    def _local_path_from_url(path: str) -> str | None:
        if path.startswith('file://'):
            return path[7:]
        if path.startswith(('http://', 'https://', 'data:video')):
            return None
        return path

    def _video_resize_hw(self, video_path: str) -> tuple[int, int] | None:
        local_path = self._local_path_from_url(video_path)
        if local_path is None:
            return None

        try:
            import cv2
            cap = cv2.VideoCapture(local_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if width > 0 and height > 0:
                size = resize_size_long_edge(width, height, self.mlvu_pre_resize_long_edge)
                if size is not None:
                    return size[1], size[0]
        except Exception:
            pass

        try:
            import decord
            vr = decord.VideoReader(local_path, ctx=decord.cpu(0))
            if len(vr) == 0:
                return None
            frame = vr[0].asnumpy()
            height, width = frame.shape[:2]
            size = resize_size_long_edge(width, height, self.mlvu_pre_resize_long_edge)
            if size is not None:
                return size[1], size[0]
        except Exception:
            return None
        return None

    def _fps_limited_nframes(self, video_path: str) -> tuple[int, int, float] | None:
        if self.fps is None or self.max_frames is None:
            return None
        local_path = self._local_path_from_url(video_path)
        if local_path is None:
            return None

        try:
            import cv2
            cap = cv2.VideoCapture(local_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            cap.release()
            if total <= 0 or source_fps <= 0:
                return None
            sampled_by_fps = int(total / source_fps * float(self.fps))
            target = min(max(sampled_by_fps, self.FRAME_FACTOR), int(self.max_frames), total)
            if total < self.FRAME_FACTOR:
                # Too few frames for smart_nframes; let qwen_vl_utils decide.
                return None
            return max(self.FRAME_FACTOR, target // self.FRAME_FACTOR * self.FRAME_FACTOR), sampled_by_fps, source_fps
        except Exception:
            return None

    def _frame_list_resize_hw(self, frames: list) -> tuple[int, int] | None:
        if not frames:
            return None
        first = frames[0]
        try:
            if isinstance(first, str):
                from PIL import Image
                local_path = self._local_path_from_url(first)
                if local_path is None:
                    return None
                with Image.open(local_path) as image:
                    width, height = image.size
            elif hasattr(first, 'shape') and len(first.shape) >= 2:
                height, width = first.shape[:2]
            elif hasattr(first, 'size') and not callable(first.size):
                width, height = first.size
            else:
                return None
            size = resize_size_long_edge(width, height, self.mlvu_pre_resize_long_edge)
            if size is not None:
                return size[1], size[0]
        except Exception:
            return None
        return None

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 32 * 32
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['min_pixels', 'max_pixels', 'total_pixels', 'resized_height', 'resized_width']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
            elif s['type'] == 'video':
                value = s['value']
                if isinstance(value, list):
                    item = {
                        'type': 'video',
                        'video': [ensure_image_url(v) if isinstance(v, str) else v for v in value],
                    }
                    if self.fps is not None and 'fps' not in item:
                        item['fps'] = self.fps
                    if self.max_frames is not None and 'max_frames' not in item:
                        item['max_frames'] = self.max_frames
                    if self._should_apply_mlvu_pre_resize(dataset):
                        resized_hw = self._frame_list_resize_hw(value)
                        if resized_hw is not None:
                            item['resized_height'], item['resized_width'] = resized_hw
                else:
                    item = {'type': 'video', 'video': ensure_video_url(value)}
                    if self._should_apply_mlvu_pre_resize(dataset):
                        resized_hw = self._video_resize_hw(value)
                        if resized_hw is not None:
                            item['resized_height'], item['resized_width'] = resized_hw
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['resized_height', 'resized_width', 'fps', 'nframes', 'sample_fps']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
                if not isinstance(value, list):
                    limited = self._fps_limited_nframes(value)
                    if limited is not None:
                        limited_nframes, sampled_by_fps, source_fps = limited
                        item['nframes'] = limited_nframes
                        print(
                            '[Qwen3VLChat] video sampling '
                            f'video={os.path.basename(value)} 总共fps={source_fps:.3f} '
                            f'按照fps采样帧数={sampled_by_fps} max_frames={self.max_frames} '
                            f'有效帧数={limited_nframes}',
                            flush=True,
                        )
                    elif self.fps is not None and 'fps' not in item:
                        item['fps'] = self.fps
                        if self.max_frames is not None and 'max_frames' not in item:
                            item['max_frames'] = self.max_frames
                        print(
                            '[Qwen3VLChat] video sampling fallback '
                            f'video={os.path.basename(value)} configured_fps={self.fps} '
                            f'max_frames={self.max_frames}',
                            flush=True,
                        )
                    elif self.nframe is not None and 'nframes' not in item:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            if new_frame_count >= self.FRAME_FACTOR:
                                print(f"use {new_frame_count} for {s['value']}")
                                item['nframes'] = new_frame_count
                            # else: cv2 frame_count unreliable (e.g. h264 without
                            # frame count atom); skip nframes, let qwen_vl_utils decide.
                        else:
                            item['nframes'] = self.nframe
            elif s['type'] == 'audio':
                item = {'type': 'audio', 'audio': s['value']}
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner_transformers(self, message, dataset=None):
        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("Please install it via 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("Please install it via 'pip install qwen-vl-utils'")
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        if is_omni:
            # For Qwen3-Omni, messages is a list of dicts
            try:
                text = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors='pt',
                padding=True,
                use_audio_in_video=self.use_audio_in_video,
            )
        else:
            try:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
                for meta in video_metadatas:
                    if isinstance(meta, dict) and 'fps' not in meta and self.fps is not None:
                        meta['fps'] = self.fps

            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                do_resize=False,
                return_tensors='pt',
                **(video_kwargs or {}),
            )
        try:
            inputs = inputs.to(self.model.device)
            if hasattr(self.model, 'dtype'):
                inputs = inputs.to(self.model.dtype)
        except Exception:
            inputs = inputs.to('cuda')

        if is_omni:
            with torch.inference_mode():
                try:
                    text_ids, _ = self.model.generate(
                        **inputs,
                        return_audio=False,
                        thinker_return_dict_in_generate=True,
                        use_audio_in_video=self.use_audio_in_video,
                    )
                except TypeError:
                    text_ids, _ = self.model.generate(
                        **inputs,
                        return_audio=False,
                        use_audio_in_video=self.use_audio_in_video,
                    )
            response = self.processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        else:
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    **self.generate_kwargs,
                )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            out = self.processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = out[0]
        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import SamplingParams
        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, 'pip install qwen-vl-utils'")
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if is_omni:
            audios, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
        else:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            stop_token_ids=None
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        if is_omni and 'audios' in locals() and audios is not None:
            mm_data['audio'] = audios

        req = {'prompt': text}
        if mm_data:
            req['multi_modal_data'] = mm_data
        if is_omni:
            req['mm_processor_kwargs'] = {"use_audio_in_video": self.use_audio_in_video}
        elif video_kwargs is not None:
            req['mm_processor_kwargs'] = video_kwargs

        outputs = self.llm.generate([req], sampling_params=sampling_params, use_tqdm=False)

        for o in outputs:
            generated_text = o.outputs[0].text

        if self.post_process:
            resp = generated_text.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                generated_text = resp[:end]

        if self.verbose:
            print(f'\033[32m{generated_text}\033[0m')
        return generated_text

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)
