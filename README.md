<p align="center">
  <img src="assets/elephant.webp" alt="FramePack-U logo" style="width: 60%;" />
</p>

<!-- # Elephant in the Fridge: Constant-Memory Frame Packing for Long Video Understanding -->
<h1 align="center">
  Elephant in the Fridge: Constant-Memory Frame Packing for Long Video Understanding
</h1>

<!-- This is the official code repo for paper *Elephant in the Fridge: Constant-Memory Frame Packing for Long Video Understanding*. -->
<p align="center">
  <a href="https://zeld0n.github.io/shuo_homepage/">Shuo Gao</a>, <a href="https://hellomuffin.github.io/">Chenhao Zheng</a>, <a href="https://jason718.github.io/">Zhongzheng Ren</a>.
</p>
<p align="center">
<a href="https://zeld0n.github.io/framepack-U/"><img src="https://img.shields.io/badge/Homepage-FramePackU-green?logo=googlechrome&logoColor=white" alt="Homepage"></a>
<a href="https://arxiv.org/abs/2610.16118"><img src="https://img.shields.io/badge/arXiv-FramePackU-red?logo=arxiv" alt="GitHub"></a>
<a href="https://github.com/ZelD0n/framepack-U"><img src="https://img.shields.io/badge/GitHub-FramePackU-blue?logo=github" alt="GitHub"></a>
<a href="https://huggingface.co/datasets/ZelD0n/FramePack-U"><img src="https://img.shields.io/badge/HuggingFace-FramePackU-yellow?logo=huggingface" alt="Hugging Face"></a>
<a href="https://www.modelscope.cn/datasets/ZelD0n/FramePack-U"><img src="https://img.shields.io/badge/ModelScope-FramePackU-purple?logo=modelscope" alt="ModelScope"></a>
</p>

<p align="center">
  <img src="assets/framepack_framework.png" alt="FramePack-U framework" style="width: 85%;"/>
</p>

## 📑 Contents

- [News](#-news)
- [Preparation](#-preparation)
  - [Environment](#environment)
  - [Dataset downloads](#dataset-downloads)
  - [Model downloads](#model-downloads)
  - [Local model and benchmark paths](#local-model-and-benchmark-paths)
  - [Offline data](#offline-data)
- [Offline Storage](#-offline-storage)
- [Quick Start](#-quick-start)
  - [Certain Resolution Scheme](#certain-resolution-scheme)
  - [Large Packing Schemes](#large-packing-schemes)
- [Supported Models](#-supported-models)
- [Supported Sampling Strategies](#-supported-sampling-strategies)
- [Acknowledgement](#-acknowledgement)
- [License](#-license)
- [Citation](#-citation)

## 📰 News
- [2026/10/12] 📦 Our paper, code, and supporting materials are now available!
- [2026/09/25] 🎉 Our paper has been accepted to **NeurIPS 2026**!


## 🛠 Preparation

After cloning the repo, install the environment, download the models and benchmark data, and link them to the local paths below.

### Environment

Run the following command to create a new environment and install the required packages:

```bash
conda create -n framepack-u python=3.10 pip -y
conda activate framepack-u
cd /path/to/framepack-U

python -m pip install pip==26.1.2
python -m pip install \
  torch==2.10.0+cu128 \
  torchvision==0.25.0+cu128 \
  torchaudio==2.10.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r requirements.txt
```

Install the matching prebuilt FlashAttention wheel:

```bash
python -m pip install --no-deps \
  "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3%2Bcu128torch2.10-cp310-cp310-linux_x86_64.whl"
```

Alternatively, [download the wheel](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3%2Bcu128torch2.10-cp310-cp310-linux_x86_64.whl)
and install it locally:

```bash
python -m pip install --no-index --no-deps \
  /path/to/flash_attn-2.8.3+cu128torch2.10-cp310-cp310-linux_x86_64.whl
```

### Dataset downloads

Download the benchmark videos and annotations from the following repositories.
Extract any archives according to the dataset instructions before linking the
prepared directories in [Local model and benchmark paths](#local-model-and-benchmark-paths).

| Dataset | Download |
| --- | --- |
| LongVideoBench | [Hugging Face](https://huggingface.co/datasets/longvideobench/LongVideoBench) |
| Video-MME | [Hugging Face](https://huggingface.co/datasets/lmms-eval/Video-MME) |
| MLVU | [Hugging Face](https://huggingface.co/datasets/MLVU/MVLU) |

### Model downloads

Download the checkpoint for the model you want to evaluate. CLIP is needed only
when generating the offline scores and features locally (see [tools/README.md](tools/README.md)).

| Model | Usage | Download |
| --- | --- | --- |
| Qwen3.5-4B | Default for `qwen3_5` | [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Qwen3.5-9B | `qwen3_5` with `MODEL_SIZE=9B` | [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-9B) |
| Qwen3.5-35B-A3B | Default for `qwen3_5_moe` | [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) |
| Qwen3.6-27B | Default for `qwen3_6` | [Hugging Face](https://huggingface.co/Qwen/Qwen3.6-27B) |
| CLIP ViT-B/32 | Offline score and feature generation | [Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32) |

### Local model and benchmark paths
Link your existing model weights and prepared benchmark directories once. The
setup script creates links under `local/`:

```bash
python scripts/setup_local_paths.py \
  --model Qwen3.6-27B=/path/to/Qwen3.6-27B \
  --benchmark LongVideoBench=/path/to/LongVideoBench \
  --clip-scores /path/to/clip_scores \
  --frame-features /path/to/frame_feature
```

Repeat `--model` and `--benchmark` to link more resources. Supported benchmark
names are `LongVideoBench`, `Video-MME` and `MLVU`. The dataset links should point to directories containing the
benchmark files and videos.

Model weights are read from `local/models/<model name>/` and datasets from
`local/benchmarks/<benchmark name>/`. Paths are resolved from the repository,
so evaluation commands work from any current directory. Missing links cause
an error before model loading. The linked benchmark directories must be writable
if the dataset needs to extract archives or generate TSV files.

### Offline data
Framepack-U uses CLIP relevance scores in `local/clip_scores/` and frame features in
`local/frame_feature/`. Both must contain data for each video with matching
frame indices. Use `--clip-scores` and `--frame-features` of `scripts/setup_local_paths.py` in the setup command
to link them. See [Offline Storage](#-offline-storage) to prepare the offline data.

## 💾 Offline Storage
To avoid massive feature extraction and similarity caculation, you can download our pre-computed scores and features. For huggingface, you can download them from [this link](https://huggingface.co/datasets/ZelD0n/FramePack-U). For modelscope, you can download them from [this link](https://modelscope.cn/datasets/ZelD0n/FramePack-U).

To use the offline data, link the downloaded score and feature directories with
`scripts/setup_local_paths.py` using `--clip-scores` and `--frame-features` as
shown in [Preparation](#-preparation), then run evaluation with our method.

You can aslo generate them locally, see [tools/README.md](tools/README.md).

## 🚀 Quick Start
### Certain Resolution Scheme
To run evaluation under certain resolution scheme, the shell command is 
```bash
sh eval.sh [model] [benchmark] [scheme] [strategy] [--verbose]
```
For example:

```bash
sh eval.sh qwen3_5 longvideobench d1f8d2f480 redi
```

Add `--verbose` to show watchdog progress and frame selection details. By
default, routine progress messages are hidden while failures remain visible.

To run evaluation with vLLM, use `eval_vllm.sh` (which enables vLLM by default)
and set the data-parallel worker count with `DP`:

```bash
DP=8 sh eval_vllm.sh qwen3_6 longvideobench d1f8d2f480 redi
```

Alternatively, set `USE_VLLM=1` when using `eval.sh`:

```bash
USE_VLLM=1 DP=8 sh eval.sh qwen3_6 longvideobench d1f8d2f480 redi
```

With `eval.sh`, setting `DP` alone does not enable vLLM.

### Large Packing Schemes

Large packing schemes may require a higher context limit and chunked prefill.
The example below assumes eight visible GPUs with `DP=4` (two GPUs per worker).
You can adjust the limits and memory budget for your hardware and input lengths, for example:

```bash
DP=4 \
VLLM_MAX_MODEL_LEN=442368 \
VLLM_MAX_PROMPT_TOKENS=440320 \
VLLM_MAX_NUM_BATCHED_TOKENS=42768 \
VLLM_ENABLE_CHUNKED_PREFILL=1 \
VLLM_GPU_MEMORY_UTILIZATION=0.95 \
MODEL_SIZE=27b \
sh eval_vllm.sh qwen3_6 longvideobench d1f64d2f512d4f1024 redi
```


## 🤖 Supported Models
FramePack-U is now available for the following models:

- Qwen3.5: `qwen3_5`
  - Qwen3.5-4B: Default
  - Qwen3.5-9B: set `MODEL_SIZE=9B`

- Qwen3.5-MoE: `qwen3_5_moe`
  - Qwen3.5-35B-A3B: Default

- Qwen3.6: `qwen3_6`
  - Qwen3.6-27B: Default


## 🎯 Supported Sampling Strategies
- ReDi (Relevance+Diversity): `redi`, our FramePack-U method.
- Bi-tree: `uniform`, a query-agnostic alternative sampling strategy by binary-tree midpoint ordering, see [our paper](https://zeld0n.github.io/framepack-U/assets/paper.pdf) for details.


## 🙏 Acknowledgement
This work is built upon [VLMEvalKit](https://github.com/open-compass/vlmevalkit), [Qwen3.5](https://github.com/QwenLM/Qwen3-VL), and inspired from [FramePack](https://github.com/lllyasviel/framepack).

Thanks for these amazing works!

## 📄 License

FramePack-U is released under the [Apache License 2.0](LICENSE).

## 📚 Citation
If you find this work helpful, please consider citing our paper:
```bibtex
@inproceedings{gao2026elephant,
  title = {Elephant in the Fridge: Constant-Memory Frame Packing for Long Video Understanding},
  author = {Gao, Shuo and Zheng, Chenhao and Ren, Zhongzheng},
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year = {2026}
}
```
