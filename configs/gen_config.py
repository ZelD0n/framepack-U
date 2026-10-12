#!/usr/bin/env python3
"""Generate an eval config JSON for FramePack experiments.

Usage:
    python gen_config.py <benchmark> <scheme> [sampling] [--model <model_name>]

    benchmark : longvideobench | videomme | mlvu | rtvbench | svbench-dialogue-en | svbench-streaming-en | streamingbench
                (case-insensitive)
    scheme    : compression scheme string, e.g. d2f512, d1f128d2f512
    sampling  : frame sampling strategy, 'redi' or 'uniform' (default: redi)
    --model   : model family, 'qwen3_5', 'qwen3_5_debug',
                'qwen3_5_moe', or 'qwen3_6'
                (default: qwen3_5)

Optional env-var overrides:
    MODEL_SIZE   : model size             (default: 4b for qwen3_5,
                                           35b-a3b for qwen3_5_moe,
                                           27b for qwen3_6)
    CLIP_SCORES  : offline relevance scores for redi (default: local/clip_scores)
    FRAME_FEATURES: frame_features_dir     (default: local/frame_feature; redi only)
    REDI_LAMBDA  : redi MMR lambda          (default: 0.7; redi only)
    PRE_RESIZE_LONG_EDGE: optional cap before FramePack scheme scaling
    MAX_NEW_TOKENS: override model max_new_tokens
    KEEP_CONFIG_IF_CHANGED: keep an existing config even if it differs from the current inputs
    DATASET_CLASS: override dataset class  (default: inferred from benchmark)

Output:
    Writes a config JSON to configs/ and prints the path to stdout.

Examples:
    python gen_config.py longvideobench d2f512 redi
    python gen_config.py videomme d1f128 uniform --model qwen3_5_moe
    MODEL_SIZE=9b python gen_config.py longvideobench d2f512 redi
"""

import json
import os
import re
import sys
from pathlib import Path

LOCAL_ROOT = Path(__file__).resolve().parents[1] / 'local'

# benchmark name (lower) -> {dataset key in config, class name, extra kwargs}
BENCHMARK_MAP = {
    "longvideobench":  {"name": "LongVideoBench",       "class": "LongVideoBench"},
    "videomme":        {"name": "Video-MME",             "class": "VideoMME"},
    "videomme-short":  {"name": "Video-MME_short",       "class": "VideoMME",      "extra": {"duration": "short"}},
    "videomme-medium": {"name": "Video-MME_medium",      "class": "VideoMME",      "extra": {"duration": "medium"}},
    "videomme-long":   {"name": "Video-MME_long",        "class": "VideoMME",      "extra": {"duration": "long"}},
    "mlvu":            {"name": "MLVU",                  "class": "MLVU"},
    "rtvbench":         {"name": "RTV-Bench",             "class": "RTVBench"},
    "rtv-bench":        {"name": "RTV-Bench",             "class": "RTVBench"},
    "svbench-dialogue-en":   {"name": "SVBench_Dialogue_EN",   "class": "SVBench"},
    "svbench-dialogue-zh":   {"name": "SVBench_Dialogue_ZH",   "class": "SVBench"},
    "svbench-streaming-en":  {"name": "SVBench_Streaming_EN",  "class": "SVBench"},
    "svbench-streaming-zh":  {"name": "SVBench_Streaming_ZH",  "class": "SVBench"},
    "streamingbench":        {"name": "StreamingBench",        "class": "StreamingBench"},
    "streaming-bench":       {"name": "StreamingBench",        "class": "StreamingBench"},
    "streamingbench-rtvu":   {
        "name": "StreamingBench_Real_Time_Visual_Understanding", "class": "StreamingBench"
    },
    "streamingbench-sqa":    {"name": "StreamingBench_Sequential_Question_Answering", "class": "StreamingBench"},
    "streamingbench-context": {"name": "StreamingBench_Contextual_Understanding", "class": "StreamingBench"},
    "streamingbench-omni":   {"name": "StreamingBench_Omni_Source_Understanding", "class": "StreamingBench"},
    "streamingbench-proactive": {"name": "StreamingBench_Proactive_Output", "class": "StreamingBench"},
}

def parse_scheme_nframe(scheme: str) -> int:
    """Sum all frame counts from a scheme string like 'd1f128d2f512' -> 640."""
    matches = re.findall(r"f(\d+)", scheme)
    if not matches:
        print(f"Error: cannot parse frame counts from scheme '{scheme}'", file=sys.stderr)
        sys.exit(1)
    return sum(int(f) for f in matches)


def main():
    # Parse --model flag anywhere in argv
    model_family = 'qwen3_5'
    args = sys.argv[1:]
    filtered_args = []
    i = 0
    while i < len(args):
        if args[i] == '--model' and i + 1 < len(args):
            model_family = args[i + 1].lower()
            i += 2
        else:
            filtered_args.append(args[i])
            i += 1
    args = filtered_args

    if len(args) < 2:
        print(
            "Usage: python gen_config.py <benchmark> <scheme> [redi|uniform] "
            "[--model qwen3_5|qwen3_5_debug|qwen3_5_moe|qwen3_6]\n"
            f"  benchmark: {' | '.join(BENCHMARK_MAP)}",
            file=sys.stderr,
        )
        sys.exit(1)

    benchmark_arg = args[0].lower()
    scheme        = args[1]
    sampling      = args[2] if len(args) > 2 else "redi"

    if benchmark_arg not in BENCHMARK_MAP:
        print(
            f"Error: unknown benchmark '{benchmark_arg}'. "
            f"Known: {', '.join(BENCHMARK_MAP)}",
            file=sys.stderr,
        )
        sys.exit(1)

    if sampling not in ("redi", "uniform"):
        print(f"Error: sampling must be 'redi' or 'uniform', got '{sampling}'", file=sys.stderr)
        sys.exit(1)

    if model_family not in ("qwen3_5", "qwen3_5_debug", "qwen3_5_moe", "qwen3_6"):
        print(f"Error: unsupported model family {model_family!r}; use qwen3_5, qwen3_5_debug, qwen3_5_moe, or qwen3_6.", file=sys.stderr)
        sys.exit(1)

    nframe = parse_scheme_nframe(scheme)

    bm_info       = BENCHMARK_MAP[benchmark_arg]
    dataset_name  = bm_info["name"]
    dataset_class = os.environ.get("DATASET_CLASS", bm_info["class"])
    clip_scores_dir = os.environ.get("CLIP_SCORES", str(LOCAL_ROOT / 'clip_scores'))
    frame_features_dir = os.environ.get("FRAME_FEATURES", str(LOCAL_ROOT / 'frame_feature'))
    redi_lambda = float(os.environ.get("REDI_LAMBDA", "0.7"))
    pre_resize_long_edge = os.environ.get("PRE_RESIZE_LONG_EDGE")
    pre_resize_long_edge = int(pre_resize_long_edge) if pre_resize_long_edge else None
    max_new_tokens_override = os.environ.get("MAX_NEW_TOKENS")
    max_new_tokens_override = int(max_new_tokens_override) if max_new_tokens_override else None

    # Data config
    data_entry = {"class": dataset_class, "nframe": nframe}
    data_entry.update(bm_info.get("extra", {}))
    data_config = {dataset_name: data_entry}

    if model_family in ('qwen3_5', 'qwen3_5_debug'):
        model_size   = os.environ.get("MODEL_SIZE", "4b").lower()
        model_name   = (
            f"Qwen3.5-{model_size.upper()}-FramePack-Debug"
            if model_family == 'qwen3_5_debug'
            else f"Qwen3.5-{model_size.upper()}-FramePack"
        )
        model_config = {
            "nframe": nframe,
            "compression_scheme": scheme,
            "max_new_tokens": 2048,
            "frame_sampling_strategy": sampling,
        }
        if benchmark_arg == "mlvu":
            model_config["mlvu_pre_resize_long_edge"] = 1280
        if pre_resize_long_edge is not None:
            model_config["pre_resize_long_edge"] = pre_resize_long_edge
        if sampling == "redi":
            model_config["clip_nframe"] = nframe
            model_config["clip_scores_dir"] = clip_scores_dir
            model_config["frame_features_dir"] = frame_features_dir
            model_config["redi_lambda"] = redi_lambda

        out_name = (
            f"config_{model_family}_{model_size}_{benchmark_arg}"
            f"_{nframe}frame_{sampling}_{scheme}.json"
        )
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_name)

    elif model_family == 'qwen3_5_moe':
        model_size = os.environ.get("MODEL_SIZE", "35b-a3b").lower()
        model_name = f"Qwen3.5-{model_size.upper()}-FramePack"
        model_config = {
            "nframe": nframe,
            "compression_scheme": scheme,
            "max_new_tokens": 2048,
            "frame_sampling_strategy": sampling,
        }
        if benchmark_arg == "mlvu":
            model_config["mlvu_pre_resize_long_edge"] = 1280
        if pre_resize_long_edge is not None:
            model_config["pre_resize_long_edge"] = pre_resize_long_edge
        if sampling == "redi":
            model_config["clip_nframe"] = nframe
            model_config["clip_scores_dir"] = clip_scores_dir
            model_config["frame_features_dir"] = frame_features_dir
            model_config["redi_lambda"] = redi_lambda

        out_name = (
            f"config_qwen3_5_moe_{model_size}_{benchmark_arg}"
            f"_{nframe}frame_{sampling}_{scheme}.json"
        )
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_name)

    elif model_family == 'qwen3_6':
        model_size = os.environ.get("MODEL_SIZE", "27b").lower()
        model_name = f"Qwen3.6-{model_size.upper()}-FramePack"
        model_config = {
            "nframe": nframe,
            "compression_scheme": scheme,
            "max_new_tokens": 2048,
            "frame_sampling_strategy": sampling,
        }
        if benchmark_arg == "mlvu":
            model_config["mlvu_pre_resize_long_edge"] = 1280
        if pre_resize_long_edge is not None:
            model_config["pre_resize_long_edge"] = pre_resize_long_edge
        if sampling == "redi":
            model_config["clip_nframe"] = nframe
            model_config["clip_scores_dir"] = clip_scores_dir
            model_config["frame_features_dir"] = frame_features_dir
            model_config["redi_lambda"] = redi_lambda

        out_name = (
            f"config_qwen3_6_{model_size}_{benchmark_arg}"
            f"_{nframe}frame_{sampling}_{scheme}.json"
        )
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_name)

    if benchmark_arg.startswith("svbench-"):
        model_config["max_new_tokens"] = 128
    if max_new_tokens_override is not None and "max_new_tokens" in model_config:
        model_config["max_new_tokens"] = max_new_tokens_override

    config = {
        "model": {model_name: model_config},
        "data": data_config,
    }

    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                existing_config = json.load(f)
        except json.JSONDecodeError as err:
            existing_config = None
            print(f"Existing config is invalid JSON ({err}); regenerating: {out_path}", file=sys.stderr)

        if existing_config == config:
            print(f"Config matches current inputs, reusing: {out_path}", file=sys.stderr)
        elif os.environ.get("KEEP_CONFIG_IF_CHANGED", "0") == "1":
            print(
                f"Config contents differ from current inputs, keeping because KEEP_CONFIG_IF_CHANGED=1: {out_path}",
                file=sys.stderr,
            )
        else:
            with open(out_path, "w") as f:
                json.dump(config, f, indent=4)
            print(f"Config contents differ from current inputs, updated: {out_path}", file=sys.stderr)
    else:
        with open(out_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Generated config: {out_path}", file=sys.stderr)

    print(out_path)


if __name__ == "__main__":
    main()
