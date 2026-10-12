#!/usr/bin/env python3
"""Visualize Qwen3.6-27B FramePack frame sampling strategies.

Randomly samples 10 video-question pairs from each of LongVideoBench, MLVU,
and Video-MME, then saves 32 frames per pair for uniform, CLIP, and ReDi
sampling under ./visualization.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "visualization"

DEFAULT_ROOTS = {
    "LongVideoBench": Path("/gaoshuo-sh/benchmarks/LongVideoBench/LongVideoBench"),
    "MLVU": Path("/gaoshuo-sh/benchmarks/MVLU/MVLU"),
    "Video-MME": Path("/gaoshuo-sh/benchmarks/Video-MME"),
}

DATASET_ALIASES = {
    "LongVideoBench": ["LongVideoBench"],
    "MLVU": ["MLVU", "MLVU_MCQ"],
    "Video-MME": ["Video-MME", "VideoMME"],
}


@dataclass
class Sample:
    benchmark: str
    row_index: int
    video_path: Path
    question: str
    lookup_query: str
    row: dict[str, Any]


def normalize_video_path(video_path: str | Path) -> str:
    path = str(video_path)
    if path.startswith("file://"):
        path = path[len("file://") :]
    return os.path.normpath(path).replace("\\", "/")


def normalize_query(query: str) -> str:
    normalized = str(query).lstrip()
    suffix = "\nAnswer with the option's letter from the given choices directly."
    if normalized.endswith(suffix):
        normalized = normalized[: -len(suffix)].rstrip()
    lines = normalized.splitlines()
    content_lines = []
    for line in lines:
        if re.match(r"^[A-Z][\.\)]\s", line.strip()):
            break
        content_lines.append(line.rstrip())
    question_only = "\n".join(content_lines).strip()
    return question_only or normalized.strip()


def literal_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.tolist()]
    return [str(x) for x in ast.literal_eval(str(value))]


def options_dot(candidates: list[str]) -> str:
    return "\n".join(f"{chr(ord('A') + idx)}. {candidate}" for idx, candidate in enumerate(candidates))


def options_paren(candidates: list[str]) -> str:
    return "\n".join(f"({chr(ord('A') + idx)}) {candidate}" for idx, candidate in enumerate(candidates))


def longvideobench_query(row: dict[str, Any]) -> str:
    candidates = literal_list(row["candidates"])
    return (
        f"{row['question']}\n{options_dot(candidates)}"
        "\nAnswer with the option's letter from the given choices directly."
    )


def mlvu_query(row: dict[str, Any]) -> str:
    candidates = literal_list(row["candidates"])
    sys_prompt = (
        "Carefully watch this video and pay attention to every detail. "
        "Based on your observations, select the best option that accurately addresses the question."
    )
    return f"{sys_prompt} Question: {row['question']}\nOptions:\n{options_paren(candidates)}\nOnly give the best option."


def videomme_query(row: dict[str, Any]) -> str:
    candidates = literal_list(row["candidates"])
    question = str(row["question"]) + "\n" + "\n".join(candidates)
    frame_prompt = (
        "These are the frames of a video. "
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option."
    )
    return f"{frame_prompt}\n Question: {question}\nAnswer: "


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset TSV: {path}")
    return pd.read_csv(path, sep="\t")


def load_samples(benchmark: str, root: Path, count: int, rng: random.Random) -> list[Sample]:
    if benchmark == "LongVideoBench":
        df = read_tsv(root / "LongVideoBench.tsv")
        make_video = lambda row: root / str(row["video_path"]).replace("./", "", 1)
        make_query = longvideobench_query
    elif benchmark == "MLVU":
        df = read_tsv(root / "MLVU_MCQ.tsv")
        make_video = lambda row: root / str(row["prefix"]).replace("./", "", 1) / str(row["video"])
        make_query = mlvu_query
    elif benchmark == "Video-MME":
        df = read_tsv(root / "Video-MME.tsv")
        make_video = lambda row: root / str(row["video_path"]).replace("./", "", 1)
        make_query = videomme_query
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")

    candidates: list[Sample] = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        video_path = make_video(row_dict)
        if not video_path.exists():
            continue
        query = make_query(row_dict)
        candidates.append(
            Sample(
                benchmark=benchmark,
                row_index=int(row_dict.get("index", idx)),
                video_path=video_path,
                question=query,
                lookup_query=normalize_query(query),
                row=row_dict,
            )
        )

    if len(candidates) < count:
        raise RuntimeError(f"{benchmark} only has {len(candidates)} usable samples; need {count}.")
    rng.shuffle(candidates)
    return candidates[:count]


def resolve_clip_dir(clip_root: Path, benchmark: str, frame_label: str = "allframes") -> Path:
    tried = []
    for ds_name in DATASET_ALIASES[benchmark]:
        for candidate in (clip_root / ds_name / frame_label, clip_root / ds_name):
            tried.append(candidate)
            if has_clip_files(candidate):
                return candidate
    for candidate in (clip_root / frame_label, clip_root):
        tried.append(candidate)
        if has_clip_files(candidate):
            return candidate
    raise FileNotFoundError(f"No offline CLIP files found for {benchmark}. Tried: {tried}")


def has_clip_files(path: Path) -> bool:
    return all((path / name).exists() for name in ("scores.json", "frames.json", "video_paths.json", "queries.json"))


def load_clip_index(clip_dir: Path) -> dict[str, Any]:
    with open(clip_dir / "scores.json") as f:
        scores = json.load(f)
    with open(clip_dir / "frames.json") as f:
        frames = json.load(f)
    with open(clip_dir / "video_paths.json") as f:
        video_paths = json.load(f)
    with open(clip_dir / "queries.json") as f:
        queries = json.load(f)

    by_query: dict[str, list[tuple[str, tuple[list[float], list[int] | None]]]] = {}
    for vpath, query, score_list, frame_list in zip(video_paths, queries, scores, frames):
        norm_query = normalize_query(str(query))
        payload = ([float(x) for x in score_list], None if frame_list is None else [int(x) for x in frame_list])
        by_query.setdefault(norm_query, []).append((normalize_video_path(vpath), payload))
    return {"dir": str(clip_dir), "by_query": by_query}


def lookup_clip(clip_index: dict[str, Any], video_path: Path, query: str) -> tuple[list[float], list[int] | None]:
    norm_path = normalize_video_path(video_path)
    matches = [
        payload
        for candidate_path, payload in clip_index["by_query"].get(normalize_query(query), [])
        if norm_path.endswith(candidate_path) or candidate_path.endswith(norm_path)
    ]
    if len(matches) != 1:
        raise KeyError(f"Offline CLIP lookup returned {len(matches)} matches for {video_path}")
    return matches[0]


def resolve_features_dir(features_root: Path, benchmark: str) -> Path:
    tried = []
    for ds_name in DATASET_ALIASES[benchmark]:
        candidate = features_root / ds_name
        tried.append(candidate)
        if (candidate / "manifest.json").exists():
            return candidate
    if (features_root / "manifest.json").exists():
        return features_root
    raise FileNotFoundError(f"No frame-feature manifest found for {benchmark}. Tried: {tried}")


def load_feature_manifest(features_dir: Path) -> dict[str, Path]:
    with open(features_dir / "manifest.json") as f:
        manifest = json.load(f)
    mapping: dict[str, Path] = {}
    for entry in manifest.get("videos", []):
        rel = entry.get("feature_path")
        if not rel:
            continue
        feat_path = Path(rel) if os.path.isabs(str(rel)) else features_dir / rel
        for raw in (entry.get("source_video_path"), entry.get("video_path")):
            if raw:
                mapping.setdefault(normalize_video_path(raw), feat_path)
    return mapping


def lookup_feature_path(mapping: dict[str, Path], video_path: Path) -> Path:
    norm_path = normalize_video_path(video_path)
    exact = mapping.get(norm_path)
    if exact is not None:
        return exact

    matches = [
        (candidate_path, feat)
        for candidate_path, feat in mapping.items()
        if norm_path.endswith(candidate_path) or candidate_path.endswith(norm_path)
    ]
    unique_feats = []
    for _, feat in matches:
        if feat not in unique_feats:
            unique_feats.append(feat)
    if len(unique_feats) == 1:
        return unique_feats[0]

    if matches:
        max_len = max(len(candidate_path) for candidate_path, _ in matches)
        longest = [feat for candidate_path, feat in matches if len(candidate_path) == max_len]
        unique_longest = []
        for feat in longest:
            if feat not in unique_longest:
                unique_longest.append(feat)
        if len(unique_longest) == 1:
            return unique_longest[0]

    raise KeyError(f"Frame-feature lookup returned {len(unique_feats)} unique matches for {video_path}")


def import_torch():
    try:
        import torch
    except Exception as err:
        raise RuntimeError(
            "ReDi sampling requires PyTorch to load offline frame-feature tensors, "
            "but importing torch failed in this environment."
        ) from err
    return torch


def load_feature_tensor(path: Path) -> tuple[Any, list[int]]:
    torch = import_torch()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    feats = blob["features"].to(torch.float32)
    if not bool(blob.get("features_are_normalized", True)):
        feats = torch.nn.functional.normalize(feats, dim=-1)
    idxs = blob["frame_indices"]
    if isinstance(idxs, torch.Tensor):
        idxs = idxs.to(torch.int64).tolist()
    else:
        idxs = [int(x) for x in idxs]
    return feats, idxs


def video_metadata(video_path: Path) -> tuple[int, float | None]:
    try:
        import decord

        vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        return len(vr), float(vr.get_avg_fps())
    except Exception:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or None
        cap.release()
        if total <= 0:
            raise RuntimeError(f"Could not read video metadata: {video_path}")
        return total, fps


def decode_frames(video_path: Path, indices: list[int]) -> list[Image.Image]:
    try:
        import decord

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        total = len(vr)
        safe = [min(max(int(i), 0), total - 1) for i in indices]
        batch = vr.get_batch(safe).asnumpy()
        return [Image.fromarray(batch[i]).convert("RGB") for i in range(len(batch))]
    except Exception:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames: list[Image.Image] = []
        for idx in indices:
            safe = min(max(int(idx), 0), total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(safe))
            ok, frame = cap.read()
            if ok:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame).convert("RGB"))
            elif frames:
                frames.append(frames[-1].copy())
        cap.release()
        return frames


def sample_uniform(total: int, nframe: int) -> tuple[list[int], list[int]]:
    n = min(nframe, total)
    indices = np.linspace(0, total - 1, n).round().astype(int).tolist()
    return indices, list(range(n))


def sample_clip(offline_scores: list[float], offline_indices: list[int] | None, nframe: int) -> tuple[list[int], list[int]]:
    n = min(nframe, len(offline_scores))
    ranked_desc = sorted(range(len(offline_scores)), key=lambda i: offline_scores[i], reverse=True)
    top_ranked = sorted(ranked_desc[:n])
    frame_indices = [offline_indices[i] for i in top_ranked] if offline_indices is not None else top_ranked
    rank_in_all = {idx: rank for rank, idx in enumerate(ranked_desc)}
    priority_ranks = [rank_in_all[i] for i in top_ranked]
    return [int(x) for x in frame_indices], [int(x) for x in priority_ranks]


def sample_redi(
    offline_scores: list[float],
    offline_indices: list[int] | None,
    feats: Any,
    feat_indices: list[int],
    nframe: int,
    lam: float,
) -> tuple[list[int], list[int]]:
    torch = import_torch()
    clip_indices = list(range(len(offline_scores))) if offline_indices is None else [int(x) for x in offline_indices]
    feat_idx_to_row = {int(idx): row for row, idx in enumerate(feat_indices)}
    cand_scores: list[float] = []
    cand_rows: list[int] = []
    cand_orig_indices: list[int] = []
    for pos, frame_idx in enumerate(clip_indices):
        row = feat_idx_to_row.get(int(frame_idx))
        if row is None:
            continue
        cand_scores.append(float(offline_scores[pos]))
        cand_rows.append(row)
        cand_orig_indices.append(int(frame_idx))

    if not cand_scores:
        raise RuntimeError("No overlap between offline CLIP frame indices and frame features.")

    target = min(nframe, len(cand_scores))
    F = feats[cand_rows].to(torch.float32)
    r = torch.tensor(cand_scores, dtype=torch.float32)

    selected: list[int] = []
    available = torch.ones(len(cand_scores), dtype=torch.bool)
    max_sim = torch.full((len(cand_scores),), float("-inf"))

    first = int(torch.argmax(r).item())
    selected.append(first)
    available[first] = False
    max_sim = torch.maximum(max_sim, F @ F[first])

    while len(selected) < target:
        score = lam * r - (1.0 - lam) * max_sim
        score = torch.where(available, score, torch.full_like(score, float("-inf")))
        nxt = int(torch.argmax(score).item())
        if not bool(available[nxt].item()):
            break
        selected.append(nxt)
        available[nxt] = False
        max_sim = torch.maximum(max_sim, F @ F[nxt])

    rank_by_local = {local: rank for rank, local in enumerate(selected)}
    selected_sorted = sorted(selected, key=lambda i: cand_orig_indices[i])
    return [cand_orig_indices[i] for i in selected_sorted], [rank_by_local[i] for i in selected_sorted]


def save_strategy_frames(out_dir: Path, frames: list[Image.Image], indices: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for pos, (frame, idx) in enumerate(zip(frames, indices)):
        frame.save(out_dir / f"frame_{pos:02d}_idx_{idx:08d}.jpg", quality=92)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clip-scores", type=Path, default=Path(os.environ.get("CLIP_SCORES", ROOT / "clip_scores")))
    parser.add_argument("--frame-features", type=Path, default=Path(os.environ.get("FRAME_FEATURES", "/gaoshuo-sh/frame_feature")))
    parser.add_argument("--nframe", type=int, default=32)
    parser.add_argument("--per-benchmark", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--redi-lambda", type=float, default=float(os.environ.get("REDI_LAMBDA", "0.7")))
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("uniform", "clip", "redi"),
        default=["uniform", "clip", "redi"],
        help="Sampling strategies to visualize. Defaults to all three.",
    )
    parser.add_argument("--longvideobench-root", type=Path, default=DEFAULT_ROOTS["LongVideoBench"])
    parser.add_argument("--mlvu-root", type=Path, default=DEFAULT_ROOTS["MLVU"])
    parser.add_argument("--videomme-root", type=Path, default=DEFAULT_ROOTS["Video-MME"])
    args = parser.parse_args()

    rng = random.Random(args.seed)
    roots = {
        "LongVideoBench": args.longvideobench_root,
        "MLVU": args.mlvu_root,
        "Video-MME": args.videomme_root,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    all_metadata: list[dict[str, Any]] = []

    for benchmark, root in roots.items():
        print(f"[{benchmark}] loading dataset rows from {root}")
        samples = load_samples(benchmark, root, args.per_benchmark, rng)
        print(f"[{benchmark}] loading offline CLIP scores from {args.clip_scores}")
        clip_index = load_clip_index(resolve_clip_dir(args.clip_scores, benchmark))
        feature_map = None
        if "redi" in args.strategies:
            print(f"[{benchmark}] loading frame-feature manifest from {args.frame_features}")
            feature_map = load_feature_manifest(resolve_features_dir(args.frame_features, benchmark))

        for sample_pos, sample in enumerate(samples):
            total, fps = video_metadata(sample.video_path)
            offline_scores, offline_frame_indices = lookup_clip(clip_index, sample.video_path, sample.lookup_query)
            strategy_indices = {}
            feat_path = None
            if "uniform" in args.strategies:
                strategy_indices["uniform"] = sample_uniform(total, args.nframe)
            if "clip" in args.strategies:
                strategy_indices["clip"] = sample_clip(offline_scores, offline_frame_indices, args.nframe)
            if "redi" in args.strategies:
                assert feature_map is not None
                feat_path = lookup_feature_path(feature_map, sample.video_path)
                feats, feat_indices = load_feature_tensor(feat_path)
                strategy_indices["redi"] = sample_redi(
                    offline_scores,
                    offline_frame_indices,
                    feats,
                    feat_indices,
                    args.nframe,
                    args.redi_lambda,
                )

            sample_dir = args.output / benchmark / f"{sample_pos:02d}_row_{sample.row_index}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            write_text(sample_dir / "question.txt", sample.question)

            metadata = {
                "benchmark": benchmark,
                "row_index": sample.row_index,
                "video_path": str(sample.video_path),
                "total_num_frames": total,
                "fps": fps,
                "question": sample.question,
                "lookup_query": sample.lookup_query,
                "clip_scores_dir": clip_index["dir"],
                "feature_path": str(feat_path) if feat_path is not None else None,
                "strategies": {},
            }

            for strategy, (indices, priority_ranks) in strategy_indices.items():
                frames = decode_frames(sample.video_path, indices)
                save_strategy_frames(sample_dir / strategy, frames, indices)
                metadata["strategies"][strategy] = {
                    "frame_indices": indices,
                    "priority_ranks": priority_ranks,
                    "saved_frames": len(frames),
                }

            (sample_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            all_metadata.append(metadata)
            print(f"[{benchmark}] saved sample {sample_pos + 1}/{len(samples)}: {sample_dir}")

    (args.output / "manifest.json").write_text(
        json.dumps(all_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Done. Saved {len(all_metadata)} video-question pairs to {args.output}")


if __name__ == "__main__":
    main()
