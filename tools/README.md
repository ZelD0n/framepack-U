# Offline CLIP calculation

## ℹ️ Overview
`save_clip_score.py` computes per-frame CLIP similarities for each benchmark
question. `save_frame_feature.py` saves normalized CLIP image features for each
video. Both use `openai/clip-vit-base-patch32` and decode all frames when
`--sample_fps 0` (by default).
`scripts/setup_local_paths.py` creates repository-local links to model weights,
benchmark datasets, and offline CLIP score and frame feature caches.

## ▶️ Running
Run these commands from the repository root:

- `scripts/setup_local_paths.py` links the dataset into `local/benchmarks/`.
  Its `--benchmark` argument takes `NAME=/path/to/dataset`.
- `tools/save_clip_score.py` computes per-frame CLIP scores.
- `tools/save_frame_feature.py` extracts and saves per-frame CLIP features.

For the last two scripts, `--benchmark` selects the linked dataset,
`--clip_model` points to a local CLIP checkpoint, and `--save_root` sets the
output directory. Use a new output directory to preserve existing caches.
For example, to regenerate LongVideoBench:

```bash
python scripts/setup_local_paths.py --benchmark LongVideoBench=/path/to/LongVideoBench
python tools/save_clip_score.py --benchmark LongVideoBench \
  --clip_model /path/to/clip-vit-base-patch32 \
  --save_root /path/to/save-root
python tools/save_frame_feature.py --benchmark LongVideoBench \
  --clip_model /path/to/clip-vit-base-patch32 \
  --save_root /path/to/save-root
```

## 📁 Cached outputs
The cached outputs look like this:
```text
<save_root>/              # Root for generated CLIP caches
└── <benchmark>/          # Outputs for the selected benchmark
    ├── allframes/         # CLIP scores, frame indices, queries, and video paths (JSON)
    ├── features/          # Per-video CLIP frame features (.pt)
    └── manifest.json      # Index and metadata for saved frame features
```

Now you can use these pre-calculated scores and features to run evaluations with `REDI` strategy.
