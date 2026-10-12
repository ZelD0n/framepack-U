#!/usr/bin/env bash
# Source from an evaluation launcher; resource paths are anchored to this repository.
FRAMEPACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FRAMEPACK_ROOT
export LMUData="${LMUData:-$FRAMEPACK_ROOT/local/cache}"
export CLIP_SCORES="${CLIP_SCORES:-$FRAMEPACK_ROOT/local/clip_scores}"
export FRAME_FEATURES="${FRAME_FEATURES:-$FRAMEPACK_ROOT/local/frame_feature}"
mkdir -p "$LMUData"
