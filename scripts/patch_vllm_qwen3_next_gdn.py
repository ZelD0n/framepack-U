#!/usr/bin/env python3
"""Patch vLLM Qwen3Next GDN backend selection for FramePack evals.

vLLM 0.17.1 forces FlashInfer's GDN prefill kernel on H100. In this
environment FlashInfer 0.6.4 may JIT-compile gdn_prefill with nvcc and fail
before the engine starts. This patch adds an environment switch:

    VLLM_QWEN3_NEXT_GDN_BACKEND=native

When set to native/fla/0/false/no/off, Qwen3Next uses vLLM's native FLA
implementation instead of FlashInfer.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MARKER = "VLLM_QWEN3_NEXT_GDN_BACKEND"

OLD_BLOCK = """        if current_platform.is_cuda() and current_platform.is_device_capability(90):
            logger.info_once(
                "Using FlashInfer GDN prefill kernel on CUDA compute capability 90"
            )
            self._forward_method = self.forward_cuda
        else:
            self._forward_method = self.forward_native
"""

NEW_BLOCK = """        gdn_backend = os.environ.get("VLLM_QWEN3_NEXT_GDN_BACKEND", "").lower()
        use_flashinfer_gdn = gdn_backend not in (
            "native", "fla", "0", "false", "no", "off"
        )
        if (
            use_flashinfer_gdn
            and current_platform.is_cuda()
            and current_platform.is_device_capability(90)
        ):
            logger.info_once(
                "Using FlashInfer GDN prefill kernel on CUDA compute capability 90"
            )
            self._forward_method = self.forward_cuda
        else:
            if not use_flashinfer_gdn:
                logger.info_once(
                    "Using native FLA GDN prefill kernel because "
                    "VLLM_QWEN3_NEXT_GDN_BACKEND=%s",
                    gdn_backend,
                )
            self._forward_method = self.forward_native
"""


def find_qwen3_next() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Could not find the vLLM package in the active Python environment.")

    for location in spec.submodule_search_locations:
        candidate = Path(location) / "model_executor" / "models" / "qwen3_next.py"
        if candidate.is_file():
            return candidate

    raise RuntimeError("Could not find vllm/model_executor/models/qwen3_next.py.")


def patch_file(path: Path) -> str:
    text = path.read_text()
    if MARKER in text and "use_flashinfer_gdn" in text:
        return f"already patched: {path}"

    if "import os\n" not in text:
        anchor = "from itertools import islice\n\n"
        if anchor not in text:
            raise RuntimeError("Could not locate import insertion point in qwen3_next.py.")
        text = text.replace(anchor, anchor + "import os\n", 1)

    if OLD_BLOCK not in text:
        raise RuntimeError(
            "Could not locate the original Qwen3Next GDN backend block. "
            "The installed vLLM may have changed."
        )

    text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    path.write_text(text)
    return f"patched: {path}"


def main() -> int:
    try:
        path = find_qwen3_next()
        print(f"[eval_vllm] qwen3_next GDN backend patch: {patch_file(path)}")
        return 0
    except Exception as exc:
        print(f"[eval_vllm] ERROR: failed to patch vLLM Qwen3Next GDN backend: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
