"""Which ONNX Runtime execution providers the models run on.

The stem and effects separators are the heaviest step of a decomposition. On CPU they keep every core busy for minutes; a GPU
build of ONNX Runtime (onnxruntime-directml on Windows, onnxruntime-gpu with CUDA) moves that work off the CPU.

Order: the providers named in REA_VIDEO_ORT_PROVIDERS (comma-separated, e.g. "DmlExecutionProvider,CPUExecutionProvider"),
otherwise every GPU provider the installed build offers, always followed by CPUExecutionProvider as the fallback. Setting
REA_VIDEO_ORT_PROVIDERS=CPUExecutionProvider forces the CPU.
"""

from __future__ import annotations

import os

GPU_PREFERENCE = ("CUDAExecutionProvider", "DmlExecutionProvider")
CPU = "CPUExecutionProvider"
ENV = "REA_VIDEO_ORT_PROVIDERS"


def pick(available: list[str], requested: str | None = None) -> list[str]:
    """Pure: the providers to pass to InferenceSession, given what the build offers and the env value."""
    if requested:
        wanted = [p.strip() for p in requested.split(",") if p.strip()]
        chosen = [p for p in wanted if p in available]
    else:
        chosen = [p for p in GPU_PREFERENCE if p in available]
    if CPU not in chosen:
        chosen.append(CPU)
    return chosen


def providers() -> list[str]:
    import onnxruntime as ort

    return pick(ort.get_available_providers(), os.environ.get(ENV))
