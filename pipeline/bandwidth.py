from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def extend_bandwidth(
    stem_paths: dict[str, Path],
    work_dir: Path,
    model_cache_dir: Path,
    device: str = "cpu",
    backend: str = "auto",
) -> dict[str, Path]:
    """Stage 3 — Bandwidth extension via AudioSR.

    Reconstructs high-frequency content above the original recording rolloff
    (~8 kHz on 78rpm sources). If AudioSR is unavailable or enabled=False,
    the input stems are returned as-is.
    """
    if backend in {"passthrough", "disabled", "none"}:
        logger.info("[bandwidth] Stage disabled — passing stems through unchanged")
        return stem_paths

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    if backend == "audiosr":
        result = _extend_audiosr(stem_paths, work_dir, model_cache_dir, device)
    elif backend == "auto":
        try:
            result = _extend_audiosr(stem_paths, work_dir, model_cache_dir, device)
        except (ImportError, ModuleNotFoundError):
            logger.warning("[bandwidth] AudioSR not installed — skipping bandwidth extension")
            return stem_paths
    else:
        raise ValueError(f"Unsupported bandwidth backend: {backend}")

    elapsed = time.perf_counter() - t0
    logger.info("[bandwidth] Done in %.1fs", elapsed)
    return result


def _extend_audiosr(
    stem_paths: dict[str, Path],
    work_dir: Path,
    model_cache_dir: Path,
    device: str,
) -> dict[str, Path]:
    """Run AudioSR super-resolution on each stem."""
    import torch  # noqa: PLC0415
    from audiosr import build_model, super_resolution, save_wave  # noqa: PLC0415

    logger.info("[bandwidth] Loading AudioSR on %s", device)
    model = build_model(model_name="basic", device=device)

    try:
        out: dict[str, Path] = {}
        for stem_name, stem_path in stem_paths.items():
            out_path = work_dir / f"{stem_path.stem}__bwe.wav"

            waveform = super_resolution(
                model,
                str(stem_path),
                seed=42,
                guidance_scale=3.5,
                ddim_steps=50,
                latent_t_per_second=12.8,
            )
            save_wave(waveform, str(out_path), target_sr=44_100)
            out[stem_name] = out_path
            logger.debug("[bandwidth] %s extended → %s", stem_name, out_path.name)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out
