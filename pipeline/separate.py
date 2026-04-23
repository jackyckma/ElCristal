from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import soundfile as sf
import torch

logger = logging.getLogger(__name__)

DEMUCS_MODEL = "htdemucs"
STEMS = ("vocals", "bass", "drums", "other")


def separate(
    input_path: Path,
    work_dir: Path,
    device: str = "cpu",
) -> dict[str, Path]:
    """Stage 1 — Source separation via Demucs htdemucs.

    Returns a mapping of stem name → WAV file path.
    Model is loaded, used, then immediately unloaded to free RAM.
    """
    from demucs.api import Separator, save_audio  # noqa: PLC0415

    input_path = Path(input_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    logger.info("[separate] Loading Demucs %s on %s for: %s", DEMUCS_MODEL, device, input_path.name)

    separator = Separator(model=DEMUCS_MODEL, device=device)
    try:
        origin, separated = separator.separate_audio_file(input_path)

        stem_paths: dict[str, Path] = {}
        for stem_name, tensor in separated.items():
            out_path = work_dir / f"{input_path.stem}__{stem_name}.wav"
            save_audio(tensor, out_path, samplerate=separator.samplerate)
            stem_paths[stem_name] = out_path
            logger.debug("[separate] Saved stem %s → %s", stem_name, out_path.name)

    finally:
        del separator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    logger.info("[separate] Done in %.1fs — stems: %s", elapsed, list(stem_paths.keys()))
    return stem_paths


def remix_stems(stem_paths: dict[str, Path], output_path: Path) -> Path:
    """Sum all stems back into a stereo mix after per-stem processing."""
    import numpy as np  # noqa: PLC0415

    mix: np.ndarray | None = None
    sr: int = 44_100

    for stem_path in stem_paths.values():
        data, sr = sf.read(str(stem_path))
        mix = data if mix is None else mix + data

    if mix is None:
        raise ValueError("No stems to remix")

    # Prevent clipping after summation
    peak = np.abs(mix).max()
    if peak > 0.98:
        mix = mix * (0.98 / peak)

    sf.write(str(output_path), mix, sr, subtype="PCM_16")
    logger.info("[separate] Remixed %d stems → %s", len(stem_paths), output_path.name)
    return output_path
