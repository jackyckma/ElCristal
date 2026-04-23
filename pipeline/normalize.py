from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from app.config import EBU_R128_TARGET_LUFS

logger = logging.getLogger(__name__)


def normalize_and_export(
    input_path: Path,
    output_path: Path,
    output_format: str = "wav",
) -> Path:
    """Stage 4 — EBU R128 loudness normalisation followed by format export.

    Parameters
    ----------
    input_path:    Remixed WAV from the previous stage.
    output_path:   Destination path (suffix determines format if output_format is empty).
    output_format: One of 'wav', 'flac', 'mp3', or '' (use output_path suffix).
    """
    t0 = time.perf_counter()

    output_format = (output_format or output_path.suffix.lstrip(".")).lower()
    if output_format not in {"wav", "flac", "mp3"}:
        output_format = "wav"

    output_path = output_path.with_suffix(f".{output_format}")

    data, sr = sf.read(str(input_path))
    data = data.astype(np.float64)

    meter = pyln.Meter(sr)
    measured_lufs = meter.integrated_loudness(data)

    if np.isfinite(measured_lufs):
        normalized = pyln.normalize.loudness(data, measured_lufs, EBU_R128_TARGET_LUFS)
        # Hard-limit to prevent inter-sample clipping
        normalized = np.clip(normalized, -1.0, 1.0)
        logger.info(
            "[normalize] %s: %.1f LUFS → %.1f LUFS",
            input_path.name,
            measured_lufs,
            EBU_R128_TARGET_LUFS,
        )
    else:
        logger.warning("[normalize] Could not measure loudness for %s — skipping normalisation", input_path.name)
        normalized = data

    _write_output(normalized.astype(np.float32), output_path, sr, output_format)

    elapsed = time.perf_counter() - t0
    logger.info("[normalize] Done in %.1fs → %s", elapsed, output_path.name)
    return output_path


def _write_output(data: np.ndarray, path: Path, sr: int, fmt: str) -> None:
    if fmt == "mp3":
        tmp_wav = path.with_suffix(".tmp_norm.wav")
        sf.write(str(tmp_wav), data, sr, subtype="PCM_16")
        _ffmpeg_encode_mp3(tmp_wav, path)
        tmp_wav.unlink(missing_ok=True)
    elif fmt == "flac":
        sf.write(str(path), data, sr, subtype="PCM_24")
    else:
        sf.write(str(path), data, sr, subtype="PCM_16")


def _ffmpeg_encode_mp3(src: Path, dst: Path, bitrate: str = "320k") -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-codec:a", "libmp3lame",
            "-b:a", bitrate,
            str(dst),
        ],
        check=True,
    )
