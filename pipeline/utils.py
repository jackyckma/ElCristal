from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from app.config import SAMPLE_RATE

logger = logging.getLogger(__name__)

AudioArray = np.ndarray  # shape (samples,) mono or (samples, channels)


def load_audio(path: Path, target_sr: int = SAMPLE_RATE) -> tuple[AudioArray, int]:
    """Load audio file, resample to target_sr, return (array, sample_rate).

    MP3 files are first converted to WAV via ffmpeg.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".mp3":
        tmp_wav = path.with_suffix(".tmp.wav")
        _ffmpeg_convert(path, tmp_wav)
        data, sr = sf.read(str(tmp_wav))
        tmp_wav.unlink(missing_ok=True)
    else:
        data, sr = sf.read(str(path))

    if sr != target_sr:
        data = _resample(data, sr, target_sr)
        sr = target_sr

    return data.astype(np.float32), sr


def save_audio(data: AudioArray, path: Path, sr: int = SAMPLE_RATE, fmt: str | None = None) -> None:
    """Write audio array to disk.  fmt overrides the path suffix (mp3/wav/flac)."""
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".")).lower()

    if fmt == "mp3":
        # soundfile cannot write MP3; write WAV first, then re-encode
        tmp_wav = path.with_suffix(".tmp.wav")
        sf.write(str(tmp_wav), data, sr, subtype="PCM_16")
        _ffmpeg_convert(tmp_wav, path)
        tmp_wav.unlink(missing_ok=True)
    elif fmt == "flac":
        sf.write(str(path.with_suffix(".flac")), data, sr, subtype="PCM_24")
    else:
        sf.write(str(path.with_suffix(".wav")), data, sr, subtype="PCM_16")


def _resample(data: AudioArray, orig_sr: int, target_sr: int) -> AudioArray:
    try:
        import resampy  # preferred: high-quality sinc resampler
        return resampy.resample(data, orig_sr, target_sr, axis=0)
    except ImportError:
        pass

    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    return resample_poly(data, up, down, axis=0).astype(np.float32)


def _ffmpeg_convert(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def stem_to_path(track_stem: str, stage: str, work_dir: Path, suffix: str = ".wav") -> Path:
    """Return a deterministic intermediate file path for a pipeline stage."""
    return work_dir / f"{track_stem}__{stage}{suffix}"


def detect_device(use_gpu: bool = False) -> str:
    """Return 'cuda' if GPU is requested and available, else 'cpu'."""
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("USE_GPU=true but CUDA is not available — falling back to CPU")
        except ImportError:
            logger.warning("torch not installed — cannot use GPU")
    return "cpu"
