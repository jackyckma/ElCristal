"""End-to-end pipeline tests.

Each stage is tested in isolation using the 5-second fixture WAV.
The full pipeline test chains all stages together.

Run with:  pytest tests/test_pipeline.py -v
"""
from __future__ import annotations

import importlib.util
import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

FIXTURE_WAV = Path(__file__).parent / "fixtures" / "test_5s_mono_44100.wav"
SR = 44_100


# ---------------------------------------------------------------------------
# Utilities (must be defined before use in skipif decorators)
# ---------------------------------------------------------------------------

def _which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _make_fixture(tmp_path: Path) -> Path:
    """Copy (or regenerate) the fixture WAV into a temp directory."""
    if FIXTURE_WAV.exists():
        dest = tmp_path / FIXTURE_WAV.name
        shutil.copy2(FIXTURE_WAV, dest)
        return dest
    # Fallback: regenerate inline without numpy
    n = SR * 5
    raw = bytearray()
    for i in range(n):
        t = i / SR
        s = int(0.3 * math.sin(2 * math.pi * 440 * t) * 32767)
        raw += struct.pack("<h", max(-32768, min(32767, s)))
    out = tmp_path / "test_5s_mono_44100.wav"
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(raw)
    return out


# ---------------------------------------------------------------------------
# Stage 4 — Normalisation (no ML dependencies; always runs)
# ---------------------------------------------------------------------------

def test_normalize_wav(tmp_path: Path) -> None:
    """Normalise produces a WAV with loudness close to -23 LUFS."""
    src = _make_fixture(tmp_path)

    import pyloudnorm as pyln
    import soundfile as sf
    from pipeline.normalize import normalize_and_export

    out = normalize_and_export(src, tmp_path / "normalized.wav", output_format="wav")

    assert out.exists(), "Output file not created"
    data, sr = sf.read(str(out))
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(data.astype("float64"))
    assert math.isfinite(lufs), "LUFS measurement returned non-finite value"
    assert abs(lufs - (-23.0)) < 2.0, f"LUFS {lufs:.1f} not within 2 LU of -23"


def test_normalize_flac(tmp_path: Path) -> None:
    """Normalise can export to FLAC."""
    src = _make_fixture(tmp_path)

    from pipeline.normalize import normalize_and_export

    out = normalize_and_export(src, tmp_path / "normalized.flac", output_format="flac")
    assert out.exists()
    assert out.suffix == ".flac"


@pytest.mark.skipif(not _which("ffmpeg"), reason="ffmpeg not installed")
def test_normalize_mp3(tmp_path: Path) -> None:
    """Normalise can export to MP3 via ffmpeg."""
    src = _make_fixture(tmp_path)

    from pipeline.normalize import normalize_and_export

    out = normalize_and_export(src, tmp_path / "normalized.mp3", output_format="mp3")
    assert out.exists()
    assert out.suffix == ".mp3"


# ---------------------------------------------------------------------------
# Stage 3 — Bandwidth extension
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _module_available("audiosr"), reason="audiosr not installed")
def test_bandwidth_extension(tmp_path: Path) -> None:
    src = _make_fixture(tmp_path)
    stems = {"other": src}

    from pipeline.bandwidth import extend_bandwidth

    result = extend_bandwidth(
        stems, tmp_path / "bwe", model_cache_dir=tmp_path / "models", device="cpu"
    )
    assert "other" in result
    assert result["other"].exists()


def test_bandwidth_extension_disabled(tmp_path: Path) -> None:
    """When disabled, original paths are returned unchanged."""
    src = _make_fixture(tmp_path)
    stems = {"other": src}

    from pipeline.bandwidth import extend_bandwidth

    result = extend_bandwidth(stems, tmp_path, model_cache_dir=tmp_path, enabled=False)
    assert result == stems


# ---------------------------------------------------------------------------
# Stage 2 — Denoising (spectral fallback always available)
# ---------------------------------------------------------------------------

def test_denoise_spectral_fallback(tmp_path: Path) -> None:
    """Spectral-subtraction fallback runs without any ML models."""
    import soundfile as sf

    src = _make_fixture(tmp_path)
    stems = {"other": src}

    from pipeline.denoise import _denoise_spectral

    result = _denoise_spectral(stems, tmp_path / "denoised")
    assert "other" in result
    out_path = result["other"]
    assert out_path.exists()

    data, sr = sf.read(str(out_path))
    assert sr == SR
    assert len(data) > 0


# ---------------------------------------------------------------------------
# Stage 1 — Source separation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _module_available("demucs"), reason="demucs not installed")
def test_separate(tmp_path: Path) -> None:
    src = _make_fixture(tmp_path)

    from pipeline.separate import separate

    stems = separate(src, tmp_path / "stems", device="cpu")
    assert len(stems) >= 1
    for stem_path in stems.values():
        assert stem_path.exists()


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_module_available("demucs") and _module_available("pyloudnorm")),
    reason="demucs and/or pyloudnorm not installed",
)
def test_full_pipeline(tmp_path: Path) -> None:
    """Run all four stages end-to-end on the 5-second fixture."""
    import soundfile as sf

    src = _make_fixture(tmp_path)
    model_cache = tmp_path / "models"
    model_cache.mkdir()

    from pipeline import bandwidth, denoise, normalize, separate

    # Stage 1
    stem_paths = separate.separate(src, tmp_path / "stems", device="cpu")

    # Stage 2
    denoised = denoise.denoise_stems(
        stem_paths, tmp_path / "denoised", model_cache_dir=model_cache, device="cpu"
    )

    # Stage 3 (disabled so test runs without audiosr)
    extended = bandwidth.extend_bandwidth(
        denoised, tmp_path / "bwe", model_cache_dir=model_cache, enabled=False
    )

    # Remix
    mix_path = tmp_path / "mix.wav"
    separate.remix_stems(extended, mix_path)

    # Stage 4
    out = normalize.normalize_and_export(mix_path, tmp_path / "out.wav", output_format="wav")

    assert out.exists()
    data, sr = sf.read(str(out))
    assert sr == SR
    assert len(data) > 0
