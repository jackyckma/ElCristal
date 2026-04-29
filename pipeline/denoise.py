from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def denoise_stems(
    stem_paths: dict[str, Path],
    work_dir: Path,
    model_cache_dir: Path,
    device: str = "cpu",
    backend: str = "auto",
) -> dict[str, Path]:
    """Stage 2 — Denoise each stem with CleanUNet (Aero as fallback).

    Returns a mapping of stem name → denoised WAV file path.
    Model is loaded once, applied to all stems, then unloaded.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    if backend == "passthrough":
        logger.info("[denoise] backend=passthrough — returning stems unchanged")
        return stem_paths

    if backend == "cleanunet":
        denoised = _denoise_cleanunet(stem_paths, work_dir, model_cache_dir, device)
        method = "CleanUNet"
    elif backend == "aero":
        denoised = _denoise_aero(stem_paths, work_dir, model_cache_dir, device)
        method = "Aero"
    elif backend == "spectral":
        denoised = _denoise_spectral(stem_paths, work_dir)
        method = "spectral-subtraction"
    elif backend == "auto":
        try:
            denoised = _denoise_cleanunet(stem_paths, work_dir, model_cache_dir, device)
            method = "CleanUNet"
        except Exception as exc:
            logger.warning(
                "[denoise] CleanUNet unavailable/failed (%s) — falling back to Aero",
                exc,
            )
            try:
                denoised = _denoise_aero(stem_paths, work_dir, model_cache_dir, device)
                method = "Aero"
            except Exception as aero_exc:
                logger.warning(
                    "[denoise] Aero unavailable/failed (%s) — falling back to spectral subtraction",
                    aero_exc,
                )
                denoised = _denoise_spectral(stem_paths, work_dir)
                method = "spectral-subtraction"
    else:
        raise ValueError(f"Unsupported denoise backend: {backend}")

    elapsed = time.perf_counter() - t0
    logger.info("[denoise] Done (%s) in %.1fs", method, elapsed)
    return denoised


# ---------------------------------------------------------------------------
# CleanUNet backend
# ---------------------------------------------------------------------------

def _denoise_cleanunet(
    stem_paths: dict[str, Path],
    work_dir: Path,
    model_cache_dir: Path,
    device: str,
) -> dict[str, Path]:
    """Denoise using NVIDIA CleanUNet loaded from Hugging Face Hub.

    Requires the CleanUNet repo to be on PYTHONPATH (see Dockerfile.worker).
    Checkpoint is downloaded automatically via huggingface_hub on first run.
    """
    import torch  # noqa: PLC0415
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    # CleanUNet repo must be cloned and on PYTHONPATH
    from network import CleanUNet as CleanUNetModel  # noqa: PLC0415  # from nvidia/CleanUNet

    checkpoint_path = hf_hub_download(
        repo_id="nvidia/CleanUNet",
        filename="CleanUNet.pkl",
        cache_dir=str(model_cache_dir),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Architecture matching the published checkpoint
    model = CleanUNetModel(
        channels_input=1,
        channels_output=1,
        channels_H=64,
        max_H=768,
        encoder_n_layers=8,
        kernel_size=4,
        stride=2,
        tsfm_n_layers=3,
        tsfm_n_head=8,
        tsfm_d_model=512,
        tsfm_d_inner=2048,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    try:
        return _apply_model_to_stems(model, stem_paths, work_dir, device, tag="cleanunet")
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _apply_model_to_stems(
    model: "torch.nn.Module",
    stem_paths: dict[str, Path],
    work_dir: Path,
    device: str,
    tag: str,
) -> dict[str, Path]:
    """Run a waveform model over each stem in chunks and write output WAVs."""
    import torch  # noqa: PLC0415

    out: dict[str, Path] = {}
    chunk_samples = 44_100 * 10  # 10-second chunks

    with torch.no_grad():
        for stem_name, stem_path in stem_paths.items():
            data, sr = sf.read(str(stem_path))

            # Convert to mono float32 tensor (1, 1, T)
            if data.ndim == 2:
                data_mono = data.mean(axis=1).astype(np.float32)
            else:
                data_mono = data.astype(np.float32)

            n_samples = len(data_mono)
            denoised_chunks: list[np.ndarray] = []

            for start in range(0, n_samples, chunk_samples):
                chunk = data_mono[start : start + chunk_samples]
                tensor = torch.tensor(chunk, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                out_tensor = model(tensor).squeeze().cpu().numpy()
                denoised_chunks.append(out_tensor)

            denoised_mono = np.concatenate(denoised_chunks)

            # Restore stereo if original was stereo
            if data.ndim == 2:
                denoised_stereo = np.stack([denoised_mono, denoised_mono], axis=1)
                out_data = denoised_stereo
            else:
                out_data = denoised_mono

            out_path = work_dir / f"{stem_path.stem}__{tag}.wav"
            sf.write(str(out_path), out_data, sr, subtype="PCM_16")
            out[stem_name] = out_path
            logger.debug("[denoise] %s denoised → %s", stem_name, out_path.name)

    return out


# ---------------------------------------------------------------------------
# Aero fallback backend
# ---------------------------------------------------------------------------

def _denoise_aero(
    stem_paths: dict[str, Path],
    work_dir: Path,
    model_cache_dir: Path,
    device: str,
) -> dict[str, Path]:
    """Denoise using Aero (Meta). Requires aero repo on PYTHONPATH.

    See Dockerfile.worker for installation.
    """
    import subprocess  # noqa: PLC0415

    out: dict[str, Path] = {}
    for stem_name, stem_path in stem_paths.items():
        out_path = work_dir / f"{stem_path.stem}__aero.wav"
        subprocess.run(
            [
                "python", "-m", "aero",
                "--in_file", str(stem_path),
                "--out_dir", str(work_dir),
            ],
            check=True,
        )
        # Aero writes to the same directory; rename to expected path
        aero_out = work_dir / stem_path.name
        if aero_out.exists() and aero_out != out_path:
            aero_out.rename(out_path)
        out[stem_name] = out_path
    return out


# ---------------------------------------------------------------------------
# Spectral subtraction fallback (no ML, CPU-only, always available)
# ---------------------------------------------------------------------------

def _denoise_spectral(
    stem_paths: dict[str, Path],
    work_dir: Path,
) -> dict[str, Path]:
    """Lightweight spectral subtraction denoising — used only when ML models are unavailable."""
    from scipy.signal import stft, istft  # noqa: PLC0415

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for stem_name, stem_path in stem_paths.items():
        data, sr = sf.read(str(stem_path))

        if data.ndim == 2:
            channels = [data[:, c] for c in range(data.shape[1])]
        else:
            channels = [data]

        denoised_channels = []
        for ch in channels:
            freqs, times, Z = stft(ch.astype(np.float32), fs=sr, nperseg=1024)
            # Estimate noise floor from first 0.5 s (assumed silence / tail-out)
            n_noise_frames = max(1, int(0.5 * sr / 512))
            noise_profile = np.mean(np.abs(Z[:, :n_noise_frames]), axis=1, keepdims=True)
            gain = np.maximum(0.0, 1.0 - 2.0 * noise_profile / (np.abs(Z) + 1e-8))
            _, denoised = istft(gain * Z, fs=sr, nperseg=1024)
            denoised_channels.append(denoised[: len(ch)].astype(np.float32))

        if len(denoised_channels) > 1:
            out_data = np.stack(denoised_channels, axis=1)
        else:
            out_data = denoised_channels[0]

        out_path = work_dir / f"{stem_path.stem}__spectral.wav"
        sf.write(str(out_path), out_data, sr, subtype="PCM_16")
        out[stem_name] = out_path
        logger.debug("[denoise] %s spectral-denoised → %s", stem_name, out_path.name)

    return out
