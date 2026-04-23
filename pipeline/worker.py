from __future__ import annotations

import gc
import logging
import shutil
import time
import uuid
from pathlib import Path

from app import config, notify
from pipeline import bandwidth, denoise, normalize, separate
from pipeline.utils import detect_device

logger = logging.getLogger(__name__)


def process_track(
    job_id: str,
    input_path: str,
    recipient_email: str,
    output_format: str,
) -> dict:
    """RQ job entry point — runs the full four-stage restoration pipeline.

    Memory pattern per stage: load → process → unload → gc.collect()
    """
    src = Path(input_path)
    work_dir = config.OUTPUT_DIR / job_id / "work"
    out_dir = config.OUTPUT_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    device = detect_device(config.USE_GPU)
    t_job_start = time.perf_counter()

    logger.info("=== Job %s started | file=%s | device=%s ===", job_id, src.name, device)

    try:
        # ------------------------------------------------------------------
        # Stage 1 — Source separation
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        stem_paths = separate.separate(src, work_dir, device=device)
        gc.collect()
        logger.info("[job %s] Stage 1 (separate) completed in %.1fs", job_id, time.perf_counter() - t0)

        # ------------------------------------------------------------------
        # Stage 2 — Denoising
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        denoised_stems = denoise.denoise_stems(
            stem_paths,
            work_dir / "denoised",
            model_cache_dir=config.MODEL_CACHE_DIR,
            device=device,
        )
        gc.collect()
        logger.info("[job %s] Stage 2 (denoise) completed in %.1fs", job_id, time.perf_counter() - t0)

        # ------------------------------------------------------------------
        # Stage 3 — Bandwidth extension
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        extended_stems = bandwidth.extend_bandwidth(
            denoised_stems,
            work_dir / "bwe",
            model_cache_dir=config.MODEL_CACHE_DIR,
            device=device,
            enabled=True,
        )
        gc.collect()
        logger.info("[job %s] Stage 3 (bandwidth) completed in %.1fs", job_id, time.perf_counter() - t0)

        # ------------------------------------------------------------------
        # Remix stems → mix
        # ------------------------------------------------------------------
        mix_path = work_dir / f"{src.stem}__mix.wav"
        separate.remix_stems(extended_stems, mix_path)
        gc.collect()

        # ------------------------------------------------------------------
        # Stage 4 — Loudness normalisation + export
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        fmt = output_format or src.suffix.lstrip(".")
        out_filename = f"{src.stem}__restored.{fmt}"
        out_path = out_dir / out_filename

        final_path = normalize.normalize_and_export(mix_path, out_path, output_format=fmt)
        gc.collect()
        logger.info("[job %s] Stage 4 (normalize) completed in %.1fs", job_id, time.perf_counter() - t0)

        # ------------------------------------------------------------------
        # Cleanup work directory
        # ------------------------------------------------------------------
        shutil.rmtree(work_dir, ignore_errors=True)

        total_elapsed = time.perf_counter() - t_job_start
        logger.info("=== Job %s completed in %.1fs | output=%s ===", job_id, total_elapsed, final_path.name)

        # ------------------------------------------------------------------
        # Email notification
        # ------------------------------------------------------------------
        download_url = f"{config.BASE_URL}/download/{job_id}/{final_path.name}"
        notify.send_completion_email(
            recipient=recipient_email,
            job_id=job_id,
            file_names=[final_path.name],
            download_urls=[download_url],
        )

        return {
            "job_id": job_id,
            "status": "complete",
            "output": str(final_path),
            "elapsed_seconds": round(total_elapsed, 1),
        }

    except Exception:
        logger.exception("Job %s failed", job_id)
        # Attempt partial cleanup
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


def run_worker() -> None:
    """Start the RQ worker — called by the Docker worker container."""
    import redis as redis_lib  # noqa: PLC0415
    from rq import Connection, Worker  # noqa: PLC0415

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    logger.info("Starting ElCristal worker (concurrency=%d, device=%s)",
                config.WORKER_CONCURRENCY,
                detect_device(config.USE_GPU))

    config.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    conn = redis_lib.from_url(config.REDIS_URL)
    with Connection(conn):
        worker = Worker(["elcristal"])
        worker.work()


if __name__ == "__main__":
    run_worker()
