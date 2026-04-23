from __future__ import annotations

import logging
import uuid
from pathlib import Path

from redis import Redis
from rq import Queue

from app import config

logger = logging.getLogger(__name__)

_redis: Redis | None = None
_queue: Queue | None = None


def _get_queue() -> Queue:
    global _redis, _queue
    if _queue is None:
        _redis = Redis.from_url(config.REDIS_URL)
        _queue = Queue("elcristal", connection=_redis)
    return _queue


def queue_job(
    input_path: Path,
    recipient_email: str,
    output_format: str,
) -> str:
    """Enqueue a single restoration job and return its job ID."""
    job_id = str(uuid.uuid4())
    q = _get_queue()

    # Import here to avoid circular deps; worker module is only needed in the worker container
    from pipeline.worker import process_track  # noqa: PLC0415

    job = q.enqueue(
        process_track,
        kwargs={
            "job_id": job_id,
            "input_path": str(input_path),
            "recipient_email": recipient_email,
            "output_format": output_format,
        },
        job_id=job_id,
        job_timeout=3600,  # 1 hour max per track
    )
    logger.info("Queued job %s for %s (format=%s)", job.id, recipient_email, output_format)
    return job.id


def get_queue_length() -> int:
    """Return the number of jobs currently waiting in the queue."""
    try:
        return len(_get_queue())
    except Exception:
        return -1


def get_job_position(job_id: str) -> int | None:
    """Return 1-based queue position of a job, or None if not found / already running."""
    try:
        q = _get_queue()
        job_ids = q.job_ids
        if job_id in job_ids:
            return job_ids.index(job_id) + 1
        return None
    except Exception:
        return None
