from __future__ import annotations

import os
from pathlib import Path


SMTP_HOST: str = os.getenv("SMTP_HOST", "")
SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER: str = os.getenv("SMTP_USER", "")
SMTP_PASS: str = os.getenv("SMTP_PASS", "")

MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "50"))
OUTPUT_TTL_HOURS: int = int(os.getenv("OUTPUT_TTL_HOURS", "72"))
WORKER_CONCURRENCY: int = int(os.getenv("WORKER_CONCURRENCY", "1"))
USE_GPU: bool = os.getenv("USE_GPU", "false").lower() == "true"
MODEL_CACHE_DIR: Path = Path(os.getenv("MODEL_CACHE_DIR", "/models"))

REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379")

STORAGE_DIR: Path = Path(os.getenv("STORAGE_DIR", "/tmp/elcristal"))
INPUT_DIR: Path = STORAGE_DIR / "inputs"
OUTPUT_DIR: Path = STORAGE_DIR / "outputs"

BASE_URL: str = os.getenv("BASE_URL", "http://localhost:7860")

ACCEPTED_FORMATS: frozenset[str] = frozenset({".mp3", ".wav", ".flac"})
SAMPLE_RATE: int = 44_100
EBU_R128_TARGET_LUFS: float = -23.0
