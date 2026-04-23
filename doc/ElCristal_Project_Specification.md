# ElCristal — Tango Audio Restoration Platform
> Project Specification · Version 1.0 · Inference Pipeline (v1 scope)

| Field | Value |
|---|---|
| **Status** | Draft — Ready for Development |
| **Scope** | v1 Inference Pipeline |
| **Stack** | Python · Docker · Gradio |
| **Access** | Open to everyone (free) |
| **Batch** | Yes — single file + folder upload |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Restoration Pipeline — Technical Details](#3-restoration-pipeline--technical-details)
4. [Web User Interface](#4-web-user-interface)
5. [Deployment](#5-deployment)
6. [Repository Structure](#6-repository-structure)
7. [Python Dependencies](#7-python-dependencies)
8. [Development Roadmap](#8-development-roadmap)
9. [Open Questions for v1](#9-open-questions-for-v1)
10. [Notes for Claude Code](#10-notes-for-claude-code)

---

## 1. Project Overview

ElCristal is a free, open-source audio restoration tool built specifically for the Argentine tango community. It uses modern deep learning models to remove noise, hiss, crackle and bandwidth limitations from golden-age tango recordings (1930s–1950s), making them cleaner and more enjoyable to listen to and dance to — without altering the musical character of the original.

The project is designed for hobbyist community members who are not technically skilled. They interact only with a simple web interface — upload a file, receive a cleaned version. All processing happens server-side, invisibly.

> **Design principle:** Restore clarity. Never alter musical identity. The bandoneon must still sound like a bandoneon.

### 1.1 Goals

- Provide free, accessible audio restoration to the global tango community
- Support batch processing so entire orchestra discographies can be processed overnight
- Run on modest server hardware (16GB RAM, 4 vCPU) with optional GPU acceleration
- Be fully open-source and self-hostable by any community
- Produce results that are clearly better than the original without introducing ML artifacts

### 1.2 Non-Goals (v1)

- Fine-tuning or model training — inference only in v1
- Real-time processing or streaming
- Audio editing or manipulation beyond restoration
- Mobile app or desktop application

---

## 2. System Architecture

ElCristal follows a simple three-layer architecture: a web frontend, an async job queue, and a sequential processing pipeline. The pipeline runs all stages in order, frees memory between stages, and delivers results via email.

| Layer | Responsibility | Technology |
|---|---|---|
| **Frontend** | File upload (single + folder), job status display, email collection | Gradio |
| **Job Queue** | Accepts jobs, sequences them, manages memory between stages | Redis + RQ |
| **Pipeline Worker** | Runs restoration stages sequentially per track | Python |
| **Stage 1: Separation** | Isolates stems for cleaner per-stem processing | Demucs |
| **Stage 2: Denoising** | Removes hiss, crackle, broadband noise | CleanUNet / Aero |
| **Stage 3: Bandwidth Ext.** | Reconstructs missing high-frequency content | AudioSR |
| **Stage 4: Normalisation** | EBU R128 loudness normalisation | pyloudnorm + FFmpeg |
| **Notification** | Sends download link to user email on job completion | SMTP / SendGrid |
| **Storage** | Temporary file storage for inputs and outputs | Local filesystem |

### 2.1 Processing Pipeline (per track)

Each track goes through the following stages in sequence. Memory is explicitly freed between stages to stay within the 16GB RAM constraint.

**Stage 1 — Source Separation:**
Demucs splits the audio into stems (vocals, bass, drums, other). The "other" stem captures bandoneon, strings and piano. Cleaning per stem then remixing consistently produces better results than cleaning the full mix.

**Stage 2 — Denoising:**
CleanUNet (primary) or Aero (fallback) removes broadband tape hiss, vinyl surface noise, and harmonic distortion. These are the dominant degradation types in 78rpm golden-age tango recordings.

**Stage 3 — Bandwidth Extension:**
AudioSR reconstructs high-frequency content above the original recording's rolloff (typically ~8kHz). This restores perceived presence and air to the audio.

**Stage 4 — Loudness Normalisation:**
pyloudnorm applies EBU R128 loudness normalisation. FFmpeg handles final format conversion and output encoding.

### 2.2 Job Queue Design

Jobs are processed sequentially (not in parallel) to avoid OOM errors on CPU-only deployments. The queue is backed by Redis with RQ workers. A single worker process is the default configuration, with the option to add GPU-backed workers later.

1. User submits files via Gradio UI
2. Each file creates one job in the Redis queue
3. Worker picks up jobs one at a time
4. On completion, worker sends email with download link
5. Output files are retained for 72 hours then automatically deleted

---

## 3. Restoration Pipeline — Technical Details

### 3.1 Model Selection

| Model | Stage | License | Why chosen |
|---|---|---|---|
| **Demucs v4 (htdemucs)** | Source separation | MIT | Best open-source music separation, strong community support, batch CLI built-in |
| **CleanUNet** | Denoising | MIT | Music-aware architecture, NVIDIA research quality, fine-tune friendly for future v2 |
| **Aero (Meta)** | Denoising fallback | CC BY-NC | Designed specifically for music enhancement, good on orchestral textures |
| **AudioSR** | Bandwidth extension | Apache 2.0 | General audio super-resolution, open weights available |
| **pyloudnorm** | Normalisation | MIT | Correct EBU R128 implementation in Python, lightweight |

### 3.2 Input / Output Formats

- Accepted input formats: **MP3, WAV, FLAC**
- Output format: matches input format by default; user may optionally select output format
- Maximum file size per upload: 100MB (configurable via `MAX_FILE_SIZE_MB`)
- Maximum files per batch: 50 (configurable via `MAX_BATCH_SIZE`)
- Sample rate: all audio resampled to 44.1kHz internally, output at original or 44.1kHz

### 3.3 Memory Management Strategy

On a 16GB RAM / CPU-only server, running all models simultaneously is not feasible. The pipeline uses an explicit **load → process → unload** pattern:

```python
for track in batch:
    load_demucs()  → separate()          → unload_demucs()    → gc.collect()
    load_denoiser() → denoise_stems()    → unload_denoiser()  → gc.collect()
    load_audiosr()  → extend_bandwidth() → unload_audiosr()   → gc.collect()
    remix_stems()   → normalize()        → export()
```

> **Note:** Peak RAM usage per track is estimated at ~8–10GB (Demucs htdemucs_ft is the heaviest at ~5GB). Do not run concurrent workers on CPU-only deployments.

---

## 4. Web User Interface

### 4.1 Framework

Gradio is the chosen framework. It requires no frontend development skills to maintain, produces a clean usable interface, and can be wrapped in a Docker container trivially. The UI is intentionally minimal — the community members are hobbyists, not engineers.

### 4.2 UI Components

| Component | Description |
|---|---|
| **Upload area** | Drag-and-drop or click to select. Accepts single files or a folder (batch). |
| **Email field** | Required. Used to notify the user when their job is complete. |
| **Output format selector** | Optional dropdown: MP3 / WAV / FLAC / Same as input. |
| **Submit button** | Queues all uploaded files as individual jobs. |
| **Status area** | Shows queue position and estimated wait time after submission. |
| **About section** | Brief explanation of what ElCristal does, link to GitHub. |

### 4.3 User Flow

1. User opens ElCristal web URL
2. User drags one or more audio files onto the upload area
3. User enters their email address
4. User clicks Submit — jobs enter the queue
5. UI shows estimated wait time
6. User receives email with download link when all files are ready
7. Download link expires after 72 hours

### 4.4 Access Control

v1 is open to everyone — no login, no registration required. Only an email address is collected for job notification. This aligns with the community hobbyist use case and avoids friction.

> **Future consideration:** If abuse becomes an issue (large batch spam), add a simple rate limit by IP or a CAPTCHA before submission. Do not add login in v1.

---

## 5. Deployment

### 5.1 Docker Architecture

ElCristal is packaged as a multi-container Docker Compose application with three services:

| Service | Role | Port |
|---|---|---|
| `elcristal-web` | Gradio frontend | 7860 |
| `elcristal-worker` | Python RQ worker, runs the pipeline | — |
| `elcristal-redis` | Redis queue backend | 6379 (internal) |

All three services share a Docker volume for temporary file storage (inputs, outputs, model weights cache).

### 5.2 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SMTP_HOST` | — | SMTP server hostname for email notifications |
| `SMTP_PORT` | `587` | SMTP port |
| `SMTP_USER` | — | SMTP username / sender address |
| `SMTP_PASS` | — | SMTP password (use Docker secret in production) |
| `MAX_FILE_SIZE_MB` | `100` | Maximum upload size per file in MB |
| `MAX_BATCH_SIZE` | `50` | Maximum files per submission |
| `OUTPUT_TTL_HOURS` | `72` | Hours before output files are auto-deleted |
| `WORKER_CONCURRENCY` | `1` | Number of concurrent pipeline workers |
| `USE_GPU` | `false` | Set to `true` to enable CUDA acceleration |
| `MODEL_CACHE_DIR` | `/models` | Path to cache downloaded model weights |

### 5.3 Hardware Requirements

| Resource | Minimum (CPU) | Recommended (GPU) |
|---|---|---|
| RAM | 16GB | 16GB |
| CPU | 4 vCPU | 4 vCPU |
| GPU | None | T4 16GB VRAM or better |
| Disk | 40GB (models + temp) | 40GB |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| Docker | 24.x+ | 24.x+ with nvidia-container-toolkit |

### 5.4 Performance Estimates

| Hardware | Time per 3-min track | 50-track batch |
|---|---|---|
| 4 vCPU, no GPU | ~20 minutes | ~17 hours |
| 4 vCPU + T4 GPU | ~1–2 minutes | ~1 hour |
| 4 vCPU + A10G | < 30 seconds | ~25 minutes |

> **Recommendation:** Start with CPU-only. If community demand grows, attach a spot/preemptible GPU instance (GCP T4 ~$0.20/hr) that auto-starts when the queue is non-empty and stops when idle.

---

## 6. Repository Structure

```
elcristal/
├── docker-compose.yml
├── Dockerfile.web
├── Dockerfile.worker
├── .env.example
├── README.md
│
├── app/
│   ├── main.py              # Gradio UI entrypoint
│   ├── queue.py             # RQ job submission
│   ├── notify.py            # Email notification (SMTP)
│   └── config.py            # Environment variable loading
│
├── pipeline/
│   ├── worker.py            # RQ worker + job orchestration
│   ├── separate.py          # Stage 1: Demucs source separation
│   ├── denoise.py           # Stage 2: CleanUNet / Aero denoising
│   ├── bandwidth.py         # Stage 3: AudioSR bandwidth extension
│   ├── normalize.py         # Stage 4: pyloudnorm + FFmpeg export
│   └── utils.py             # Shared audio I/O, format conversion
│
├── models/                  # Downloaded weights cache (gitignored)
│
└── tests/
    ├── test_pipeline.py     # End-to-end pipeline test
    └── fixtures/            # Short test audio clips
```

---

## 7. Python Dependencies

| Package | Purpose | Install |
|---|---|---|
| `torch` / `torchaudio` | Core ML framework | `pip install torch torchaudio` |
| `demucs` | Source separation | `pip install demucs` |
| `gradio` | Web UI | `pip install gradio` |
| `redis` / `rq` | Job queue | `pip install redis rq` |
| `pyloudnorm` | Loudness normalisation | `pip install pyloudnorm` |
| `ffmpeg-python` | Audio format I/O | `pip install ffmpeg-python` |
| `soundfile` | WAV / FLAC I/O | `pip install soundfile` |
| `pydub` | MP3 handling | `pip install pydub` |
| `audiosr` | Bandwidth extension | `pip install audiosr` |
| `numpy` / `scipy` | Signal processing utilities | `pip install numpy scipy` |

> **Note:** CleanUNet and Aero do not have PyPI packages. Clone from GitHub and install locally. Model weights download automatically on first run via Hugging Face Hub.

---

## 8. Development Roadmap

| Phase | Scope | Effort |
|---|---|---|
| **v1.0 — Pipeline** | Demucs + CleanUNet + AudioSR + pyloudnorm, sequential batch, CPU-only | 1–2 weekends |
| **v1.1 — Web UI** | Gradio frontend, Redis queue, email notification, Docker Compose | 1 weekend |
| **v1.2 — Hardening** | Error handling, file cleanup, logging, rate limiting, `.env` config | 1 weekend |
| **v1.3 — GPU support** | CUDA device detection, docker-compose GPU profile, benchmark comparison | 1 weekend |
| **v2.0 — Fine-tuning** | Synthetic data generation, CleanUNet fine-tune on tango noise profiles | 4–6 weekends |

---

## 9. Open Questions for v1

- **Model artifact evaluation:** Test CleanUNet vs Aero on 5–10 representative tango tracks before committing to primary model. Key test: does the bandoneon timbre survive denoising?
- **AudioSR bandwidth extension:** Evaluate whether it adds genuine value or introduces artifacts on mono 78rpm sources. May be optional / disabled by default in v1.
- **Email provider:** SendGrid free tier (100 emails/day) is sufficient for community use. Plain SMTP relay is the fallback.
- **Output file hosting:** Local filesystem is simplest for v1. Consider S3-compatible object storage (Cloudflare R2 free tier) if disk space becomes a concern.
- **Stem remixing balance:** After cleaning individual stems, the remix levels need validation. Demucs separation is not perfect and some bleed may affect remix quality.

---

## 10. Notes for Claude Code

> **Context:** The developer is an experienced data scientist and AI specialist comfortable with Python, ML pipelines, Docker, and fine-tuning. No hand-holding needed on technical fundamentals.

### Implementation priorities

- **Memory safety first** — always unload models and call `gc.collect()` between pipeline stages
- Use **Python type hints** throughout
- Write the pipeline so each stage (`separate` / `denoise` / `bandwidth` / `normalize`) is **independently testable**
- Gradio UI should be **minimal and clean** — this is for community hobbyists
- Docker Compose should work out-of-the-box with `docker compose up` — no manual setup steps
- All configuration via `.env` file — no hardcoded paths or credentials
- Include a short test fixture (5-second mono WAV) and a test that runs the full pipeline on it
- **Log processing time per stage per track** for future benchmarking

### Fine-tuning (v2 — not in scope for v1)

When v2 comes, the plan is:
1. Generate synthetic paired data using `audiomentations` — add vinyl crackle, hiss, frequency rolloff (~8kHz cutoff), wow/flutter on top of clean modern tango recordings
2. Fine-tune CleanUNet on those pairs
3. Evaluate on held-out golden-age recordings

The goal is a model specifically tuned to the noise profile of Odeon/Victor 78rpm pressings recorded in Buenos Aires between 1935 and 1955.
