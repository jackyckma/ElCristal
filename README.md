# ElCristal — Tango Audio Restoration

Free, open-source AI restoration for golden-age Argentine tango recordings (1930s–1950s).

Upload a track → receive a cleaned version by email. No login required.

---

## What it does

ElCristal runs a four-stage pipeline on each uploaded track:

| Stage | Model | What it fixes |
|---|---|---|
| 1. Source separation | Demucs htdemucs | Splits into stems for cleaner per-stem processing |
| 2. Denoising | CleanUNet (Aero fallback) | Tape hiss, vinyl crackle, broadband noise |
| 3. Bandwidth extension | AudioSR | Reconstructs high-frequency content lost in 78rpm recording |
| 4. Loudness normalisation | pyloudnorm (EBU R128) | Consistent, broadcast-ready levels |

The bandoneon must still sound like a bandoneon. No musical character is altered.

---

## Quick start

```bash
# 1. Clone the repo
git clone https://github.com/jackyckma/elcristal.git
cd elcristal

# 2. Configure email (optional — skip for testing without notifications)
cp .env.example .env
$EDITOR .env

# 3. Start all services
docker compose up --build

# 4. Open the UI
open http://localhost:7860
```

The first run downloads model weights (~5 GB) to the `elcristal-models` Docker volume.

### Fast iteration workflow (modular services)

The stack is split into independent services:

- `web`: Gradio frontend + queue client (fast build/redeploy)
- `worker`: heavy ML runtime (slow build, changes less often)
- `redis`: queue broker

So UI-only changes do **not** require rebuilding the heavy worker image.

---

## Configuration

All settings live in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `SMTP_HOST` | — | SMTP server for email notifications |
| `SMTP_USER` | — | SMTP username / sender address |
| `SMTP_PASS` | — | SMTP password |
| `MAX_FILE_SIZE_MB` | `100` | Per-file upload size limit |
| `MAX_BATCH_SIZE` | `50` | Max files per submission |
| `OUTPUT_TTL_HOURS` | `72` | Hours before output files are deleted |
| `USE_GPU` | `false` | Set `true` to enable CUDA (see GPU section) |
| `BASE_URL` | `http://localhost:7860` | Public URL used in notification emails |
| `SEPARATION_BACKEND` | `demucs` | `demucs` or `passthrough` |
| `DENOISE_BACKEND` | `auto` | `auto`, `cleanunet`, `aero`, `spectral`, `passthrough` |
| `BANDWIDTH_BACKEND` | `auto` | `auto`, `audiosr`, `passthrough` |

### Stage backend selection

You can tune/replace stages independently via environment variables:

```bash
# examples
SEPARATION_BACKEND=demucs
DENOISE_BACKEND=spectral
BANDWIDTH_BACKEND=passthrough
```

This lets you iterate on one component at a time without touching orchestration.

---

## GPU acceleration

1. Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Set `USE_GPU=true` in `.env`
3. Uncomment the `deploy.resources` block in `docker-compose.yml`
4. Rebuild: `docker compose up --build`

Performance: ~20 min/track (CPU) → ~1–2 min/track (T4 GPU)

---

## Running tests

```bash
# Install test dependencies
pip install pytest pyloudnorm soundfile numpy scipy

# Run (ML stages are skipped if models are not installed)
pytest tests/ -v
```

---

## Repository layout

```
├── app/
│   ├── config.py      # Environment variable loading
│   ├── main.py        # Gradio UI
│   ├── notify.py      # Email notifications
│   └── queue.py       # Redis/RQ job submission
├── pipeline/
│   ├── worker.py      # RQ worker + job orchestration
│   ├── separate.py    # Stage 1: Demucs
│   ├── denoise.py     # Stage 2: CleanUNet / Aero
│   ├── bandwidth.py   # Stage 3: AudioSR
│   ├── normalize.py   # Stage 4: pyloudnorm + FFmpeg
│   └── utils.py       # Shared audio I/O
├── tests/
│   ├── test_pipeline.py
│   └── fixtures/
├── docker-compose.yml
├── Dockerfile.web
├── Dockerfile.worker
└── .env.example
```

---

## Hardware requirements

| Resource | Minimum (CPU) | Recommended (GPU) |
|---|---|---|
| RAM | 16 GB | 16 GB |
| CPU | 4 vCPU | 4 vCPU |
| GPU | None | NVIDIA T4 16 GB |
| Disk | 40 GB | 40 GB |

---

## License

MIT — free for everyone, especially the tango community.
