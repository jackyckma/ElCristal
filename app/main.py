from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path

import gradio as gr

from app import config, queue

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

config.INPUT_DIR.mkdir(parents=True, exist_ok=True)
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FORMATS = ["Same as input", "MP3", "WAV", "FLAC"]
ACCEPTED_EXTENSIONS = {".mp3", ".wav", ".flac"}
MAX_SIZE_BYTES = config.MAX_FILE_SIZE_MB * 1024 * 1024


def _validate_files(files: list) -> tuple[list[Path], list[str]]:
    """Return accepted paths and any validation error messages."""
    accepted: list[Path] = []
    errors: list[str] = []

    if not files:
        return accepted, errors

    for f in files[: config.MAX_BATCH_SIZE]:
        path = Path(f.name if hasattr(f, "name") else str(f))
        if path.suffix.lower() not in ACCEPTED_EXTENSIONS:
            errors.append(f"{path.name}: unsupported format (use MP3, WAV or FLAC)")
            continue
        size = path.stat().st_size
        if size > MAX_SIZE_BYTES:
            errors.append(f"{path.name}: file too large ({size // 1_048_576} MB > {config.MAX_FILE_SIZE_MB} MB limit)")
            continue
        accepted.append(path)

    if len(files) > config.MAX_BATCH_SIZE:
        errors.append(
            f"Only the first {config.MAX_BATCH_SIZE} files were accepted "
            f"(batch limit is {config.MAX_BATCH_SIZE})."
        )

    return accepted, errors


def submit(files: list, email: str, output_format: str) -> str:
    """Validate inputs, copy uploads to storage, enqueue jobs."""
    email = (email or "").strip()
    if not email or "@" not in email:
        return "Please enter a valid email address."

    if not files:
        return "Please upload at least one audio file."

    accepted, errors = _validate_files(files)
    if not accepted:
        return "No valid files to process.\n" + "\n".join(errors)

    fmt = output_format if output_format != "Same as input" else ""

    job_ids: list[str] = []
    for src in accepted:
        dest_name = f"{uuid.uuid4()}_{src.name}"
        dest = config.INPUT_DIR / dest_name
        shutil.copy2(src, dest)
        job_id = queue.queue_job(dest, email, fmt.lower() if fmt else "")
        job_ids.append(job_id)

    queue_len = queue.get_queue_length()
    # Rough estimate: 20 min per track on CPU
    est_minutes = queue_len * 20

    status_lines = [
        f"✓ {len(job_ids)} track(s) queued for restoration.",
        f"Queue position: ~{queue_len} job(s) ahead.",
        f"Estimated wait: {est_minutes} – {est_minutes + len(job_ids) * 20} minutes (CPU) or significantly less with GPU.",
        f"You'll receive a download link at {email} when your tracks are ready.",
        f"Download links expire after {config.OUTPUT_TTL_HOURS} hours.",
    ]
    if errors:
        status_lines.append("\nSkipped files:")
        status_lines.extend(f"  • {e}" for e in errors)

    return "\n".join(status_lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ElCristal — Tango Audio Restoration", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
# ElCristal — Tango Audio Restoration

Restore golden-age tango recordings (1930s–1950s) using modern AI.
Removes hiss, crackle and noise — without altering the musical character of the original.

Upload one track or an entire folder. Your restored files will be emailed to you.
"""
        )

        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(
                    label="Audio files (MP3 / WAV / FLAC)",
                    file_count="multiple",
                    file_types=[".mp3", ".wav", ".flac"],
                )
                email_input = gr.Textbox(
                    label="Your email address",
                    placeholder="you@example.com",
                )
                format_input = gr.Dropdown(
                    label="Output format",
                    choices=OUTPUT_FORMATS,
                    value="Same as input",
                )
                submit_btn = gr.Button("Restore my tracks", variant="primary")

            with gr.Column(scale=1):
                status_output = gr.Textbox(
                    label="Status",
                    lines=8,
                    interactive=False,
                    placeholder="Submit your files to see queue status here.",
                )

        submit_btn.click(
            fn=submit,
            inputs=[file_input, email_input, format_input],
            outputs=status_output,
        )

        gr.Markdown(
            """
---
### About ElCristal

ElCristal uses a four-stage AI pipeline:

1. **Source separation** (Demucs) — isolates stems for cleaner per-stem processing
2. **Denoising** (CleanUNet) — removes tape hiss, vinyl surface noise and harmonic distortion
3. **Bandwidth extension** (AudioSR) — reconstructs high-frequency content lost in 78rpm recording
4. **Loudness normalisation** (EBU R128) — delivers consistent, broadcast-ready levels

Processing takes ~20 minutes per track on CPU, ~1–2 minutes with a GPU.

**Open source** · [GitHub](https://github.com/jackyckma/elcristal) · Free for the tango community
"""
        )

    return app


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
