from __future__ import annotations

import logging
import os
import queue as stdlib_queue
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator

import gradio as gr

from app import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

config.INPUT_DIR.mkdir(parents=True, exist_ok=True)
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FORMATS = ["Same as input", "MP3", "WAV", "FLAC"]
ACCEPTED_EXTENSIONS = {".mp3", ".wav", ".flac"}
MAX_SIZE_BYTES = config.MAX_FILE_SIZE_MB * 1024 * 1024

# How often to poll RQ for job status updates while the user waits.
POLL_INTERVAL_SECONDS = 2.0
# Hard cap on how long the UI will wait for a single submission. Matches the
# per-job timeout enforced in queue.py multiplied by a small safety factor.
MAX_WAIT_SECONDS = 3600 * 2


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


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"


def submit(files: list, output_format: str) -> Iterator[tuple[str, list[str] | None]]:
    """Validate inputs, run/enqueue jobs, and stream status + finished file paths.

    Dispatches to inline-mode or RQ-queued mode depending on ``config.INLINE_MODE``.
    Yields ``(status_markdown, output_file_paths)`` tuples. ``output_file_paths``
    is ``None`` until at least one job finishes, then a list of restored files.
    """
    if not files:
        yield "Please upload at least one audio file.", None
        return

    accepted, errors = _validate_files(files)
    if not accepted:
        yield "No valid files to process.\n" + "\n".join(errors), None
        return

    fmt = output_format if output_format != "Same as input" else ""
    fmt_arg = fmt.lower() if fmt else ""

    if config.INLINE_MODE:
        yield from _submit_inline(accepted, fmt_arg, errors)
    else:
        yield from _submit_queued(accepted, fmt_arg, errors)


def _submit_queued(
    accepted: list[Path],
    fmt: str,
    errors: list[str],
) -> Iterator[tuple[str, list[str] | None]]:
    """Original RQ/Redis path: enqueue jobs and poll for completion."""
    from rq.job import Job  # noqa: PLC0415 — avoid importing rq when unused
    from app import queue  # noqa: PLC0415

    job_meta: list[tuple[str, str]] = []
    try:
        for src in accepted:
            dest_name = f"{uuid.uuid4()}_{src.name}"
            dest = config.INPUT_DIR / dest_name
            shutil.copy2(src, dest)
            # Empty recipient: notify.send_completion_email() short-circuits when SMTP
            # is unconfigured, which is the case for this in-UI test mode.
            job_id = queue.queue_job(dest, "", fmt)
            job_meta.append((job_id, src.name))
        redis_conn = queue.get_redis()
    except Exception as exc:
        logger.exception("Cannot enqueue job — likely a Redis configuration issue")
        yield (
            "Failed to enqueue restoration job.\n"
            f"Reason: {exc}\n\n"
            "This usually means the worker queue (Redis) is unreachable or "
            "requires a password that isn't being passed via REDIS_URL.\n"
            f"Current REDIS_URL host: {config.REDIS_URL.split('@')[-1] or '(unset)'}\n\n"
            "Tip: set INLINE_MODE=true to run the pipeline in-process and skip Redis."
        ), None
        return

    pending: set[str] = {jid for jid, _ in job_meta}
    done: dict[str, str] = {}
    failed: dict[str, str] = {}
    started_at = time.monotonic()

    def render_status() -> str:
        elapsed = time.monotonic() - started_at
        header = (
            f"Processing {len(job_meta)} track(s) — elapsed {_format_elapsed(elapsed)}.\n"
            f"Restoration takes ~20 min/track on CPU, ~1–2 min on GPU.\n"
        )
        if errors:
            header += "Skipped: " + "; ".join(errors) + "\n"
        lines = []
        for jid, name in job_meta:
            if jid in done:
                lines.append(f"  ✓ {name} — ready")
            elif jid in failed:
                lines.append(f"  ✗ {name} — failed: {failed[jid]}")
            else:
                try:
                    job = Job.fetch(jid, connection=redis_conn)
                    status = job.get_status(refresh=False) or "queued"
                except Exception:
                    status = "unknown"
                lines.append(f"  … {name} — {status}")
        return header + "\n".join(lines)

    yield render_status(), None

    while pending:
        if time.monotonic() - started_at > MAX_WAIT_SECONDS:
            for jid in pending:
                failed[jid] = "timed out waiting for worker"
            pending.clear()
            break

        time.sleep(POLL_INTERVAL_SECONDS)
        for jid in list(pending):
            try:
                job = Job.fetch(jid, connection=redis_conn)
            except Exception:
                continue
            status = job.get_status(refresh=True)
            if status == "finished":
                pending.discard(jid)
                result = job.result or {}
                output_path = result.get("output")
                if output_path and Path(output_path).exists():
                    done[jid] = output_path
                else:
                    failed[jid] = "worker reported success but output file is missing"
            elif status == "failed":
                pending.discard(jid)
                exc = (job.exc_info or "").strip().splitlines()
                failed[jid] = exc[-1] if exc else "worker raised an exception"

        ready_files = list(done.values()) if done else None
        yield render_status(), ready_files

    elapsed = time.monotonic() - started_at
    summary_parts = [
        f"Done in {_format_elapsed(elapsed)}.",
        f"{len(done)}/{len(job_meta)} track(s) restored.",
    ]
    if failed:
        summary_parts.append("Failures:")
        summary_parts.extend(f"  • {jid[:8]}…: {msg}" for jid, msg in failed.items())
    summary_parts.append("")
    summary_parts.append(render_status())
    yield "\n".join(summary_parts), list(done.values()) or None


def _submit_inline(
    accepted: list[Path],
    fmt: str,
    errors: list[str],
) -> Iterator[tuple[str, list[str] | None]]:
    """Inline path: run the full pipeline in a worker thread, stream stage progress.

    No Redis/RQ involved. Designed for single-container deploys (e.g. Zeabur).
    Tracks are processed sequentially because the ML stages are RAM-heavy.
    """
    try:
        from pipeline.worker import process_track  # noqa: PLC0415 — heavy ML imports
    except Exception as exc:
        logger.exception("Inline mode requested but pipeline cannot be imported")
        yield (
            "INLINE_MODE is set but the ML pipeline could not be loaded.\n"
            f"Reason: {exc}\n\n"
            "The image probably doesn't have the worker dependencies "
            "(torch, demucs, audiosr, etc.) installed."
        ), None
        return

    track_states: list[dict] = []
    for src in accepted:
        job_id = str(uuid.uuid4())
        dest_name = f"{job_id}_{src.name}"
        dest = config.INPUT_DIR / dest_name
        shutil.copy2(src, dest)
        track_states.append(
            {
                "job_id": job_id,
                "name": src.name,
                "input_path": dest,
                "stage": "queued",
                "progress": 0.0,
                "output": None,
                "error": None,
            }
        )

    started_at = time.monotonic()

    def render_status(active_idx: int | None) -> str:
        elapsed = time.monotonic() - started_at
        header_lines = [
            f"Processing {len(track_states)} track(s) inline — elapsed {_format_elapsed(elapsed)}.",
            "Restoration takes ~20 min/track on CPU, ~1–2 min on GPU.",
        ]
        if errors:
            header_lines.append("Skipped: " + "; ".join(errors))
        lines = []
        for i, t in enumerate(track_states):
            if t["output"]:
                lines.append(f"  ✓ {t['name']} — ready")
            elif t["error"]:
                lines.append(f"  ✗ {t['name']} — failed: {t['error']}")
            elif i == active_idx:
                pct = int(t["progress"] * 100)
                lines.append(f"  … {t['name']} — {t['stage']} ({pct}%)")
            else:
                lines.append(f"  · {t['name']} — {t['stage']}")
        return "\n".join(header_lines) + "\n" + "\n".join(lines)

    yield render_status(active_idx=None), None

    for idx, track in enumerate(track_states):
        # Cross-thread progress messages: (label, progress_0_to_1)
        progress_q: stdlib_queue.Queue[tuple[str, float] | None] = stdlib_queue.Queue()
        result_holder: dict = {}

        def _cb(label: str, pct: float) -> None:
            progress_q.put((label, pct))

        def _runner() -> None:
            try:
                result = process_track(
                    job_id=track["job_id"],
                    input_path=str(track["input_path"]),
                    recipient_email="",
                    output_format=fmt,
                    progress_cb=_cb,
                )
                result_holder["result"] = result
            except Exception as exc:
                logger.exception("Inline job %s failed", track["job_id"])
                result_holder["error"] = exc
            finally:
                progress_q.put(None)

        worker = threading.Thread(target=_runner, name=f"restore-{track['job_id'][:8]}", daemon=True)
        worker.start()

        while True:
            try:
                msg = progress_q.get(timeout=POLL_INTERVAL_SECONDS)
            except stdlib_queue.Empty:
                yield render_status(active_idx=idx), _ready_outputs(track_states)
                continue

            if msg is None:
                break
            label, pct = msg
            track["stage"] = label
            track["progress"] = pct
            yield render_status(active_idx=idx), _ready_outputs(track_states)

        worker.join(timeout=5)

        if "error" in result_holder:
            track["error"] = str(result_holder["error"]) or type(result_holder["error"]).__name__
            track["stage"] = "failed"
        else:
            res = result_holder.get("result") or {}
            output_path = res.get("output")
            if output_path and Path(output_path).exists():
                track["output"] = output_path
                track["stage"] = "complete"
                track["progress"] = 1.0
            else:
                track["error"] = "pipeline finished but output file is missing"
                track["stage"] = "failed"

        yield render_status(active_idx=idx), _ready_outputs(track_states)

    elapsed = time.monotonic() - started_at
    successes = [t for t in track_states if t["output"]]
    failures = [t for t in track_states if t["error"]]
    summary = [
        f"Done in {_format_elapsed(elapsed)}.",
        f"{len(successes)}/{len(track_states)} track(s) restored.",
    ]
    if failures:
        summary.append("Failures:")
        summary.extend(f"  • {t['name']}: {t['error']}" for t in failures)
    summary.append("")
    summary.append(render_status(active_idx=None))
    yield "\n".join(summary), _ready_outputs(track_states)


def _ready_outputs(track_states: list[dict]) -> list[str] | None:
    files = [t["output"] for t in track_states if t["output"]]
    return files or None


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ElCristal — Tango Audio Restoration", theme=gr.themes.Soft()) as app:
        mode_label = "inline" if config.INLINE_MODE else "queued"
        gr.Markdown(
            f"""
# ElCristal — Tango Audio Restoration (test mode · {mode_label})

Restore golden-age tango recordings (1930s–1950s) using modern AI.
Removes hiss, crackle and noise — without altering the musical character of the original.

Upload one or more tracks. The page will stay open while your files are processed
and offer them as direct downloads when ready. **No email required.**
"""
        )

        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(
                    label="Audio files (MP3 / WAV / FLAC)",
                    file_count="multiple",
                    file_types=[".mp3", ".wav", ".flac"],
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
                    lines=10,
                    interactive=False,
                    placeholder="Submit your files to see live progress here.",
                )
                download_output = gr.Files(
                    label="Restored tracks",
                    interactive=False,
                )

        submit_btn.click(
            fn=submit,
            inputs=[file_input, format_input],
            outputs=[status_output, download_output],
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
    port = int(os.environ.get("PORT", "7860"))
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        # Gradio serves downloadable files via its cache and blocks paths outside
        # cwd/tmp unless explicitly allowlisted.
        allowed_paths=[str(config.OUTPUT_DIR)],
    )
