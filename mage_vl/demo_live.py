#!/usr/bin/env python
"""Local Gradio demo for Mage-VL incremental codec streaming.

The UI launches ``inference_live.py`` as a process group, streams its JSONL events
into a table, and can stop the model plus its FFmpeg child safely.  It binds to the
requested interface with password authentication and never creates a public Gradio
share link.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


DEFAULT_CHECKPOINT = "microsoft/Mage-VL"
DEFAULT_REVISION = "d88b153285f1633a61b2f693c59c8576693af185"
TABLE_HEADERS = [
    "Segment", "Stream time", "Gate p", "Decision", "Response",
    "Prep (s)", "Gate (s)", "Gen (s)", "Lag (s)", "Source live",
    "State (KiB)", "CUDA allocated (GiB)",
]
MAX_UI_ROWS = 500


@dataclass
class ActiveRun:
    process: subprocess.Popen
    run_dir: Path
    source_secret: str


_LOCK = threading.Lock()
_ACTIVE: ActiveRun | None = None


def redact_source(source: str) -> str:
    """Remove RTSP credentials while keeping enough endpoint context for the UI."""
    try:
        parsed = urlsplit(source)
    except ValueError:
        return "<source>"
    if parsed.scheme.lower() not in ("rtsp", "rtsps"):
        return source
    host = parsed.hostname or "camera"
    if parsed.port:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def terminate_process_group(process: subprocess.Popen, timeout: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=min(3.0, timeout))
        return
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=min(3.0, timeout))
        return
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def stop_active() -> str:
    global _ACTIVE
    with _LOCK:
        active = _ACTIVE
    if active is None or active.process.poll() is not None:
        return "### No active stream"
    terminate_process_group(active.process)
    return "### Stop requested — model and FFmpeg process group terminated"


def _cleanup_at_exit() -> None:
    with _LOCK:
        active = _ACTIVE
    if active is not None:
        terminate_process_group(active.process, timeout=2.0)


atexit.register(_cleanup_at_exit)


def records_from(path: Path, offset: int, partial: bytes) -> tuple[list[dict], int, bytes]:
    if not path.exists():
        return [], offset, partial
    with path.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read()
        offset = handle.tell()
    complete = partial + chunk
    lines = complete.split(b"\n")
    partial = lines.pop()
    records = []
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace")
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records, offset, partial


def table_row(record: dict) -> list:
    return [
        record.get("segment_index"),
        f"{record.get('start_s', 0):.1f}–{record.get('end_s', 0):.1f}s",
        round(float(record.get("gate_probability") or 0), 3),
        record.get("decision"),
        record.get("response") or "",
        record.get("preprocess_s"),
        record.get("gate_s"),
        record.get("generation_s"),
        record.get("source_lag_s"),
        record.get("source_process_running"),
        round(int(record.get("gate_state_bytes") or 0) / 1024, 1),
        round(int(record.get("cuda_allocated_bytes") or 0) / 2**30, 3),
    ]


def build_command(
    source: str,
    run_dir: Path,
    segment_sec: float,
    threshold: float,
    max_new_tokens: int,
    max_segments: int,
    realtime: bool,
    prompt: str,
    checkpoint: str,
    revision: str,
    device: str,
    runner_python: str,
) -> tuple[list[str], Path]:
    output = run_dir / "events.jsonl"
    command = [
        runner_python,
        "-u",
        str(Path(__file__).with_name("inference_live.py")),
        "--source", source,
        "--checkpoint", checkpoint,
        "--revision", revision,
        "--device", device,
        "--video-backend", "codec",
        "--codec-engine", "hevc",
        "--attn-impl", "sdpa",
        "--segment-sec", str(segment_sec),
        "--gate-threshold", str(threshold),
        "--max-new-tokens", str(int(max_new_tokens)),
        "--max-segments", str(int(max_segments)),
        "--max-backlog-segments", "4",
        "--prompt", prompt,
        "--spool-dir", str(run_dir / "spool"),
        "--output", str(output),
    ]
    command.append("--realtime" if realtime else "--no-realtime")
    return command, output


def run_stream(
    video_path,
    rtsp_url: str,
    segment_sec: float,
    threshold: float,
    max_new_tokens: int,
    max_segments: int,
    realtime: bool,
    prompt: str,
    checkpoint: str,
    revision: str,
    device: str,
    cv_preinfer_bin: str,
    runner_python: str,
):
    """Yield status, confidence, latest text, rows, logs, and final JSONL."""
    global _ACTIVE
    source = str(video_path or "").strip()
    rtsp_url = (rtsp_url or "").strip()
    empty = ("### Error", {}, "", [], "", None)
    if rtsp_url:
        if any(character in rtsp_url for character in ("\x00", "\r", "\n")):
            yield ("### Error: invalid control character in RTSP URL", *empty[1:])
            return
        parsed = urlsplit(rtsp_url)
        if parsed.scheme.lower() not in ("rtsp", "rtsps") or not parsed.hostname:
            yield ("### Error: enter a valid `rtsp://` or `rtsps://` URL", *empty[1:])
            return
        source = rtsp_url
    if not source:
        yield ("### Error: upload a video or enter an RTSP URL", *empty[1:])
        return
    if not rtsp_url and not Path(source).is_file():
        yield ("### Error: uploaded video is unavailable", *empty[1:])
        return
    if len(prompt or "") > 2000:
        yield ("### Error: prompt is limited to 2,000 characters", *empty[1:])
        return

    with _LOCK:
        if _ACTIVE is not None and _ACTIVE.process.poll() is None:
            yield ("### Busy: another stream is already using the GPU", *empty[1:])
            return

    run_dir = Path(tempfile.mkdtemp(prefix="mage-live-demo-"))
    run_dir.chmod(0o700)
    public_source = redact_source(source)
    source_secret = source
    if not rtsp_url:
        uploaded = Path(source).resolve()
        if uploaded.is_symlink() or not uploaded.is_file():
            yield ("### Error: upload must be a regular file", *empty[1:])
            shutil.rmtree(run_dir, ignore_errors=True)
            return
        if uploaded.stat().st_size > 2 * 1024**3:
            yield ("### Error: upload exceeds 2 GiB demo limit", *empty[1:])
            shutil.rmtree(run_dir, ignore_errors=True)
            return
        copied = run_dir / ("input" + (uploaded.suffix or ".mp4"))
        shutil.copy2(uploaded, copied)
        source = str(copied)
        public_source = f"uploaded:{uploaded.name}"

    command, output_path = build_command(
        source, run_dir, segment_sec, threshold, max_new_tokens, max_segments,
        realtime, prompt, checkpoint, revision, device, runner_python,
    )
    environment = os.environ.copy()
    environment["CV_PREINFER_BIN"] = cv_preinfer_bin
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
        start_new_session=True,
    )
    active = ActiveRun(process, run_dir, source_secret)
    with _LOCK:
        _ACTIVE = active

    rows: list[list] = []
    confidence: dict[str, float] = {}
    latest_response = ""
    log_lines = [f"source: {public_source}", "starting pinned Mage-VL runtime..."]
    record_offset = 0
    partial = b""
    yield (
        "### Running — waiting for first finalized segment",
        confidence,
        latest_response,
        rows,
        "\n".join(log_lines),
        None,
    )

    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip().replace(source_secret, public_source).replace(
                source, public_source
            )
            if line:
                log_lines.append(line)
                log_lines = log_lines[-200:]
            new_records, record_offset, partial = records_from(
                output_path, record_offset, partial
            )
            for record in new_records:
                rows.append(table_row(record))
                probability = float(record.get("gate_probability") or 0)
                confidence = {"response": probability, "silence": 1 - probability}
                if record.get("response"):
                    latest_response = str(record["response"])
            rows = rows[-MAX_UI_ROWS:]
            status = f"### Running — {len(rows)} event(s) received"
            yield status, confidence, latest_response, rows, "\n".join(log_lines), None

        new_records, record_offset, partial = records_from(output_path, record_offset, partial)
        for record in new_records:
            rows.append(table_row(record))
            probability = float(record.get("gate_probability") or 0)
            confidence = {"response": probability, "silence": 1 - probability}
            if record.get("response"):
                latest_response = str(record["response"])
        rows = rows[-MAX_UI_ROWS:]
        return_code = process.wait()
        if return_code == 0:
            status = f"### Finished — {len(rows)} event(s)"
        else:
            status = f"### Failed with exit code {return_code} — inspect log below"
        download = None
        if output_path.exists():
            results_dir = Path(".mage_live/demo_results").resolve()
            results_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            download_path = results_dir / f"{run_dir.name}.jsonl"
            shutil.copy2(output_path, download_path)
            download = str(download_path)
        yield status, confidence, latest_response, rows, "\n".join(log_lines), download
    finally:
        terminate_process_group(process)
        with _LOCK:
            if _ACTIVE is active:
                _ACTIVE = None
        shutil.rmtree(run_dir, ignore_errors=True)


def build_demo(args):
    import gradio as gr

    with gr.Blocks(
        title="Mage-VL Live Codec Streaming",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            "# Mage-VL Live Codec Streaming\n"
            "Incremental H.264/HEVC preprocessing with persistent StreamMind state. "
            "Events below are emitted before the source ends."
        )
        with gr.Row():
            bundled_sample = Path(__file__).parent / "assets/examples/soccer-broadcast.mp4"
            video = gr.Video(
                value=str(bundled_sample),
                label="Video (bundled sample is ready)",
                sources=["upload"],
                format="mp4",
            )
            with gr.Column():
                rtsp = gr.Textbox(
                    label="RTSP URL (takes precedence over upload)",
                    type="password",
                    placeholder="rtsp://user:password@camera/live",
                )
                prompt = gr.Textbox(
                    label="Generation prompt",
                    value="Please describe the current video event concisely.",
                    lines=2,
                )
                with gr.Row():
                    segment = gr.Slider(2, 12, value=4, step=1, label="Segment seconds")
                    threshold = gr.Slider(0, 1, value=0.5, step=0.05, label="Gate threshold")
                with gr.Row():
                    tokens = gr.Slider(4, 128, value=32, step=4, label="Max new tokens")
                    max_segments = gr.Number(value=4, precision=0, label="Max segments (0 = unlimited)")
                realtime = gr.Checkbox(value=True, label="Replay uploaded file at real-time speed")
                with gr.Row():
                    start = gr.Button("Start live inference", variant="primary")
                    stop = gr.Button("Stop", variant="stop")

        status = gr.Markdown("### Idle")
        with gr.Row():
            confidence = gr.Label(label="Latest gate confidence", num_top_classes=2)
            latest = gr.Textbox(label="Latest generated response", interactive=False)
        events = gr.Dataframe(
            headers=TABLE_HEADERS,
            datatype=["number", "str", "number", "str", "str"]
            + ["number"] * 4 + ["bool", "number", "number"],
            interactive=False,
            wrap=True,
            label="Live events",
        )
        logs = gr.Textbox(label="Sanitized runtime log", lines=14, interactive=False)
        download = gr.File(label="Completed event JSONL", interactive=False)

        run_event = start.click(
            fn=lambda *values: run_stream(
                *values,
                checkpoint=args.checkpoint,
                revision=args.revision,
                device=args.device,
                cv_preinfer_bin=args.cv_preinfer_bin,
                runner_python=args.runner_python,
            ),
            inputs=[video, rtsp, segment, threshold, tokens, max_segments, realtime, prompt],
            outputs=[status, confidence, latest, events, logs, download],
            concurrency_limit=1,
            api_visibility="private",
        )
        stop.click(
            fn=stop_active,
            outputs=status,
            cancels=[run_event],
            queue=False,
            api_visibility="private",
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cv-preinfer-bin",
        default=os.environ.get("CV_PREINFER_BIN", "cv-preinfer"),
    )
    parser.add_argument("--runner-python", default=sys.executable)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--auth-user", default="mage")
    parser.add_argument("--auth-password", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    password = args.auth_password or secrets.token_urlsafe(12)
    print(
        f"Mage live demo: http://127.0.0.1:{args.server_port}\n"
        f"username: {args.auth_user}\npassword: {password}\n"
        "No public share link is enabled.",
        flush=True,
    )
    demo = build_demo(args)
    demo.queue(default_concurrency_limit=1, max_size=4, api_open=False).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=False,
        auth=(args.auth_user, password),
        show_error=True,
    )


if __name__ == "__main__":
    main()
