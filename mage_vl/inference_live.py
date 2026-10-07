#!/usr/bin/env python
"""Incremental codec-native Mage-VL inference for files and RTSP streams.

Unlike ``inference_streaming.py`` (whole-file causal replay), this runner keeps one
FFmpeg process alive, consumes each finalized keyframe-aligned segment immediately,
and preserves StreamMind's recurrent Mamba state between segments.  Only the current
segment and a constant-size gate cache live on the GPU.

The codec preprocessor is still segment-granular because codec-video-prep 0.2.5 has
no packet/GOP push API.  Consequently the minimum decision latency is one completed
segment plus preprocessing and model time; this is real bounded-memory micro-batch
streaming, not frame-by-frame decoding.
"""

from __future__ import annotations

import argparse
import builtins
import csv
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


DEFAULT_CHECKPOINT = "microsoft/Mage-VL"
DEFAULT_REVISION = "d88b153285f1633a61b2f693c59c8576693af185"
DEFAULT_PROMPT = "Please describe the current video event concisely."


@dataclass(frozen=True)
class Segment:
    index: int
    path: Path
    start_s: float
    end_s: float
    ready_wall_time: float
    discontinuity: bool = False


@dataclass
class LiveRecord:
    stream_id: str
    segment_index: int
    start_s: float
    end_s: float
    discontinuity: bool
    gate_probability: float
    decision: str
    response: str | None
    epfe_steps: int
    total_epfe_steps: int
    gate_state_bytes: int
    preprocess_s: float
    gate_s: float
    generation_s: float
    source_lag_s: float
    source_process_running: bool
    cuda_allocated_bytes: int
    cuda_reserved_bytes: int
    emitted_at: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="Video file or RTSP URL")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attn-impl", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa"
    )
    parser.add_argument("--video-backend", choices=("codec", "frames"), default="codec")
    parser.add_argument("--codec-engine", choices=("hevc", "dcvc-rt"), default="hevc")
    parser.add_argument("--segment-sec", type=float, default=8.0)
    parser.add_argument("--target-canvas", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--images-per-group", type=int, default=4)
    parser.add_argument("--max-pixels", type=int, default=150000)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--cur-fps", type=float, default=2.0)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-segments", type=int, default=0, help="0 means unbounded")
    parser.add_argument("--max-stream-steps", type=int, default=1_000_000)
    parser.add_argument(
        "--output", default=None,
        help="JSONL output; defaults to <spool session>/events.jsonl",
    )
    parser.add_argument("--spool-dir", default=".mage_live")
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument(
        "--max-backlog-segments",
        type=int,
        default=0,
        help="drop older finalized segments and reset gate state when backlog exceeds this; 0 is lossless",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--rtsp-transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rate-limit local files to source speed; ignored for RTSP",
    )
    parser.add_argument(
        "--copy-codec",
        action="store_true",
        help="copy an already H.264/HEVC stream instead of normalizing to H.264",
    )
    parser.add_argument("--encoder", default="libx264")
    parser.add_argument("--keep-segments", action="store_true")
    parser.add_argument("--keep-codec-cache", action="store_true")
    args = parser.parse_args()

    if args.segment_sec <= 0:
        parser.error("--segment-sec must be positive")
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")
    if args.max_backlog_segments < 0:
        parser.error("--max-backlog-segments cannot be negative")
    if not 0 <= args.gate_threshold <= 1:
        parser.error("--gate-threshold must be in [0, 1]")
    if args.target_canvas <= 0 or args.group_size <= 0 or args.images_per_group <= 0:
        parser.error("codec grouping parameters must be positive")
    return args


class FFmpegSegmentSource:
    """Yield finalized MP4 segments written by one persistent FFmpeg process."""

    def __init__(
        self,
        source: str,
        session_dir: Path,
        segment_sec: float,
        ffmpeg_bin: str = "ffmpeg",
        poll_interval: float = 0.1,
        realtime: bool = True,
        copy_codec: bool = False,
        encoder: str = "libx264",
        rtsp_transport: str = "tcp",
        max_backlog_segments: int = 4,
    ):
        self.source = source
        self.session_dir = session_dir
        self.segment_sec = float(segment_sec)
        self.ffmpeg_bin = ffmpeg_bin
        self.poll_interval = float(poll_interval)
        self.realtime = bool(realtime)
        self.copy_codec = bool(copy_codec)
        self.encoder = encoder
        self.rtsp_transport = rtsp_transport
        self.max_backlog_segments = int(max_backlog_segments)
        self.playlist = session_dir / "segments.csv"
        self.stderr_log = session_dir / "ffmpeg.log"
        self.process: subprocess.Popen | None = None
        self._seen: set[str] = set()

    @property
    def is_rtsp(self) -> bool:
        return self.source.lower().startswith(("rtsp://", "rtsps://"))

    def command(self) -> list[str]:
        pattern = self.session_dir / "segment_%08d.mp4"
        cmd = [self.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "warning", "-nostdin"]
        if self.is_rtsp:
            cmd += ["-rtsp_transport", self.rtsp_transport]
        elif self.realtime:
            cmd += ["-re"]
        cmd += ["-i", self.source, "-map", "0:v:0", "-an"]
        if self.copy_codec:
            cmd += ["-c:v", "copy"]
        else:
            cmd += [
                "-c:v", self.encoder,
                "-preset", "veryfast",
                "-tune", "zerolatency",
                "-pix_fmt", "yuv420p",
                "-sc_threshold", "0",
                "-force_key_frames", f"expr:gte(t,n_forced*{self.segment_sec:g})",
            ]
        cmd += [
            "-f", "segment",
            "-segment_time", f"{self.segment_sec:g}",
            "-reset_timestamps", "1",
            "-segment_list", str(self.playlist),
            "-segment_list_type", "csv",
            "-segment_list_size", "0",
            "-segment_format", "mp4",
            str(pattern),
        ]
        return cmd

    def _entries(self) -> list[Segment]:
        if not self.playlist.exists():
            return []
        result = []
        # FFmpeg rewrites this short playlist atomically enough for read_text; an
        # incomplete last CSV row is ignored and read on the next poll.
        lines = self.playlist.read_text(encoding="utf-8", errors="replace").splitlines()
        for row in csv.reader(lines):
            if len(row) < 3:
                continue
            name = row[0]
            path = Path(name)
            if not path.is_absolute():
                path = self.session_dir / path
            key = str(path.resolve())
            if key in self._seen:
                continue
            try:
                start_s, end_s = float(row[1]), float(row[2])
            except ValueError:
                continue
            if not path.exists() or path.stat().st_size <= 0:
                continue
            result.append(
                Segment(
                    len(self._seen) + len(result),
                    path,
                    start_s,
                    end_s,
                    path.stat().st_mtime,
                )
            )
        return result

    @staticmethod
    def _tail(path: Path, limit: int = 4000) -> str:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-limit:]

    def _bounded_entries(self, entries: list[Segment]) -> list[Segment]:
        if not self.max_backlog_segments or len(entries) <= self.max_backlog_segments:
            return entries
        dropped = entries[:-self.max_backlog_segments]
        entries = entries[-self.max_backlog_segments:]
        for segment in dropped:
            self._seen.add(str(segment.path.resolve()))
            segment.path.unlink(missing_ok=True)
        first = entries[0]
        entries[0] = Segment(
            first.index,
            first.path,
            first.start_s,
            first.end_s,
            first.ready_wall_time,
            discontinuity=True,
        )
        print(
            f"[warn] dropped {len(dropped)} stale segment(s); "
            "resetting recurrent gate state",
            file=sys.stderr,
            flush=True,
        )
        return entries

    def __iter__(self) -> Iterator[Segment]:
        if shutil.which(self.ffmpeg_bin) is None and not Path(self.ffmpeg_bin).is_file():
            raise FileNotFoundError(f"ffmpeg binary not found: {self.ffmpeg_bin}")
        self.session_dir.mkdir(parents=True, exist_ok=False)
        stderr_handle = self.stderr_log.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command(),
            stdout=subprocess.DEVNULL,
            stderr=stderr_handle,
            text=True,
        )
        try:
            while True:
                emitted = False
                entries = self._bounded_entries(self._entries())
                for segment in entries:
                    self._seen.add(str(segment.path.resolve()))
                    emitted = True
                    yield segment
                return_code = self.process.poll()
                if return_code is not None:
                    # Drain playlist entries written during process shutdown.
                    pending = self._bounded_entries(self._entries())
                    for segment in pending:
                        self._seen.add(str(segment.path.resolve()))
                        yield segment
                    if return_code != 0:
                        raise RuntimeError(
                            f"ffmpeg exited with code {return_code}:\n{self._tail(self.stderr_log)}"
                        )
                    return
                if not emitted:
                    time.sleep(self.poll_interval)
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            stderr_handle.close()


def resolve_checkpoint(checkpoint: str, revision: str | None) -> str:
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        return str(path.resolve())
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=checkpoint, revision=revision)


@contextmanager
def confirm_pinned_processor_code():
    """Auto-confirm nested Transformers prompts for an already pinned local snapshot.

    MageVLProcessor currently drops ``trust_remote_code`` before its nested tokenizer
    load. The outer caller has explicitly trusted a content-addressed local snapshot,
    so answer that redundant prompt non-interactively and restore ``input`` at once.
    """
    original_input = builtins.input

    def trusted_input(prompt: str = "") -> str:
        print(
            "[trust] executing custom code from the pinned local Mage-VL snapshot",
            file=sys.stderr,
            flush=True,
        )
        return "y"

    builtins.input = trusted_input
    try:
        yield
    finally:
        builtins.input = original_input


def preflight(args, checkpoint_path: str) -> None:
    missing = []
    for binary in (args.ffmpeg_bin, "ffprobe"):
        if shutil.which(binary) is None and not Path(binary).is_file():
            missing.append(binary)
    if args.video_backend == "codec" and args.codec_engine == "hevc":
        cv_preinfer = os.environ.get("CV_PREINFER_BIN", "cv-preinfer")
        if shutil.which(cv_preinfer) is None and not Path(cv_preinfer).is_file():
            missing.append(cv_preinfer)
    if missing:
        raise RuntimeError(
            "missing streaming prerequisite(s): " + ", ".join(missing)
            + "; see mage_vl/LIVE_STREAMING.md"
        )
    if not args.copy_codec:
        encoders = subprocess.run(
            [args.ffmpeg_bin, "-hide_banner", "-encoders"],
            text=True,
            capture_output=True,
            timeout=30,
        )
        if encoders.returncode != 0 or args.encoder not in encoders.stdout:
            raise RuntimeError(
                f"ffmpeg encoder {args.encoder!r} is unavailable; install an FFmpeg "
                "build with that encoder or use --copy-codec for H.264/HEVC input"
            )
    if args.video_backend == "codec" and args.codec_engine == "dcvc-rt":
        neural = Path(checkpoint_path) / "neural_codec"
        required = (
            neural / "dcvc_readiness_gen.py",
            neural / "dcvc_rt_intra.tar",
            neural / "dcvc_rt_inter.tar",
        )
        absent = [str(path) for path in required if not path.is_file()]
        if absent:
            raise RuntimeError(
                "DCVC-RT needs a complete local checkpoint snapshot; missing: "
                + ", ".join(absent)
            )


def to_device(inputs, device: str, dtype):
    result = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            result[key] = value
        elif key == "pixel_values":
            result[key] = value.to(device=device, dtype=dtype)
        else:
            result[key] = value.to(device=device)
    return result


def prepare_segment(processor, path: Path, args, checkpoint_path: str):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": args.prompt},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    kwargs = {
        "text": [prompt],
        "videos": [str(path)],
        "video_backend": args.video_backend,
        "return_tensors": "pt",
        "padding": False,
    }
    if args.video_backend == "codec":
        codec_config = {
            "engine": args.codec_engine,
            "target_canvas": args.target_canvas,
            "group_size": args.group_size,
            "images_per_group": args.images_per_group,
            "patch": 16,
            "max_pixels": args.max_pixels,
        }
        if args.codec_engine == "dcvc-rt":
            codec_config["dcvc"] = {
                "pkg_dir": str(Path(checkpoint_path) / "neural_codec"),
                "device": args.device,
            }
        kwargs.update(max_pixels=args.max_pixels, codec_config=codec_config)
    else:
        kwargs.update(num_frames=args.num_frames, target_fps=args.cur_fps)
    return processor(**kwargs)


def generate_current_segment(model, processor, inputs, max_new_tokens: int) -> str:
    import torch
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from streaming_session import IncrementalStreamMindSession

    checkpoint_path = resolve_checkpoint(args.checkpoint, args.revision)
    preflight(args, checkpoint_path)
    with confirm_pinned_processor_code():
        processor = AutoProcessor.from_pretrained(
            checkpoint_path, trust_remote_code=True
        )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    ).to(args.device).eval()

    session = IncrementalStreamMindSession(
        model,
        max_stream_steps=args.max_stream_steps,
    )

    root = Path(args.spool_dir).resolve()
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
    session_dir = root / run_name
    codec_cache = session_dir / "codec_cache"
    # Set before the first processor call; the remote processor reads this lazily.
    os.environ["ONLINE_CODEC_CACHE_DIR"] = str(codec_cache)

    source = FFmpegSegmentSource(
        source=args.source,
        session_dir=session_dir / "segments",
        segment_sec=args.segment_sec,
        ffmpeg_bin=args.ffmpeg_bin,
        poll_interval=args.poll_interval,
        realtime=args.realtime,
        copy_codec=args.copy_codec,
        encoder=args.encoder,
        rtsp_transport=args.rtsp_transport,
        max_backlog_segments=args.max_backlog_segments,
    )

    output_path = Path(args.output) if args.output else session_dir / "events.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with output_path.open("a", encoding="utf-8", buffering=1) as output_file:
        try:
            for segment in source:
                if args.max_segments and processed >= args.max_segments:
                    break
                if segment.discontinuity:
                    session.reset()
                preprocess_start = time.perf_counter()
                inputs = prepare_segment(processor, segment.path, args, checkpoint_path)
                inputs = to_device(inputs, args.device, model.dtype)
                preprocess_s = time.perf_counter() - preprocess_start

                gate_start = time.perf_counter()
                gate_step = session.push_segment(inputs)
                gate_s = time.perf_counter() - gate_start
                speak = gate_step.probability >= args.gate_threshold

                response = None
                generation_s = 0.0
                if speak:
                    generation_start = time.perf_counter()
                    response = generate_current_segment(
                        model, processor, inputs, args.max_new_tokens
                    )
                    generation_s = time.perf_counter() - generation_start

                record = LiveRecord(
                    stream_id=run_name,
                    segment_index=segment.index,
                    start_s=round(segment.start_s, 3),
                    end_s=round(segment.end_s, 3),
                    discontinuity=segment.discontinuity,
                    gate_probability=round(gate_step.probability, 6),
                    decision="response" if speak else "silence",
                    response=response,
                    epfe_steps=gate_step.epfe_steps,
                    total_epfe_steps=gate_step.total_epfe_steps,
                    gate_state_bytes=gate_step.state_bytes,
                    preprocess_s=round(preprocess_s, 4),
                    gate_s=round(gate_s, 4),
                    generation_s=round(generation_s, 4),
                    source_lag_s=round(max(0.0, time.time() - segment.ready_wall_time), 4),
                    source_process_running=(
                        source.process is not None and source.process.poll() is None
                    ),
                    cuda_allocated_bytes=(
                        int(torch.cuda.memory_allocated(args.device))
                        if torch.cuda.is_available() else 0
                    ),
                    cuda_reserved_bytes=(
                        int(torch.cuda.memory_reserved(args.device))
                        if torch.cuda.is_available() else 0
                    ),
                    emitted_at=datetime.now(timezone.utc).isoformat(),
                )
                output_file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                suffix = f" -> {response}" if response else ""
                print(
                    f"[t={record.start_s:.1f}-{record.end_s:.1f}s] "
                    f"gate={record.decision} (p={record.gate_probability:.2f}) "
                    f"prep={record.preprocess_s:.2f}s gate={record.gate_s:.2f}s"
                    f"{suffix}",
                    flush=True,
                )
                processed += 1

                del inputs
                gc.collect()
                if not args.keep_codec_cache:
                    clear_directory(codec_cache)
                if not args.keep_segments:
                    segment.path.unlink(missing_ok=True)
        except KeyboardInterrupt:
            print("interrupted; flushed records are valid and resumable", file=sys.stderr)

    if not args.keep_segments:
        for stale_segment in (session_dir / "segments").glob("segment_*.mp4"):
            stale_segment.unlink(missing_ok=True)

    print(
        f"stream ended: segments={processed} epfe_steps={session.total_steps} "
        f"gate_state={session.state_bytes}B output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
