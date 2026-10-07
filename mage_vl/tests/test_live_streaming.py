from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MAGE_VL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAGE_VL))

from inference_live import (  # noqa: E402
    FFmpegSegmentSource,
    confirm_pinned_processor_code,
    preflight,
)


class FFmpegSegmentSourceTest(unittest.TestCase):
    def make_source(self, source: str, **kwargs) -> FFmpegSegmentSource:
        return FFmpegSegmentSource(
            source=source,
            session_dir=Path("/tmp/mage-live-test"),
            segment_sec=4.0,
            **kwargs,
        )

    def test_file_command_is_realtime_normalized_h264(self):
        command = self.make_source("demo.mp4").command()
        self.assertIn("-re", command)
        self.assertIn("libx264", command)
        self.assertIn("-force_key_frames", command)
        self.assertIn("segments.csv", " ".join(command))

    def test_rtsp_command_uses_transport_without_re_flag(self):
        command = self.make_source("rtsp://camera/live", rtsp_transport="tcp").command()
        self.assertNotIn("-re", command)
        self.assertEqual(command[command.index("-rtsp_transport") + 1], "tcp")

    def test_copy_codec_does_not_reencode(self):
        command = self.make_source("demo.mp4", copy_codec=True).command()
        self.assertEqual(command[command.index("-c:v") + 1], "copy")
        self.assertNotIn("-force_key_frames", command)

    def test_playlist_yields_each_absolute_or_relative_path_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "segment_00000000.mp4"
            second = root / "segment_00000001.mp4"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            (root / "segments.csv").write_text(
                f"{first},0.0,4.0\n{second.name},4.0,8.0\n",
                encoding="utf-8",
            )
            source = FFmpegSegmentSource("demo.mp4", root, 4.0)
            entries = source._entries()
            self.assertEqual([entry.path for entry in entries], [first, second])
            source._seen.add(str(first.resolve()))
            self.assertEqual([entry.path for entry in source._entries()], [second])

    def test_backlog_drop_marks_discontinuity_and_deletes_stale_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / f"segment_{index:08d}.mp4" for index in range(3)]
            for path in paths:
                path.write_bytes(path.name.encode())
            (root / "segments.csv").write_text(
                "".join(
                    f"{path.name},{index * 4.0},{(index + 1) * 4.0}\n"
                    for index, path in enumerate(paths)
                ),
                encoding="utf-8",
            )
            source = FFmpegSegmentSource(
                "demo.mp4", root, 4.0, max_backlog_segments=2
            )
            kept = source._bounded_entries(source._entries())
            self.assertEqual([entry.path for entry in kept], paths[1:])
            self.assertTrue(kept[0].discontinuity)
            self.assertFalse(paths[0].exists())

    def test_preflight_fails_before_model_load_when_tools_are_missing(self):
        args = SimpleNamespace(
            ffmpeg_bin="missing-ffmpeg",
            video_backend="codec",
            codec_engine="hevc",
        )
        with mock.patch("inference_live.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "missing streaming prerequisite"):
                preflight(args, "/unused/checkpoint")

    def test_pinned_processor_confirmation_is_scoped(self):
        import builtins

        original = builtins.input
        with confirm_pinned_processor_code():
            self.assertEqual(builtins.input("ignored"), "y")
        self.assertIs(builtins.input, original)


class IncrementalSessionTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            from mamba_ssm.utils.generation import InferenceParams  # noqa: F401
        except ImportError as error:
            self.skipTest(str(error))
        self.torch = torch

    def test_tokenwise_state_is_constant_and_resettable(self):
        from streaming_session import IncrementalStreamMindSession

        torch = self.torch

        class Identity:
            def __call__(self, value):
                return value

        class Accumulator:
            def __call__(self, value, inference_params):
                state = inference_params.key_value_memory_dict.get("state")
                if state is None:
                    state = torch.zeros_like(value[:, :1])
                    inference_params.key_value_memory_dict["state"] = state
                output = state + value.cumsum(dim=1)
                state.copy_(output[:, -1:])
                return output

        class Gate:
            pre_net = Identity()
            mamba_model = Accumulator()
            post_net = Identity()

        class Core:
            def _load_streammind_gate(self):
                return Gate()

            def _streammind_vision_tokens(self, *_args, **_kwargs):
                raise AssertionError("not used in this unit test")

        session = IncrementalStreamMindSession(Core(), max_stream_steps=10)
        session._classify_last_token = lambda token: torch.cat(
            (torch.zeros_like(token[..., :1]), token[..., :1]), dim=-1
        ).reshape(1, 2)

        first = torch.ones(1, 2, 3, 4)
        second = torch.full((1, 1, 3, 4), 2.0)
        one = session.push_vision_tokens(first)
        bytes_after_first = session.state_bytes
        two = session.push_vision_tokens(second)

        self.assertEqual(one.total_epfe_steps, 2)
        self.assertEqual(two.total_epfe_steps, 3)
        self.assertEqual(session.inference_params.seqlen_offset, 3)
        self.assertEqual(session.state_bytes, bytes_after_first)
        self.assertGreater(two.probability, one.probability)

        session.reset()
        self.assertEqual(session.total_steps, 0)
        self.assertEqual(session.state_bytes, 0)
        self.assertEqual(session.inference_params.seqlen_offset, 0)


@unittest.skipUnless(
    os.environ.get("MAGE_RUN_GPU_TESTS") == "1",
    "set MAGE_RUN_GPU_TESTS=1 for the real StreamMind gate test",
)
class RealGateEquivalenceTest(unittest.TestCase):
    def test_incremental_boundaries_match_one_shot_gate(self):
        import torch
        from safetensors.torch import load_file
        from streaming_session import IncrementalStreamMindSession

        checkpoint = Path(os.environ["MAGE_CHECKPOINT"]).resolve()
        sys.path.insert(0, str(checkpoint))
        from streammind_gate import StreamMindGate

        torch.manual_seed(7)
        gate = StreamMindGate(hidden_size=2560)
        gate.load_state_dict(
            load_file(str(checkpoint / "streammind_gate.safetensors")), strict=True
        )
        gate.to(device="cuda:0", dtype=torch.bfloat16).eval()

        segments = [
            torch.randn(1, length, 2, 2560, device="cuda:0", dtype=torch.bfloat16)
            for length in (3, 2, 4)
        ]
        boundaries = [3, 5, 9]
        with torch.inference_mode():
            offline = gate(
                torch.cat(segments, dim=1), response_positions=boundaries
            )[:, [value - 1 for value in boundaries]]

        class Core:
            def _load_streammind_gate(self):
                return gate

            def _streammind_vision_tokens(self, *_args, **_kwargs):
                raise AssertionError("not used")

        session = IncrementalStreamMindSession(Core(), max_stream_steps=32)
        actual = []
        cache_sizes = []
        for segment in segments:
            step = session.push_vision_tokens(segment)
            actual.append(step.logits)
            cache_sizes.append(step.state_bytes)
        actual = torch.stack(actual, dim=1)

        torch.testing.assert_close(actual, offline, atol=2e-3, rtol=2e-3)
        self.assertGreater(cache_sizes[0], 0)
        self.assertEqual(len(set(cache_sizes)), 1)


if __name__ == "__main__":
    unittest.main()
