from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MAGE_VL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAGE_VL))

from demo_live import (  # noqa: E402
    build_command,
    records_from,
    redact_source,
    run_stream,
    terminate_process_group,
)


class LiveDemoSupervisorTest(unittest.TestCase):
    def test_rtsp_redaction_removes_credentials_and_query(self):
        source = "rtsp://user:secret@camera.example:8554/live?token=hidden"
        redacted = redact_source(source)
        self.assertEqual(redacted, "rtsp://camera.example:8554/live")
        self.assertNotIn("secret", redacted)
        self.assertNotIn("hidden", redacted)

    def test_source_is_one_argv_element_without_shell_interpretation(self):
        source = "rtsp://camera/live;touch /tmp/never-created"
        command, output = build_command(
            source=source,
            run_dir=Path("/tmp/demo-run"),
            segment_sec=4,
            threshold=0.5,
            max_new_tokens=8,
            max_segments=1,
            realtime=True,
            prompt="describe",
            checkpoint="/pinned/checkpoint",
            revision="abc",
            device="cuda:0",
            runner_python="/pinned/python",
        )
        self.assertEqual(command[command.index("--source") + 1], source)
        self.assertEqual(command[0], "/pinned/python")
        self.assertNotIn("shell=True", command)
        self.assertEqual(output, Path("/tmp/demo-run/events.jsonl"))

    def test_jsonl_tail_handles_partial_utf8_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            encoded = json.dumps({"response": "café"}, ensure_ascii=False).encode("utf-8")
            split = len(encoded) - 2
            path.write_bytes(encoded[:split])
            records, offset, partial = records_from(path, 0, b"")
            self.assertEqual(records, [])
            self.assertTrue(partial)

            with path.open("ab") as handle:
                handle.write(encoded[split:] + b"\n")
            records, offset, partial = records_from(path, offset, partial)
            self.assertEqual(records, [{"response": "café"}])
            self.assertEqual(partial, b"")

    def test_empty_start_yields_six_outputs(self):
        generator = run_stream(
            None, "", 4, 0.5, 8, 1, True, "describe",
            "/pinned", "abc", "cuda:0", "/cv-preinfer", "/python",
        )
        result = next(generator)
        self.assertEqual(len(result), 6)
        self.assertIn("upload a video", result[0])

    def test_process_group_termination_is_idempotent(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        terminate_process_group(process, timeout=2)
        self.assertIsNotNone(process.poll())
        terminate_process_group(process, timeout=2)


if __name__ == "__main__":
    unittest.main()
