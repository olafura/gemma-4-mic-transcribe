import argparse
import tempfile
import unittest
from pathlib import Path

from gemma_4_mic_transcribe.cli import DEFAULT_MODEL, build_parser, validate_args


class CliTests(unittest.TestCase):
    def test_defaults(self):
        args = build_parser().parse_args([])
        config = validate_args(args)

        self.assertTrue(config.model.endswith(DEFAULT_MODEL.removeprefix("~")))
        self.assertEqual(config.window_seconds, 5.0)
        self.assertEqual(config.stride_seconds, 2.5)
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.backend, "gpu")
        self.assertEqual(config.audio_backend, "gpu")
        self.assertEqual(config.audio_position, "before")
        self.assertEqual(config.skip_windows, 0)
        self.assertEqual(config.request_timeout_seconds, 30.0)

    def test_custom_system_message_and_device(self):
        args = build_parser().parse_args(
            [
                "--device",
                "3",
                "--system-message",
                "Transcribe in Icelandic.",
                "--window-seconds",
                "8",
                "--stride-seconds",
                "1",
            ]
        )
        config = validate_args(args)

        self.assertEqual(config.device, 3)
        self.assertEqual(config.system_message, "Transcribe in Icelandic.")
        self.assertEqual(config.window_seconds, 8.0)
        self.assertEqual(config.stride_seconds, 1.0)

    def test_wav_and_system_message_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "sample.wav"
            system_path = temp_path / "system.txt"
            wav_path.write_bytes(b"RIFF")
            system_path.write_text("Transcribe exactly.\n", encoding="utf-8")

            args = build_parser().parse_args(
                [
                    "--wav",
                    str(wav_path),
                    "--skip-windows",
                    "2",
                    "--max-windows",
                    "1",
                    "--system-message-file",
                    str(system_path),
                ]
            )
            config = validate_args(args)

        self.assertEqual(config.wav, str(wav_path))
        self.assertEqual(config.skip_windows, 2)
        self.assertEqual(config.max_windows, 1)
        self.assertEqual(config.system_message, "Transcribe exactly.")

    def test_invalid_window(self):
        args = argparse.Namespace(
            model=DEFAULT_MODEL,
            wav=None,
            skip_windows=0,
            max_windows=None,
            device=None,
            system_message=None,
            system_message_file=None,
            prompt="prompt",
            window_seconds=0,
            stride_seconds=1,
            sample_rate=16000,
            backend="gpu",
            audio_backend="gpu",
            audio_position="before",
            request_timeout_seconds=30.0,
            debug_response=False,
        )

        with self.assertRaisesRegex(ValueError, "window-seconds"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
