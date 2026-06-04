import argparse
import unittest

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
        self.assertEqual(config.audio_backend, "cpu")

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

    def test_invalid_window(self):
        args = argparse.Namespace(
            model=DEFAULT_MODEL,
            device=None,
            system_message=None,
            prompt="prompt",
            window_seconds=0,
            stride_seconds=1,
            sample_rate=16000,
            backend="gpu",
            audio_backend="cpu",
            no_file_fallback=False,
        )

        with self.assertRaisesRegex(ValueError, "window-seconds"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
