import unittest
import wave
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from gemma_4_mic_transcribe.audio import (
    RollingPcmBuffer,
    format_device_list,
    frames_to_timestamp,
    iter_wav_windows,
    list_input_devices,
    pcm16_to_wav_bytes,
)


class FakeSoundDevice:
    default = type("Default", (), {"device": (2, None)})

    @staticmethod
    def query_devices():
        return [
            {"name": "Output only", "max_input_channels": 0},
            {"name": "USB Mic", "max_input_channels": 1},
            {"name": "Built-in Mic", "max_input_channels": 2},
        ]


class AudioTests(unittest.TestCase):
    def test_rolling_buffer_trims_to_window(self):
        buffer = RollingPcmBuffer(sample_rate=4, window_seconds=1.0)

        buffer.append(b"\x01\x00" * 2)
        buffer.append(b"\x02\x00" * 4)

        pcm, start, end = buffer.snapshot()
        self.assertEqual(pcm, b"\x02\x00" * 4)
        self.assertEqual(start, 2)
        self.assertEqual(end, 6)

    def test_pcm16_to_wav_bytes(self):
        wav_bytes = pcm16_to_wav_bytes(b"\x00\x00\xff\x7f", sample_rate=16000)

        with wave.open(BytesIO(wav_bytes), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), 16000)
            self.assertEqual(wav.readframes(2), b"\x00\x00\xff\x7f")

    def test_iter_wav_windows_splits_valid_wav_chunks(self):
        with TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "sample.wav"
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(4)
                wav.writeframes(b"\x00\x00\x00\x00" * 8)

            windows = iter_wav_windows(
                str(wav_path),
                window_seconds=1.0,
                stride_seconds=0.5,
                target_sample_rate=4,
            )

        self.assertEqual([(window.start_frame, window.end_frame) for window in windows], [(0, 4), (2, 6), (4, 8)])
        for window in windows:
            with wave.open(BytesIO(window.wav_bytes), "rb") as wav:
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertEqual(wav.getframerate(), 4)

    def test_iter_wav_windows_resamples_to_target_rate(self):
        with TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "sample.wav"
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(8)
                wav.writeframes(b"\x00\x00" * 8)

            windows = iter_wav_windows(
                str(wav_path),
                window_seconds=1.0,
                stride_seconds=1.0,
                target_sample_rate=4,
            )

        self.assertEqual(len(windows), 1)
        self.assertEqual((windows[0].start_frame, windows[0].end_frame), (0, 4))
        with wave.open(BytesIO(windows[0].wav_bytes), "rb") as wav:
            self.assertEqual(wav.getframerate(), 4)
            self.assertEqual(wav.getnframes(), 4)

    def test_device_listing_and_format(self):
        devices = list_input_devices(FakeSoundDevice)

        self.assertEqual([device.index for device in devices], [1, 2])
        self.assertTrue(devices[1].default)
        self.assertIn("2: Built-in Mic [2 input channels] (default)", format_device_list(devices))

    def test_frames_to_timestamp(self):
        self.assertEqual(frames_to_timestamp(100, 10), "00:10.0")
        self.assertEqual(frames_to_timestamp(650, 10), "01:05.0")


if __name__ == "__main__":
    unittest.main()
