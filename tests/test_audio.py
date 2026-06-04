import unittest
import wave
from io import BytesIO

from gemma_4_mic_transcribe.audio import (
    RollingPcmBuffer,
    format_device_list,
    frames_to_timestamp,
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
