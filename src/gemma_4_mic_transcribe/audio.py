"""Audio device listing, rolling capture buffers, and WAV encoding."""

from __future__ import annotations

import io
import threading
import wave
from collections import deque
from dataclasses import dataclass


SAMPLE_WIDTH_BYTES = 2


@dataclass(frozen=True)
class InputDevice:
    """A sounddevice input device entry."""

    index: int
    name: str
    channels: int
    default: bool = False


@dataclass(frozen=True)
class WavWindow:
    """A valid WAV byte window plus source-frame bounds."""

    wav_bytes: bytes
    start_frame: int
    end_frame: int
    sample_rate: int


class RollingPcmBuffer:
    """Thread-safe rolling mono PCM16 buffer."""

    def __init__(self, sample_rate: int, window_seconds: float) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

        self.sample_rate = sample_rate
        self.max_frames = max(1, int(sample_rate * window_seconds))
        self._chunks: deque[bytes] = deque()
        self._frames = 0
        self._total_frames = 0
        self._condition = threading.Condition()

    @property
    def total_frames(self) -> int:
        with self._condition:
            return self._total_frames

    @property
    def frames_available(self) -> int:
        with self._condition:
            return self._frames

    def append(self, pcm16: bytes) -> None:
        if len(pcm16) % SAMPLE_WIDTH_BYTES != 0:
            raise ValueError("PCM16 byte length must be divisible by 2")
        if not pcm16:
            return

        frames = len(pcm16) // SAMPLE_WIDTH_BYTES
        with self._condition:
            self._chunks.append(pcm16)
            self._frames += frames
            self._total_frames += frames
            self._trim_locked()
            self._condition.notify_all()

    def wait_until_total_frames(self, target_total_frames: int, timeout: float = 0.25) -> bool:
        with self._condition:
            if self._total_frames >= target_total_frames:
                return True
            self._condition.wait(timeout=timeout)
            return self._total_frames >= target_total_frames

    def snapshot(self) -> tuple[bytes, int, int]:
        """Return ``(pcm_bytes, start_frame, end_frame)`` for the current window."""

        with self._condition:
            pcm = b"".join(self._chunks)
            end_frame = self._total_frames
            start_frame = max(0, end_frame - self._frames)
            return pcm, start_frame, end_frame

    def _trim_locked(self) -> None:
        while self._frames > self.max_frames and self._chunks:
            excess_frames = self._frames - self.max_frames
            chunk = self._chunks[0]
            chunk_frames = len(chunk) // SAMPLE_WIDTH_BYTES
            if chunk_frames <= excess_frames:
                self._chunks.popleft()
                self._frames -= chunk_frames
                continue

            drop_bytes = excess_frames * SAMPLE_WIDTH_BYTES
            self._chunks[0] = chunk[drop_bytes:]
            self._frames -= excess_frames


def pcm16_to_wav_bytes(pcm16: bytes, sample_rate: int) -> bytes:
    """Wrap mono PCM16 bytes in a WAV container."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if len(pcm16) % SAMPLE_WIDTH_BYTES != 0:
        raise ValueError("PCM16 byte length must be divisible by 2")

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16)
    return output.getvalue()


def iter_wav_windows(
    path: str,
    window_seconds: float,
    stride_seconds: float,
    target_sample_rate: int = 16000,
) -> list[WavWindow]:
    """Split a PCM16 WAV file into valid mono WAV byte windows."""

    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive")

    import numpy as np

    with wave.open(path, "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()
        pcm = wav.readframes(frames)

    if sample_width != SAMPLE_WIDTH_BYTES:
        raise ValueError(f"only PCM16 WAV files are supported, got sample width {sample_width}")
    if channels <= 0:
        raise ValueError("WAV file has no audio channels")

    samples = np.frombuffer(pcm, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).astype(np.int32).mean(axis=1).clip(-32768, 32767).astype("<i2")

    if sample_rate != target_sample_rate and samples.size:
        duration_seconds = samples.shape[0] / sample_rate
        target_frames = max(1, int(round(duration_seconds * target_sample_rate)))
        source_positions = np.linspace(0, samples.shape[0] - 1, num=target_frames)
        samples = np.interp(source_positions, np.arange(samples.shape[0]), samples.astype(np.float32))
        samples = samples.clip(-32768, 32767).astype("<i2")
        sample_rate = target_sample_rate

    total_frames = int(samples.shape[0])
    window_frames = max(1, int(sample_rate * window_seconds))
    stride_frames = max(1, int(sample_rate * stride_seconds))

    windows: list[WavWindow] = []
    start_frame = 0
    while start_frame < total_frames:
        end_frame = min(total_frames, start_frame + window_frames)
        window_pcm = samples[start_frame:end_frame].astype("<i2", copy=False).tobytes()
        windows.append(
            WavWindow(
                wav_bytes=pcm16_to_wav_bytes(window_pcm, sample_rate),
                start_frame=start_frame,
                end_frame=end_frame,
                sample_rate=sample_rate,
            )
        )
        if end_frame == total_frames:
            break
        start_frame += stride_frames

    return windows


def list_input_devices(sounddevice_module: object | None = None) -> list[InputDevice]:
    """Return available input devices using the same index space as sounddevice."""

    if sounddevice_module is None:
        import sounddevice as sounddevice_module

    devices = sounddevice_module.query_devices()
    default_input = sounddevice_module.default.device[0]
    input_devices: list[InputDevice] = []
    for index, device in enumerate(devices):
        channels = int(device.get("max_input_channels", 0))
        if channels <= 0:
            continue
        input_devices.append(
            InputDevice(
                index=index,
                name=str(device.get("name", "")),
                channels=channels,
                default=index == default_input,
            )
        )
    return input_devices


def format_device_list(devices: list[InputDevice]) -> str:
    lines = ["Available input devices:"]
    if not devices:
        lines.append("  No input devices found.")
        return "\n".join(lines)

    for device in devices:
        default = " (default)" if device.default else ""
        plural = "" if device.channels == 1 else "s"
        lines.append(f"  {device.index}: {device.name} [{device.channels} input channel{plural}]{default}")
    return "\n".join(lines)


def frames_to_timestamp(frames: int, sample_rate: int) -> str:
    seconds = frames / sample_rate
    minutes = int(seconds // 60)
    remaining = seconds - (minutes * 60)
    return f"{minutes:02d}:{remaining:04.1f}"
