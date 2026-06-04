"""Command-line interface for microphone transcription."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

from .audio import RollingPcmBuffer, format_device_list, frames_to_timestamp, list_input_devices, pcm16_to_wav_bytes
from .litert import DEFAULT_TRANSCRIBE_PROMPT, LiteRtTranscriber


DEFAULT_MODEL = "~/.litert-lm/models/gemma4-12b/model.litertlm"


@dataclass(frozen=True)
class RunConfig:
    model: str
    device: int | None
    system_message: str | None
    prompt: str
    window_seconds: float
    stride_seconds: float
    sample_rate: int
    backend: str
    audio_backend: str
    no_file_fallback: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gemma-4-mic-transcribe",
        description="Transcribe microphone audio locally with Gemma 4 and LiteRT-LM.",
    )
    parser.add_argument("--list-devices", action="store_true", help="List available audio input devices and exit")
    parser.add_argument("--device", type=int, default=None, help="Input device ID from --list-devices")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Path to model.litertlm (default: {DEFAULT_MODEL})")
    parser.add_argument("--system-message", default=None, help="System instruction sent with every audio window")
    parser.add_argument("--prompt", default=DEFAULT_TRANSCRIBE_PROMPT, help="User prompt paired with each audio window")
    parser.add_argument("--window-seconds", type=float, default=5.0, help="Rolling audio window to send per request")
    parser.add_argument("--stride-seconds", type=float, default=2.5, help="Seconds between requests")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Microphone sample rate to request")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu", help="LiteRT-LM text backend")
    parser.add_argument("--audio-backend", choices=("cpu", "gpu"), default="cpu", help="LiteRT-LM audio backend")
    parser.add_argument(
        "--no-file-fallback",
        action="store_true",
        help="Fail instead of falling back to temporary AudioFile input if AudioBytes is unavailable",
    )
    return parser


def validate_args(args: argparse.Namespace) -> RunConfig:
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be positive")
    if args.stride_seconds <= 0:
        raise ValueError("--stride-seconds must be positive")
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")

    return RunConfig(
        model=os.path.expanduser(args.model),
        device=args.device,
        system_message=args.system_message,
        prompt=args.prompt,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        sample_rate=args.sample_rate,
        backend=args.backend,
        audio_backend=args.audio_backend,
        no_file_fallback=args.no_file_fallback,
    )


def _make_callback(buffer: RollingPcmBuffer):
    def callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"audio warning: {status}", file=sys.stderr)

        import numpy as np

        mono = indata[:, 0] if getattr(indata, "ndim", 1) > 1 else indata
        pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2", copy=False).tobytes()
        buffer.append(pcm16)

    return callback


def run_transcription(config: RunConfig) -> int:
    if not os.path.exists(config.model):
        print(f"Model not found: {config.model}", file=sys.stderr)
        print("Run scripts/setup.sh for install/import guidance.", file=sys.stderr)
        return 2

    import sounddevice as sd

    buffer = RollingPcmBuffer(config.sample_rate, config.window_seconds)
    stride_frames = max(1, int(config.sample_rate * config.stride_seconds))
    next_send_at = stride_frames

    with LiteRtTranscriber(
        config.model,
        backend=config.backend,
        audio_backend=config.audio_backend,
        system_message=config.system_message,
        prompt=config.prompt,
        allow_file_fallback=not config.no_file_fallback,
    ) as transcriber:
        with sd.InputStream(
            samplerate=config.sample_rate,
            device=config.device,
            channels=1,
            dtype="float32",
            callback=_make_callback(buffer),
        ):
            print("Recording. Press Ctrl+C to stop.", file=sys.stderr)
            while True:
                if not buffer.wait_until_total_frames(next_send_at):
                    continue

                pcm16, start_frame, end_frame = buffer.snapshot()
                if not pcm16:
                    continue

                wav_bytes = pcm16_to_wav_bytes(pcm16, config.sample_rate)
                transcript = transcriber.transcribe(wav_bytes)
                if transcript:
                    start = frames_to_timestamp(start_frame, config.sample_rate)
                    end = frames_to_timestamp(end_frame, config.sample_rate)
                    print(f"[{start}-{end}] {transcript}", flush=True)

                next_send_at = max(next_send_at + stride_frames, buffer.total_frames + stride_frames)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        print(format_device_list(list_input_devices()))
        return 0

    try:
        config = validate_args(args)
        return run_transcription(config)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
