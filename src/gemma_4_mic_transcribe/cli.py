"""Command-line interface for microphone transcription."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

from .audio import (
    RollingPcmBuffer,
    format_device_list,
    frames_to_timestamp,
    iter_wav_windows,
    list_input_devices,
    pcm16_to_wav_bytes,
)
from .litert import DEFAULT_TRANSCRIBE_PROMPT
from .worker import InferenceWorker, WorkerConfig


DEFAULT_MODEL = "~/.litert-lm/models/gemma4-12b/model.litertlm"


@dataclass(frozen=True)
class RunConfig:
    model: str
    wav: str | None
    skip_windows: int
    max_windows: int | None
    device: int | None
    system_message: str | None
    prompt: str
    window_seconds: float
    stride_seconds: float
    sample_rate: int
    backend: str
    audio_backend: str
    audio_position: str
    request_timeout_seconds: float | None
    debug_response: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gemma-4-mic-transcribe",
        description="Transcribe microphone audio locally with Gemma 4 and LiteRT-LM.",
    )
    parser.add_argument("--list-devices", action="store_true", help="List available audio input devices and exit")
    parser.add_argument("--wav", default=None, help="Transcribe a WAV file once and exit")
    parser.add_argument("--skip-windows", type=int, default=0, help="Skip the first N WAV windows before transcribing")
    parser.add_argument("--max-windows", type=int, default=None, help="Limit WAV transcription to the first N windows")
    parser.add_argument("--device", type=int, default=None, help="Input device ID from --list-devices")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Path to model.litertlm (default: {DEFAULT_MODEL})")
    parser.add_argument("--system-message", default=None, help="System instruction sent with every audio window")
    parser.add_argument("--system-message-file", default=None, help="Read the system instruction from a UTF-8 text file")
    parser.add_argument("--prompt", default=DEFAULT_TRANSCRIBE_PROMPT, help="User prompt paired with each audio window")
    parser.add_argument("--window-seconds", type=float, default=5.0, help="Rolling audio window to send per request")
    parser.add_argument("--stride-seconds", type=float, default=2.5, help="Seconds between requests")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Microphone sample rate to request")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu", help="LiteRT-LM text backend")
    parser.add_argument("--audio-backend", choices=("cpu", "gpu"), default="gpu", help="LiteRT-LM audio backend")
    parser.add_argument(
        "--audio-position",
        choices=("before", "after"),
        default="before",
        help="Place AudioBytes before or after the text prompt",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=30.0,
        help="Maximum seconds to wait for one audio window; set 0 to disable",
    )
    parser.add_argument("--debug-response", action="store_true", help="Print raw LiteRT-LM response data to stderr")
    return parser


def validate_args(args: argparse.Namespace) -> RunConfig:
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be positive")
    if args.stride_seconds <= 0:
        raise ValueError("--stride-seconds must be positive")
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    if args.request_timeout_seconds is not None and args.request_timeout_seconds < 0:
        raise ValueError("--request-timeout-seconds must be zero or positive")

    system_message = args.system_message
    if args.system_message_file:
        if system_message:
            raise ValueError("use either --system-message or --system-message-file, not both")
        with open(os.path.expanduser(args.system_message_file), encoding="utf-8") as system_file:
            system_message = system_file.read().strip()

    wav = os.path.expanduser(args.wav) if args.wav else None
    if wav and not os.path.exists(wav):
        raise ValueError(f"--wav file not found: {wav}")
    if args.skip_windows < 0:
        raise ValueError("--skip-windows must be zero or positive")
    if args.max_windows is not None and args.max_windows <= 0:
        raise ValueError("--max-windows must be positive")

    return RunConfig(
        model=os.path.expanduser(args.model),
        wav=wav,
        skip_windows=args.skip_windows,
        max_windows=args.max_windows,
        device=args.device,
        system_message=system_message,
        prompt=args.prompt,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        sample_rate=args.sample_rate,
        backend=args.backend,
        audio_backend=args.audio_backend,
        audio_position=args.audio_position,
        request_timeout_seconds=args.request_timeout_seconds or None,
        debug_response=args.debug_response,
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

    worker_config = WorkerConfig(
        config.model,
        backend=config.backend,
        audio_backend=config.audio_backend,
        system_message=config.system_message,
        prompt=config.prompt,
        audio_position=config.audio_position,
        debug_response=config.debug_response,
    )

    print(
        f"Using LiteRT-LM backends: text={config.backend}, audio={config.audio_backend}",
        file=sys.stderr,
        flush=True,
    )

    if config.wav:
        windows = iter_wav_windows(
            config.wav,
            config.window_seconds,
            config.stride_seconds,
            target_sample_rate=config.sample_rate,
        )
        if config.skip_windows:
            windows = windows[config.skip_windows :]
        if config.max_windows is not None:
            windows = windows[: config.max_windows]

        print(
            f"Transcribing {config.wav} as {len(windows)} AudioBytes window(s)...",
            file=sys.stderr,
            flush=True,
        )
        if not windows:
            print("No WAV windows selected.", file=sys.stderr, flush=True)
            return 3

        with InferenceWorker(worker_config) as transcriber:
            returned_anything = False
            for index, window in enumerate(windows, start=1):
                start = frames_to_timestamp(window.start_frame, window.sample_rate)
                end = frames_to_timestamp(window.end_frame, window.sample_rate)
                print(f"Sending window {index}/{len(windows)} [{start}-{end}]...", file=sys.stderr, flush=True)
                print(f"[{start}-{end}] ", end="", flush=True)
                saw_text = False
                for chunk in transcriber.transcribe_chunks(
                    window.wav_bytes,
                    timeout_seconds=config.request_timeout_seconds,
                ):
                    saw_text = True
                    returned_anything = True
                    print(chunk, end="", flush=True)
                if saw_text:
                    print(flush=True)
                else:
                    print("<no transcript>", flush=True)

        return 0 if returned_anything else 3

    import sounddevice as sd

    buffer = RollingPcmBuffer(config.sample_rate, config.window_seconds)
    stride_frames = max(1, int(config.sample_rate * config.stride_seconds))
    next_send_at = stride_frames

    with InferenceWorker(worker_config) as transcriber:
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
                start = frames_to_timestamp(start_frame, config.sample_rate)
                end = frames_to_timestamp(end_frame, config.sample_rate)
                saw_text = False
                for chunk in transcriber.transcribe_chunks(
                    wav_bytes,
                    timeout_seconds=config.request_timeout_seconds,
                ):
                    if not saw_text:
                        print(f"[{start}-{end}] ", end="", flush=True)
                    saw_text = True
                    print(chunk, end="", flush=True)
                if saw_text:
                    print(flush=True)

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
