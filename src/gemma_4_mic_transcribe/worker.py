"""Run LiteRT-LM inference in a child process so Ctrl+C can stop promptly."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass

from .litert import LiteRtTranscriber


@dataclass(frozen=True)
class WorkerConfig:
    model: str
    backend: str
    audio_backend: str
    system_message: str | None
    prompt: str
    audio_position: str
    debug_response: bool


def _worker_main(config: WorkerConfig, requests: mp.Queue, responses: mp.Queue) -> None:
    try:
        with LiteRtTranscriber(
            config.model,
            backend=config.backend,
            audio_backend=config.audio_backend,
            system_message=config.system_message,
            prompt=config.prompt,
            audio_position=config.audio_position,
            debug_response=config.debug_response,
        ) as transcriber:
            responses.put(("ready", None))
            while True:
                job = requests.get()
                if job is None:
                    return

                job_id, wav_bytes = job
                try:
                    for chunk in transcriber.transcribe_chunks(wav_bytes):
                        responses.put(("chunk", job_id, chunk))
                    responses.put(("done", job_id))
                except BaseException as exc:
                    responses.put(("error", job_id, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
    except BaseException as exc:
        responses.put(("fatal", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


class InferenceWorker:
    """Own a LiteRT-LM child process and synchronous request/response API."""

    def __init__(self, config: WorkerConfig) -> None:
        self._context = mp.get_context("spawn")
        self._requests = self._context.Queue()
        self._responses = self._context.Queue()
        self._process = self._context.Process(
            target=_worker_main,
            args=(config, self._requests, self._responses),
            daemon=True,
        )
        self._next_job_id = 0

    def __enter__(self) -> "InferenceWorker":
        self._process.start()
        try:
            self._wait_until_ready()
        except BaseException:
            self.close(terminate=True)
            raise
        return self

    def __exit__(self, exc_type, exc, traceback_obj) -> None:
        self.close(terminate=exc_type is KeyboardInterrupt)

    def transcribe_chunks(self, wav_bytes: bytes, timeout_seconds: float | None = None) -> Iterator[str]:
        if not self._process.is_alive():
            raise RuntimeError("LiteRT-LM worker is not running")

        self._next_job_id += 1
        job_id = self._next_job_id
        self._requests.put((job_id, wav_bytes))
        deadline = time.monotonic() + timeout_seconds if timeout_seconds else None

        while True:
            self._raise_if_dead()
            if deadline is not None and time.monotonic() >= deadline:
                self.close(terminate=True)
                raise TimeoutError(f"LiteRT-LM request timed out after {timeout_seconds:g} seconds")
            try:
                message = self._responses.get(timeout=0.1)
            except queue.Empty:
                continue

            kind = message[0]
            if kind == "chunk" and message[1] == job_id:
                yield message[2]
                continue
            if kind == "done" and message[1] == job_id:
                return
            if kind == "error" and message[1] == job_id:
                raise RuntimeError(message[2])
            if kind == "fatal":
                raise RuntimeError(message[1])

    def transcribe(self, wav_bytes: bytes, timeout_seconds: float | None = None) -> str:
        return "".join(self.transcribe_chunks(wav_bytes, timeout_seconds=timeout_seconds)).strip()

    def close(self, *, terminate: bool = False) -> None:
        if not self._process.is_alive():
            self._process.join(timeout=0.2)
            return

        if terminate:
            self._process.terminate()
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2)
            return

        self._requests.put(None)
        self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)

    def _wait_until_ready(self) -> None:
        while True:
            self._raise_if_dead()
            try:
                message = self._responses.get(timeout=0.1)
            except queue.Empty:
                continue

            if message[0] == "ready":
                return
            if message[0] == "fatal":
                raise RuntimeError(message[1])

    def _raise_if_dead(self) -> None:
        if self._process.is_alive():
            return
        exitcode = self._process.exitcode
        raise RuntimeError(f"LiteRT-LM worker exited unexpectedly with code {exitcode}")
