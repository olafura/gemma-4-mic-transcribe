"""Small compatibility wrapper around the LiteRT-LM Python API."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import Any


DEFAULT_TRANSCRIBE_PROMPT = (
    "Transcribe the spoken audio exactly. Return only the transcript for this audio window."
)


def text_from_response(response: Any) -> str:
    """Extract text from LiteRT-LM response dictionaries and chunks."""

    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if not isinstance(response, dict):
        return str(response)

    parts: list[str] = []
    for item in response.get("content", []):
        if isinstance(item, dict):
            text = item.get("text")
            if text:
                parts.append(str(text))
        elif isinstance(item, str):
            parts.append(item)
    return "".join(parts)


def _backend(litert_lm: Any, name: str) -> Any:
    factory = getattr(litert_lm.Backend, name.upper())
    return factory()


def _audio_bytes_content(litert_lm: Any, wav_bytes: bytes) -> Any:
    audio_bytes = getattr(litert_lm.Content, "AudioBytes", None)
    if audio_bytes is None:
        raise AttributeError("litert_lm.Content.AudioBytes is not available")

    attempts = (
        lambda: audio_bytes(wav_bytes),
        lambda: audio_bytes(data=wav_bytes),
        lambda: audio_bytes(audio_bytes=wav_bytes),
        lambda: audio_bytes(bytes=wav_bytes),
    )
    errors: list[Exception] = []
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            errors.append(exc)

    raise TypeError(f"Could not construct AudioBytes content: {errors[-1]}")


@contextmanager
def _audio_content(litert_lm: Any, wav_bytes: bytes, allow_file_fallback: bool) -> Iterator[Any]:
    try:
        yield _audio_bytes_content(litert_lm, wav_bytes)
        return
    except (AttributeError, TypeError):
        if not allow_file_fallback:
            raise

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio.write(wav_bytes)
        temp_path = temp_audio.name
    try:
        yield litert_lm.Content.AudioFile(absolute_path=temp_path)
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass


class LiteRtTranscriber:
    """Owns the LiteRT-LM engine and transcribes audio windows."""

    def __init__(
        self,
        model_path: str,
        *,
        backend: str = "gpu",
        audio_backend: str = "cpu",
        system_message: str | None = None,
        prompt: str = DEFAULT_TRANSCRIBE_PROMPT,
        allow_file_fallback: bool = True,
        litert_lm_module: Any | None = None,
    ) -> None:
        self.model_path = os.path.expanduser(model_path)
        self.backend = backend
        self.audio_backend = audio_backend
        self.system_message = system_message
        self.prompt = prompt
        self.allow_file_fallback = allow_file_fallback
        self._litert_lm = litert_lm_module
        self._engine_context: Any | None = None
        self._engine: Any | None = None

    def __enter__(self) -> "LiteRtTranscriber":
        if self._litert_lm is None:
            import litert_lm as litert_lm_module

            self._litert_lm = litert_lm_module

        if hasattr(self._litert_lm, "set_min_log_severity") and hasattr(self._litert_lm, "LogSeverity"):
            self._litert_lm.set_min_log_severity(self._litert_lm.LogSeverity.ERROR)

        engine_kwargs = {
            "backend": _backend(self._litert_lm, self.backend),
            "audio_backend": _backend(self._litert_lm, self.audio_backend),
        }
        self._engine_context = self._litert_lm.Engine(self.model_path, **engine_kwargs)
        self._engine = self._engine_context.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._engine_context is None:
            return None
        return self._engine_context.__exit__(exc_type, exc, traceback)

    def transcribe(self, wav_bytes: bytes) -> str:
        if self._engine is None or self._litert_lm is None:
            raise RuntimeError("LiteRtTranscriber must be used as a context manager")

        messages = []
        if self.system_message:
            messages.append(self._litert_lm.Message.system(self.system_message))

        with _audio_content(self._litert_lm, wav_bytes, self.allow_file_fallback) as audio_content:
            contents = self._litert_lm.Contents.of(self.prompt, audio_content)
            with self._engine.create_conversation(messages=messages) as conversation:
                if hasattr(conversation, "send_message_async"):
                    return "".join(text_from_response(chunk) for chunk in conversation.send_message_async(contents)).strip()
                return text_from_response(conversation.send_message(contents)).strip()
