"""Small compatibility wrapper around the LiteRT-LM Python API."""

from __future__ import annotations

import os
from collections.abc import Iterator
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
    if isinstance(response, list | tuple):
        return "".join(text_from_response(item) for item in response)
    if not isinstance(response, dict):
        return str(response)

    direct_text = response.get("text")
    if direct_text:
        return str(direct_text)

    content = response.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return text_from_response(content)

    parts: list[str] = []
    for item in content or []:
        if isinstance(item, dict):
            parts.append(text_from_response(item))
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
    return audio_bytes(bytes=wav_bytes)


class LiteRtTranscriber:
    """Owns the LiteRT-LM engine and transcribes audio windows."""

    def __init__(
        self,
        model_path: str,
        *,
        backend: str = "gpu",
        audio_backend: str = "gpu",
        system_message: str | None = None,
        prompt: str = DEFAULT_TRANSCRIBE_PROMPT,
        audio_position: str = "before",
        debug_response: bool = False,
        litert_lm_module: Any | None = None,
    ) -> None:
        self.model_path = os.path.expanduser(model_path)
        self.backend = backend
        self.audio_backend = audio_backend
        self.system_message = system_message
        self.prompt = prompt
        if audio_position not in {"before", "after"}:
            raise ValueError("audio_position must be 'before' or 'after'")
        self.audio_position = audio_position
        self.debug_response = debug_response
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

    def transcribe_chunks(self, wav_bytes: bytes) -> Iterator[str]:
        if self._engine is None or self._litert_lm is None:
            raise RuntimeError("LiteRtTranscriber must be used as a context manager")

        conversation_kwargs = {}
        if self.system_message:
            conversation_kwargs["system_message"] = self.system_message

        audio_content = _audio_bytes_content(self._litert_lm, wav_bytes)
        if self.audio_position == "after":
            contents = self._litert_lm.Contents.of(self.prompt, audio_content)
        else:
            contents = self._litert_lm.Contents.of(audio_content, self.prompt)
        with self._engine.create_conversation(**conversation_kwargs) as conversation:
            for chunk in conversation.send_message_async(contents):
                if self.debug_response:
                    print(f"LiteRT-LM raw chunk: {chunk!r}", file=__import__("sys").stderr, flush=True)
                text = text_from_response(chunk)
                if text:
                    yield text

    def transcribe(self, wav_bytes: bytes) -> str:
        return "".join(self.transcribe_chunks(wav_bytes)).strip()
