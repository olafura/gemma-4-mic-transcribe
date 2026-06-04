import unittest

from gemma_4_mic_transcribe.litert import LiteRtTranscriber, text_from_response


class FakeBackend:
    @staticmethod
    def GPU():
        return "gpu-backend"

    @staticmethod
    def CPU():
        return "cpu-backend"


class FakeMessage:
    @staticmethod
    def system(text):
        return {"role": "system", "content": text}


class FakeAudioBytes:
    def __init__(self, data):
        self.data = data


class FakeContent:
    AudioBytes = FakeAudioBytes


class FakeContents:
    last_args = None

    @staticmethod
    def of(*args):
        FakeContents.last_args = args
        return {"contents": args}


class FakeConversation:
    messages = None

    def __init__(self, messages):
        FakeConversation.messages = messages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def send_message_async(self, contents):
        yield {"content": [{"type": "text", "text": "hello "}]}
        yield {"content": [{"type": "text", "text": "world"}]}


class FakeEngineContext:
    kwargs = None

    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        FakeEngineContext.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def create_conversation(self, messages=None):
        return FakeConversation(messages)


class FakeLiteRtLm:
    Backend = FakeBackend
    Message = FakeMessage
    Content = FakeContent
    Contents = FakeContents
    Engine = FakeEngineContext


class LiteRtTests(unittest.TestCase):
    def test_text_from_response(self):
        self.assertEqual(text_from_response({"content": [{"text": "a"}, {"text": "b"}]}), "ab")
        self.assertEqual(text_from_response("plain"), "plain")

    def test_transcriber_uses_audio_bytes_and_system_message(self):
        with LiteRtTranscriber(
            "/tmp/model.litertlm",
            system_message="You transcribe.",
            litert_lm_module=FakeLiteRtLm,
        ) as transcriber:
            text = transcriber.transcribe(b"RIFF")

        self.assertEqual(text, "hello world")
        self.assertEqual(FakeEngineContext.kwargs["backend"], "gpu-backend")
        self.assertEqual(FakeEngineContext.kwargs["audio_backend"], "cpu-backend")
        self.assertEqual(FakeConversation.messages, [{"role": "system", "content": "You transcribe."}])
        self.assertIsInstance(FakeContents.last_args[1], FakeAudioBytes)
        self.assertEqual(FakeContents.last_args[1].data, b"RIFF")


if __name__ == "__main__":
    unittest.main()
