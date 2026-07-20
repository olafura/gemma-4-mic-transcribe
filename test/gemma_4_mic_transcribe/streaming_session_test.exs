defmodule Gemma4MicTranscribe.StreamingSessionTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.StreamingSession

  defmodule FakeRuntime do
    def load(_opts), do: {:ok, :runtime}

    def generate(:runtime, input, _opts) do
      assert input.prompt =~ "Transcribe."
      assert input.audio.token_count in [1, 50]
      {:ok, "hello world"}
    end
  end

  defmodule FailingRuntime do
    def load(_opts), do: raise("runtime should not load")
  end

  test "silence produces no final event and does not load the runtime" do
    {:ok, session} =
      StreamingSession.start_link(
        runtime_module: FailingRuntime,
        sample_rate: 1_000,
        frame_ms: 20.0,
        speech_threshold: 0.1,
        prompt: "Transcribe."
      )

    assert {:ok, []} = StreamingSession.push_audio(session, List.duplicate(0.0, 200), 0.0)
    assert {:ok, []} = StreamingSession.flush(session)
  end

  test "speech followed by silence emits one self-contained final event" do
    {:ok, session} = start_test_session(partials: false)

    samples = List.duplicate(0.2, 60) ++ List.duplicate(0.0, 60)
    assert {:ok, events} = StreamingSession.push_audio(session, samples, 0.0)

    assert Enum.any?(events, &(&1.type == "speech_start"))

    assert %{
             type: "final",
             text: "hello world",
             stable: true,
             send_to_llm: true,
             start_ms: 0,
             end_ms: 60
           } = Enum.find(events, &(&1.type == "final"))
  end

  test "partial events are unstable and not marked for LLM delivery" do
    {:ok, session} =
      start_test_session(
        speech_end_silence_ms: 1_000.0,
        partial_interval_ms: 40.0,
        partials: true
      )

    assert {:ok, events} = StreamingSession.push_audio(session, List.duplicate(0.2, 100), 0.0)

    assert %{
             type: "partial",
             text: "hello world",
             stable: false,
             send_to_llm: false
           } = Enum.find(events, &(&1.type == "partial"))
  end

  test "recent matching TTS text suppresses a final transcript" do
    {:ok, session} = start_test_session(partials: false)

    assert {:ok, [%{type: "tts"}]} = StreamingSession.push_tts(session, "Hello, world!", 0.0)

    samples = List.duplicate(0.2, 60) ++ List.duplicate(0.0, 60)
    assert {:ok, events} = StreamingSession.push_audio(session, samples, 0.0)

    assert %{
             type: "suppressed",
             reason: "tts_echo",
             text: "hello world",
             send_to_llm: false
           } = Enum.find(events, &(&1.type == "suppressed"))

    refute Enum.any?(events, &(&1.type == "final"))
  end

  defp start_test_session(opts) do
    defaults = [
      runtime_module: FakeRuntime,
      sample_rate: 1_000,
      frame_ms: 20.0,
      speech_threshold: 0.1,
      speech_start_ms: 40.0,
      speech_end_silence_ms: 40.0,
      min_utterance_ms: 20.0,
      speech_min_active_ratio: 0.1,
      prompt: "Transcribe."
    ]

    StreamingSession.start_link(Keyword.merge(defaults, opts))
  end
end
