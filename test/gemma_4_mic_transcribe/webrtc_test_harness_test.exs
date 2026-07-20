defmodule Gemma4MicTranscribe.WebRTCTestHarnessTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.StreamingSession
  alias Gemma4MicTranscribe.WebRTC.TestHarness

  defmodule FakeRuntime do
    def load(_opts), do: {:ok, :runtime}
    def generate(:runtime, _input, _opts), do: {:ok, "hello world"}
  end

  test "forwards f32le audio payloads into the streaming session" do
    {:ok, session} =
      StreamingSession.start_link(
        runtime_module: FakeRuntime,
        sample_rate: 1_000,
        frame_ms: 20.0,
        speech_threshold: 0.1,
        speech_start_ms: 40.0,
        speech_end_silence_ms: 40.0,
        min_utterance_ms: 20.0,
        speech_min_active_ratio: 0.1,
        prompt: "Transcribe.",
        partials: false
      )

    payload =
      (List.duplicate(0.2, 60) ++ List.duplicate(0.0, 60))
      |> Enum.map(&<<&1::little-float-32>>)
      |> IO.iodata_to_binary()

    assert {:ok, events} = TestHarness.push_audio_payload(session, payload, 0.0)
    assert Enum.any?(events, &(&1.type == "final" and &1.text == "hello world"))
  end
end
