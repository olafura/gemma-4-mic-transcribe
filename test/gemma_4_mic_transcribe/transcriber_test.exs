defmodule Gemma4MicTranscribe.TranscriberTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Audio.Window
  alias Gemma4MicTranscribe.Transcriber

  defmodule FakeRuntime do
    def load(_opts), do: {:ok, :runtime}

    def generate(:runtime, input, _opts) do
      assert input.prompt =~ "Transcribe."
      assert input.audio.token_count == 1
      {:ok, "hallo"}
    end
  end

  test "transcribes windows through an injected runtime" do
    windows = [
      %Window{
        samples: List.duplicate(0.0, 640),
        start_frame: 0,
        end_frame: 640,
        sample_rate: 16_000
      }
    ]

    assert {:ok, [{:ok, _window, "hallo"}]} =
             Transcriber.transcribe_windows(windows,
               model_name: "google/gemma-4-12B-it",
               prompt: "Transcribe.",
               runtime_module: FakeRuntime
             )
  end
end
