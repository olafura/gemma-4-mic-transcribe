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

  test "emits each window result as soon as it is generated" do
    test_pid = self()

    windows = [
      %Window{
        samples: List.duplicate(0.0, 640),
        start_frame: 0,
        end_frame: 640,
        sample_rate: 16_000
      },
      %Window{
        samples: List.duplicate(0.0, 640),
        start_frame: 640,
        end_frame: 1280,
        sample_rate: 16_000
      }
    ]

    assert {:ok, [{:ok, _, "hallo"}, {:ok, _, "hallo"}]} =
             Transcriber.transcribe_windows(windows,
               model_name: "google/gemma-4-12B-it",
               prompt: "Transcribe.",
               runtime_module: FakeRuntime,
               on_window_result: fn {:ok, window, text} ->
                 send(test_pid, {:window_result, window.start_frame, text})
               end
             )

    assert_received {:window_result, 0, "hallo"}
    assert_received {:window_result, 640, "hallo"}
  end
end
