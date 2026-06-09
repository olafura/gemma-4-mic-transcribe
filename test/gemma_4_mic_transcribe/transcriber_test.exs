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

  defmodule FailingRuntime do
    def load(_opts), do: raise("runtime should not load")
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
               speech_gate: false,
               runtime_module: FakeRuntime
             )
  end

  test "skips too-short windows before loading the runtime" do
    windows = [
      %Window{
        samples: List.duplicate(0.2, 160),
        start_frame: 0,
        end_frame: 160,
        sample_rate: 16_000
      }
    ]

    assert {:ok, []} =
             Transcriber.transcribe_windows(windows,
               model_name: "google/gemma-4-12B-it",
               prompt: "Transcribe.",
               runtime_module: FailingRuntime
             )
  end

  test "skips silent windows before loading the runtime" do
    windows = [
      %Window{
        samples: List.duplicate(0.0, 16_000),
        start_frame: 0,
        end_frame: 16_000,
        sample_rate: 16_000
      }
    ]

    assert {:ok, []} =
             Transcriber.transcribe_windows(windows,
               model_name: "google/gemma-4-12B-it",
               prompt: "Transcribe.",
               runtime_module: FailingRuntime
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
               speech_gate: false,
               on_window_result: fn {:ok, window, text} ->
                 send(test_pid, {:window_result, window.start_frame, text})
               end
             )

    assert_received {:window_result, 0, "hallo"}
    assert_received {:window_result, 640, "hallo"}
  end
end
