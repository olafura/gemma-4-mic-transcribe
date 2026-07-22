defmodule Gemma4MicTranscribe.CascadeRuntimeTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.CascadeRuntime

  defmodule FastRuntime do
    def load(opts), do: {:ok, Keyword.get(opts, :fast_text, "fast transcript")}
    def generate(text, _input, _opts), do: {:ok, text}
    def warmup(_runtime, _opts), do: :ok
  end

  defmodule AccurateRuntime do
    def load(_opts), do: {:ok, :accurate}

    def generate(:accurate, _input, _opts) do
      send(:cascade_runtime_test, :accurate_called)
      {:ok, "accurate transcript"}
    end

    def warmup(_runtime, _opts), do: :ok
  end

  setup do
    Process.register(self(), :cascade_runtime_test)
    :ok
  end

  test "accepts a usable fast transcript without calling the accurate model" do
    cascade = load_cascade(fast_text: "fast transcript")

    assert {:ok, "fast transcript"} =
             CascadeRuntime.generate(cascade, %{samples: List.duplicate(0.1, 16_000)})

    refute_received :accurate_called
  end

  test "escalates empty fast transcripts" do
    cascade = load_cascade(fast_text: "")

    assert {:ok, "accurate transcript"} =
             CascadeRuntime.generate(cascade, %{samples: List.duplicate(0.1, 16_000)})

    assert_received :accurate_called
  end

  test "optionally escalates transcripts with implausibly low character density" do
    cascade = load_cascade(fast_text: "hi", cascade_min_chars_per_second: 2.0)

    assert {:ok, "accurate transcript"} =
             CascadeRuntime.generate(cascade, %{samples: List.duplicate(0.1, 32_000)})

    assert_received :accurate_called
  end

  defp load_cascade(opts) do
    opts =
      Keyword.merge(
        [
          model_name: "accurate",
          fast_runtime_module: FastRuntime,
          accurate_runtime_module: AccurateRuntime
        ],
        opts
      )

    assert {:ok, cascade} = CascadeRuntime.load(opts)
    cascade
  end
end
