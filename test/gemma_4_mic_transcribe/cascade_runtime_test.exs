defmodule Gemma4MicTranscribe.CascadeRuntimeTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.CascadeRuntime

  defmodule FastRuntime do
    def load(opts), do: {:ok, Keyword.get(opts, :fast_text, "fast transcript")}
    def generate(text, _input, _opts), do: {:ok, text}
    def warmup(_runtime, _opts), do: :ok
  end

  defmodule AccurateRuntime do
    def load(opts) do
      send(:cascade_runtime_test, {:accurate_load_opts, opts})
      {:ok, :accurate}
    end

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

  def handle_route(event, measurements, metadata, test_pid) do
    send(test_pid, {:route, event, measurements, metadata})
  end

  test "accepts a usable fast transcript without calling the accurate model" do
    cascade = load_cascade(fast_text: "fast transcript")

    assert {:ok, "fast transcript"} =
             CascadeRuntime.generate(cascade, %{samples: List.duplicate(0.1, 16_000)})

    refute_received :accurate_called

    assert %{requests: 1, accepted: 1, escalated: 0, fast_errors: 0} =
             CascadeRuntime.stats(cascade)
  end

  test "uses compatibility-only ROCm preflight for the second model load" do
    _cascade = load_cascade(fast_text: "fast transcript")

    assert_received {:accurate_load_opts, opts}
    assert opts[:rocm_preflight] == :compatibility_only
  end

  test "escalates empty fast transcripts" do
    cascade = load_cascade(fast_text: "")

    assert {:ok, "accurate transcript"} =
             CascadeRuntime.generate(cascade, %{samples: List.duplicate(0.1, 16_000)})

    assert_received :accurate_called
    assert %{requests: 1, accepted: 0, escalated: 1} = CascadeRuntime.stats(cascade)
  end

  test "emits route telemetry with model timings" do
    test_pid = self()
    handler_id = "cascade-route-#{System.unique_integer()}"

    :ok =
      :telemetry.attach(
        handler_id,
        [:gemma_4_mic_transcribe, :cascade, :route],
        &__MODULE__.handle_route/4,
        test_pid
      )

    on_exit(fn -> :telemetry.detach(handler_id) end)
    cascade = load_cascade(fast_text: "fast transcript")
    assert {:ok, _text} = CascadeRuntime.generate(cascade, %{samples: []})

    assert_received {:route, [:gemma_4_mic_transcribe, :cascade, :route],
                     %{fast_ms: fast_ms, accurate_ms: 0}, %{route: :fast, reason: nil}}

    assert is_integer(fast_ms) and fast_ms >= 0
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
