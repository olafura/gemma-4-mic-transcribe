defmodule Gemma4MicTranscribe.Transcriber do
  @moduledoc false

  require Logger

  alias Gemma4MicTranscribe.Gemma4Unified.Input
  alias Gemma4MicTranscribe.Gemma4Unified.Runtime

  def transcribe_windows(windows, opts \\ []) do
    runtime_module = Keyword.get(opts, :runtime_module, Runtime)
    total_windows = length(windows)

    debug(opts, fn ->
      "transcriber: loading runtime module=#{inspect(runtime_module)} model=#{inspect(Keyword.get(opts, :model_name))} " <>
        "backend=#{inspect(Keyword.get(opts, :backend))} windows=#{total_windows}"
    end)

    with {:ok, runtime} <-
           timed_debug(opts, "transcriber: runtime load", fn -> runtime_module.load(opts) end) do
      windows
      |> Enum.with_index(1)
      |> Enum.map(fn {window, index} ->
        debug(opts, fn ->
          "transcriber: window #{index}/#{total_windows} samples=#{length(window.samples)} " <>
            "frames=#{window.start_frame}..#{window.end_frame}"
        end)

        input =
          timed_debug(opts, "transcriber: window #{index}/#{total_windows} input build", fn ->
            Input.build(window.samples,
              prompt: Keyword.fetch!(opts, :prompt),
              system_message: Keyword.get(opts, :system_message)
            )
          end)

        debug(opts, fn ->
          "transcriber: window #{index}/#{total_windows} prompt_bytes=#{byte_size(input.prompt)} " <>
            "audio_tokens=#{input.audio.token_count}"
        end)

        case timed_debug(opts, "transcriber: window #{index}/#{total_windows} generate", fn ->
               runtime_module.generate(runtime, input,
                 timeout_seconds: Keyword.get(opts, :request_timeout_seconds)
               )
             end) do
          {:ok, text} ->
            debug(opts, fn ->
              "transcriber: window #{index}/#{total_windows} transcript_bytes=#{byte_size(text)}"
            end)

            {:ok, window, String.trim(text)}

          {:error, reason} ->
            {:error, reason}
        end
      end)
      |> collect_results()
    end
  end

  defp timed_debug(opts, label, fun) do
    if Keyword.get(opts, :debug, false) do
      started_at = System.monotonic_time(:millisecond)
      Logger.debug("#{label}: start")

      result = fun.()

      elapsed_ms = System.monotonic_time(:millisecond) - started_at

      case result do
        {:error, reason} ->
          Logger.debug("#{label}: error after #{elapsed_ms}ms reason=#{inspect(reason)}")

        _ ->
          Logger.debug("#{label}: done in #{elapsed_ms}ms")
      end

      result
    else
      fun.()
    end
  end

  defp debug(opts, message_fun) do
    if Keyword.get(opts, :debug, false) do
      Logger.debug(message_fun)
    end
  end

  defp collect_results(results) do
    case Enum.find(results, &match?({:error, _reason}, &1)) do
      nil -> {:ok, results}
      {:error, reason} -> {:error, reason}
    end
  end
end
