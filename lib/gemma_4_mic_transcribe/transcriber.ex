defmodule Gemma4MicTranscribe.Transcriber do
  @moduledoc false

  require Logger

  alias Gemma4MicTranscribe.Gemma4Unified.Input
  alias Gemma4MicTranscribe.Gemma4Unified.Runtime
  alias Gemma4MicTranscribe.SpeechGate

  def transcribe_windows(windows, opts \\ []) do
    runtime_module = Keyword.get(opts, :runtime_module, Runtime)
    total_windows = length(windows)
    selected_windows = speech_windows(windows, opts)

    debug(opts, fn ->
      "transcriber: loading runtime module=#{inspect(runtime_module)} model=#{inspect(Keyword.get(opts, :model_name))} " <>
        "backend=#{inspect(Keyword.get(opts, :backend))} windows=#{length(selected_windows)}/#{total_windows}"
    end)

    case selected_windows do
      [] ->
        debug(opts, fn -> "transcriber: speech gate selected no windows; runtime load skipped" end)

        {:ok, []}

      selected_windows ->
        with {:ok, runtime} <-
               timed_debug(opts, "transcriber: runtime load", fn -> runtime_module.load(opts) end) do
          results =
            selected_windows
            |> Enum.reduce_while([], fn {window, index}, acc ->
              case transcribe_window(runtime_module, runtime, window, index, total_windows, opts) do
                {:ok, _window, _text} = result ->
                  emit_window_result(result, opts)
                  {:cont, [result | acc]}

                {:error, reason} ->
                  {:halt, {:error, reason}}
              end
            end)

          case results do
            {:error, reason} -> {:error, reason}
            results -> {:ok, Enum.reverse(results)}
          end
        end
    end
  end

  defp speech_windows(windows, opts) do
    windows
    |> Enum.with_index(1)
    |> Enum.filter(fn {window, index} -> speech_window?(window, index, length(windows), opts) end)
  end

  defp speech_window?(window, index, total_windows, opts) do
    if Keyword.get(opts, :speech_gate, true) do
      analysis =
        SpeechGate.analyze(window.samples,
          sample_rate: window.sample_rate,
          min_speech_seconds: Keyword.get(opts, :min_speech_seconds, 0.25),
          threshold: Keyword.get(opts, :speech_threshold, 0.01),
          min_active_ratio: Keyword.get(opts, :speech_min_active_ratio, 0.2),
          max_zero_crossing_rate: Keyword.get(opts, :speech_max_zero_crossing_rate, 0.35)
        )

      debug(opts, fn ->
        "transcriber: window #{index}/#{total_windows} speech_gate=#{if analysis.speech?, do: "pass", else: "skip"} " <>
          "reason=#{analysis.reason} samples=#{analysis.sample_count} min_samples=#{analysis.min_samples} " <>
          "rms=#{format_float(analysis.rms)} peak=#{format_float(analysis.peak)} " <>
          "active_ratio=#{format_float(analysis.active_ratio)} zcr=#{format_float(analysis.zero_crossing_rate)}"
      end)

      analysis.speech?
    else
      true
    end
  end

  defp transcribe_window(runtime_module, runtime, window, index, total_windows, opts) do
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
  end

  defp emit_window_result({:ok, _window, _text} = result, opts) do
    case Keyword.get(opts, :on_window_result) do
      callback when is_function(callback, 1) -> callback.(result)
      _other -> :ok
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

  defp format_float(value) do
    :io_lib.format("~.5f", [value]) |> IO.iodata_to_binary()
  end
end
