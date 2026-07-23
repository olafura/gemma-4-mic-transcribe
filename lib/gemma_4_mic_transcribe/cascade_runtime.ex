defmodule Gemma4MicTranscribe.CascadeRuntime do
  @moduledoc false

  require Logger

  alias Gemma4MicTranscribe.Gemma4Unified.Runtime
  alias Gemma4MicTranscribe.StreamingSession

  defstruct [
    :fast_module,
    :fast_runtime,
    :accurate_module,
    :accurate_runtime,
    :min_chars_per_second,
    :min_logit_margin,
    :min_handoff_confidence,
    :sample_rate,
    :counters
  ]

  @accepted 1
  @escalated 2
  @fast_errors 3
  @fast_ms 4
  @accurate_ms 5

  def load(opts) do
    fast_module = Keyword.get(opts, :fast_runtime_module, Runtime)
    accurate_module = Keyword.get(opts, :accurate_runtime_module, Runtime)

    default_fast_model =
      if Keyword.get(opts, :handoff_probe_artifact), do: "gemma4-e2b", else: "gemma4-e4b"

    fast_opts =
      opts
      |> Keyword.put(
        :model_name,
        Keyword.get(opts, :cascade_fast_model_name) || default_fast_model
      )
      |> Keyword.put(:fused_ffn, false)

    # The first load performs the full VRAM safety check. Once EXLA has created
    # its allocator, rocm-smi reports that allocator's reservation as used VRAM;
    # the second load must not mistake our own reservation for another workload.
    accurate_opts =
      opts
      |> Keyword.delete(:handoff_probe_artifact)
      |> Keyword.delete(:capture_layer)
      |> Keyword.put(:rocm_preflight, :compatibility_only)

    with {:ok, fast_runtime} <- fast_module.load(fast_opts),
         {:ok, accurate_runtime} <- accurate_module.load(accurate_opts) do
      {:ok,
       %__MODULE__{
         fast_module: fast_module,
         fast_runtime: fast_runtime,
         accurate_module: accurate_module,
         accurate_runtime: accurate_runtime,
         min_chars_per_second: Keyword.get(opts, :cascade_min_chars_per_second, 0.0),
         min_logit_margin: Keyword.get(opts, :cascade_min_logit_margin, 0.0),
         min_handoff_confidence: Keyword.get(opts, :cascade_min_handoff_confidence, 0.0),
         sample_rate: Keyword.get(opts, :sample_rate, 16_000),
         counters: :atomics.new(5, [])
       }}
    end
  end

  def warmup(%__MODULE__{} = cascade, opts) do
    with :ok <- maybe_warmup(cascade.fast_module, cascade.fast_runtime, opts),
         :ok <- maybe_warmup(cascade.accurate_module, cascade.accurate_runtime, opts) do
      :ok
    end
  end

  def generate(%__MODULE__{} = cascade, input, opts \\ []) do
    {fast_result, fast_ms} =
      timed(fn -> generate_fast(cascade, input, opts) end)

    :atomics.add(cascade.counters, @fast_ms, fast_ms)

    case fast_result do
      {:ok, text, confidence} ->
        input = Map.put_new(input, :sample_rate, cascade.sample_rate)

        case escalation_reason(
               text,
               input,
               cascade.min_chars_per_second,
               confidence,
               cascade.min_logit_margin,
               cascade.min_handoff_confidence
             ) do
          nil ->
            :atomics.add(cascade.counters, @accepted, 1)
            Logger.info("cascade: accepting fast transcript #{format_confidence(confidence)}")
            emit_route(:fast, nil, fast_ms, 0, confidence)
            {:ok, text}

          reason ->
            Logger.info(
              "cascade: escalating fast transcript to accurate model reason=#{reason} " <>
                format_confidence(confidence)
            )

            run_accurate(cascade, input, opts, reason, fast_ms, confidence)
        end

      {:error, reason} ->
        Logger.warning("cascade: fast model failed, escalating reason=#{inspect(reason)}")
        :atomics.add(cascade.counters, @fast_errors, 1)
        run_accurate(cascade, input, opts, :fast_error, fast_ms, nil)
    end
  end

  def stats(%__MODULE__{} = cascade) do
    accepted = :atomics.get(cascade.counters, @accepted)
    escalated = :atomics.get(cascade.counters, @escalated)

    %{
      accepted: accepted,
      escalated: escalated,
      fast_errors: :atomics.get(cascade.counters, @fast_errors),
      fast_ms: :atomics.get(cascade.counters, @fast_ms),
      accurate_ms: :atomics.get(cascade.counters, @accurate_ms),
      requests: accepted + escalated
    }
  end

  @doc false
  def escalate?(text, input, min_chars_per_second) when is_binary(text) do
    escalation_reason(text, input, min_chars_per_second, nil, 0.0, 0.0) != nil
  end

  defp escalation_reason(
         text,
         input,
         min_chars_per_second,
         confidence,
         min_logit_margin,
         min_handoff_confidence
       ) do
    normalized = String.trim(text)

    cond do
      normalized == "" -> :empty
      StreamingSession.refusal?(normalized) -> :refusal
      String.contains?(normalized, ["<|", "|>"]) -> :malformed_control_token
      String.contains?(normalized, "�") -> :replacement_character
      below_density?(normalized, input, min_chars_per_second) -> :low_character_density
      low_logit_margin?(confidence, min_logit_margin) -> :low_logit_margin
      low_handoff_confidence?(confidence, min_handoff_confidence) -> :low_handoff_confidence
      true -> nil
    end
  end

  defp below_density?(_text, _input, threshold) when threshold <= 0, do: false

  defp below_density?(text, input, threshold) do
    samples = Map.get(input, :samples, [])
    sample_rate = Map.get(input, :sample_rate, 16_000)
    duration_seconds = length(samples) / sample_rate

    duration_seconds >= 1.0 and String.length(text) / duration_seconds < threshold
  end

  defp maybe_warmup(module, runtime, opts) do
    if function_exported?(module, :warmup, 2), do: module.warmup(runtime, opts), else: :ok
  end

  defp generate_fast(cascade, input, opts) do
    if confidence_required?(cascade) do
      if function_exported?(cascade.fast_module, :generate_with_confidence, 3) do
        case cascade.fast_module.generate_with_confidence(cascade.fast_runtime, input, opts) do
          {:ok, %{text: text, confidence: confidence}} -> {:ok, text, confidence}
          {:error, reason} -> {:error, reason}
        end
      else
        {:error, :confidence_unavailable}
      end
    else
      case cascade.fast_module.generate(cascade.fast_runtime, input, opts) do
        {:ok, text} -> {:ok, text, nil}
        {:error, reason} -> {:error, reason}
      end
    end
  end

  defp confidence_required?(cascade) do
    cascade.min_logit_margin > 0 or cascade.min_handoff_confidence > 0
  end

  defp low_logit_margin?(_confidence, threshold) when threshold <= 0, do: false
  defp low_logit_margin?(nil, _threshold), do: true

  defp low_logit_margin?(confidence, threshold) do
    Map.get(confidence, :min_logit_margin, 0.0) < threshold
  end

  defp low_handoff_confidence?(_confidence, threshold) when threshold <= 0, do: false
  defp low_handoff_confidence?(nil, _threshold), do: true

  defp low_handoff_confidence?(confidence, threshold) do
    case Map.get(confidence, :handoff_confidence) do
      value when is_number(value) -> value < threshold
      _other -> true
    end
  end

  defp run_accurate(cascade, input, opts, reason, fast_ms, confidence) do
    {result, accurate_ms} =
      timed(fn -> cascade.accurate_module.generate(cascade.accurate_runtime, input, opts) end)

    :atomics.add(cascade.counters, @escalated, 1)
    :atomics.add(cascade.counters, @accurate_ms, accurate_ms)
    emit_route(:accurate, reason, fast_ms, accurate_ms, confidence)
    result
  end

  defp emit_route(route, reason, fast_ms, accurate_ms, confidence) do
    :telemetry.execute(
      [:gemma_4_mic_transcribe, :cascade, :route],
      %{fast_ms: fast_ms, accurate_ms: accurate_ms},
      %{route: route, reason: reason, confidence: confidence}
    )
  end

  defp format_confidence(nil), do: "confidence=unavailable"

  defp format_confidence(confidence) do
    handoff = Map.get(confidence, :handoff_confidence)
    margin = Map.get(confidence, :min_logit_margin)

    "handoff_confidence=#{format_number(handoff)} min_logit_margin=#{format_number(margin)}"
  end

  defp format_number(value) when is_number(value),
    do: :io_lib.format("~.4f", [value]) |> IO.iodata_to_binary()

  defp format_number(_value), do: "unavailable"

  defp timed(fun) do
    started_at = System.monotonic_time(:millisecond)
    result = fun.()
    {result, System.monotonic_time(:millisecond) - started_at}
  end
end
