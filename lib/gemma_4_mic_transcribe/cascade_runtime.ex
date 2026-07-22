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
    :sample_rate
  ]

  def load(opts) do
    fast_module = Keyword.get(opts, :fast_runtime_module, Runtime)
    accurate_module = Keyword.get(opts, :accurate_runtime_module, Runtime)

    fast_opts =
      opts
      |> Keyword.put(:model_name, Keyword.get(opts, :cascade_fast_model_name, "gemma4-e4b"))
      |> Keyword.put(:fused_ffn, false)

    # The first load performs the full VRAM safety check. Once EXLA has created
    # its allocator, rocm-smi reports that allocator's reservation as used VRAM;
    # the second load must not mistake our own reservation for another workload.
    accurate_opts = Keyword.put(opts, :rocm_preflight, :compatibility_only)

    with {:ok, fast_runtime} <- fast_module.load(fast_opts),
         {:ok, accurate_runtime} <- accurate_module.load(accurate_opts) do
      {:ok,
       %__MODULE__{
         fast_module: fast_module,
         fast_runtime: fast_runtime,
         accurate_module: accurate_module,
         accurate_runtime: accurate_runtime,
         min_chars_per_second: Keyword.get(opts, :cascade_min_chars_per_second, 0.0),
         sample_rate: Keyword.get(opts, :sample_rate, 16_000)
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
    case cascade.fast_module.generate(cascade.fast_runtime, input, opts) do
      {:ok, text} ->
        input = Map.put_new(input, :sample_rate, cascade.sample_rate)

        if escalate?(text, input, cascade.min_chars_per_second) do
          Logger.info("cascade: escalating E4B transcript to accurate model")
          cascade.accurate_module.generate(cascade.accurate_runtime, input, opts)
        else
          {:ok, text}
        end

      {:error, reason} ->
        Logger.warning("cascade: E4B failed, escalating reason=#{inspect(reason)}")
        cascade.accurate_module.generate(cascade.accurate_runtime, input, opts)
    end
  end

  @doc false
  def escalate?(text, input, min_chars_per_second) when is_binary(text) do
    normalized = String.trim(text)

    normalized == "" or
      StreamingSession.refusal?(normalized) or
      String.contains?(normalized, ["<|", "|>", "�"]) or
      below_density?(normalized, input, min_chars_per_second)
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
end
