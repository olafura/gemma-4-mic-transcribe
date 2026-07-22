defmodule Gemma4MicTranscribe.Gemma4.DecoderPipeline do
  @moduledoc """
  Splits a unified Gemma 4 model into a raw-input prefix and replaceable tail.

  The prefix embeds text and audio and runs every decoder block before the
  tail boundary. The tail owns the remaining blocks, final norm, and vocabulary
  head. Together they preserve the full model computation while making the
  boundary hidden state explicit.
  """

  alias Gemma4MicTranscribe.Config
  alias Gemma4MicTranscribe.Gemma4.DecoderBlocks
  alias Gemma4MicTranscribe.Gemma4Unified.Input
  alias Gemma4MicTranscribe.Gemma4Unified.Model
  alias Gemma4MicTranscribe.Gemma4Unified.Runtime

  defmodule Prefix do
    @moduledoc "The embeddings, audio projection, and leading decoder blocks."

    @enforce_keys [
      :last_layer,
      :parameter_count,
      :model,
      :params,
      :predict_fun,
      :backend
    ]
    defstruct @enforce_keys
  end

  @enforce_keys [:prefix, :tail, :input_context, :parameter_count]
  defstruct @enforce_keys

  @doc "Extracts a raw-input prefix and final decoder tail from a loaded runtime."
  def extract(runtime, tail_layers) do
    tail_layers = Enum.to_list(tail_layers)

    with :ok <- validate_boundary(tail_layers),
         {:ok, tail} <- DecoderBlocks.extract_tail(runtime, tail_layers),
         {:ok, model_info} <- fetch(runtime, :model_info),
         {:ok, spec} <- fetch(model_info, :spec),
         :ok <- validate_spec(spec),
         {:ok, source_params} <- fetch(model_info, :params),
         prefix_end = hd(tail_layers) - 1,
         {:ok, prefix_params} <- extract_prefix_params(source_params, prefix_end) do
      backend = Map.get(runtime, :backend)
      prefix_model = Model.decoder_prefix_model(spec, prefix_end)
      {_init_fun, prefix_predict_fun} = Axon.build(prefix_model, build_opts(backend))

      prefix = %Prefix{
        last_layer: prefix_end,
        parameter_count: parameter_count(prefix_params.data),
        model: prefix_model,
        params: prefix_params,
        predict_fun: prefix_predict_fun,
        backend: backend
      }

      input_context = %{
        backend: backend,
        tokenizer: Map.get(runtime, :tokenizer),
        e4b?: false,
        model_info: %{spec: spec}
      }

      {:ok,
       %__MODULE__{
         prefix: prefix,
         tail: tail,
         input_context: input_context,
         parameter_count: prefix.parameter_count + tail.parameter_count
       }}
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  @doc "Like `extract/2`, but raises when extraction fails."
  def extract!(runtime, tail_layers) do
    case extract(runtime, tail_layers) do
      {:ok, pipeline} -> pipeline
      {:error, reason} -> raise ArgumentError, reason
    end
  end

  @doc "Prepares a unified input, runs the split pipeline, and returns next-token candidates."
  def top_k(%__MODULE__{} = pipeline, input, k) do
    with {:ok, prepared} <- Runtime.prepare_input(pipeline.input_context, input) do
      top_k_prepared(pipeline, prepared, k)
    end
  end

  @doc "Builds a unified input directly from 16 kHz samples and returns candidates."
  def top_k_samples(%__MODULE__{} = pipeline, samples, k, opts \\ []) do
    input_opts =
      opts
      |> Keyword.put_new(:prompt, Config.default_prompt())
      |> Keyword.take([:prompt, :system_message, :audio_token_count, :max_tokens])

    samples
    |> Input.build(input_opts)
    |> then(&top_k(pipeline, &1, k))
  end

  @doc "Runs the split pipeline from already-prepared model tensors."
  def top_k_prepared(%__MODULE__{} = pipeline, prepared, k) when is_map(prepared) do
    with {:ok, hidden_state} <- run_prefix(pipeline.prefix, prepared) do
      DecoderBlocks.top_k(pipeline.tail, hidden_state, k,
        position_ids: prepared["position_ids"],
        attention_mask: prepared["attention_mask"]
      )
    end
  end

  @doc "Runs only the extracted prefix and returns the tail-boundary hidden state."
  def run_prefix(%Prefix{} = prefix, prepared) when is_map(prepared) do
    {:ok, prefix.predict_fun.(prefix.params, prepared)}
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  defp extract_prefix_params(%Axon.ModelState{data: data}, last_layer) do
    required = ["embedder.token_embedding", "audio_embedder.projection"]

    missing =
      Enum.reject(required, &Map.has_key?(data, &1)) ++
        Enum.reject(0..last_layer, fn layer ->
          prefix = "decoder.blocks.#{layer}."
          Enum.any?(data, fn {name, _params} -> String.starts_with?(name, prefix) end)
        end)

    if missing == [] do
      selected =
        Map.filter(data, fn {name, _params} ->
          name in required or decoder_layer_in_prefix?(name, last_layer)
        end)

      {:ok, Axon.ModelState.new(selected)}
    else
      {:error, "runtime is missing prefix parameters: #{inspect(missing)}"}
    end
  end

  defp extract_prefix_params(other, _last_layer),
    do: {:error, "expected Axon model parameters, got: #{inspect(other)}"}

  defp decoder_layer_in_prefix?("decoder.blocks." <> rest, last_layer) do
    case Integer.parse(rest) do
      {layer, "." <> _parameter} -> layer <= last_layer
      _other -> false
    end
  end

  defp decoder_layer_in_prefix?(_name, _last_layer), do: false

  defp parameter_count(data) do
    Enum.reduce(data, 0, fn {_name, parameters}, count ->
      count + Enum.reduce(parameters, 0, fn {_name, tensor}, sum -> sum + Nx.size(tensor) end)
    end)
  end

  defp validate_boundary([first | _rest]) when is_integer(first) and first > 0, do: :ok

  defp validate_boundary(_layers),
    do: {:error, "decoder pipeline tail must start after layer 0"}

  defp validate_spec(%Model{}), do: :ok

  defp validate_spec(spec),
    do: {:error, "decoder pipelines currently require Gemma4Unified.Model, got: #{inspect(spec)}"}

  defp fetch(map, key) when is_map(map) do
    case Map.fetch(map, key) do
      {:ok, value} -> {:ok, value}
      :error -> {:error, "decoder pipeline runtime is missing #{key}"}
    end
  end

  defp build_opts(EXLA.Backend), do: [compiler: EXLA]

  defp build_opts({EXLA.Backend, backend_opts}),
    do: [compiler: EXLA] ++ Keyword.take(backend_opts, [:client, :device_id])

  defp build_opts(_backend), do: []
end
