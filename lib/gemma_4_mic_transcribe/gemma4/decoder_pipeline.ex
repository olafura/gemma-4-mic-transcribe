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
  alias Gemma4MicTranscribe.Gemma4Unified.ChannelState
  alias Gemma4MicTranscribe.Gemma4Unified.Input
  alias Gemma4MicTranscribe.Gemma4Unified.Model
  alias Gemma4MicTranscribe.Gemma4Unified.Runtime
  alias Gemma4MicTranscribe.Gemma4Unified.TokenSelection
  alias Gemma4MicTranscribe.Gemma4Unified.Transcript

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

  @enforce_keys [
    :prefix,
    :tail,
    :input_context,
    :generation,
    :cached_prefix_model,
    :cached_prefix_predict_fun,
    :cached_tail_model,
    :cached_tail_predict_fun,
    :parameter_count
  ]
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
      cached_prefix_model = Model.cached_decoder_prefix_model(spec, prefix_end)

      {_init_fun, cached_prefix_predict_fun} =
        Axon.build(cached_prefix_model, build_opts(backend))

      cached_tail_model = Model.cached_decoder_tail_model(spec, tail_layers)
      {_init_fun, cached_tail_predict_fun} = Axon.build(cached_tail_model, build_opts(backend))

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

      generation = %{
        spec: spec,
        suppression_mask: Map.get(runtime, :suppression_mask),
        inside_channel_suppression_mask: Map.get(runtime, :inside_channel_suppression_mask),
        content_suppression_mask: Map.get(runtime, :content_suppression_mask),
        channel_token_ids: Map.get(runtime, :channel_token_ids),
        eos_token_ids: generation_eos_token_ids(runtime, spec),
        no_repeat_ngram_size: Map.get(runtime, :no_repeat_ngram_size, 0)
      }

      {:ok,
       %__MODULE__{
         prefix: prefix,
         tail: tail,
         input_context: input_context,
         generation: generation,
         cached_prefix_model: cached_prefix_model,
         cached_prefix_predict_fun: cached_prefix_predict_fun,
         cached_tail_model: cached_tail_model,
         cached_tail_predict_fun: cached_tail_predict_fun,
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

  @doc "Greedily generates a transcript and token ids from a unified input."
  def generate(%__MODULE__{} = pipeline, input, opts \\ []) do
    with {:ok, prepared} <- Runtime.prepare_input(pipeline.input_context, input),
         {:ok, token_ids} <- generate_prepared(pipeline, prepared, opts) do
      {:ok,
       %{
         token_ids: token_ids,
         text: Transcript.decode(pipeline.tail.tokenizer, token_ids)
       }}
    end
  end

  @doc "Builds a unified input from 16 kHz samples and greedily generates a transcript."
  def generate_samples(%__MODULE__{} = pipeline, samples, opts \\ []) do
    input_opts =
      opts
      |> Keyword.put_new(:prompt, Config.default_prompt())
      |> Keyword.take([:prompt, :system_message, :audio_token_count, :max_tokens])

    samples
    |> Input.build(input_opts)
    |> then(&generate(pipeline, &1, opts))
  end

  @doc "Runs cache-aware split generation from model-ready tensors."
  def generate_prepared(%__MODULE__{} = pipeline, prepared, opts \\ []) do
    max_new_tokens = Keyword.get(opts, :max_new_tokens, 32)
    min_new_tokens = Keyword.get(opts, :min_new_tokens, 0)

    cond do
      not (is_integer(max_new_tokens) and max_new_tokens >= 0) ->
        {:error, ":max_new_tokens must be a non-negative integer"}

      not (is_integer(min_new_tokens) and min_new_tokens >= 0) ->
        {:error, ":min_new_tokens must be a non-negative integer"}

      max_new_tokens == 0 ->
        {:ok, []}

      true ->
        generate_cached(pipeline, prepared, max_new_tokens, min_new_tokens)
    end
  rescue
    exception -> {:error, Exception.message(exception)}
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

  defp generate_cached(pipeline, prepared, max_new_tokens, min_new_tokens) do
    sequence_length = Nx.axis_size(prepared["input_ids"], 1)
    max_cache_length = sequence_length + max_new_tokens
    backend = pipeline.prefix.backend || Nx.BinaryBackend

    cache =
      Nx.with_default_backend(backend, fn ->
        Model.init_cache(pipeline.generation.spec, 1, max_cache_length, %{})
      end)

    prefix_inputs = Map.put(prepared, "cache", cache)
    prefix_outputs = pipeline.cached_prefix_predict_fun.(pipeline.prefix.params, prefix_inputs)

    tail_outputs =
      pipeline.cached_tail_predict_fun.(pipeline.tail.params, %{
        "hidden_state" => prefix_outputs.hidden_state,
        "position_ids" => prepared["position_ids"],
        "attention_mask" => prefix_outputs.attention_mask,
        "cache" => prefix_outputs.cache
      })

    channel_state = ChannelState.content()
    suppression_mask = suppression_mask(pipeline, channel_state)
    token_id = TokenSelection.next_token_id_from_sequence(tail_outputs.logits, suppression_mask)

    if stop_token?(pipeline, token_id) and min_new_tokens <= 1 do
      {:ok, []}
    else
      content_length = prepared["attention_mask"] |> Nx.sum() |> Nx.to_number()

      decode_cached(
        pipeline,
        tail_outputs.cache,
        token_id,
        content_length,
        [token_id],
        1,
        max_new_tokens,
        min_new_tokens,
        channel_state
      )
    end
  end

  defp decode_cached(
         _pipeline,
         _cache,
         _previous_token_id,
         _prompt_length,
         generated,
         generated_count,
         max_new_tokens,
         _min_new_tokens,
         _channel_state
       )
       when generated_count >= max_new_tokens,
       do: {:ok, Enum.reverse(generated)}

  defp decode_cached(
         pipeline,
         cache,
         previous_token_id,
         prompt_length,
         generated,
         generated_count,
         max_new_tokens,
         min_new_tokens,
         channel_state
       ) do
    backend = pipeline.prefix.backend || Nx.BinaryBackend
    position_id = prompt_length + generated_count - 1

    prefix_inputs =
      Nx.with_default_backend(backend, fn ->
        %{
          "input_ids" => Nx.tensor([[previous_token_id]], type: :s64),
          "attention_mask" => Nx.tensor([[1]], type: :s64),
          "position_ids" => Nx.tensor([[position_id]], type: :s64),
          "input_features" => Nx.broadcast(0.0, {1, 1, pipeline.generation.spec.audio_embed_dim}),
          "input_features_mask" => Nx.tensor([[0]], type: :s64),
          "cache" => cache
        }
      end)

    prefix_outputs = pipeline.cached_prefix_predict_fun.(pipeline.prefix.params, prefix_inputs)

    tail_outputs =
      pipeline.cached_tail_predict_fun.(pipeline.tail.params, %{
        "hidden_state" => prefix_outputs.hidden_state,
        "position_ids" => prefix_inputs["position_ids"],
        "attention_mask" => prefix_outputs.attention_mask,
        "cache" => prefix_outputs.cache
      })

    suppression_mask = suppression_mask(pipeline, channel_state)

    banned_ids =
      Runtime.banned_ngram_token_ids(generated, pipeline.generation.no_repeat_ngram_size)

    token_id =
      TokenSelection.next_allowed_token_id_from_sequence(
        tail_outputs.logits,
        suppression_mask,
        banned_ids
      )

    step = generated_count + 1

    if stop_token?(pipeline, token_id) and step >= min_new_tokens do
      {:ok, Enum.reverse(generated)}
    else
      decode_cached(
        pipeline,
        tail_outputs.cache,
        token_id,
        prompt_length,
        [token_id | generated],
        generated_count + 1,
        max_new_tokens,
        min_new_tokens,
        ChannelState.advance(channel_state, token_id, pipeline.generation.channel_token_ids)
      )
    end
  end

  defp suppression_mask(pipeline, :before_content),
    do: pipeline.generation.suppression_mask

  defp suppression_mask(pipeline, :inside_channel),
    do: pipeline.generation.inside_channel_suppression_mask

  defp suppression_mask(pipeline, :content),
    do: pipeline.generation.content_suppression_mask

  defp stop_token?(pipeline, token_id) do
    token_id == pipeline.generation.spec.pad_token_id or
      token_id in pipeline.generation.eos_token_ids
  end

  defp generation_eos_token_ids(runtime, spec) do
    case get_in(runtime, [Access.key(:generation_config), Access.key(:eos_token_id)]) do
      nil -> List.wrap(spec.eos_token_id)
      ids -> List.wrap(ids)
    end
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
