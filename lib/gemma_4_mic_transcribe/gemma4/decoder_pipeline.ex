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

  @cache_length_step 256

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
    :generation_model,
    :generation_params,
    :generation_predict_fun,
    :parameter_count
  ]
  defstruct @enforce_keys ++
              [prefill_generation_model: nil, prefill_generation_predict_fun: nil]

  @doc "Extracts a raw-input prefix and final decoder tail from a loaded runtime."
  def extract(runtime, tail_layers) do
    tail_layers = Enum.to_list(tail_layers)

    with :ok <- validate_boundary(tail_layers),
         {:ok, tail} <- DecoderBlocks.extract_tail(runtime, tail_layers),
         {:ok, model_info} <- fetch(runtime, :model_info),
         {:ok, spec} <- fetch(model_info, :spec),
         :ok <- validate_spec(spec),
         {:ok, generation_model} <- fetch(model_info, :model),
         {:ok, source_params} <- fetch(model_info, :params),
         {:ok, generation_predict_fun} <- fetch(runtime, :predict_fun),
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
         generation_model: generation_model,
         generation_params: source_params,
         generation_predict_fun: generation_predict_fun,
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

  @doc """
  Replaces one decoder slot with the weights from another compatible layer.

  The graph and KV-cache slot remain those of `target_layer`; only its parameter
  tensors are replaced. Source and target must use the same attention type and
  expose identical parameter names, shapes, and types. This makes the returned
  pipeline a shape-safe "Frankenstein" model without recompiling its graph.
  """
  def transplant_layer(%__MODULE__{} = pipeline, source_layer, target_layer) do
    spec = pipeline.generation.spec

    with :ok <- validate_layer_index(source_layer, spec.num_blocks, "source"),
         :ok <- validate_layer_index(target_layer, spec.num_blocks, "target"),
         :ok <- validate_layer_types(spec, source_layer, target_layer),
         {:ok, replacements} <-
           layer_replacements(pipeline.generation_params.data, source_layer, target_layer) do
      generation_params = replace_params(pipeline.generation_params, replacements)

      prefix =
        if target_layer <= pipeline.prefix.last_layer do
          %{pipeline.prefix | params: replace_params(pipeline.prefix.params, replacements)}
        else
          pipeline.prefix
        end

      tail =
        if target_layer in pipeline.tail.layer_indices do
          %{pipeline.tail | params: replace_params(pipeline.tail.params, replacements)}
        else
          pipeline.tail
        end

      {:ok,
       %{
         pipeline
         | prefix: prefix,
           tail: tail,
           generation_params: generation_params
       }}
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  @doc "Like `transplant_layer/3`, but raises when the layers are incompatible."
  def transplant_layer!(%__MODULE__{} = pipeline, source_layer, target_layer) do
    case transplant_layer(pipeline, source_layer, target_layer) do
      {:ok, pipeline} -> pipeline
      {:error, reason} -> raise ArgumentError, reason
    end
  end

  @doc """
  Blends a compatible source layer into a target layer without changing the graph.

  `source_weight` is the fraction contributed by the source tensors. A weight of
  `0.0` preserves the target and `1.0` is equivalent to a full transplant.
  """
  def blend_layer(%__MODULE__{} = pipeline, source_layer, target_layer, source_weight)
      when is_number(source_weight) and source_weight >= 0.0 and source_weight <= 1.0 do
    spec = pipeline.generation.spec

    with :ok <- validate_layer_index(source_layer, spec.num_blocks, "source"),
         :ok <- validate_layer_index(target_layer, spec.num_blocks, "target"),
         :ok <- validate_layer_types(spec, source_layer, target_layer),
         {:ok, source_params} <-
           layer_replacements(pipeline.generation_params.data, source_layer, target_layer) do
      replacements =
        blend_replacements(
          source_params,
          pipeline.generation_params.data,
          source_weight
        )

      generation_params = replace_params(pipeline.generation_params, replacements)

      prefix =
        if target_layer <= pipeline.prefix.last_layer do
          %{pipeline.prefix | params: replace_params(pipeline.prefix.params, replacements)}
        else
          pipeline.prefix
        end

      tail =
        if target_layer in pipeline.tail.layer_indices do
          %{pipeline.tail | params: replace_params(pipeline.tail.params, replacements)}
        else
          pipeline.tail
        end

      {:ok,
       %{
         pipeline
         | prefix: prefix,
           tail: tail,
           generation_params: generation_params
       }}
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  def blend_layer(%__MODULE__{}, _source_layer, _target_layer, source_weight),
    do: {:error, "source weight must be between 0.0 and 1.0, got: #{inspect(source_weight)}"}

  @doc "Like `blend_layer/4`, but raises when the layers or weight are invalid."
  def blend_layer!(%__MODULE__{} = pipeline, source_layer, target_layer, source_weight) do
    case blend_layer(pipeline, source_layer, target_layer, source_weight) do
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
      |> Keyword.take([
        :prompt,
        :system_message,
        :audio_token_count,
        :max_tokens,
        :thought_channel
      ])

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
      |> Keyword.take([
        :prompt,
        :system_message,
        :audio_token_count,
        :max_tokens,
        :thought_channel
      ])

    samples
    |> Input.build(input_opts)
    |> then(&generate(pipeline, &1, opts))
  end

  @doc """
  Runs cache-aware split generation from model-ready tensors.

  `thought_channel: false` says the prompt stopped at the model turn instead
  of closing an empty thought channel, so generation starts before the content
  channel and the model may open a thought channel of its own.

  `logits_index: i` continues from prompt position `i` instead of the last
  one, for a prompt padded on the right to a fixed bucket. It is ignored when
  prefill returns a single position, which is what the `:split` execution's
  tail does.

  `scores: true` also returns, for every generated token, its log-probability
  and its margin over the best other candidate (both under the suppression
  mask the step used), as `{:ok, token_ids, scores}`. The picks are the same
  as without it.

  `on_token: {fun, acc}` streams the reply: `fun.(token_id, score, acc)` is
  called with each token as soon as it is final (not with the stop token that
  ends the reply), with its score when `scores: true` and `nil` otherwise,
  and returns `{:cont, acc}` to go on or `{:halt, acc}` to end generation
  there, keeping that token. The picks are the same as without it.
  """
  def generate_prepared(%__MODULE__{} = pipeline, prepared, opts \\ []) do
    max_new_tokens = Keyword.get(opts, :max_new_tokens, 32)
    min_new_tokens = Keyword.get(opts, :min_new_tokens, 0)
    execution = Keyword.get(opts, :execution, :composed)
    logits_index = Keyword.get(opts, :logits_index)
    scores = if Keyword.get(opts, :scores, false), do: [], else: nil
    on_token = Keyword.get(opts, :on_token)

    channel_state =
      if Keyword.get(opts, :thought_channel, true),
        do: ChannelState.content(),
        else: ChannelState.initial()

    cond do
      not (is_integer(max_new_tokens) and max_new_tokens >= 0) ->
        {:error, ":max_new_tokens must be a non-negative integer"}

      not (is_integer(min_new_tokens) and min_new_tokens >= 0) ->
        {:error, ":min_new_tokens must be a non-negative integer"}

      execution not in [:composed, :split] ->
        {:error, ":execution must be :composed or :split"}

      not (is_nil(logits_index) or (is_integer(logits_index) and logits_index >= 0)) ->
        {:error, ":logits_index must be a non-negative integer"}

      not (is_nil(on_token) or match?({fun, _acc} when is_function(fun, 3), on_token)) ->
        {:error, ":on_token must be {fun/3, acc}"}

      max_new_tokens == 0 ->
        finish([], scores)

      true ->
        generate_cached(
          pipeline,
          prepared,
          max_new_tokens,
          min_new_tokens,
          execution,
          channel_state,
          logits_index,
          scores,
          on_token
        )
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

  defp generate_cached(
         pipeline,
         prepared,
         max_new_tokens,
         min_new_tokens,
         execution,
         channel_state,
         logits_index,
         scores,
         on_token
       ) do
    sequence_length = Nx.axis_size(prepared["input_ids"], 1)
    max_cache_length = cache_length(sequence_length, max_new_tokens)
    backend = pipeline.prefix.backend || Nx.BinaryBackend

    cache =
      Nx.with_default_backend(backend, fn ->
        Model.init_cache(pipeline.generation.spec, 1, max_cache_length, %{})
      end)

    started = step_clock()
    outputs = predict_cached(pipeline, Map.put(prepared, "cache", cache), execution, :prefill)
    predicted = step_clock(outputs)

    suppression_mask = suppression_mask(pipeline, channel_state)

    index = prefill_index(outputs.logits, logits_index)
    token_id = TokenSelection.next_token_id_from_sequence(outputs.logits, suppression_mask, index)
    scores = record_score(scores, outputs.logits, suppression_mask, token_id, index)
    report_step(:prefill, started, predicted)

    if stop_token?(pipeline, token_id) and min_new_tokens <= 1 do
      finish([], scores)
    else
      case stream_token(on_token, token_id, scores) do
        :halt ->
          finish([token_id], scores)

        on_token ->
          content_length = prepared["attention_mask"] |> Nx.sum() |> Nx.to_number()

          decode_cached(
            pipeline,
            outputs.cache,
            token_id,
            content_length,
            [token_id],
            1,
            max_new_tokens,
            min_new_tokens,
            ChannelState.advance(channel_state, token_id, pipeline.generation.channel_token_ids),
            execution,
            scores,
            on_token
          )
      end
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
         _channel_state,
         _execution,
         scores,
         _on_token
       )
       when generated_count >= max_new_tokens,
       do: finish(Enum.reverse(generated), scores)

  defp decode_cached(
         pipeline,
         cache,
         previous_token_id,
         prompt_length,
         generated,
         generated_count,
         max_new_tokens,
         min_new_tokens,
         channel_state,
         execution,
         scores,
         on_token
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

    started = step_clock()
    outputs = predict_cached(pipeline, prefix_inputs, execution, :decode)
    predicted = step_clock(outputs)

    suppression_mask = suppression_mask(pipeline, channel_state)

    banned_ids =
      Runtime.banned_ngram_token_ids(generated, pipeline.generation.no_repeat_ngram_size)

    token_id =
      TokenSelection.next_allowed_token_id_from_sequence(
        outputs.logits,
        suppression_mask,
        banned_ids
      )

    step = generated_count + 1
    scores = record_score(scores, outputs.logits, suppression_mask, token_id, -1)
    report_step(:decode, started, predicted)

    if stop_token?(pipeline, token_id) and step >= min_new_tokens do
      finish(Enum.reverse(generated), scores)
    else
      case stream_token(on_token, token_id, scores) do
        :halt ->
          finish(Enum.reverse([token_id | generated]), scores)

        on_token ->
          decode_cached(
            pipeline,
            outputs.cache,
            token_id,
            prompt_length,
            [token_id | generated],
            generated_count + 1,
            max_new_tokens,
            min_new_tokens,
            ChannelState.advance(channel_state, token_id, pipeline.generation.channel_token_ids),
            execution,
            scores,
            on_token
          )
      end
    end
  end

  # Hands a token that is now part of the reply to the `:on_token` callback,
  # with the score `record_score` just put at the head of `scores`. Returns
  # the callback with its new accumulator, or `:halt`.
  defp stream_token(nil, _token_id, _scores), do: nil

  defp stream_token({fun, acc}, token_id, scores) do
    score = if is_list(scores), do: hd(scores)

    case fun.(token_id, score, acc) do
      {:cont, acc} -> {fun, acc}
      {:halt, _acc} -> :halt
    end
  end

  # GEMMA_DECODE_TIMING=1 prints, for every prefill and decode step, the time
  # spent in the model and in token selection, to profile new hardware. The
  # model time waits for the logits, so it no longer overlaps selection.
  defp step_clock, do: if(step_timing?(), do: System.monotonic_time(:microsecond))

  defp step_clock(outputs) do
    if step_timing?() do
      outputs.logits |> Nx.reduce_max() |> Nx.to_number()
      System.monotonic_time(:microsecond)
    end
  end

  defp report_step(_phase, nil, _predicted), do: :ok

  defp report_step(phase, started, predicted) do
    selected = System.monotonic_time(:microsecond)

    IO.puts(
      :stderr,
      ~s({"event":"step_timing","phase":"#{phase}","model_us":#{predicted - started},) <>
        ~s("select_us":#{selected - predicted}})
    )
  end

  defp step_timing?, do: System.get_env("GEMMA_DECODE_TIMING") == "1"

  # A stop token ends generation without being returned, so its score, the
  # last one recorded, is dropped with it.
  defp finish(token_ids, nil), do: {:ok, token_ids}

  defp finish(token_ids, scores),
    do: {:ok, token_ids, scores |> Enum.reverse() |> Enum.take(length(token_ids))}

  defp record_score(nil, _logits, _suppression_mask, _token_id, _index), do: nil

  defp record_score(scores, logits, suppression_mask, token_id, index) do
    candidates = TokenSelection.scored_candidates(logits, suppression_mask, 4, index)

    score =
      case List.keyfind(candidates, token_id, 0) do
        {^token_id, logprob} ->
          runner_up =
            Enum.find_value(candidates, fn {id, lp} -> if id != token_id, do: lp end)

          %{logprob: logprob, margin: if(runner_up, do: logprob - runner_up, else: nil)}

        nil ->
          %{logprob: nil, margin: nil}
      end

    [score | scores]
  end

  # Every step replaces the KV cache, and nothing but this loop holds the old
  # one, so its device buffers are freed here rather than whenever the terms
  # holding them happen to be collected. Those terms are too small to prompt a
  # collection, so otherwise several stale caches (1.75 GB each at 1024 tokens
  # on the 12B) pile up and overrun a 24 GB card. A collection per step would
  # also do it, but costs 5-12 ms because it copies the whole process heap.
  defp predict_cached(pipeline, inputs, execution, phase) do
    outputs = run_cached(pipeline, inputs, execution, phase)
    release_stale(inputs["cache"], outputs.cache)
    outputs
  end

  defp run_cached(pipeline, inputs, :composed, :prefill) do
    predict_fun =
      pipeline.prefill_generation_predict_fun || pipeline.generation_predict_fun

    predict_fun.(pipeline.generation_params, inputs)
  end

  defp run_cached(pipeline, inputs, :composed, :decode) do
    pipeline.generation_predict_fun.(pipeline.generation_params, inputs)
  end

  defp run_cached(pipeline, inputs, :split, _phase) do
    prefix_outputs = pipeline.cached_prefix_predict_fun.(pipeline.prefix.params, inputs)

    outputs =
      pipeline.cached_tail_predict_fun.(pipeline.tail.params, %{
        "hidden_state" => prefix_outputs.hidden_state,
        "position_ids" => inputs["position_ids"],
        "attention_mask" => prefix_outputs.attention_mask,
        "cache" => prefix_outputs.cache
      })

    release_stale(prefix_outputs.cache, outputs.cache)
    outputs
  end

  # Frees the tensors of `old` whose buffers `new` does not share.
  defp release_stale(old, new) do
    kept = new |> cache_tensors() |> MapSet.new(& &1.data)

    for tensor <- cache_tensors(old), not MapSet.member?(kept, tensor.data) do
      Nx.backend_deallocate(tensor)
    end

    :ok
  end

  defp cache_tensors(cache),
    do:
      Nx.Defn.Composite.reduce(cache, [], &if(is_struct(&1, Nx.Tensor), do: [&1 | &2], else: &2))

  # Prefill of an unpadded prompt, and the tail model's already-sliced output,
  # both continue at the last position.
  defp prefill_index(_logits, nil), do: -1

  defp prefill_index(logits, index) do
    if Nx.rank(logits) == 3 and Nx.axis_size(logits, 1) > index, do: index, else: -1
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

  defp cache_length(prompt_length, max_new_tokens) do
    needed = prompt_length + max_new_tokens
    div(needed + @cache_length_step - 1, @cache_length_step) * @cache_length_step
  end

  defp validate_layer_index(layer, count, _label)
       when is_integer(layer) and layer >= 0 and layer < count,
       do: :ok

  defp validate_layer_index(layer, count, label),
    do: {:error, "#{label} layer must be in 0..#{count - 1}, got: #{inspect(layer)}"}

  defp validate_layer_types(spec, source_layer, target_layer) do
    source_type = layer_type(spec, source_layer)
    target_type = layer_type(spec, target_layer)

    if source_type == target_type do
      :ok
    else
      {:error,
       "cannot transplant #{source_type} layer #{source_layer} into #{target_type} layer #{target_layer}"}
    end
  end

  defp layer_type(spec, layer) do
    spec.layer_types
    |> Kernel.||(default_layer_types(spec.num_blocks))
    |> Enum.fetch!(layer)
  end

  defp default_layer_types(count) do
    Enum.map(0..(count - 1), fn index ->
      if rem(index + 1, 6) == 0, do: :full_attention, else: :sliding_attention
    end)
  end

  defp layer_replacements(data, source_layer, target_layer) do
    source_prefix = "decoder.blocks.#{source_layer}."
    target_prefix = "decoder.blocks.#{target_layer}."

    replacements =
      data
      |> Enum.filter(fn {name, _params} -> String.starts_with?(name, source_prefix) end)
      |> Map.new(fn {name, params} ->
        suffix = String.replace_prefix(name, source_prefix, "")
        {target_prefix <> suffix, params}
      end)

    target = Map.filter(data, fn {name, _params} -> String.starts_with?(name, target_prefix) end)

    cond do
      map_size(replacements) == 0 ->
        {:error, "source layer #{source_layer} has no parameters"}

      Map.keys(replacements) |> Enum.sort() != Map.keys(target) |> Enum.sort() ->
        {:error, "layers #{source_layer} and #{target_layer} expose different parameter nodes"}

      true ->
        validate_replacement_layouts(replacements, target, source_layer, target_layer)
    end
  end

  defp validate_replacement_layouts(replacements, target, source_layer, target_layer) do
    mismatch =
      Enum.find(replacements, fn {name, source_params} ->
        parameter_layout(source_params) != parameter_layout(Map.fetch!(target, name))
      end)

    case mismatch do
      nil ->
        {:ok, replacements}

      {name, _params} ->
        {:error,
         "layers #{source_layer} and #{target_layer} have incompatible parameters at #{name}"}
    end
  end

  defp blend_replacements(source, target_data, source_weight) do
    target_weight = 1.0 - source_weight

    Map.new(source, fn {node_name, source_params} ->
      target_params = Map.fetch!(target_data, node_name)

      blended =
        Map.new(source_params, fn {param_name, source_tensor} ->
          target_tensor = Map.fetch!(target_params, param_name)
          source_tensor = transfer_to_tensor_backend(source_tensor, target_tensor)
          target_weight_tensor = scalar_like(target_weight, target_tensor)
          source_weight_tensor = scalar_like(source_weight, target_tensor)

          {param_name,
           Nx.add(
             Nx.multiply(target_tensor, target_weight_tensor),
             Nx.multiply(source_tensor, source_weight_tensor)
           )}
        end)

      {node_name, blended}
    end)
  end

  defp scalar_like(value, tensor) do
    value
    |> Nx.tensor(type: Nx.type(tensor))
    |> transfer_to_tensor_backend(tensor)
  end

  defp transfer_to_tensor_backend(source, target) do
    source_backend = source.data.__struct__
    target_backend = target.data.__struct__

    if source_backend == target_backend do
      source
    else
      Nx.backend_copy(source, target_backend)
    end
  end

  defp parameter_layout(params) do
    params
    |> Enum.map(fn {name, tensor} -> {name, Nx.shape(tensor), Nx.type(tensor)} end)
    |> Enum.sort()
  end

  defp replace_params(%Axon.ModelState{data: data} = state, replacements) do
    selected = Map.take(replacements, Map.keys(data))
    %{state | data: Map.merge(data, selected)}
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
