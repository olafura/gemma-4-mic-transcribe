defmodule Gemma4MicTranscribe.Gemma4.ExtractedGenerator do
  @moduledoc """
  Greedy text generation through independently extracted Gemma 4 decoder layers.

  Prefill processes the complete prompt once. Every later step processes only
  the preceding token against fixed-shape per-layer attention K/V caches, so
  XLA can reuse the one-token decode executable.
  """

  alias Gemma4MicTranscribe.Gemma4.ExpertCaller
  alias Gemma4MicTranscribe.Gemma4.ExtractedOutputHead
  alias Gemma4MicTranscribe.Gemma4.ExtractedSparseDecoderLayer
  alias Gemma4MicTranscribe.Gemma4.RoutedExpertCache

  @default_eos_token_ids [1, 106]
  @default_expert_cache_bytes 16 * 1024 * 1024 * 1024

  defstruct [
    :artifact_prefix,
    :expert_artifact,
    :head_artifact,
    :backend,
    :head,
    :first_layer,
    :expert_cache,
    sparse_layers: %{}
  ]

  @doc """
  Loads the reusable extracted-model resources.

  The returned handle owns its output head, complete first layer, lazily loaded
  sparse decoder shells, and routed-expert cache. Call `release/1` when the
  long-running owner terminates.
  """
  def load!(opts) do
    prefix = Keyword.fetch!(opts, :artifact_prefix)
    expert_artifact = Keyword.fetch!(opts, :expert_artifact)
    head_artifact = Keyword.fetch!(opts, :head_artifact)
    backend = Keyword.fetch!(opts, :backend)
    expert_cache_bytes = Keyword.get(opts, :expert_cache_bytes, @default_expert_cache_bytes)

    unless is_integer(expert_cache_bytes) and expert_cache_bytes >= 0 do
      raise ArgumentError, "expert_cache_bytes must be a non-negative integer"
    end

    head = ExtractedOutputHead.load!(head_artifact, backend)

    try do
      first_layer =
        ExpertCaller.load_layer!(
          layer_artifact(prefix, 0, "caller"),
          layer_artifact(prefix, 0, "moe"),
          expert_artifact,
          backend
        )

      %__MODULE__{
        artifact_prefix: prefix,
        expert_artifact: expert_artifact,
        head_artifact: head_artifact,
        backend: backend,
        head: head,
        first_layer: first_layer,
        expert_cache: RoutedExpertCache.new(expert_cache_bytes)
      }
    rescue
      exception ->
        Nx.backend_deallocate(head.params)
        reraise exception, __STACKTRACE__
    end
  end

  @doc """
  Greedily generates tokens while retaining reusable model resources.

  Returns `{result, updated_model}` because sparse shells and routed experts
  discovered by this request become part of the reusable model handle. KV
  caches and token state are always fresh and are released before returning.
  """
  def generate!(%__MODULE__{} = model, opts) do
    input_text = Keyword.fetch!(opts, :input_text)
    max_new_tokens = Keyword.get(opts, :max_new_tokens, 1)
    expert_scale = Keyword.get(opts, :expert_scale, 1.0)
    eos_token_ids = Keyword.get(opts, :eos_token_ids, @default_eos_token_ids)

    unless is_integer(max_new_tokens) and max_new_tokens > 0 do
      raise ArgumentError, "max_new_tokens must be a positive integer"
    end

    embedding_data = ExpertCaller.prepare_embeddings!(model.first_layer, input_text)

    prompt_embeddings =
      embedding_data.input
      |> Nx.as_type(:bf16)
      |> transfer(model.backend)

    prompt_length = Nx.axis_size(prompt_embeddings, 0)
    started_at = System.monotonic_time(:microsecond)

    state = %{
      input: prompt_embeddings,
      offset: 0,
      cache_length: prompt_length + max_new_tokens,
      caches: %{},
      sparse_layers: model.sparse_layers,
      expert_cache: model.expert_cache,
      generated_ids: [],
      steps: [],
      override_route_count: 0,
      first_layer: model.first_layer
    }

    state =
      Enum.reduce_while(1..max_new_tokens, state, fn step, state ->
        {token, state} =
          decode_one!(
            state,
            model.artifact_prefix,
            model.expert_artifact,
            model.head,
            embedding_data.tokenizer,
            model.backend,
            expert_scale,
            step
          )

        if token.id in eos_token_ids do
          {:halt, state}
        else
          {:cont, state}
        end
      end)

    elapsed_us = System.monotonic_time(:microsecond) - started_at
    generated_ids = Enum.reverse(state.generated_ids)
    Nx.backend_deallocate(state.input)
    Nx.backend_deallocate(state.caches)
    expert_cache_stats = RoutedExpertCache.stats(state.expert_cache)

    result =
      %{
        input_tokens: embedding_data.tokens,
        generated_ids: generated_ids,
        generated_text: Bumblebee.Tokenizer.decode(embedding_data.tokenizer, generated_ids),
        steps: Enum.reverse(state.steps),
        override_route_count: state.override_route_count,
        elapsed_us: elapsed_us,
        mean_token_us: elapsed_us / length(generated_ids),
        expert_cache: expert_cache_stats,
        eos?: List.last(generated_ids) in eos_token_ids
      }

    updated_model = %{
      model
      | sparse_layers: state.sparse_layers,
        expert_cache: state.expert_cache
    }

    {result, updated_model}
  end

  @doc """
  One-shot compatibility wrapper.

  Long-running callers should prefer `load!/1` plus `generate!/2`, or use
  `ExtractedGeneratorServer`, so reusable XLA resources survive each request.
  """
  def generate!(opts) when is_list(opts) do
    model = load!(opts)

    try do
      {result, model} = generate!(model, opts)
      :ok = release(model)
      result
    rescue
      exception ->
        release(model)
        reraise exception, __STACKTRACE__
    end
  end

  @doc "Returns measurements for resources retained between requests."
  def stats(%__MODULE__{} = model) do
    %{
      sparse_layers: map_size(model.sparse_layers),
      expert_cache: RoutedExpertCache.stats(model.expert_cache)
    }
  end

  @doc "Explicitly releases all backend buffers owned by the reusable model."
  def release(%__MODULE__{} = model) do
    Nx.backend_deallocate(model.head.params)
    unload_first_layer(model.first_layer)
    Enum.each(model.sparse_layers, fn {_index, layer} -> unload_decoder_layer(layer) end)
    RoutedExpertCache.release(model.expert_cache)
  end

  defp decode_one!(
         state,
         prefix,
         _expert_artifact,
         head,
         tokenizer,
         backend,
         expert_scale,
         step
       ) do
    started_at = System.monotonic_time(:microsecond)

    first_layer = state.first_layer

    first_cache =
      Map.get_lazy(state.caches, 0, fn ->
        ExpertCaller.init_cache(first_layer, state.cache_length)
      end)

    first =
      ExpertCaller.call_layer_cached_output_device!(
        first_layer,
        state.input,
        first_cache,
        state.offset,
        expert_scale: expert_scale
      )

    route_count =
      first.override_route_count
      |> Nx.backend_copy(Nx.BinaryBackend)
      |> Nx.to_number()

    Nx.backend_deallocate(first.override_route_count)

    caches = put_cache(state.caches, 0, first_cache, first)

    {final_output, caches, sparse_layers, expert_cache} =
      Enum.reduce(
        1..29,
        {first.output, caches, state.sparse_layers, state.expert_cache},
        fn layer_index, {input, caches, sparse_layers, expert_cache} ->
          {layer, cache, result, expert_cache} =
            run_later_layer!(
              prefix,
              layer_index,
              backend,
              input,
              caches,
              sparse_layers,
              expert_cache,
              state.offset,
              state.cache_length
            )

          sparse_layers = Map.put(sparse_layers, layer_index, layer)

          Nx.backend_deallocate(input)

          {
            result.output,
            put_cache(caches, layer_index, cache, result),
            sparse_layers,
            expert_cache
          }
        end
      )

    prediction = ExtractedOutputHead.run(head, final_output, top_k: 5)
    candidates = prediction_rows(prediction, tokenizer)
    token = hd(candidates)
    next_input = ExtractedOutputHead.embedding(head, token.id)
    next_offset = state.offset + Nx.axis_size(state.input, 0)

    Nx.backend_deallocate(final_output)
    Nx.backend_deallocate(prediction)
    Nx.backend_deallocate(state.input)

    elapsed_us = System.monotonic_time(:microsecond) - started_at

    token =
      Map.merge(token, %{
        step: step,
        elapsed_us: elapsed_us,
        prefix_tokens: next_offset + 1
      })

    {
      token,
      %{
        input: next_input,
        offset: next_offset,
        cache_length: state.cache_length,
        caches: caches,
        sparse_layers: sparse_layers,
        expert_cache: expert_cache,
        generated_ids: [token.id | state.generated_ids],
        steps: [Map.put(token, :candidates, candidates) | state.steps],
        override_route_count: state.override_route_count + route_count,
        first_layer: first_layer
      }
    }
  end

  defp put_cache(caches, layer_index, previous, result) do
    Nx.backend_deallocate(previous)
    Map.put(caches, layer_index, %{key: result.key_cache, value: result.value_cache})
  end

  defp run_later_layer!(
         prefix,
         layer_index,
         backend,
         input,
         caches,
         sparse_layers,
         expert_cache,
         0,
         cache_length
       ) do
    layer =
      Map.get_lazy(sparse_layers, layer_index, fn ->
        ExtractedSparseDecoderLayer.load!(
          layer_artifact(prefix, layer_index, "caller"),
          layer_artifact(prefix, layer_index, "moe"),
          backend
        )
      end)

    cache =
      Map.get_lazy(caches, layer_index, fn ->
        ExtractedSparseDecoderLayer.init_cache(layer, cache_length)
      end)

    # A multi-token prompt touches more routed experts than the bounded cache
    # can usually retain. Admitting that one-pass scan evicts the recurring
    # one-token decode working set and causes cyclic cache thrashing on the next
    # request, so prefill's compact expert banks remain ephemeral.
    result = ExtractedSparseDecoderLayer.run_cached(layer, input, cache, 0)

    {layer, cache, result, expert_cache}
  end

  defp run_later_layer!(
         prefix,
         layer_index,
         backend,
         input,
         caches,
         sparse_layers,
         expert_cache,
         offset,
         cache_length
       ) do
    layer =
      Map.get_lazy(sparse_layers, layer_index, fn ->
        ExtractedSparseDecoderLayer.load!(
          layer_artifact(prefix, layer_index, "caller"),
          layer_artifact(prefix, layer_index, "moe"),
          backend,
          verify_checksum: false
        )
      end)

    cache =
      Map.get_lazy(caches, layer_index, fn ->
        ExtractedSparseDecoderLayer.init_cache(layer, cache_length)
      end)

    result =
      ExtractedSparseDecoderLayer.run_cached(layer, input, cache, offset,
        expert_cache: expert_cache
      )

    {layer, cache, result, result.expert_cache}
  end

  defp prediction_rows(result, tokenizer) do
    ids = result.top_k_indices |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list()
    logits = result.top_k_values |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list()

    raw_logits =
      result.raw_top_k_values
      |> Nx.backend_copy(Nx.BinaryBackend)
      |> Nx.to_flat_list()

    Enum.zip_with([ids, logits, raw_logits], fn [id, logit, raw_logit] ->
      %{
        id: id,
        token:
          Bumblebee.Tokenizer.id_to_token(tokenizer, id) ||
            Bumblebee.Tokenizer.decode(tokenizer, [id]),
        logit: logit,
        raw_logit: raw_logit
      }
    end)
  end

  defp unload_first_layer(layer) do
    Nx.backend_deallocate(layer.attention_params)
    Nx.backend_deallocate(layer.moe_params)
    Nx.backend_deallocate(layer.expert.params)
  end

  defp unload_decoder_layer(layer) do
    Nx.backend_deallocate(layer.attention_params)
    Nx.backend_deallocate(layer.moe_params)
  end

  defp layer_artifact(prefix, layer, kind), do: "#{prefix}-layer#{layer}-#{kind}"

  defp transfer(tensor, nil), do: tensor
  defp transfer(tensor, Nx.BinaryBackend), do: Nx.backend_copy(tensor, Nx.BinaryBackend)
  defp transfer(tensor, backend), do: Nx.backend_transfer(tensor, backend)
end
