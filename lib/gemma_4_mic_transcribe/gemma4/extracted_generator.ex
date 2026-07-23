defmodule Gemma4MicTranscribe.Gemma4.ExtractedGenerator do
  @moduledoc """
  Greedy text generation through independently extracted Gemma 4 decoder layers.

  This initial generator recomputes the complete token prefix for each new
  token. It deliberately uses the output-only layer entry points so it executes
  one model path and does not retain diagnostic routing tensors.
  """

  alias Gemma4MicTranscribe.Gemma4.ExpertCaller
  alias Gemma4MicTranscribe.Gemma4.ExtractedDecoderLayer
  alias Gemma4MicTranscribe.Gemma4.ExtractedOutputHead

  @default_eos_token_ids [1, 106]

  @doc "Greedily generates tokens from a complete set of extracted artifacts."
  def generate!(opts) do
    prefix = Keyword.fetch!(opts, :artifact_prefix)
    expert_artifact = Keyword.fetch!(opts, :expert_artifact)
    head_artifact = Keyword.fetch!(opts, :head_artifact)
    input_text = Keyword.fetch!(opts, :input_text)
    backend = Keyword.fetch!(opts, :backend)
    max_new_tokens = Keyword.get(opts, :max_new_tokens, 1)
    expert_scale = Keyword.get(opts, :expert_scale, 1.0)
    eos_token_ids = Keyword.get(opts, :eos_token_ids, @default_eos_token_ids)

    unless is_integer(max_new_tokens) and max_new_tokens > 0 do
      raise ArgumentError, "max_new_tokens must be a positive integer"
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

      embedding_data = ExpertCaller.prepare_embeddings!(first_layer, input_text)

      embeddings =
        embedding_data.input
        |> Nx.as_type(:bf16)
        |> transfer(backend)

      started_at = System.monotonic_time(:microsecond)

      state = %{
        embeddings: embeddings,
        generated_ids: [],
        steps: [],
        override_route_count: 0,
        first_layer: first_layer
      }

      state =
        Enum.reduce_while(1..max_new_tokens, state, fn step, state ->
          {token, state} =
            decode_one!(
              state,
              prefix,
              expert_artifact,
              head,
              embedding_data.tokenizer,
              backend,
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
      Nx.backend_deallocate(state.embeddings)

      %{
        input_tokens: embedding_data.tokens,
        generated_ids: generated_ids,
        generated_text: Bumblebee.Tokenizer.decode(embedding_data.tokenizer, generated_ids),
        steps: Enum.reverse(state.steps),
        override_route_count: state.override_route_count,
        elapsed_us: elapsed_us,
        mean_token_us: elapsed_us / length(generated_ids),
        eos?: List.last(generated_ids) in eos_token_ids
      }
    after
      Nx.backend_deallocate(head.params)
    end
  end

  defp decode_one!(
         state,
         prefix,
         expert_artifact,
         head,
         tokenizer,
         backend,
         expert_scale,
         step
       ) do
    started_at = System.monotonic_time(:microsecond)

    first_layer =
      state.first_layer ||
        ExpertCaller.load_layer!(
          layer_artifact(prefix, 0, "caller"),
          layer_artifact(prefix, 0, "moe"),
          expert_artifact,
          backend
        )

    first =
      ExpertCaller.call_layer_output_device!(
        first_layer,
        state.embeddings,
        expert_scale: expert_scale
      )

    route_count =
      first.override_route_count
      |> Nx.backend_copy(Nx.BinaryBackend)
      |> Nx.to_number()

    unload_first_layer(first_layer)
    Nx.backend_deallocate(first.override_route_count)

    final_output =
      Enum.reduce(1..29, first.output, fn layer_index, input ->
        layer =
          ExtractedDecoderLayer.load!(
            layer_artifact(prefix, layer_index, "caller"),
            layer_artifact(prefix, layer_index, "moe"),
            backend
          )

        output = ExtractedDecoderLayer.run_output(layer, input)
        unload_decoder_layer(layer)
        Nx.backend_deallocate(input)
        output
      end)

    prediction = ExtractedOutputHead.run(head, final_output, top_k: 5)
    candidates = prediction_rows(prediction, tokenizer)
    token = hd(candidates)
    token_embedding = ExtractedOutputHead.embedding(head, token.id)
    next_embeddings = Nx.concatenate([state.embeddings, token_embedding], axis: 0)

    Nx.backend_deallocate(final_output)
    Nx.backend_deallocate(prediction)
    Nx.backend_deallocate(state.embeddings)

    elapsed_us = System.monotonic_time(:microsecond) - started_at

    token =
      Map.merge(token, %{
        step: step,
        elapsed_us: elapsed_us,
        prefix_tokens: Nx.axis_size(next_embeddings, 0)
      })

    {
      token,
      %{
        embeddings: next_embeddings,
        generated_ids: [token.id | state.generated_ids],
        steps: [Map.put(token, :candidates, candidates) | state.steps],
        override_route_count: state.override_route_count + route_count,
        first_layer: nil
      }
    }
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
