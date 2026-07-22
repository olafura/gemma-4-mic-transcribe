defmodule Gemma4MicTranscribe.Gemma4.DecoderPipelineTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4.DecoderPipeline
  alias Gemma4MicTranscribe.Gemma4Unified.Model

  test "splits prepared-input inference at a replaceable decoder boundary" do
    {runtime, inputs} = runtime()
    pipeline = DecoderPipeline.extract!(runtime, [1])

    assert pipeline.prefix.last_layer == 0
    assert pipeline.tail.layer_indices == [1]

    assert Map.keys(pipeline.prefix.params.data)
           |> Enum.all?(fn name ->
             name in ["embedder.token_embedding", "audio_embedder.projection"] or
               String.starts_with?(name, "decoder.blocks.0.")
           end)

    refute Enum.any?(
             Map.keys(pipeline.prefix.params.data),
             &String.starts_with?(&1, "decoder.blocks.1.")
           )

    assert {:ok, candidates} = DecoderPipeline.top_k_prepared(pipeline, inputs, 3)

    expected_logits = runtime.predict_fun.(runtime.model_info.params, inputs).logits[0][-1]
    {_scores, expected_ids} = Nx.top_k(expected_logits, k: 3)

    assert Enum.map(candidates, & &1.token_id) == Nx.to_flat_list(expected_ids)
  end

  test "requires at least one layer on each side of the boundary" do
    {runtime, _inputs} = runtime()

    assert {:error, message} = DecoderPipeline.extract(runtime, 0..1)
    assert message =~ "start after layer 0"
  end

  test "passes one global KV cache through split prefill and decode" do
    {runtime, inputs} = runtime()
    pipeline = DecoderPipeline.extract!(runtime, [1])
    cache = Model.init_cache(runtime.model_info.spec, 1, 6, %{})

    prefill = runtime.predict_fun.(runtime.model_info.params, Map.put(inputs, "cache", cache))
    first_id = prefill.logits[0][-1] |> Nx.argmax() |> Nx.to_number()

    decode_inputs = %{
      "input_ids" => Nx.tensor([[first_id]], type: :s64),
      "attention_mask" => Nx.tensor([[1]], type: :s64),
      "position_ids" => Nx.tensor([[4]], type: :s64),
      "input_features" => Nx.broadcast(0.0, {1, 1, 4}),
      "input_features_mask" => Nx.tensor([[0]], type: :s64),
      "cache" => prefill.cache
    }

    second = runtime.predict_fun.(runtime.model_info.params, decode_inputs)
    second_id = second.logits[0][-1] |> Nx.argmax() |> Nx.to_number()

    for execution <- [:composed, :split] do
      assert {:ok, [^first_id, ^second_id]} =
               DecoderPipeline.generate_prepared(pipeline, inputs,
                 max_new_tokens: 2,
                 min_new_tokens: 3,
                 execution: execution
               )
    end
  end

  test "rejects unknown generation execution modes" do
    {runtime, inputs} = runtime()
    pipeline = DecoderPipeline.extract!(runtime, [1])

    assert {:error, ":execution must be :composed or :split"} =
             DecoderPipeline.generate_prepared(pipeline, inputs, execution: :unknown)
  end

  test "compiles the composed generation graph with XLA" do
    assert {:ok, _started} = Application.ensure_all_started(:exla)
    {runtime, inputs} = runtime({EXLA.Backend, client: :host})
    pipeline = DecoderPipeline.extract!(runtime, [1])

    assert {:ok, token_ids} =
             DecoderPipeline.generate_prepared(pipeline, inputs,
               max_new_tokens: 2,
               min_new_tokens: 3
             )

    assert length(token_ids) == 2
  end

  defp runtime(backend \\ nil)

  defp runtime(nil), do: build_runtime(nil)

  defp runtime(backend) do
    Nx.with_default_backend(backend, fn -> build_runtime(backend) end)
  end

  defp build_runtime(backend) do
    spec =
      Bumblebee.configure(Model,
        vocab_size: 32,
        max_positions: 16,
        hidden_size: 8,
        intermediate_size: 16,
        num_blocks: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        num_global_key_value_heads: 1,
        attention_head_size: 4,
        global_attention_head_size: 4,
        layer_types: [:sliding_attention, :full_attention],
        attention_window_size: 4,
        audio_embed_dim: 4,
        audio_token_id: 7,
        boa_token_id: 6,
        eoa_token_id: 8,
        final_logit_softcapping: nil
      )

    inputs = %{
      "input_ids" => Nx.tensor([[6, 7, 8, 3]], type: :s64),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1]], type: :s64),
      "position_ids" => Nx.tensor([[0, 1, 2, 3]], type: :s64),
      "input_features" => Nx.tensor([[[0.1, 0.2, 0.3, 0.4]]]),
      "input_features_mask" => Nx.tensor([[1]], type: :s64)
    }

    model = Bumblebee.build_model(spec)
    build_opts = if backend, do: [compiler: EXLA, client: :host], else: []
    {init_fun, predict_fun} = Axon.build(model, build_opts)
    params = init_fun.(inputs, Axon.ModelState.empty())

    {%{
       model_name: "tiny-gemma4",
       backend: backend,
       tokenizer: nil,
       suppression_mask: Nx.broadcast(0, {32}) |> Nx.as_type(:u8),
       inside_channel_suppression_mask: Nx.broadcast(0, {32}) |> Nx.as_type(:u8),
       content_suppression_mask: Nx.broadcast(0, {32}) |> Nx.as_type(:u8),
       channel_token_ids: %{start: 30, end: 31},
       generation_config: %{eos_token_id: [1]},
       no_repeat_ngram_size: 0,
       predict_fun: predict_fun,
       model_info: %{model: model, params: params, spec: spec}
     }, inputs}
  end
end
