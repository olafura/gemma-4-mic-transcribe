defmodule Gemma4MicTranscribe.Gemma4.DecoderPipelineTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4.DecoderPipeline
  alias Gemma4MicTranscribe.Gemma4.DecoderBlockArtifact
  alias Gemma4MicTranscribe.Gemma4.DecoderPipelineArtifact
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

  test "transplants compatible layer weights into a compiled pipeline" do
    {runtime, inputs} = runtime(nil, num_blocks: 3)
    pipeline = DecoderPipeline.extract!(runtime, [1, 2])
    source_name = "decoder.blocks.0.ffn.gate"
    target_name = "decoder.blocks.1.ffn.gate"
    source_kernel = pipeline.generation_params.data[source_name]["kernel"]
    original_target = pipeline.generation_params.data[target_name]["kernel"]

    refute Nx.to_binary(source_kernel) == Nx.to_binary(original_target)
    assert {:ok, frankenstein} = DecoderPipeline.transplant_layer(pipeline, 0, 1)

    transplanted = frankenstein.generation_params.data[target_name]["kernel"]
    standalone_tail = frankenstein.tail.params.data[target_name]["kernel"]

    assert Nx.to_binary(transplanted) == Nx.to_binary(source_kernel)
    assert Nx.to_binary(standalone_tail) == Nx.to_binary(source_kernel)
    assert frankenstein.generation_predict_fun == pipeline.generation_predict_fun

    assert {:ok, composed_ids} =
             DecoderPipeline.generate_prepared(frankenstein, inputs,
               max_new_tokens: 2,
               min_new_tokens: 3,
               execution: :composed
             )

    assert {:ok, split_ids} =
             DecoderPipeline.generate_prepared(frankenstein, inputs,
               max_new_tokens: 2,
               min_new_tokens: 3,
               execution: :split
             )

    assert length(composed_ids) == 2
    assert split_ids == composed_ids
  end

  test "rejects transplants between different attention types" do
    {runtime, _inputs} = runtime()
    pipeline = DecoderPipeline.extract!(runtime, [1])
    spec = %{pipeline.generation.spec | layer_types: [:sliding_attention, :full_attention]}
    pipeline = %{pipeline | generation: %{pipeline.generation | spec: spec}}

    assert {:error, message} = DecoderPipeline.transplant_layer(pipeline, 0, 1)
    assert message =~ "cannot transplant sliding_attention layer 0"
  end

  test "blends compatible layer weights into a compiled pipeline" do
    {runtime, _inputs} = runtime(nil, num_blocks: 3)
    pipeline = DecoderPipeline.extract!(runtime, [1, 2])
    source_name = "decoder.blocks.0.ffn.gate"
    target_name = "decoder.blocks.1.ffn.gate"
    source = pipeline.generation_params.data[source_name]["kernel"]
    target = pipeline.generation_params.data[target_name]["kernel"]

    binary_source_params =
      Map.new(pipeline.generation_params.data[source_name], fn {name, tensor} ->
        {name, Nx.backend_copy(tensor, Nx.BinaryBackend)}
      end)

    generation_params = %{
      pipeline.generation_params
      | data: Map.put(pipeline.generation_params.data, source_name, binary_source_params)
    }

    pipeline = %{pipeline | generation_params: generation_params}

    blended = DecoderPipeline.blend_layer!(pipeline, 0, 1, 0.25)
    actual = blended.generation_params.data[target_name]["kernel"]
    expected = Nx.add(Nx.multiply(target, 0.75), Nx.multiply(source, 0.25))

    assert Nx.all_close(actual, expected)

    assert Nx.to_binary(pipeline.generation_params.data[target_name]["kernel"]) ==
             Nx.to_binary(target)

    assert blended.generation_predict_fun == pipeline.generation_predict_fun
  end

  test "rejects an invalid layer blend weight" do
    {runtime, _inputs} = runtime(nil, num_blocks: 3)
    pipeline = DecoderPipeline.extract!(runtime, [1, 2])

    assert {:error, message} = DecoderPipeline.blend_layer(pipeline, 0, 1, 1.1)
    assert message =~ "source weight must be between"
  end

  test "saves and reloads a transplanted pipeline independently" do
    {runtime, inputs} = runtime(nil, num_blocks: 3)

    pipeline =
      runtime
      |> DecoderPipeline.extract!([1, 2])
      |> DecoderPipeline.transplant_layer!(0, 1)

    path =
      Path.join(
        System.tmp_dir!(),
        "gemma-decoder-artifact-#{System.unique_integer([:positive])}"
      )

    on_exit(fn -> File.rm_rf(path) end)

    DecoderPipelineArtifact.save!(pipeline, path,
      transplants: [%{source: 0, target: 1}],
      blends: [%{source: 0, target: 1, weight: 0.1}]
    )

    backend = {Torchx.Backend, device: :cpu}
    artifact = DecoderPipelineArtifact.load!(path, backend, load_tokenizer: false)
    reloaded = DecoderPipelineArtifact.build_pipeline!(artifact, backend)

    target_name = "decoder.blocks.1.ffn.gate"
    source_name = "decoder.blocks.0.ffn.gate"

    assert Nx.to_binary(reloaded.generation_params.data[target_name]["kernel"]) ==
             Nx.to_binary(reloaded.generation_params.data[source_name]["kernel"])

    assert artifact.manifest.transplants == [%{source: 0, target: 1}]
    assert artifact.manifest.blends == [%{source: 0, target: 1, weight: 0.1}]

    assert {:ok, token_ids} =
             DecoderPipeline.generate_prepared(reloaded, inputs,
               max_new_tokens: 2,
               min_new_tokens: 3
             )

    assert length(token_ids) == 2
  end

  test "rebuilds cache-aware generation from separate prefix and tail components" do
    {runtime, inputs} = runtime()
    pipeline = DecoderPipeline.extract!(runtime, [1])
    path = Path.join(System.tmp_dir!(), "gemma-prefix-#{System.unique_integer([:positive])}")
    on_exit(fn -> File.rm_rf(path) end)

    DecoderBlockArtifact.save_prefix!(pipeline, path)
    backend = {Torchx.Backend, device: :cpu}
    prefix = DecoderBlockArtifact.load_prefix!(path, backend)
    split = DecoderBlockArtifact.build_split_pipeline!(prefix, pipeline.tail, backend)

    assert {:ok, expected} =
             DecoderPipeline.generate_prepared(pipeline, inputs,
               max_new_tokens: 2,
               min_new_tokens: 3,
               execution: :split
             )

    for execution <- [:composed, :split] do
      assert {:ok, ^expected} =
               DecoderPipeline.generate_prepared(split, inputs,
                 max_new_tokens: 2,
                 min_new_tokens: 3,
                 execution: execution
               )
    end

    bypassed =
      DecoderBlockArtifact.build_split_pipeline!(prefix, pipeline.tail, backend,
        bypass_layers: [0]
      )

    assert {:ok, bypassed_ids} =
             DecoderPipeline.generate_prepared(bypassed, inputs,
               max_new_tokens: 2,
               min_new_tokens: 3,
               execution: :composed
             )

    assert length(bypassed_ids) == 2
  end

  test "preserves packed weights and scales in separate prefix and tail artifacts" do
    {runtime, _inputs} = runtime()
    pipeline = DecoderPipeline.extract!(runtime, [1])
    packed = Nx.tensor([[0x01234567, 0x89ABCDEF]], type: :u32)
    scales = Nx.tensor([[0.25, 0.5]], type: :bf16)

    pipeline =
      put_artifact_parameters(
        pipeline,
        :prefix,
        "decoder.blocks.0.ffn.gate",
        packed,
        scales
      )

    pipeline =
      put_artifact_parameters(
        pipeline,
        :tail,
        "decoder.blocks.1.ffn.gate",
        packed,
        scales
      )

    root =
      Path.join(System.tmp_dir!(), "gemma-packed-split-#{System.unique_integer([:positive])}")

    prefix_path = Path.join(root, "prefix")
    tail_path = Path.join(root, "tail")
    on_exit(fn -> File.rm_rf(root) end)

    DecoderBlockArtifact.save_prefix!(pipeline, prefix_path)
    DecoderBlockArtifact.save_tail!(pipeline.tail, tail_path, verification_sequence_length: 3)

    prefix = DecoderBlockArtifact.load_prefix!(prefix_path, Nx.BinaryBackend)
    tail = DecoderBlockArtifact.load_tail!(tail_path, Nx.BinaryBackend)

    for {params, node} <- [
          {prefix.prefix.params, "decoder.blocks.0.ffn.gate"},
          {tail.params, "decoder.blocks.1.ffn.gate"}
        ] do
      assert Nx.type(params.data[node]["packed"]) == {:u, 32}
      assert Nx.type(params.data[node]["scales"]) == {:bf, 16}
      assert Nx.to_binary(params.data[node]["packed"]) == Nx.to_binary(packed)
      assert Nx.to_binary(params.data[node]["scales"]) == Nx.to_binary(scales)
    end
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

  defp put_artifact_parameters(pipeline, component, node, packed, scales) do
    part = Map.fetch!(pipeline, component)

    params =
      Map.update!(part.params.data, node, fn parameters ->
        parameters
        |> Map.put("packed", packed)
        |> Map.put("scales", scales)
      end)

    Map.put(pipeline, component, %{part | params: %{part.params | data: params}})
  end

  defp runtime(backend \\ nil, opts \\ [])

  defp runtime(nil, opts), do: build_runtime(nil, opts)

  defp runtime(backend, opts) do
    Nx.with_default_backend(backend, fn -> build_runtime(backend, opts) end)
  end

  defp build_runtime(backend, opts) do
    num_blocks = Keyword.get(opts, :num_blocks, 2)

    layer_types =
      if num_blocks == 3,
        do: [:sliding_attention, :sliding_attention, :full_attention],
        else: [:sliding_attention, :full_attention]

    spec =
      Bumblebee.configure(Model,
        vocab_size: 32,
        max_positions: 16,
        hidden_size: 8,
        intermediate_size: 16,
        num_blocks: num_blocks,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        num_global_key_value_heads: 1,
        attention_head_size: 4,
        global_attention_head_size: 4,
        layer_types: layer_types,
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
