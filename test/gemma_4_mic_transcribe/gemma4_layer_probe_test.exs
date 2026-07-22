defmodule Gemma4MicTranscribe.Gemma4.LayerProbeTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4.LayerProbe
  alias Gemma4MicTranscribe.Gemma4E4B
  alias Gemma4MicTranscribe.Gemma4Unified.Model

  test "captures bounded decoder contributions at semantic positions" do
    {runtime, inputs} = unified_runtime()

    assert {:ok, report} =
             LayerProbe.run(runtime, inputs,
               layers: [0, 1],
               positions: [:audio_begin, :first_audio, :audio_end, :last],
               capture: [:attention, :ffn, :per_layer_input, :hidden_state],
               include_activations: true
             )

    assert Enum.map(report.positions, &{&1.label, &1.index, &1.token_id}) == [
             {:audio_begin, 0, 6},
             {:first_audio, 1, 7},
             {:audio_end, 2, 8},
             {:last, 3, 3}
           ]

    assert report.unavailable == [{0, :per_layer_input}, {1, :per_layer_input}]
    assert report.layers |> Enum.map(& &1.attention) == [:sliding_attention, :full_attention]

    first_layer = hd(report.layers)
    assert length(first_layer.metrics.attention) == 4
    assert Enum.all?(first_layer.metrics.ffn, &is_float(&1.norm))
    assert Enum.all?(first_layer.metrics.ffn, &is_float(&1.hidden_norm_ratio))

    assert Nx.shape(report.activations["0:hidden_state"]) == {1, 4, 8}
    assert length(report.hidden_state_similarity) == 1
  end

  test "applies a bounded logit lens to captured hidden states" do
    {runtime, inputs} = unified_runtime()

    assert {:ok, report} =
             LayerProbe.run(runtime, inputs,
               layers: [0, 1],
               positions: [0, :last],
               capture: [:hidden_state],
               top_k_logits: 3
             )

    assert Enum.map(report.logit_lens, & &1.layer) == [0, 1]

    for layer <- report.logit_lens,
        position <- layer.positions do
      assert length(position.candidates) == 3
      assert Enum.all?(position.candidates, &is_integer(&1.token_id))
      assert Enum.all?(position.candidates, &is_float(&1.score))
    end

    outputs = runtime.predict_fun.(runtime.model_info.params, inputs)
    {expected_scores, expected_ids} = Nx.top_k(outputs.logits[0][3], k: 3)

    last_candidates =
      report.logit_lens
      |> Enum.find(&(&1.layer == 1))
      |> then(&Enum.find(&1.positions, fn position -> position.position == :last end))
      |> Map.fetch!(:candidates)

    assert Enum.map(last_candidates, & &1.token_id) == Nx.to_flat_list(expected_ids)

    Enum.zip(last_candidates, Nx.to_flat_list(expected_scores))
    |> Enum.each(fn {candidate, expected} ->
      assert_in_delta candidate.score, expected, 1.0e-5
    end)
  end

  test "captures E4B's actual gated per-layer embedding contribution" do
    {runtime, inputs} = e4b_runtime()

    assert {:ok, report} =
             LayerProbe.run(runtime, inputs,
               layers: [0, 1],
               positions: [:first_audio],
               capture: [:per_layer_input, :hidden_state]
             )

    assert report.unavailable == []
    assert {0, :per_layer_input} in report.available
    assert {1, :per_layer_input} in report.available

    for layer <- report.layers do
      assert [%{position: :first_audio, norm: norm}] = layer.metrics.per_layer_input
      assert is_float(norm)
    end
  end

  test "validates selectors, layer bounds, and logit-lens dependencies" do
    {runtime, inputs} = unified_runtime()

    assert {:error, message} = LayerProbe.run(runtime, inputs, layers: [2])
    assert message =~ "layer index"

    assert {:error, message} =
             LayerProbe.run(runtime, inputs,
               capture: [:ffn],
               top_k_logits: 2
             )

    assert message =~ "requires :hidden_state"

    assert {:error, message} = LayerProbe.run(runtime, inputs, positions: ["missing"])
    assert message =~ "without a tokenizer"
  end

  defp unified_runtime do
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

    runtime(spec, inputs)
  end

  defp e4b_runtime do
    spec =
      struct(
        Gemma4E4B.Model,
        Map.from_struct(%Gemma4E4B.Spec{
          vocab_size: 32,
          vocab_size_per_layer_input: 32,
          hidden_size: 8,
          hidden_size_per_layer_input: 4,
          intermediate_size: 16,
          num_blocks: 2,
          num_attention_heads: 4,
          num_key_value_heads: 2,
          num_global_key_value_heads: 2,
          attention_head_size: 4,
          global_attention_head_size: 4,
          attention_window_size: 4,
          num_kv_shared_layers: 0,
          max_positions: 32,
          layer_types: [:sliding_attention, :full_attention],
          audio_hidden_size: 8,
          audio_num_blocks: 1,
          audio_num_attention_heads: 2,
          audio_conv_kernel_size: 3,
          audio_subsampling_conv_channels: [4, 2],
          audio_attention_chunk_size: 2,
          audio_attention_context_left: 2,
          audio_mel_bins: 8,
          audio_token_id: 7,
          boa_token_id: 6,
          eoa_token_id: 8,
          final_logit_softcapping: nil
        })
      )

    inputs = %{
      "input_ids" => Nx.tensor([[6, 7, 7, 8]], type: :s64),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1]], type: :s64),
      "position_ids" => Nx.tensor([[0, 1, 2, 3]], type: :s64),
      "input_features" => Nx.broadcast(0.1, {1, 8, 8})
    }

    runtime(spec, inputs)
  end

  defp runtime(spec, inputs) do
    model = Bumblebee.build_model(spec)
    {init_fun, predict_fun} = Axon.build(model)
    params = init_fun.(inputs, Axon.ModelState.empty())

    {%{
       model_name: "tiny-gemma4",
       backend: nil,
       tokenizer: nil,
       predict_fun: predict_fun,
       model_info: %{model: model, params: params, spec: spec}
     }, inputs}
  end
end
