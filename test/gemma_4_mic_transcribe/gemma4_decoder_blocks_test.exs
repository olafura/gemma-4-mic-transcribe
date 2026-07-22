defmodule Gemma4MicTranscribe.Gemma4.DecoderBlocksTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4.DecoderBlocks
  alias Gemma4MicTranscribe.Gemma4.LayerProbe
  alias Gemma4MicTranscribe.Gemma4Unified.Model

  test "extracts one block and reproduces its in-model output" do
    {runtime, inputs} = runtime()

    assert {:ok, report} =
             LayerProbe.run(runtime, inputs,
               layers: [0],
               positions: :all,
               capture: [:block_input, :hidden_state],
               include_activations: true
             )

    block = DecoderBlocks.extract!(runtime, 0)

    assert block.id == "language_model.layer.0"
    assert block.layer_type == :sliding_attention
    assert block.input_size == 8
    assert block.parameter_count > 0

    assert Map.keys(block.params.data)
           |> Enum.all?(&String.starts_with?(&1, "decoder.blocks.0."))

    block_input = report.activations["0:block_input"]

    output =
      DecoderBlocks.run!(block, block_input,
        position_ids: inputs["position_ids"],
        attention_mask: inputs["attention_mask"]
      )

    assert Nx.all_close(output, report.activations["0:hidden_state"], atol: 1.0e-5, rtol: 1.0e-5)
           |> Nx.to_number() == 1
  end

  test "generates contiguous positions and an all-visible mask by default" do
    {runtime, inputs} = runtime()

    assert {:ok, report} =
             LayerProbe.run(runtime, inputs,
               layers: [0],
               positions: :all,
               capture: [:block_input],
               include_activations: true
             )

    block = DecoderBlocks.extract!(runtime, 0)
    block_input = report.activations["0:block_input"]

    assert {:ok, default_output} = DecoderBlocks.run(block, block_input)

    assert {:ok, explicit_output} =
             DecoderBlocks.run(block, block_input,
               position_ids: inputs["position_ids"],
               attention_mask: inputs["attention_mask"]
             )

    assert Nx.all_close(default_output, explicit_output) |> Nx.to_number() == 1
  end

  defp runtime do
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
