defmodule Gemma4MicTranscribe.Gemma4.FFNsTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4
  alias Gemma4MicTranscribe.Gemma4.FFNs
  alias Gemma4MicTranscribe.Gemma4.FFNs.Descriptor
  alias Gemma4MicTranscribe.Gemma4Unified.Model

  @dense_config %{
    "text_config" => %{
      "enable_moe_block" => false,
      "num_hidden_layers" => 2,
      "hidden_size" => 8,
      "intermediate_size" => 16,
      "hidden_activation" => "gelu_pytorch_tanh"
    }
  }

  test "lists each dense layer FFN with checkpoint and Axon layouts" do
    assert [first, second] = Gemma4.list_ffns(@dense_config)

    assert %Descriptor{
             id: "language_model.layer.0.ffn",
             kind: :dense,
             layer_index: 0,
             input_size: 8,
             intermediate_size: 16,
             output_size: 8,
             activation: "gelu_pytorch_tanh",
             parameter_count: 384,
             operation: "down(activation(gate(x)) * up(x))",
             weights: %{
               gate: %{
                 checkpoint_tensor: "model.language_model.layers.0.mlp.gate_proj.weight",
                 checkpoint_shape: {16, 8},
                 axon_parameter: "decoder.blocks.0.ffn.gate.kernel",
                 axon_shape: {8, 16}
               },
               up: %{
                 checkpoint_tensor: "model.language_model.layers.0.mlp.up_proj.weight"
               },
               down: %{
                 checkpoint_shape: {8, 16},
                 axon_shape: {16, 8}
               }
             },
             context: %{
               input: :after_pre_feedforward_rms_norm,
               output: :before_post_feedforward_rms_norm_and_residual,
               residual: true
             }
           } = first

    assert second.layer_index == 1
  end

  test "can list selected layers" do
    assert [%{layer_index: 1}] = FFNs.list(@dense_config, layers: [1])

    assert_raise ArgumentError, ~r/layer index/, fn ->
      FFNs.list(@dense_config, layers: [2])
    end
  end

  test "uses double-wide FFNs only in the KV-sharing suffix" do
    config =
      @dense_config
      |> put_in(["text_config", "use_double_wide_mlp"], true)
      |> put_in(["text_config", "num_kv_shared_layers"], 1)

    assert [
             %{layer_index: 0, intermediate_size: 16, parameter_count: 384},
             %{layer_index: 1, intermediate_size: 32, parameter_count: 768}
           ] = FFNs.list(config)
  end

  test "identifies the dense FFN in an MoE layer as the shared expert" do
    config = put_in(@dense_config, ["text_config", "enable_moe_block"], true)

    assert Enum.all?(FFNs.list(config), &(&1.kind == :shared))
  end

  test "accepts model-info and runtime-shaped maps" do
    spec = %Model{num_blocks: 1, hidden_size: 8, intermediate_size: 16}

    assert [%{kind: :dense, layer_index: 0}] =
             FFNs.list(%{model_info: %{spec: spec}})
  end

  test "expert and FFN convenience functions use separate catalogs" do
    assert length(Gemma4.list_ffns(@dense_config)) == 2
    assert Gemma4.list_experts(@dense_config) == []
  end
end
