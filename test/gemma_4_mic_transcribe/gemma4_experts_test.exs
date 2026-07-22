defmodule Gemma4MicTranscribe.Gemma4.ExpertsTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4.Experts
  alias Gemma4MicTranscribe.Gemma4.Experts.Descriptor
  alias Gemma4MicTranscribe.Gemma4Unified.Model

  @moe_config %{
    "text_config" => %{
      "enable_moe_block" => true,
      "num_hidden_layers" => 2,
      "num_experts" => 3,
      "top_k_experts" => 2,
      "hidden_size" => 8,
      "intermediate_size" => 12,
      "moe_intermediate_size" => 4,
      "hidden_activation" => "gelu_pytorch_tanh"
    }
  }

  test "lists the shared and routed experts in every MoE layer" do
    experts = Experts.list(@moe_config)

    assert length(experts) == 8

    assert %Descriptor{
             id: "language_model.layer.0.shared",
             kind: :shared,
             layer_index: 0,
             expert_index: :shared,
             input_size: 8,
             intermediate_size: 12,
             output_size: 8,
             parameter_count: 288,
             router: nil,
             weights: %{
               gate: %{
                 tensor: "model.language_model.layers.0.mlp.gate_proj.weight",
                 checkpoint_shape: {12, 8}
               }
             }
           } = hd(experts)

    assert %Descriptor{
             id: "language_model.layer.0.expert.1",
             kind: :routed,
             expert_index: 1,
             parameter_count: 96,
             weights: %{
               gate_up: %{
                 tensor: "model.language_model.layers.0.experts.gate_up_proj",
                 checkpoint_shape: {3, 8, 8},
                 slice: %{axis: 0, index: 1, shape: {8, 8}},
                 splits: %{
                   gate: %{axis: 0, start: 0, length: 4},
                   up: %{axis: 0, start: 4, length: 4}
                 }
               },
               down: %{
                 tensor: "model.language_model.layers.0.experts.down_proj",
                 checkpoint_shape: {3, 8, 4},
                 slice: %{axis: 0, index: 1, shape: {8, 4}}
               }
             },
             router: %{
               top_k: 2,
               projection: "model.language_model.layers.0.router.proj.weight",
               per_expert_scale: "model.language_model.layers.0.router.per_expert_scale"
             }
           } = Enum.at(experts, 2)
  end

  test "can omit shared experts" do
    experts = Experts.list(@moe_config, include_shared: false)

    assert length(experts) == 6
    assert Enum.all?(experts, &(&1.kind == :routed))
  end

  test "returns no experts for dense Gemma models" do
    assert Experts.list(%Model{}) == []
    assert Experts.list(%{"text_config" => %{"enable_moe_block" => false}}) == []
  end

  test "accepts a loaded model-info or runtime-shaped map" do
    spec = %Model{
      enable_moe_block: true,
      num_blocks: 1,
      num_experts: 2,
      top_k_experts: 1,
      hidden_size: 8,
      intermediate_size: 12,
      moe_intermediate_size: 4
    }

    assert length(Experts.list(%{model_info: %{spec: spec}})) == 3
  end

  test "rejects an impossible router configuration" do
    config = put_in(@moe_config, ["text_config", "top_k_experts"], 4)

    assert_raise ArgumentError, ~r/cannot exceed num_experts/, fn ->
      Experts.list(config)
    end
  end
end
