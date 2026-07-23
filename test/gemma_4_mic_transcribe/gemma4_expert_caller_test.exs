defmodule Gemma4MicTranscribe.Gemma4.ExpertCallerTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4.ExpertCaller
  alias Gemma4MicTranscribe.Gemma4.ExtractedDecoderLayer

  test "produces the routed expert input after layer-0 attention" do
    embeddings =
      Nx.tensor(
        [
          [0.25, -0.5, 0.75, 1.0],
          [-1.0, 0.125, 0.5, -0.25],
          [0.5, 0.75, -0.25, 0.125]
        ],
        type: :bf16
      )

    attention_params = %{
      input_norm: Nx.tensor([1.0, 0.75, 1.25, 0.875], type: :bf16),
      post_attention_norm: Nx.tensor([0.875, 1.0, 0.75, 1.25], type: :bf16),
      query: ramp({4, 4}, 13),
      key: ramp({2, 4}, 17),
      value: ramp({2, 4}, 19),
      output: ramp({4, 4}, 23),
      query_norm: Nx.tensor([1.0, 0.75], type: :bf16),
      key_norm: Nx.tensor([0.875, 1.25], type: :bf16)
    }

    moe_params = %{
      router_proj: ramp({4, 4}, 29),
      router_scale: Nx.tensor([0.75, 1.25, 0.5, 1.5], type: :bf16),
      router_per_expert_scale: Nx.tensor([0.5, 0.75, 1.25, 1.5], type: :bf16),
      norm_pre_experts: Nx.tensor([1.25, 0.75, 1.0, 0.875], type: :bf16)
    }

    result =
      ExpertCaller.forward(embeddings, attention_params, moe_params,
        top_k: 2,
        eps: 1.0e-6,
        router_scalar: 0.5,
        embedding_scalar: 2.0,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        rope_theta: 10_000.0,
        rotary_angles: 1,
        alternative_attention: false,
        sliding_window: 2
      )

    assert Nx.shape(result.residual_after_attention) == {3, 4}
    assert Nx.shape(result.expert_input) == {3, 4}
    assert Nx.shape(result.router_probabilities) == {3, 4}
    assert Nx.shape(result.top_k_indices) == {3, 2}
    assert Nx.shape(result.top_k_weights) == {3, 2}

    refute result.expert_input
           |> Nx.as_type(:f32)
           |> Nx.is_nan()
           |> Nx.any()
           |> Nx.to_number() == 1
  end

  test "feeds a preceding hidden state through a complete later decoder layer" do
    input =
      Nx.tensor(
        [
          [0.25, -0.5, 0.75, 1.0],
          [-1.0, 0.125, 0.5, -0.25]
        ],
        type: :bf16
      )

    attention_params = %{
      input_norm: Nx.tensor([1.0, 0.75, 1.25, 0.875], type: :bf16),
      post_attention_norm: Nx.tensor([0.875, 1.0, 0.75, 1.25], type: :bf16),
      query: ramp({4, 4}, 13),
      key: ramp({2, 4}, 17),
      value: ramp({2, 4}, 19),
      output: ramp({4, 4}, 23),
      query_norm: Nx.tensor([1.0, 0.75], type: :bf16),
      key_norm: Nx.tensor([0.875, 1.25], type: :bf16)
    }

    moe_params = %{
      experts_gate_up: ramp({4, 6, 4}, 97),
      experts_down: ramp({4, 4, 3}, 83),
      shared_gate: ramp({6, 4}, 71),
      shared_up: ramp({6, 4}, 61),
      shared_down: ramp({4, 6}, 53),
      router_proj: ramp({4, 4}, 29),
      router_scale: Nx.tensor([0.75, 1.25, 0.5, 1.5], type: :bf16),
      router_per_expert_scale: Nx.tensor([0.5, 0.75, 1.25, 1.5], type: :bf16),
      norm_pre_shared: Nx.tensor([0.75, 1.0, 1.25, 1.5], type: :bf16),
      norm_post_shared: Nx.tensor([1.0, 0.875, 1.125, 0.75], type: :bf16),
      norm_pre_experts: Nx.tensor([1.25, 0.75, 1.0, 0.875], type: :bf16),
      norm_post_experts: Nx.tensor([0.875, 1.0, 0.75, 1.25], type: :bf16),
      norm_post_combined: Nx.tensor([1.0, 1.25, 0.875, 0.75], type: :bf16),
      layer_scalar: Nx.tensor([0.875], type: :bf16)
    }

    result =
      ExtractedDecoderLayer.forward(input, attention_params, moe_params,
        top_k: 2,
        eps: 1.0e-6,
        router_scalar: 0.5,
        embedding_scalar: 1.0,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        rope_theta: 10_000.0,
        rotary_angles: 1,
        alternative_attention: false,
        sliding_window: 2
      )

    assert Nx.shape(result.output) == {2, 4}
    assert Nx.shape(result.residual_after_attention) == {2, 4}
    assert Nx.shape(result.top_k_indices) == {2, 2}

    assert result.output
           |> Nx.as_type(:f32)
           |> Nx.subtract(Nx.as_type(input, :f32))
           |> Nx.abs()
           |> Nx.reduce_max()
           |> Nx.to_number() > 0.0
  end

  test "runs proportional partial RoPE with the full-attention shared K/V projection" do
    embeddings = Nx.tensor([[0.25, -0.5, 0.75, 1.0]], type: :bf16)

    attention_params = %{
      input_norm: Nx.tensor([1.0, 0.75, 1.25, 0.875], type: :bf16),
      post_attention_norm: Nx.tensor([0.875, 1.0, 0.75, 1.25], type: :bf16),
      query: ramp({8, 4}, 13),
      key: ramp({4, 4}, 17),
      output: ramp({4, 8}, 23),
      query_norm: Nx.tensor([1.0, 0.75, 1.25, 0.875], type: :bf16),
      key_norm: Nx.tensor([0.875, 1.25, 1.0, 0.75], type: :bf16)
    }

    moe_params = %{
      router_proj: ramp({4, 4}, 29),
      router_scale: Nx.tensor([0.75, 1.25, 0.5, 1.5], type: :bf16),
      router_per_expert_scale: Nx.tensor([0.5, 0.75, 1.25, 1.5], type: :bf16),
      norm_pre_experts: Nx.tensor([1.25, 0.75, 1.0, 0.875], type: :bf16)
    }

    result =
      ExpertCaller.forward(embeddings, attention_params, moe_params,
        top_k: 2,
        eps: 1.0e-6,
        router_scalar: 0.5,
        embedding_scalar: 1.0,
        heads: 2,
        kv_heads: 1,
        head_dim: 4,
        rope_theta: 1_000_000.0,
        partial_rotary_factor: 0.25,
        rotary_angles: 0,
        alternative_attention: true,
        sliding_window: 2_147_483_647
      )

    assert Nx.shape(result.residual_after_attention) == {1, 4}
    assert Nx.shape(result.top_k_indices) == {1, 2}
  end

  test "output-only decoder execution preserves the device result contract" do
    layer = %ExtractedDecoderLayer{
      manifest: %{hidden_size: 4},
      attention_params: %{},
      moe_params: %{},
      output_predict_fun: fn input, %{}, %{} -> Nx.add(input, 1) end,
      backend: Nx.BinaryBackend
    }

    input = Nx.tensor([[0.25, -0.5, 0.75, 1.0]], type: :f32)
    output = ExtractedDecoderLayer.run_output(layer, input)

    assert Nx.type(output) == {:bf, 16}
    assert Nx.to_flat_list(output) == [1.25, 0.5, 1.75, 2.0]

    assert_raise ArgumentError, ~r/expected decoder input shape/, fn ->
      ExtractedDecoderLayer.run_output(layer, Nx.tensor([[1.0, 2.0]]))
    end
  end

  defp ramp(shape, divisor) do
    shape
    |> Nx.iota(type: :f32)
    |> Nx.add(1)
    |> Nx.divide(divisor)
    |> Nx.as_type(:bf16)
  end
end
