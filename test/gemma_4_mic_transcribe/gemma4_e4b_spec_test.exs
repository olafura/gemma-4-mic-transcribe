defmodule Gemma4MicTranscribe.Gemma4E4BSpecTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4E4B.Spec

  # Trimmed copy of google/gemma-4-E4B-it config.json.
  defp e4b_config do
    %{
      "audio_config" => %{
        "attention_chunk_size" => 12,
        "attention_context_left" => 13,
        "attention_context_right" => 0,
        "attention_logit_cap" => 50.0,
        "conv_kernel_size" => 5,
        "hidden_act" => "silu",
        "hidden_size" => 1024,
        "num_attention_heads" => 8,
        "num_hidden_layers" => 12,
        "output_proj_dims" => 1536,
        "residual_weight" => 0.5,
        "rms_norm_eps" => 1.0e-6,
        "subsampling_conv_channels" => [128, 32],
        "use_clipped_linears" => true
      },
      "audio_token_id" => 258_881,
      "boa_token_id" => 256_000,
      "eoa_token_index" => 258_883,
      "eos_token_id" => [1, 106],
      "text_config" => %{
        "attention_k_eq_v" => false,
        "final_logit_softcapping" => 30.0,
        "global_head_dim" => 512,
        "head_dim" => 256,
        "hidden_activation" => "gelu_pytorch_tanh",
        "hidden_size" => 2560,
        "hidden_size_per_layer_input" => 256,
        "intermediate_size" => 10_240,
        "layer_types" => List.duplicate("sliding_attention", 42),
        "num_attention_heads" => 8,
        "num_hidden_layers" => 42,
        "num_key_value_heads" => 2,
        "num_kv_shared_layers" => 18,
        "rms_norm_eps" => 1.0e-6,
        "rope_parameters" => %{
          "full_attention" => %{"partial_rotary_factor" => 0.25, "rope_theta" => 1_000_000.0},
          "sliding_attention" => %{"rope_theta" => 10_000.0}
        },
        "sliding_window" => 512,
        "tie_word_embeddings" => true,
        "use_double_wide_mlp" => false,
        "vocab_size" => 262_144
      }
    }
  end

  defp load, do: Bumblebee.HuggingFace.Transformers.Config.load(%Spec{}, e4b_config())

  test "loads the E4B decoder configuration" do
    spec = load()

    assert spec.hidden_size == 2560
    assert spec.num_blocks == 42
    assert spec.num_attention_heads == 8
    assert spec.num_key_value_heads == 2
    assert spec.attention_head_size == 256
    assert spec.global_attention_head_size == 512
    assert spec.intermediate_size == 10_240
    assert spec.attention_window_size == 512
    assert spec.rotary_embedding_base == 1_000_000.0
    assert spec.rotary_embedding_base_local == 10_000.0
    assert spec.full_attention_rotary_percentage == 0.25
    refute spec.attention_k_eq_v
  end

  test "loads the features that distinguish E4B from the 12B unified model" do
    spec = load()

    # per-layer input embeddings and KV sharing do not exist in the 12B path
    assert spec.hidden_size_per_layer_input == 256
    assert spec.num_kv_shared_layers == 18
    refute spec.use_double_wide_mlp
  end

  test "loads the conformer audio encoder configuration" do
    spec = load()

    assert spec.audio_hidden_size == 1024
    assert spec.audio_num_blocks == 12
    assert spec.audio_num_attention_heads == 8
    assert spec.audio_conv_kernel_size == 5
    assert spec.audio_subsampling_conv_channels == [128, 32]
    assert spec.audio_output_proj_dims == 1536
    assert spec.audio_attention_chunk_size == 12
    assert spec.audio_attention_context_left == 13
    assert spec.audio_attention_context_right == 0
    assert spec.audio_attention_logit_cap == 50.0
    assert spec.audio_residual_weight == 0.5
    assert spec.audio_activation == :silu
    assert spec.audio_use_clipped_linears
  end

  test "the last blocks share key values with the last computing block" do
    spec = load()

    # 42 blocks, last 18 shared -> blocks 0..23 compute, 24..41 reuse block 23
    refute Spec.kv_shared_layer?(spec, 23)
    assert Spec.kv_shared_layer?(spec, 24)
    assert Spec.kv_shared_layer?(spec, 41)
    assert Spec.kv_source_layer(spec) == 23
  end

  test "double-wide MLPs are restricted to the KV-sharing suffix" do
    spec = %{load() | use_double_wide_mlp: true}

    refute Spec.double_wide_mlp_layer?(spec, 23)
    assert Spec.double_wide_mlp_layer?(spec, 24)
    assert Spec.double_wide_mlp_layer?(spec, 41)
  end

  test "audio subsampling halves the time axis once per conv" do
    spec = load()

    # two convs -> a quarter of the input frames, rounding up
    assert Spec.audio_subsampled_length(spec, 400) == 100
    assert Spec.audio_subsampled_length(spec, 401) == 101
    assert Spec.audio_subsampled_length(spec, 1) == 1
  end

  test "layer types normalize to atoms" do
    spec = load()

    assert Enum.all?(spec.layer_types, &(&1 == :sliding_attention))
    assert length(spec.layer_types) == 42
  end
end
