defmodule Gemma4MicTranscribe.Gemma4UnifiedTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4Unified.AudioFeatureExtractor
  alias Gemma4MicTranscribe.Gemma4Unified.Input
  alias Gemma4MicTranscribe.Gemma4Unified.Model
  alias Gemma4MicTranscribe.ModelCatalog
  alias Gemma4MicTranscribe.Gemma4Unified.Prompt

  test "audio feature extractor chunks raw 16 kHz audio into 640-sample soft tokens" do
    features = AudioFeatureExtractor.extract(List.duplicate(0.25, 641))

    assert features.token_count == 2
    assert Nx.shape(features.input_features) == {2, 640}
    assert Nx.to_flat_list(features.attention_mask) == [1, 1]
    assert features.input_features[1][1] |> Nx.to_number() == 0.0
  end

  test "audio feature extractor truncates at max token count" do
    features = AudioFeatureExtractor.extract(List.duplicate(0.0, 2_000), max_tokens: 2)

    assert features.token_count == 2
    assert Nx.shape(features.input_features) == {2, 640}
  end

  test "prompt inserts one audio placeholder per audio token" do
    prompt = Prompt.build("System", "Transcribe.", 3)

    assert prompt =~ "<|audio>"
    assert prompt =~ "<audio|>"
    assert prompt =~ "<|turn>user"
    assert prompt =~ "<|turn>model"
    assert prompt =~ "System"
    assert prompt =~ "Transcribe."
    assert prompt |> String.split(Prompt.audio_placeholder()) |> length() == 4
  end

  test "input builder combines prompt and audio features" do
    input = Input.build(List.duplicate(0.0, 640), prompt: "Transcribe.")

    assert input.audio.token_count == 1
    assert input.prompt =~ Prompt.audio_placeholder()
  end

  test "local Gemma4Unified model graph runs with a tiny config" do
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
        attention_window_size: 4,
        layer_types: [:sliding_attention, :full_attention],
        audio_token_id: 7,
        audio_embed_dim: 4,
        final_logit_softcapping: nil
      )

    model = Bumblebee.build_model(spec)
    {init_fun, predict_fun} = Axon.build(model)

    inputs = %{
      "input_ids" => Nx.tensor([[2, 7, 3]], type: :s64),
      "attention_mask" => Nx.tensor([[1, 1, 1]], type: :s64),
      "position_ids" => Nx.tensor([[0, 1, 2]], type: :s64),
      "input_features" => Nx.tensor([[[0.0, 0.1, 0.2, 0.3]]], type: {:f, 32}),
      "input_features_mask" => Nx.tensor([[1]], type: :s64)
    }

    params = init_fun.(inputs, Axon.ModelState.empty())
    outputs = predict_fun.(params, inputs)

    assert Nx.shape(outputs.logits) == {1, 3, 32}
  end

  test "model config loads Gemma4Unified composite config fields" do
    spec =
      Bumblebee.HuggingFace.Transformers.Config.load(%Model{}, %{
        "audio_token_id" => 12,
        "eos_token_id" => [1, 50],
        "initializer_range" => 0.01,
        "audio_config" => %{"audio_embed_dim" => 4, "rms_norm_eps" => 1.0e-5},
        "text_config" => %{
          "vocab_size" => 32,
          "max_position_embeddings" => 128,
          "hidden_size" => 8,
          "intermediate_size" => 16,
          "num_hidden_layers" => 2,
          "num_attention_heads" => 2,
          "num_key_value_heads" => 1,
          "num_global_key_value_heads" => 1,
          "head_dim" => 4,
          "global_head_dim" => 4,
          "hidden_activation" => "gelu_pytorch_tanh",
          "rms_norm_eps" => 1.0e-5,
          "sliding_window" => 8,
          "layer_types" => ["sliding_attention", "full_attention"],
          "final_logit_softcapping" => nil,
          "rope_parameters" => %{
            "sliding_attention" => %{"rope_theta" => 10_000.0},
            "full_attention" => %{
              "rope_theta" => 1_000_000.0,
              "partial_rotary_factor" => 0.25
            }
          }
        }
      })

    assert spec.vocab_size == 32
    assert spec.hidden_size == 8
    assert spec.audio_token_id == 12
    assert spec.audio_embed_dim == 4
    assert spec.layer_types == [:sliding_attention, :full_attention]
  end

  test "model catalog resolves the friendly Gemma4 alias" do
    assert ModelCatalog.resolve("gemma4-12b-unified") == "google/gemma-4-12B-it"
    assert ModelCatalog.resolve("google/gemma-4-12B-it") == "google/gemma-4-12B-it"
  end
end
