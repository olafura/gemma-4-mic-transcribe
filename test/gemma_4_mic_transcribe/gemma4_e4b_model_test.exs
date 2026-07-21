defmodule Gemma4MicTranscribe.Gemma4E4BModelTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4E4B.Model
  alias Gemma4MicTranscribe.Gemma4E4B.Spec

  defp tiny_model do
    struct(
      Model,
      Map.from_struct(%Spec{
        vocab_size: 32,
        vocab_size_per_layer_input: 32,
        hidden_size: 8,
        hidden_size_per_layer_input: 4,
        intermediate_size: 16,
        num_blocks: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        attention_head_size: 4,
        global_attention_head_size: 4,
        attention_window_size: 4,
        num_kv_shared_layers: 2,
        max_positions: 32,
        layer_types: [:sliding_attention, :full_attention, :sliding_attention, :full_attention],
        audio_hidden_size: 8,
        audio_num_blocks: 1,
        audio_num_attention_heads: 2,
        audio_conv_kernel_size: 3,
        audio_subsampling_conv_channels: [4, 2],
        audio_attention_chunk_size: 2,
        audio_attention_context_left: 2,
        audio_mel_bins: 8,
        audio_token_id: 7,
        final_logit_softcapping: nil
      })
    )
  end

  test "model runs end to end over text and audio tokens" do
    model_spec = tiny_model()
    graph = Model.model(model_spec)
    {init_fun, predict_fun} = Axon.build(graph)

    # 8 mel frames subsample to 2 encoder frames, so two audio placeholders
    inputs = %{
      "input_ids" => Nx.tensor([[2, 7, 7, 3]], type: :s64),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1]], type: :s64),
      "position_ids" => Nx.tensor([[0, 1, 2, 3]], type: :s64),
      "input_features" => Nx.broadcast(0.1, {1, 8, 8})
    }

    params = init_fun.(inputs, Axon.ModelState.empty())
    outputs = predict_fun.(params, inputs)

    assert Nx.shape(outputs.logits) == {1, 4, model_spec.vocab_size}
    assert outputs.logits |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
  end

  test "audio changes only the placeholder positions" do
    model_spec = tiny_model()
    graph = Model.model(model_spec)
    {init_fun, predict_fun} = Axon.build(graph)

    base = %{
      "input_ids" => Nx.tensor([[2, 7, 7, 3]], type: :s64),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1]], type: :s64),
      "position_ids" => Nx.tensor([[0, 1, 2, 3]], type: :s64),
      "input_features" => Nx.broadcast(0.1, {1, 8, 8})
    }

    params = init_fun.(base, Axon.ModelState.empty())
    quiet = predict_fun.(params, base).logits

    # a structurally different signal, not merely a louder one: the subsample
    # LayerNorm cancels any uniform rescaling of the same pattern
    other_features =
      Nx.iota({1, 8, 8}, type: :f32) |> Nx.remainder(3) |> Nx.multiply(0.3)

    loud = predict_fun.(params, %{base | "input_features" => other_features}).logits

    # position 0 precedes the audio, so causal attention keeps it unchanged
    assert Nx.all_close(quiet[[0, 0]], loud[[0, 0]]) |> Nx.to_number() == 1
    # the final position attends over the audio, so it must move
    refute Nx.all_close(quiet[[0, 3]], loud[[0, 3]]) |> Nx.to_number() == 1
  end

  test "params mapping targets the published checkpoint names" do
    mapping = Bumblebee.HuggingFace.Transformers.Model.params_mapping(tiny_model())

    assert mapping["embedder.token_embedding"] == "model.language_model.embed_tokens"

    assert mapping["embedder.per_layer_embedding"] ==
             "model.language_model.embed_tokens_per_layer"

    assert mapping["decoder.blocks.{n}.per_layer.input_gate"] ==
             "model.language_model.layers.{n}.per_layer_input_gate"

    # conv kernels are transposed from PyTorch layout, so they map through a
    # builder rather than a bare name
    assert %{
             "kernel" =>
               {[{"model.audio_tower.layers.{n}.lconv1d.depthwise_conv1d", "weight"}], _}
           } =
             mapping["audio_encoder.blocks.{n}.conv.depthwise_conv1d"]

    assert mapping["embed_audio.embedding_projection"] ==
             "model.embed_audio.embedding_projection"
  end

  test "clipped linears map their nested weight and their bounds" do
    mapping = Bumblebee.HuggingFace.Transformers.Model.params_mapping(tiny_model())
    entry = mapping["audio_encoder.blocks.{n}.attention.query"]

    # the weight lives under a nested "linear", the bounds sit beside it
    assert %{"kernel" => {[{source, "weight"}], _}} = entry
    assert source == "model.audio_tower.layers.{n}.self_attn.q_proj.linear"

    for bound <- ["input_min", "input_max", "output_min", "output_max"] do
      assert %{^bound => {[{"model.audio_tower.layers.{n}.self_attn.q_proj", ^bound}], _}} = entry
    end
  end

  test "cache allocates entries only for blocks that compute key values" do
    model_spec = tiny_model()
    cache = Model.init_cache(model_spec, 1, 16, %{})

    computing = elem(cache.blocks, 0).self_attention.key
    shared = elem(cache.blocks, 3).self_attention.key

    assert Nx.shape(computing) == {1, 16, 4, 4}
    # shared blocks reuse another block's state, so they hold a placeholder
    assert Nx.shape(shared) == {1, 1, 1, 1}
  end
end
