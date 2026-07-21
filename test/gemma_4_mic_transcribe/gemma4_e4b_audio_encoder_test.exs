defmodule Gemma4MicTranscribe.Gemma4E4BAudioEncoderTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4E4B.AudioEncoder
  alias Gemma4MicTranscribe.Gemma4E4B.Spec

  defp tiny_spec do
    %Spec{
      hidden_size: 16,
      audio_hidden_size: 8,
      audio_num_blocks: 2,
      audio_num_attention_heads: 2,
      audio_conv_kernel_size: 3,
      audio_subsampling_conv_channels: [4, 2],
      audio_attention_chunk_size: 2,
      audio_attention_context_left: 2,
      audio_attention_context_right: 0
    }
  end

  test "chunk mask keeps a query inside its chunk plus left context" do
    mask =
      AudioEncoder.chunk_mask(length: 6, chunk_size: 2, context_left: 2, context_right: 0)
      |> Nx.to_flat_list()
      |> Enum.chunk_every(6)

    # query 0 is in chunk [0,1]; left context reaches back 2 before the chunk
    # start (nothing there), so it sees its own chunk only
    assert Enum.at(mask, 0) == [1, 1, 0, 0, 0, 0]

    # query 2 opens chunk [2,3] and reaches back to frame 0
    assert Enum.at(mask, 2) == [1, 1, 1, 1, 0, 0]

    # query 5 is in chunk [4,5] and reaches back to frame 2
    assert Enum.at(mask, 5) == [0, 0, 1, 1, 1, 1]
  end

  test "chunk mask can look right when context_right is set" do
    mask =
      AudioEncoder.chunk_mask(length: 4, chunk_size: 2, context_left: 0, context_right: 1)
      |> Nx.to_flat_list()
      |> Enum.chunk_every(4)

    # chunk [0,1] plus one frame after the chunk ends
    assert Enum.at(mask, 0) == [1, 1, 1, 0]
    assert Enum.at(mask, 1) == [1, 1, 1, 0]
  end

  test "relative shift aligns each row to its own offset" do
    # rows are distinct so the shift is visible: {1, 1, 3, 3}
    scores =
      Nx.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])

    shifted = AudioEncoder.relative_shift(scores)

    assert Nx.shape(shifted) == {1, 1, 3, 3}
    # every query keeps three scores, drawn from progressively earlier offsets
    assert shifted |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
    refute Nx.all_close(scores, shifted) |> Nx.to_number() == 1
  end

  test "attention uses relative key and per dim scale parameters" do
    spec = tiny_spec()

    features = Axon.input("input_features", shape: {nil, nil, 8})
    model = AudioEncoder.encode(features, spec)
    {init_fun, _predict} = Axon.build(model)

    inputs = %{"input_features" => Nx.broadcast(0.1, {1, 8, 8})}
    params = init_fun.(inputs, Axon.ModelState.empty())

    # the checkpoint carries relative_k_proj and per_dim_scale per audio block
    assert Map.has_key?(params.data, "audio_encoder.blocks.0.attention.relative_key")

    chunked = params.data["audio_encoder.blocks.0.attention.chunked"]
    head_size = div(spec.audio_hidden_size, spec.audio_num_attention_heads)
    assert Nx.shape(chunked["per_dim_scale"]) == {head_size}
  end

  test "clipped linears carry learned bounds" do
    spec = tiny_spec()

    features = Axon.input("input_features", shape: {nil, nil, 8})
    model = AudioEncoder.encode(features, spec)
    {init_fun, _predict} = Axon.build(model)

    inputs = %{"input_features" => Nx.broadcast(0.1, {1, 8, 8})}
    params = init_fun.(inputs, Axon.ModelState.empty())

    layer = params.data["audio_encoder.blocks.0.ffn_start.intermediate"]

    for bound <- ["input_min", "input_max", "output_min", "output_max"] do
      assert Nx.shape(layer[bound]) == {1}
    end
  end

  test "encoder subsamples the time axis and projects into decoder width" do
    spec = tiny_spec()

    features = Axon.input("input_features", shape: {nil, nil, 8})
    model = AudioEncoder.encode(features, spec)

    {init_fun, predict_fun} = Axon.build(model)

    inputs = %{"input_features" => Nx.broadcast(0.1, {1, 8, 8})}
    params = init_fun.(inputs, Axon.ModelState.empty())
    out = predict_fun.(params, inputs)

    # two stride-2 convs over 8 frames -> 2 frames, projected to hidden_size
    assert Nx.shape(out) == {1, 2, spec.hidden_size}
    assert Spec.audio_subsampled_length(spec, 8) == 2
  end

  test "encoder output is finite for a range of inputs" do
    spec = tiny_spec()

    features = Axon.input("input_features", shape: {nil, nil, 8})
    model = AudioEncoder.encode(features, spec)
    {init_fun, predict_fun} = Axon.build(model)

    inputs = %{"input_features" => Nx.iota({1, 16, 8}, type: :f32) |> Nx.divide(16)}
    params = init_fun.(inputs, Axon.ModelState.empty())
    out = predict_fun.(params, inputs)

    assert Nx.shape(out) == {1, 4, spec.hidden_size}
    assert out |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 0
    assert out |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
  end
end
