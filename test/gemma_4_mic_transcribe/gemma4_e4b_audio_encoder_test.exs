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

  test "chunk mask gives each query its own sliding window, not the whole chunk" do
    # chunk 2, max_past 2, max_future 0 -> context 4, keys start 2 before block.
    # max_past counts the query itself, so back reach is one frame.
    mask =
      AudioEncoder.chunk_mask(length: 6, blocks: 3, chunk_size: 2, max_past: 2, max_future: 0)

    assert Nx.shape(mask) == {3, 2, 4}

    rows = mask |> Nx.to_flat_list() |> Enum.chunk_every(4)

    # block 0, query 0 is frame 0: its window covers frames -1 and 0, so only
    # the offset landing on frame 0 is valid
    assert Enum.at(rows, 0) == [0, 0, 1, 0]

    # block 0, query 1 is frame 1: it sees frames 0 and 1
    assert Enum.at(rows, 1) == [0, 0, 1, 1]

    # block 1, query 0 is frame 2: it sees frames 1 and 2 - reaching back past
    # its own chunk start, but never frame 0, which sits max_past away
    assert Enum.at(rows, 2) == [0, 1, 1, 0]
  end

  test "chunk mask can look right, stopping short of max_future" do
    # the forward bound is exclusive too: max_future 2 reaches one frame ahead
    mask =
      AudioEncoder.chunk_mask(length: 4, blocks: 2, chunk_size: 2, max_past: 1, max_future: 2)

    assert Nx.shape(mask) == {2, 2, 5}

    rows = mask |> Nx.to_flat_list() |> Enum.chunk_every(5)

    # query 0 of block 0 is frame 0: itself plus one frame ahead
    assert Enum.at(rows, 0) == [0, 1, 1, 0, 0]
    assert Enum.at(rows, 1) == [0, 0, 1, 1, 0]
  end

  test "relative shift realigns blocked scores to the context width" do
    # {batch, heads, blocks, chunk, positions} -> context 3
    scores = Nx.iota({1, 1, 1, 2, 2}, type: :f32)

    shifted = AudioEncoder.relative_shift(scores, context_size: 3)

    assert Nx.shape(shifted) == {1, 1, 1, 2, 3}
    assert shifted |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
  end

  test "attention uses relative key and per dim scale parameters" do
    spec = tiny_spec()

    features = Axon.input("input_features", shape: {nil, nil, 8})
    model = AudioEncoder.encode(features, spec)
    {init_fun, _predict} = Axon.build(model)

    inputs = %{"input_features" => Nx.broadcast(0.1, {1, 8, 8})}
    params = init_fun.(inputs, Axon.ModelState.empty())

    # the checkpoint carries relative_k_proj and per_dim_scale per audio block,
    # both as parameters of the attention layer itself
    chunked = params.data["audio_encoder.blocks.0.attention.chunked"]
    head_size = div(spec.audio_hidden_size, spec.audio_num_attention_heads)

    assert Nx.shape(chunked["per_dim_scale"]) == {head_size}

    assert Nx.shape(chunked["relative_key"]) ==
             {spec.audio_hidden_size, spec.audio_hidden_size}
  end

  test "clipped linears carry learned bounds" do
    spec = tiny_spec()

    features = Axon.input("input_features", shape: {nil, nil, 8})
    model = AudioEncoder.encode(features, spec)
    {init_fun, _predict} = Axon.build(model)

    inputs = %{"input_features" => Nx.broadcast(0.1, {1, 8, 8})}
    params = init_fun.(inputs, Axon.ModelState.empty())

    layer = params.data["audio_encoder.blocks.0.ffn_start.intermediate"]

    # the checkpoint stores these as scalars
    for bound <- ["input_min", "input_max", "output_min", "output_max"] do
      assert Nx.shape(layer[bound]) == {}
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
