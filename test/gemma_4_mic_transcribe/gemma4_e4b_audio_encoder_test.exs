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
