defmodule Gemma4MicTranscribe.Gemma4E4BDecoderTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4E4B.Decoder
  alias Gemma4MicTranscribe.Gemma4E4B.Spec

  defp tiny_spec(opts \\ []) do
    defaults = [
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
      layer_types: [:sliding_attention, :full_attention, :sliding_attention, :full_attention]
    ]

    struct!(%Spec{}, Keyword.merge(defaults, opts))
  end

  defp build(spec, sequence) do
    hidden_state = Axon.input("hidden_state", shape: {nil, nil, spec.hidden_size})
    per_layer = Axon.input("per_layer_inputs", shape: {nil, nil, nil, nil})
    position_ids = Axon.input("position_ids", shape: {nil, nil})
    attention_mask = Axon.input("attention_mask", shape: {nil, nil})
    # optional and never supplied, so the decoder takes its no-cache path
    cache = Axon.input("cache", optional: true)

    outputs =
      Decoder.decode(hidden_state, per_layer, position_ids, attention_mask, cache, spec)

    model = Axon.container(%{hidden_state: outputs.hidden_state})
    {init_fun, predict_fun} = Axon.build(model)

    inputs = %{
      "hidden_state" => Nx.broadcast(0.1, {1, sequence, spec.hidden_size}),
      "per_layer_inputs" =>
        Nx.broadcast(0.05, {1, sequence, spec.num_blocks, spec.hidden_size_per_layer_input}),
      "position_ids" => Nx.iota({1, sequence}, axis: 1, type: :s64),
      "attention_mask" => Nx.broadcast(1, {1, sequence})
    }

    params = init_fun.(inputs, Axon.ModelState.empty())
    {predict_fun.(params, inputs), params}
  end

  test "decoder runs and preserves shape" do
    spec = tiny_spec()
    {outputs, _params} = build(spec, 5)

    assert Nx.shape(outputs.hidden_state) == {1, 5, spec.hidden_size}
    assert outputs.hidden_state |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
  end

  test "shared blocks reuse cached key values instead of projecting their own" do
    spec = tiny_spec()
    {_outputs, params} = build(spec, 4)
    data = params.data

    # 4 blocks with the last 2 shared: only 0 and 1 compute key/value
    assert Map.has_key?(data, "decoder.blocks.0.self_attention.key")
    assert Map.has_key?(data, "decoder.blocks.1.self_attention.key")
    refute Map.has_key?(data, "decoder.blocks.2.self_attention.key")
    refute Map.has_key?(data, "decoder.blocks.3.self_attention.key")

    # every block still projects its own query
    for index <- 0..3 do
      assert Map.has_key?(data, "decoder.blocks.#{index}.self_attention.query")
    end
  end

  test "shared blocks reuse a computing block of their own attention type" do
    # sliding and full alternate, last two shared: block 2 (sliding) must
    # reuse block 0 (sliding), not block 1 (full), since head sizes differ
    spec =
      tiny_spec(
        attention_head_size: 4,
        global_attention_head_size: 8,
        layer_types: [:sliding_attention, :full_attention, :sliding_attention, :full_attention],
        num_kv_shared_layers: 2
      )

    {outputs, _params} = build(spec, 4)

    # a mismatched share would fail to build at all, so reaching here with the
    # right shape is the assertion
    assert Nx.shape(outputs.hidden_state) == {1, 4, spec.hidden_size}
  end

  test "shared blocks leave their placeholder caches untouched" do
    spec = tiny_spec()
    sequence = 3
    max_length = 8

    hidden_state = Axon.input("hidden_state", shape: {nil, nil, spec.hidden_size})
    per_layer = Axon.input("per_layer_inputs", shape: {nil, nil, nil, nil})
    position_ids = Axon.input("position_ids", shape: {nil, nil})
    attention_mask = Axon.input("attention_mask", shape: {nil, nil})
    cache = Axon.input("cache", optional: true)

    outputs =
      Decoder.decode(hidden_state, per_layer, position_ids, attention_mask, cache, spec)

    model = Axon.container(%{hidden_state: outputs.hidden_state, cache: outputs.cache})
    {init_fun, predict_fun} = Axon.build(model)

    computing_cache = fn ->
      zeros = Nx.broadcast(0.0, {1, max_length, spec.num_attention_heads, 4})
      cross = Nx.broadcast(0.0, {1, 1, 1, 1})

      %{
        self_attention: %{key: zeros, value: zeros},
        cross_attention: %{key: cross, value: cross}
      }
    end

    placeholder_cache = fn ->
      zeros = Nx.broadcast(0.0, {1, 1, 1, 1})

      %{
        self_attention: %{key: zeros, value: zeros},
        cross_attention: %{key: zeros, value: zeros}
      }
    end

    decoder_cache = %{
      blocks:
        {computing_cache.(), computing_cache.(), placeholder_cache.(), placeholder_cache.()},
      offset: Nx.tensor(0),
      attention_mask: Nx.broadcast(0, {1, max_length})
    }

    inputs = %{
      "hidden_state" => Nx.broadcast(0.1, {1, sequence, spec.hidden_size}),
      "per_layer_inputs" =>
        Nx.broadcast(0.05, {1, sequence, spec.num_blocks, spec.hidden_size_per_layer_input}),
      "position_ids" => Nx.iota({1, sequence}, axis: 1, type: :s64),
      "attention_mask" => Nx.broadcast(1, {1, sequence}),
      "cache" => decoder_cache
    }

    params = init_fun.(inputs, Axon.ModelState.empty())
    result = predict_fun.(params, inputs)

    for index <- 2..3 do
      shared_key = elem(result.cache.blocks, index).self_attention.key
      assert Nx.shape(shared_key) == {1, 1, 1, 1}
      assert Nx.to_number(Nx.sum(Nx.abs(shared_key))) == 0.0
    end
  end

  test "per-layer inputs are gated against the block state" do
    spec = tiny_spec()
    {_outputs, params} = build(spec, 3)

    for index <- 0..(spec.num_blocks - 1) do
      # gate maps hidden -> per_layer_size, projection maps back
      gate = params.data["decoder.blocks.#{index}.per_layer.input_gate"]["kernel"]
      projection = params.data["decoder.blocks.#{index}.per_layer.projection"]["kernel"]

      assert Nx.shape(gate) == {spec.hidden_size, spec.hidden_size_per_layer_input}
      assert Nx.shape(projection) == {spec.hidden_size_per_layer_input, spec.hidden_size}
    end
  end

  test "per-layer inputs change the output" do
    spec = tiny_spec()

    hidden_state = Axon.input("hidden_state", shape: {nil, nil, spec.hidden_size})
    per_layer = Axon.input("per_layer_inputs", shape: {nil, nil, nil, nil})
    position_ids = Axon.input("position_ids", shape: {nil, nil})
    attention_mask = Axon.input("attention_mask", shape: {nil, nil})
    # optional and never supplied, so the decoder takes its no-cache path
    cache = Axon.input("cache", optional: true)

    outputs =
      Decoder.decode(hidden_state, per_layer, position_ids, attention_mask, cache, spec)

    model = Axon.container(%{hidden_state: outputs.hidden_state})
    {init_fun, predict_fun} = Axon.build(model)

    base = %{
      "hidden_state" => Nx.broadcast(0.1, {1, 3, spec.hidden_size}),
      "per_layer_inputs" =>
        Nx.broadcast(0.0, {1, 3, spec.num_blocks, spec.hidden_size_per_layer_input}),
      "position_ids" => Nx.iota({1, 3}, axis: 1, type: :s64),
      "attention_mask" => Nx.broadcast(1, {1, 3})
    }

    params = init_fun.(base, Axon.ModelState.empty())
    zeroed = predict_fun.(params, base).hidden_state

    varied =
      predict_fun.(
        params,
        %{
          base
          | "per_layer_inputs" =>
              Nx.iota({1, 3, spec.num_blocks, spec.hidden_size_per_layer_input}, type: :f32)
        }
      ).hidden_state

    refute Nx.all_close(zeroed, varied) |> Nx.to_number() == 1
  end

  test "query heads outnumber key value heads" do
    spec = tiny_spec()
    {_outputs, params} = build(spec, 3)

    query = params.data["decoder.blocks.0.self_attention.query"]["kernel"]
    key = params.data["decoder.blocks.0.self_attention.key"]["kernel"]

    # 4 query heads and 2 key/value heads at head size 4
    assert Nx.shape(query) == {spec.hidden_size, 4 * 4}
    assert Nx.shape(key) == {spec.hidden_size, 2 * 4}
  end
end
