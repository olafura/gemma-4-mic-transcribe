defmodule Gemma4MicTranscribe.Gemma4.MoeLayerArtifactTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4.ExtractedMoeLayer
  alias Gemma4MicTranscribe.Gemma4.MoeLayerArtifact

  @hidden 4
  @shared 6
  @expert 3
  @num_experts 4
  @top_k 2
  @eps 1.0e-6

  @config %{
    "text_config" => %{
      "enable_moe_block" => true,
      "num_hidden_layers" => 1,
      "num_experts" => @num_experts,
      "top_k_experts" => @top_k,
      "hidden_size" => @hidden,
      "intermediate_size" => @shared,
      "moe_intermediate_size" => @expert,
      "hidden_activation" => "gelu_pytorch_tanh",
      "rms_norm_eps" => @eps
    }
  }

  @tag :tmp_dir
  test "range-extracts and runs the complete routed feed-forward shell", %{tmp_dir: tmp_dir} do
    tensors = synthetic_tensors()

    source =
      tensors
      |> Safetensors.dump()
      |> IO.iodata_to_binary()

    index = %{
      "metadata" => %{"total_size" => 51_611_872_412},
      "weight_map" => Map.new(tensors, fn {name, _tensor} -> {name, "model.safetensors"} end)
    }

    fetch_json = fn url ->
      if String.ends_with?(url, "config.json"), do: @config, else: index
    end

    fetch_range = fn _url, first, last ->
      binary_part(source, first, last - first + 1)
    end

    artifact = Path.join(tmp_dir, "moe-layer")

    manifest =
      MoeLayerArtifact.extract!(artifact,
        layer: 0,
        fetch_json: fetch_json,
        fetch_range: fetch_range
      )

    assert manifest.layer_index == 0
    assert manifest.hidden_size == @hidden
    assert manifest.num_experts == @num_experts
    assert manifest.top_k_experts == @top_k
    assert manifest.parameter_type == {:bf, 16}
    assert manifest.parameter_bytes < byte_size(source)

    {_, params} = MoeLayerArtifact.load!(artifact, Torchx.Backend)

    assert_all_close(
      params.router_proj,
      tensors["model.language_model.layers.0.router.proj.weight"]
    )

    {_, router_params} = MoeLayerArtifact.load_router!(artifact, Torchx.Backend)

    assert router_params |> Map.keys() |> Enum.sort() ==
             [:router_per_expert_scale, :router_proj, :router_scale]

    input =
      Nx.tensor(
        [
          [0.25, -0.5, 0.75, 1.0],
          [-1.0, 0.125, 0.5, -0.25]
        ],
        type: :bf16
      )

    layer = ExtractedMoeLayer.load!(artifact)
    expected = reference_forward(input, params)
    actual = ExtractedMoeLayer.run(layer, input)

    assert Nx.shape(actual.output) == {2, @hidden}
    assert Nx.shape(actual.router_probabilities) == {2, @num_experts}
    assert Nx.shape(actual.top_k_indices) == {2, @top_k}
    assert Nx.to_flat_list(actual.top_k_indices) == Nx.to_flat_list(expected.top_k_indices)
    assert_all_close(actual.router_probabilities, expected.router_probabilities, 1.0e-5)
    assert_all_close(actual.top_k_weights, expected.top_k_weights, 1.0e-5)
    assert_all_close(actual.shared_output, expected.shared_output, 2.0e-2)
    assert_all_close(actual.routed_output, expected.routed_output, 2.0e-2)
    assert_all_close(actual.output, expected.output, 2.0e-2)

    override_expert =
      actual.top_k_indices
      |> Nx.slice([0, 0], [1, 1])
      |> Nx.squeeze()
      |> Nx.to_number()

    {_, override_params} = MoeLayerArtifact.load!(artifact, Torchx.Backend)

    override_input =
      Nx.tensor(
        [
          [0.25, -0.5, 0.75, 1.0],
          [-1.0, 0.125, 0.5, -0.25]
        ],
        type: :bf16
      )

    override_output = standalone_expert_output(override_input, override_params, override_expert)

    overridden =
      ExtractedMoeLayer.forward_with_override(
        override_input,
        override_params,
        override_expert,
        override_output,
        top_k: @top_k,
        eps: @eps,
        router_scalar: @hidden ** -0.5
      )

    expected_routes =
      actual.top_k_indices
      |> Nx.equal(override_expert)
      |> Nx.as_type(:s64)
      |> Nx.sum()
      |> Nx.to_number()

    assert Nx.to_number(overridden.override_route_count) == expected_routes
    assert_all_close(overridden.baseline_output, actual.output, 2.0e-2)
    assert_all_close(overridden.output, actual.output, 2.0e-2)

    ablated =
      ExtractedMoeLayer.forward_with_override(
        override_input,
        override_params,
        override_expert,
        Nx.broadcast(0.0, Nx.shape(override_input)),
        top_k: @top_k,
        eps: @eps,
        router_scalar: @hidden ** -0.5
      )

    assert max_abs_difference(ablated.output, ablated.baseline_output) > 0.0
  end

  @tag :tmp_dir
  test "concatenates MoE tensors that cross checkpoint shards", %{tmp_dir: tmp_dir} do
    tensors = synthetic_tensors()

    {first_tensors, second_tensors} =
      Enum.split_with(tensors, fn {name, _tensor} ->
        String.contains?(name, ".experts.") or
          String.ends_with?(name, ".router.scale") or
          String.ends_with?(name, ".router.per_expert_scale") or
          String.ends_with?(name, ".layer_scalar")
      end)

    sources = %{
      "model-00001-of-00002.safetensors" =>
        first_tensors |> Map.new() |> Safetensors.dump() |> IO.iodata_to_binary(),
      "model-00002-of-00002.safetensors" =>
        second_tensors |> Map.new() |> Safetensors.dump() |> IO.iodata_to_binary()
    }

    first_names = MapSet.new(first_tensors, &elem(&1, 0))

    weight_map =
      Map.new(tensors, fn {name, _tensor} ->
        shard =
          if MapSet.member?(first_names, name),
            do: "model-00001-of-00002.safetensors",
            else: "model-00002-of-00002.safetensors"

        {name, shard}
      end)

    index = %{
      "metadata" => %{"total_size" => Enum.sum(Enum.map(sources, &byte_size(elem(&1, 1))))},
      "weight_map" => weight_map
    }

    fetch_json = fn url ->
      if String.ends_with?(url, "config.json"), do: @config, else: index
    end

    fetch_range = fn url, first, last ->
      {shard, source} =
        Enum.find(sources, fn {shard, _source} -> String.ends_with?(url, shard) end)

      assert is_binary(shard)
      binary_part(source, first, last - first + 1)
    end

    artifact = Path.join(tmp_dir, "multi-shard-moe-layer")

    manifest =
      MoeLayerArtifact.extract!(artifact,
        layer: 0,
        fetch_json: fetch_json,
        fetch_range: fetch_range
      )

    assert manifest.source_shard == nil
    assert manifest.source_range == nil
    assert manifest.source_shards == Enum.sort(Map.keys(sources))
    assert Enum.map(manifest.source_ranges, & &1.parameter_offset) |> hd() == 0
    assert length(manifest.source_ranges) == 2

    {_, params} = MoeLayerArtifact.load!(artifact, Torchx.Backend)

    assert_all_close(
      params.experts_gate_up,
      tensors["model.language_model.layers.0.experts.gate_up_proj"]
    )

    assert_all_close(
      params.shared_down,
      tensors["model.language_model.layers.0.mlp.down_proj.weight"]
    )
  end

  defp synthetic_tensors do
    prefix = "model.language_model.layers.0"

    %{
      "#{prefix}.experts.gate_up_proj" => ramp({@num_experts, 2 * @expert, @hidden}, 97),
      "#{prefix}.experts.down_proj" => ramp({@num_experts, @hidden, @expert}, 83),
      "#{prefix}.mlp.gate_proj.weight" => ramp({@shared, @hidden}, 71),
      "#{prefix}.mlp.up_proj.weight" => ramp({@shared, @hidden}, 61),
      "#{prefix}.mlp.down_proj.weight" => ramp({@hidden, @shared}, 53),
      "#{prefix}.router.proj.weight" =>
        Nx.tensor(
          [
            [0.5, -0.25, 0.125, 0.75],
            [-0.5, 0.75, 0.25, -0.125],
            [0.25, 0.5, -0.75, 0.375],
            [-0.125, -0.5, 0.625, 0.25]
          ],
          type: :bf16
        ),
      "#{prefix}.router.scale" => Nx.tensor([0.75, 1.25, 0.5, 1.5], type: :bf16),
      "#{prefix}.router.per_expert_scale" => Nx.tensor([0.5, 0.75, 1.25, 1.5], type: :bf16),
      "#{prefix}.pre_feedforward_layernorm.weight" =>
        Nx.tensor([0.75, 1.0, 1.25, 1.5], type: :bf16),
      "#{prefix}.post_feedforward_layernorm_1.weight" =>
        Nx.tensor([1.0, 0.875, 1.125, 0.75], type: :bf16),
      "#{prefix}.pre_feedforward_layernorm_2.weight" =>
        Nx.tensor([1.25, 0.75, 1.0, 0.875], type: :bf16),
      "#{prefix}.post_feedforward_layernorm_2.weight" =>
        Nx.tensor([0.875, 1.0, 0.75, 1.25], type: :bf16),
      "#{prefix}.post_feedforward_layernorm.weight" =>
        Nx.tensor([1.0, 1.25, 0.875, 0.75], type: :bf16),
      "#{prefix}.layer_scalar" => Nx.tensor([0.875], type: :bf16)
    }
  end

  defp ramp(shape, divisor) do
    shape
    |> Nx.iota(type: :bf16)
    |> Nx.remainder(13)
    |> Nx.subtract(6)
    |> Nx.divide(divisor)
  end

  defp reference_forward(input, params) do
    shared_input = rms_norm(input, params.norm_pre_shared)

    shared =
      expert_forward(
        shared_input,
        params.shared_gate,
        params.shared_up,
        params.shared_down
      )
      |> rms_norm(params.norm_post_shared)

    router_input =
      input
      |> rms_norm_without_scale()
      |> Nx.multiply(params.router_scale)
      |> Nx.divide(:math.sqrt(@hidden))

    router_probabilities =
      router_input
      |> Nx.dot(Nx.transpose(params.router_proj))
      |> softmax()

    {top_k_weights, top_k_indices} = Nx.top_k(router_probabilities, k: @top_k)

    top_k_weights =
      Nx.divide(top_k_weights, Nx.sum(top_k_weights, axes: [1], keep_axes: true))

    top_k_weights =
      Nx.multiply(top_k_weights, Nx.take(params.router_per_expert_scale, top_k_indices))

    routed_input = rms_norm(input, params.norm_pre_experts)

    routed =
      0..(Nx.axis_size(input, 0) - 1)
      |> Enum.map(fn token_index ->
        token = Nx.slice_along_axis(routed_input, token_index, 1, axis: 0)

        0..(@top_k - 1)
        |> Enum.map(fn route_index ->
          expert_index =
            top_k_indices
            |> Nx.slice([token_index, route_index], [1, 1])
            |> Nx.squeeze()
            |> Nx.to_number()

          weight =
            top_k_weights
            |> Nx.slice([token_index, route_index], [1, 1])
            |> Nx.squeeze()

          gate_up =
            params.experts_gate_up
            |> Nx.slice_along_axis(expert_index, 1, axis: 0)
            |> Nx.squeeze(axes: [0])

          gate = Nx.slice_along_axis(gate_up, 0, @expert, axis: 0)
          up = Nx.slice_along_axis(gate_up, @expert, @expert, axis: 0)

          down =
            params.experts_down
            |> Nx.slice_along_axis(expert_index, 1, axis: 0)
            |> Nx.squeeze(axes: [0])

          token
          |> expert_forward(gate, up, down)
          |> Nx.as_type(:f32)
          |> Nx.multiply(weight)
        end)
        |> Enum.reduce(&Nx.add/2)
        |> Nx.as_type(:bf16)
      end)
      |> Nx.concatenate(axis: 0)
      |> rms_norm(params.norm_post_experts)

    combined = rms_norm(Nx.add(shared, routed), params.norm_post_combined)

    %{
      output: Nx.multiply(Nx.add(input, combined), params.layer_scalar),
      shared_output: shared,
      routed_output: routed,
      router_probabilities: router_probabilities,
      top_k_indices: top_k_indices,
      top_k_weights: top_k_weights
    }
  end

  defp expert_forward(input, gate, up, down) do
    gated =
      input
      |> Nx.dot(Nx.transpose(gate))
      |> Bumblebee.Layers.gelu_approx_tanh()

    hidden = Nx.multiply(gated, Nx.dot(input, Nx.transpose(up)))
    Nx.dot(hidden, Nx.transpose(down))
  end

  defp standalone_expert_output(input, params, expert_index) do
    gate_up =
      params.experts_gate_up
      |> Nx.slice_along_axis(expert_index, 1, axis: 0)
      |> Nx.squeeze(axes: [0])

    gate = Nx.slice_along_axis(gate_up, 0, @expert, axis: 0)
    up = Nx.slice_along_axis(gate_up, @expert, @expert, axis: 0)

    down =
      params.experts_down
      |> Nx.slice_along_axis(expert_index, 1, axis: 0)
      |> Nx.squeeze(axes: [0])

    input
    |> rms_norm(params.norm_pre_experts)
    |> expert_forward(gate, up, down)
  end

  defp rms_norm(input, weight) do
    input
    |> rms_norm_without_scale()
    |> Nx.as_type(:f32)
    |> Nx.multiply(Nx.as_type(weight, :f32))
    |> Nx.as_type(Nx.type(input))
  end

  defp rms_norm_without_scale(input) do
    input_f32 = Nx.as_type(input, :f32)

    mean_squared =
      input_f32
      |> Nx.pow(2)
      |> Nx.mean(axes: [1], keep_axes: true)
      |> Nx.add(@eps)

    Nx.as_type(Nx.multiply(input_f32, Nx.pow(mean_squared, -0.5)), Nx.type(input))
  end

  defp softmax(input) do
    input = Nx.as_type(input, :f32)
    shifted = Nx.subtract(input, Nx.reduce_max(input, axes: [1], keep_axes: true))
    exponents = Nx.exp(shifted)
    Nx.divide(exponents, Nx.sum(exponents, axes: [1], keep_axes: true))
  end

  defp assert_all_close(left, right, tolerance \\ 1.0e-6) do
    max_difference = max_abs_difference(left, right)

    assert max_difference <= tolerance,
           "maximum absolute difference #{max_difference} exceeds #{tolerance}"
  end

  defp max_abs_difference(left, right) do
    left
    |> Nx.as_type(:f32)
    |> Nx.subtract(Nx.as_type(right, :f32))
    |> Nx.abs()
    |> Nx.reduce_max()
    |> Nx.to_number()
  end
end
