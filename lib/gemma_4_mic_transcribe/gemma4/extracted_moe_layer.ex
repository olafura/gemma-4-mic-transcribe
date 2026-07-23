defmodule Gemma4MicTranscribe.Gemma4.ExtractedMoeLayer do
  @moduledoc """
  Runs a complete extracted Gemma 4 MoE feed-forward layer.

  Input is the residual stream immediately after attention. The result includes
  the shared and sparse routed paths, all feed-forward norms, residual addition,
  and layer scalar. Attention itself is outside this artifact.
  """

  import Nx.Defn

  alias Gemma4MicTranscribe.Gemma4.MoeLayerArtifact

  @default_backend Application.compile_env(:nx, :default_backend, Nx.BinaryBackend)

  defstruct [:manifest, :params, :predict_fun, :backend]

  @doc "Loads the extracted shell and builds a backend-specific predictor."
  def load!(path, backend \\ @default_backend, opts \\ []) do
    {manifest, params} = MoeLayerArtifact.load!(path, backend, opts)
    top_k = manifest.top_k_experts
    eps = manifest.rms_norm_eps
    router_scalar = manifest.hidden_size ** -0.5

    predict_fun =
      Nx.Defn.jit(
        fn input, params ->
          forward(input, params, top_k: top_k, eps: eps, router_scalar: router_scalar)
        end,
        build_opts(backend)
      )

    %__MODULE__{
      manifest: manifest,
      params: params,
      predict_fun: predict_fun,
      backend: backend
    }
  end

  @doc "Runs the MoE feed-forward shell over `[tokens, hidden]` residual states."
  def run(%__MODULE__{} = layer, input) do
    input = Nx.to_tensor(input)

    unless Nx.rank(input) == 2 and elem(Nx.shape(input), 1) == layer.manifest.hidden_size do
      raise ArgumentError,
            "expected MoE input shape {tokens, #{layer.manifest.hidden_size}}, got #{inspect(Nx.shape(input))}"
    end

    input =
      input
      |> Nx.as_type(layer.manifest.parameter_type)
      |> transfer(layer.backend)

    layer.predict_fun.(input, layer.params)
  end

  @doc "Compiles and synchronizes the shell for a fixed token count."
  def warmup(%__MODULE__{} = layer, token_count \\ 1) do
    result = run(layer, Nx.broadcast(0.0, {token_count, layer.manifest.hidden_size}))
    _output = Nx.backend_copy(result.output, Nx.BinaryBackend)
    :ok
  end

  @doc false
  defn forward(input, params, opts \\ []) do
    top_k = opts[:top_k]
    eps = opts[:eps]
    router_scalar = opts[:router_scalar]
    residual = input

    shared_input = rms_norm(residual, params.norm_pre_shared, eps)

    shared =
      expert_forward(shared_input, params.shared_gate, params.shared_up, params.shared_down)

    shared = rms_norm(shared, params.norm_post_shared, eps)

    routing =
      route(residual, params,
        top_k: top_k,
        eps: eps,
        router_scalar: router_scalar
      )

    routed_input = rms_norm(residual, params.norm_pre_experts, eps)

    routed =
      routed_experts(
        routed_input,
        routing.top_k_indices,
        routing.top_k_weights,
        params
      )

    routed = rms_norm(routed, params.norm_post_experts, eps)

    combined = rms_norm(shared + routed, params.norm_post_combined, eps)
    output = (residual + combined) * params.layer_scalar

    %{
      output: output,
      shared_output: shared,
      routed_output: routed,
      router_probabilities: routing.router_probabilities,
      top_k_indices: routing.top_k_indices,
      top_k_weights: routing.top_k_weights
    }
  end

  @doc false
  defn prepare_sparse(input, params, top_k_indices, top_k_weights, opts \\ []) do
    eps = opts[:eps]
    residual = input

    shared_input = rms_norm(residual, params.norm_pre_shared, eps)

    shared =
      expert_forward(shared_input, params.shared_gate, params.shared_up, params.shared_down)

    %{
      residual: residual,
      shared_output: rms_norm(shared, params.norm_post_shared, eps),
      routed_input: rms_norm(residual, params.norm_pre_experts, eps),
      top_k_indices: top_k_indices,
      top_k_weights: top_k_weights
    }
  end

  @doc false
  defn finish_sparse(prepared, params, expert_params, opts \\ []) do
    eps = opts[:eps]

    routed =
      prepared.routed_input
      |> compact_routed_expert_outputs(
        expert_params.experts_gate_up,
        expert_params.experts_down
      )
      |> combine_routed_expert_outputs(
        prepared.top_k_weights,
        Nx.type(prepared.residual)
      )
      |> rms_norm(params.norm_post_experts, eps)

    combined =
      rms_norm(prepared.shared_output + routed, params.norm_post_combined, eps)

    %{
      output: (prepared.residual + combined) * params.layer_scalar,
      shared_output: prepared.shared_output,
      routed_output: routed,
      top_k_indices: prepared.top_k_indices,
      top_k_weights: prepared.top_k_weights
    }
  end

  @doc """
  Runs the complete shell while replacing one routed expert's raw output.

  `override_output` has shape `[tokens, hidden]`. It is inserted only where
  `override_expert` occurs in the router's top-k selection. The unchanged
  baseline is returned from the same routed-expert calculation for comparison.
  """
  defn forward_with_override(input, params, override_expert, override_output, opts \\ []) do
    top_k = opts[:top_k]
    eps = opts[:eps]
    router_scalar = opts[:router_scalar]
    residual = input

    shared_input = rms_norm(residual, params.norm_pre_shared, eps)

    shared =
      expert_forward(shared_input, params.shared_gate, params.shared_up, params.shared_down)

    shared = rms_norm(shared, params.norm_post_shared, eps)

    routing =
      route(residual, params,
        top_k: top_k,
        eps: eps,
        router_scalar: router_scalar
      )

    routed_input = rms_norm(residual, params.norm_pre_experts, eps)

    expert_outputs =
      routed_expert_outputs(
        routed_input,
        routing.top_k_indices,
        params
      )

    override_mask = Nx.equal(routing.top_k_indices, override_expert)

    overridden_expert_outputs =
      Nx.select(
        override_mask |> Nx.new_axis(-1) |> Nx.broadcast(Nx.shape(expert_outputs)),
        override_output |> Nx.new_axis(1) |> Nx.broadcast(Nx.shape(expert_outputs)),
        expert_outputs
      )

    baseline_routed =
      expert_outputs
      |> combine_routed_expert_outputs(routing.top_k_weights, Nx.type(input))
      |> rms_norm(params.norm_post_experts, eps)

    overridden_routed =
      overridden_expert_outputs
      |> combine_routed_expert_outputs(routing.top_k_weights, Nx.type(input))
      |> rms_norm(params.norm_post_experts, eps)

    baseline_combined = rms_norm(shared + baseline_routed, params.norm_post_combined, eps)
    overridden_combined = rms_norm(shared + overridden_routed, params.norm_post_combined, eps)

    %{
      output: (residual + overridden_combined) * params.layer_scalar,
      baseline_output: (residual + baseline_combined) * params.layer_scalar,
      shared_output: shared,
      routed_output: overridden_routed,
      baseline_routed_output: baseline_routed,
      router_probabilities: routing.router_probabilities,
      top_k_indices: routing.top_k_indices,
      top_k_weights: routing.top_k_weights,
      override_route_count: override_mask |> Nx.as_type(:s64) |> Nx.sum()
    }
  end

  @doc false
  defn route(input, params, opts \\ []) do
    top_k = opts[:top_k]
    eps = opts[:eps]
    router_scalar = opts[:router_scalar]

    router_input = rms_norm_without_scale(input, eps)
    router_input = router_input * params.router_scale * router_scalar
    router_logits = Nx.dot(router_input, Nx.transpose(params.router_proj))
    router_probabilities = softmax_f32(router_logits)
    {top_k_weights, top_k_indices} = Nx.top_k(router_probabilities, k: top_k)
    top_k_weights = top_k_weights / Nx.sum(top_k_weights, axes: [-1], keep_axes: true)

    per_expert_scale = Nx.take(params.router_per_expert_scale, top_k_indices)
    top_k_weights = top_k_weights * Nx.as_type(per_expert_scale, :f32)

    %{
      router_probabilities: router_probabilities,
      top_k_indices: top_k_indices,
      top_k_weights: top_k_weights
    }
  end

  defnp routed_experts(input, indices, weights, params) do
    input
    |> routed_expert_outputs(indices, params)
    |> combine_routed_expert_outputs(weights, Nx.type(input))
  end

  defnp routed_expert_outputs(input, indices, params) do
    selected_gate_up = Nx.take(params.experts_gate_up, indices, axis: 0)
    selected_down = Nx.take(params.experts_down, indices, axis: 0)
    token_count = Nx.axis_size(input, 0)
    top_k = Nx.axis_size(indices, 1)
    hidden_size = Nx.axis_size(input, 1)

    expert_input =
      input
      |> Nx.new_axis(1)
      |> Nx.broadcast({token_count, top_k, hidden_size})
      |> Nx.new_axis(-1)

    gate_up =
      Nx.dot(selected_gate_up, [3], [0, 1], expert_input, [2], [0, 1])
      |> Nx.squeeze(axes: [3])

    intermediate_size = div(Nx.axis_size(gate_up, 2), 2)
    gate = Nx.slice_along_axis(gate_up, 0, intermediate_size, axis: 2)
    up = Nx.slice_along_axis(gate_up, intermediate_size, intermediate_size, axis: 2)

    hidden =
      (Bumblebee.Layers.gelu_approx_tanh(gate) * up)
      |> Nx.new_axis(-1)

    Nx.dot(selected_down, [3], [0, 1], hidden, [2], [0, 1])
    |> Nx.squeeze(axes: [3])
  end

  defnp compact_routed_expert_outputs(input, experts_gate_up, experts_down) do
    token_count = Nx.axis_size(input, 0)
    top_k = Nx.axis_size(experts_gate_up, 0)
    hidden_size = Nx.axis_size(input, 1)
    selected_gate_up = Nx.new_axis(experts_gate_up, 0)
    selected_down = Nx.new_axis(experts_down, 0)

    expert_input =
      input
      |> Nx.new_axis(1)
      |> Nx.broadcast({token_count, top_k, hidden_size})
      |> Nx.new_axis(-1)

    gate_up =
      Nx.dot(selected_gate_up, [3], [0, 1], expert_input, [2], [0, 1])
      |> Nx.squeeze(axes: [3])

    intermediate_size = div(Nx.axis_size(gate_up, 2), 2)
    gate = Nx.slice_along_axis(gate_up, 0, intermediate_size, axis: 2)
    up = Nx.slice_along_axis(gate_up, intermediate_size, intermediate_size, axis: 2)

    hidden =
      (Bumblebee.Layers.gelu_approx_tanh(gate) * up)
      |> Nx.new_axis(-1)

    Nx.dot(selected_down, [3], [0, 1], hidden, [2], [0, 1])
    |> Nx.squeeze(axes: [3])
  end

  defnp combine_routed_expert_outputs(expert_outputs, weights, output_type) do
    expert_outputs
    |> Nx.as_type(:f32)
    |> Nx.multiply(Nx.new_axis(weights, -1))
    |> Nx.sum(axes: [1])
    |> Nx.as_type(output_type)
  end

  defnp expert_forward(input, gate_weight, up_weight, down_weight) do
    gate = Nx.dot(input, Nx.transpose(gate_weight))
    up = Nx.dot(input, Nx.transpose(up_weight))
    hidden = Bumblebee.Layers.gelu_approx_tanh(gate) * up
    Nx.dot(hidden, Nx.transpose(down_weight))
  end

  defnp rms_norm(input, weight, eps) do
    input
    |> rms_norm_without_scale(eps)
    |> Nx.as_type(:f32)
    |> Nx.multiply(Nx.as_type(weight, :f32))
    |> Nx.as_type(Nx.type(input))
  end

  defnp rms_norm_without_scale(input, eps) do
    input_f32 = Nx.as_type(input, :f32)
    mean_squared = Nx.mean(Nx.pow(input_f32, 2), axes: [-1], keep_axes: true) + eps
    Nx.as_type(input_f32 * Nx.pow(mean_squared, -0.5), Nx.type(input))
  end

  defnp softmax_f32(input) do
    input = Nx.as_type(input, :f32)
    shifted = input - Nx.reduce_max(input, axes: [-1], keep_axes: true)
    exponentials = Nx.exp(shifted)
    exponentials / Nx.sum(exponentials, axes: [-1], keep_axes: true)
  end

  defp transfer(tensor, nil), do: tensor
  defp transfer(tensor, Nx.BinaryBackend), do: Nx.backend_copy(tensor, Nx.BinaryBackend)
  defp transfer(tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp build_opts(EXLA.Backend), do: [compiler: EXLA]

  defp build_opts({EXLA.Backend, backend_opts}),
    do: [compiler: EXLA] ++ Keyword.take(backend_opts, [:client, :device_id])

  defp build_opts(_backend), do: []
end
