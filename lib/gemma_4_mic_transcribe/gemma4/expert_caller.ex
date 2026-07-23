defmodule Gemma4MicTranscribe.Gemma4.ExpertCaller do
  @moduledoc """
  Reconstructs the layer-0 path from text embeddings to a routed expert call.

  The caller runs embedding scaling, layer-0 attention, the attention residual,
  router selection, and the routed-expert input norm. It then invokes one
  standalone extracted expert only for token positions where the router
  selected that expert.
  """

  import Nx.Defn

  alias Gemma4MicTranscribe.Gemma4.ExpertCallerArtifact
  alias Gemma4MicTranscribe.Gemma4.ExtractedExpert
  alias Gemma4MicTranscribe.Gemma4.ExtractedMoeLayer
  alias Gemma4MicTranscribe.Gemma4.MathExpertProfiler
  alias Gemma4MicTranscribe.Gemma4.MoeLayerArtifact

  defstruct [
    :manifest,
    :attention_params,
    :moe_manifest,
    :moe_params,
    :expert,
    :predict_fun,
    :layer_predict_fun,
    :backend,
    :moe_artifact
  ]

  @doc "Loads the extracted attention prefix, MoE caller tensors, and expert."
  def load!(caller_artifact, moe_artifact, expert_artifact, backend) do
    load_with_mode!(caller_artifact, moe_artifact, expert_artifact, backend, :caller)
  end

  @doc "Loads the attention prefix, complete MoE shell, and standalone expert override."
  def load_layer!(caller_artifact, moe_artifact, expert_artifact, backend) do
    load_with_mode!(caller_artifact, moe_artifact, expert_artifact, backend, :complete_layer)
  end

  defp load_with_mode!(caller_artifact, moe_artifact, expert_artifact, backend, mode) do
    {manifest, attention_params} = ExpertCallerArtifact.load!(caller_artifact, backend)

    {moe_manifest, moe_params} =
      case mode do
        :caller -> MoeLayerArtifact.load_caller!(moe_artifact, backend)
        :complete_layer -> MoeLayerArtifact.load!(moe_artifact, backend)
      end

    expert = ExtractedExpert.load!(expert_artifact, backend)

    unless manifest.layer_index == moe_manifest.layer_index and
             expert.manifest.layer_index == manifest.layer_index do
      raise ArgumentError, "caller, MoE layer, and expert artifacts must use the same layer"
    end

    top_k = moe_manifest.top_k_experts
    eps = manifest.rms_norm_eps
    router_scalar = manifest.hidden_size ** -0.5
    embedding_scalar = if manifest.layer_index == 0, do: manifest.hidden_size ** 0.5, else: 1.0

    forward_opts = [
      top_k: top_k,
      eps: eps,
      router_scalar: router_scalar,
      embedding_scalar: embedding_scalar,
      heads: manifest.num_attention_heads,
      kv_heads: manifest.num_key_value_heads,
      head_dim: manifest.head_dim,
      rope_theta: manifest.rope_theta,
      partial_rotary_factor: Map.get(manifest, :partial_rotary_factor, 1.0),
      rotary_angles:
        trunc(Map.get(manifest, :partial_rotary_factor, 1.0) * div(manifest.head_dim, 2)),
      alternative_attention: Map.get(manifest, :alternative_attention, false),
      sliding_window: Map.get(manifest, :sliding_window, 1024)
    ]

    predict_fun =
      Nx.Defn.jit(
        fn embeddings, attention_params, moe_params ->
          forward(embeddings, attention_params, moe_params, forward_opts)
        end,
        build_opts(backend)
      )

    layer_predict_fun =
      if mode == :complete_layer do
        expert_index = expert.manifest.expert_index

        Nx.Defn.jit(
          fn embeddings, attention_params, moe_params, expert_params, expert_scale ->
            capture = forward(embeddings, attention_params, moe_params, forward_opts)

            standalone_output =
              capture.expert_input
              |> ExtractedExpert.forward(expert_params)
              |> Nx.as_type(:f32)
              |> Nx.multiply(expert_scale)
              |> Nx.as_type(Nx.type(capture.expert_input))

            layer =
              ExtractedMoeLayer.forward_with_override(
                capture.residual_after_attention,
                moe_params,
                expert_index,
                standalone_output,
                top_k: top_k,
                eps: eps,
                router_scalar: router_scalar
              )

            %{
              output: layer.output,
              baseline_output: layer.baseline_output,
              shared_output: layer.shared_output,
              routed_output: layer.routed_output,
              baseline_routed_output: layer.baseline_routed_output,
              router_probabilities: layer.router_probabilities,
              top_k_indices: layer.top_k_indices,
              top_k_weights: layer.top_k_weights,
              override_route_count: layer.override_route_count,
              expert_input: capture.expert_input,
              standalone_output: standalone_output,
              residual_after_attention: capture.residual_after_attention
            }
          end,
          build_opts(backend)
        )
      end

    %__MODULE__{
      manifest: manifest,
      attention_params: attention_params,
      moe_manifest: moe_manifest,
      moe_params: moe_params,
      expert: expert,
      predict_fun: predict_fun,
      layer_predict_fun: layer_predict_fun,
      backend: backend,
      moe_artifact: Path.expand(moe_artifact)
    }
  end

  @doc """
  Runs text through layer 0 and inserts the standalone expert into the MoE shell.

  `:expert_scale` defaults to `1.0`. Other values provide a simple controlled
  Frankenstein-model experiment while retaining the unchanged baseline output.
  """
  def call_layer_text!(
        %__MODULE__{layer_predict_fun: layer_predict_fun} = caller,
        text,
        opts \\ []
      )
      when is_function(layer_predict_fun) do
    {embedding_data, result} = call_layer_device!(caller, text, opts)
    result = Nx.backend_copy(result, Nx.BinaryBackend)

    expert_index = caller.expert.manifest.expert_index
    selected = selected_calls(result, embedding_data.tokens, expert_index)
    positions = Enum.map(selected, & &1.position)

    delta =
      result.output
      |> Nx.as_type(:f32)
      |> Nx.subtract(Nx.as_type(result.baseline_output, :f32))

    %{
      text: text,
      tokens: embedding_data.tokens,
      expert: expert_index,
      expert_scale: Keyword.get(opts, :expert_scale, 1.0),
      selected_calls: selected,
      expert_inputs: take_rows(result.expert_input, positions),
      expert_outputs: take_rows(result.standalone_output, positions),
      layer_output: result.output,
      baseline_layer_output: result.baseline_output,
      layer_output_delta_mean_abs: delta |> Nx.abs() |> Nx.mean() |> Nx.to_number(),
      layer_output_delta_max_abs: delta |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number(),
      override_route_count: Nx.to_number(result.override_route_count),
      shared_output: result.shared_output,
      routed_output: result.routed_output,
      baseline_routed_output: result.baseline_routed_output,
      residual_after_attention: result.residual_after_attention,
      router_probabilities: result.router_probabilities,
      top_k_indices: result.top_k_indices,
      top_k_weights: result.top_k_weights
    }
  end

  @doc "Runs the layer override while keeping all activation tensors on its backend."
  def call_layer_device!(
        %__MODULE__{layer_predict_fun: layer_predict_fun} = caller,
        text,
        opts \\ []
      )
      when is_function(layer_predict_fun) do
    require_text_input_layer!(caller)
    embedding_data = embedding_inputs!(caller, text, opts)

    embeddings =
      embedding_data.input
      |> Nx.as_type(:bf16)
      |> transfer(caller.backend)

    expert_scale =
      opts
      |> Keyword.get(:expert_scale, 1.0)
      |> Nx.tensor(type: :f32)
      |> transfer(caller.backend)

    result =
      layer_predict_fun.(
        embeddings,
        caller.attention_params,
        caller.moe_params,
        caller.expert.params,
        expert_scale
      )

    {embedding_data, result}
  end

  @doc "Tokenizes text, captures real layer-0 expert inputs, and calls the selected expert."
  def call_text!(%__MODULE__{} = caller, text, opts \\ []) do
    require_text_input_layer!(caller)
    embedding_data = embedding_inputs!(caller, text, opts)

    embeddings =
      embedding_data.input
      |> Nx.as_type(:bf16)
      |> transfer(caller.backend)

    capture =
      caller.predict_fun.(embeddings, caller.attention_params, caller.moe_params)
      |> Nx.backend_copy(Nx.BinaryBackend)

    expert_index = caller.expert.manifest.expert_index
    selected = selected_calls(capture, embedding_data.tokens, expert_index)

    expert_outputs =
      case Enum.map(selected, & &1.position) do
        [] ->
          nil

        positions ->
          caller
          |> expert_inputs(capture.expert_input, positions)
          |> Nx.backend_copy(Nx.BinaryBackend)
      end

    %{
      text: text,
      tokens: embedding_data.tokens,
      expert: expert_index,
      selected_calls: selected,
      expert_inputs: take_rows(capture.expert_input, Enum.map(selected, & &1.position)),
      expert_outputs: expert_outputs,
      residual_after_attention: capture.residual_after_attention,
      router_probabilities: capture.router_probabilities,
      top_k_indices: capture.top_k_indices,
      top_k_weights: capture.top_k_weights
    }
  end

  defp embedding_inputs!(caller, text, opts) do
    MathExpertProfiler.embedding_inputs!(
      caller.moe_artifact,
      [text],
      opts
      |> Keyword.put_new(:prepend_bos, true)
      |> Keyword.put_new(:max_concurrency, 8)
    )
  end

  defp require_text_input_layer!(caller) do
    unless caller.manifest.layer_index == 0 do
      raise ArgumentError, "text input is valid only for decoder layer 0"
    end
  end

  defp expert_inputs(caller, input, positions) do
    input =
      input
      |> take_rows(positions)
      |> transfer(caller.backend)

    ExtractedExpert.run(caller.expert, input)
  end

  defp selected_calls(capture, tokens, expert) do
    indices = Nx.to_list(capture.top_k_indices)
    weights = Nx.to_list(capture.top_k_weights)
    probabilities = Nx.to_list(capture.router_probabilities)

    tokens
    |> Enum.with_index()
    |> Enum.flat_map(fn {token, position} ->
      case Enum.find_index(Enum.at(indices, position), &(&1 == expert)) do
        nil ->
          []

        route ->
          [
            %{
              position: position,
              token_id: token.id,
              token: token.token,
              route: route,
              router_probability: probabilities |> Enum.at(position) |> Enum.at(expert),
              routed_weight: weights |> Enum.at(position) |> Enum.at(route)
            }
          ]
      end
    end)
  end

  defp take_rows(tensor, []), do: Nx.broadcast(0.0, {0, Nx.axis_size(tensor, 1)})

  defp take_rows(tensor, positions) do
    Nx.take(
      tensor,
      Nx.tensor(positions, type: :s64, backend: Nx.BinaryBackend)
    )
  end

  @doc false
  defn forward(embeddings, attention_params, moe_params, opts \\ []) do
    eps = opts[:eps]
    alternative_attention = opts[:alternative_attention]
    rotary_angles = opts[:rotary_angles]
    hidden = embeddings * opts[:embedding_scalar]
    normed = rms_norm(hidden, attention_params.input_norm, eps)
    token_count = Nx.axis_size(hidden, 0)

    query =
      normed
      |> Nx.dot(Nx.transpose(attention_params.query))
      |> Nx.reshape({token_count, opts[:heads], opts[:head_dim]})
      |> rms_norm(attention_params.query_norm, eps)

    key_projection =
      normed
      |> Nx.dot(Nx.transpose(attention_params.key))
      |> Nx.reshape({token_count, opts[:kv_heads], opts[:head_dim]})

    key =
      key_projection
      |> rms_norm(attention_params.key_norm, eps)

    value =
      if alternative_attention do
        rms_norm_without_scale(key_projection, eps)
      else
        normed
        |> Nx.dot(Nx.transpose(attention_params.value))
        |> Nx.reshape({token_count, opts[:kv_heads], opts[:head_dim]})
        |> rms_norm_without_scale(eps)
      end

    {query, key} =
      apply_rope(
        query,
        key,
        opts[:head_dim],
        opts[:rope_theta],
        rotary_angles
      )

    groups = div(opts[:heads], opts[:kv_heads])
    key = repeat_kv(key, groups, opts[:heads], opts[:head_dim])
    value = repeat_kv(value, groups, opts[:heads], opts[:head_dim])

    query = Nx.transpose(query, axes: [1, 0, 2])
    key = Nx.transpose(key, axes: [1, 0, 2])
    value = Nx.transpose(value, axes: [1, 0, 2])

    scores =
      Nx.dot(query, [2], [0], key, [2], [0])
      |> Nx.as_type(:f32)

    query_positions = Nx.iota({token_count, 1}, axis: 0)
    key_positions = Nx.iota({1, token_count}, axis: 1)

    causal_mask =
      Nx.logical_and(
        Nx.less_equal(key_positions, query_positions),
        Nx.greater_equal(key_positions, query_positions - opts[:sliding_window] + 1)
      )
      |> Nx.new_axis(0)
      |> Nx.broadcast({opts[:heads], token_count, token_count})

    scores = Nx.select(causal_mask, scores, Nx.tensor(-1.0e30))
    weights = softmax_f32(scores)

    attention =
      Nx.dot(weights, [2], [0], value, [1], [0])
      |> Nx.transpose(axes: [1, 0, 2])
      |> Nx.reshape({token_count, opts[:heads] * opts[:head_dim]})
      |> Nx.dot(Nx.transpose(attention_params.output))
      |> rms_norm(attention_params.post_attention_norm, eps)

    residual_after_attention = hidden + attention

    routing =
      ExtractedMoeLayer.route(residual_after_attention, moe_params,
        top_k: opts[:top_k],
        eps: eps,
        router_scalar: opts[:router_scalar]
      )

    expert_input = rms_norm(residual_after_attention, moe_params.norm_pre_experts, eps)

    %{
      expert_input: expert_input,
      residual_after_attention: residual_after_attention,
      router_probabilities: routing.router_probabilities,
      top_k_indices: routing.top_k_indices,
      top_k_weights: routing.top_k_weights
    }
  end

  defnp apply_rope(query, key, head_dim, theta, rotary_angles) do
    token_count = Nx.axis_size(query, 0)
    half = div(head_dim, 2)
    frequency_index = Nx.iota({half}, type: :f32)

    inverse_frequency =
      Nx.select(
        Nx.less(frequency_index, rotary_angles),
        Nx.pow(theta, frequency_index * (-2.0 / head_dim)),
        0.0
      )

    positions = Nx.iota({token_count}, type: :f32)
    frequency = Nx.new_axis(positions, 1) * Nx.new_axis(inverse_frequency, 0)
    embedding = Nx.concatenate([frequency, frequency], axis: 1)
    cosine = embedding |> Nx.cos() |> Nx.new_axis(1)
    sine = embedding |> Nx.sin() |> Nx.new_axis(1)

    {
      query * cosine + rotate_half(query, half) * sine,
      key * cosine + rotate_half(key, half) * sine
    }
  end

  defnp rotate_half(tensor, half) do
    first = Nx.slice_along_axis(tensor, 0, half, axis: 2)
    second = Nx.slice_along_axis(tensor, half, half, axis: 2)
    Nx.concatenate([-second, first], axis: 2)
  end

  defnp repeat_kv(tensor, groups, heads, head_dim) do
    token_count = Nx.axis_size(tensor, 0)
    kv_heads = Nx.axis_size(tensor, 1)

    tensor
    |> Nx.new_axis(2)
    |> Nx.broadcast({token_count, kv_heads, groups, head_dim})
    |> Nx.reshape({token_count, heads, head_dim})
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
    shifted = input - Nx.reduce_max(input, axes: [-1], keep_axes: true)
    exponentials = Nx.exp(shifted)
    exponentials / Nx.sum(exponentials, axes: [-1], keep_axes: true)
  end

  defp transfer(tensor, Nx.BinaryBackend), do: Nx.backend_copy(tensor, Nx.BinaryBackend)
  defp transfer(tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp build_opts(EXLA.Backend), do: [compiler: EXLA]

  defp build_opts({EXLA.Backend, backend_opts}),
    do: [compiler: EXLA] ++ Keyword.take(backend_opts, [:client, :device_id])

  defp build_opts(_backend), do: []
end
