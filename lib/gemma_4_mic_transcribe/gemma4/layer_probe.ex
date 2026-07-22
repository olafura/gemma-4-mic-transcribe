defmodule Gemma4MicTranscribe.Gemma4.LayerProbe do
  @moduledoc """
  Captures selected Gemma 4 decoder activations for layer analysis.

  The probe rewrites the existing Axon graph to return small slices from named
  intermediate nodes. It uses the already-loaded model parameters and only
  copies the requested token positions out of the compiled graph.

  Supported capture points are:

    * `:attention` - the post-normalized attention residual contribution
    * `:ffn` - the post-normalized FFN residual contribution
    * `:per_layer_input` - E4B's gated, projected per-layer embedding contribution
    * `:hidden_state` - the completed decoder layer output

  `:per_layer_input` is reported as unavailable for models without per-layer
  embeddings. Attention probabilities are not exposed by the current model
  graph; `:attention` is the contribution attention makes to the residual stream.
  """

  @capture_nodes %{
    attention: "post_attention_norm",
    ffn: "post_ffn_norm",
    per_layer_input: "per_layer.post_norm",
    hidden_state: "layer_scalar"
  }
  @captures Map.keys(@capture_nodes)

  @doc """
  Runs a layer probe over model-ready input tensors.

  `runtime` must contain `model_info.model`, `model_info.params`, and
  `model_info.spec`, matching `Gemma4Unified.Runtime`. Inputs are the same map
  accepted by the model's Axon prediction function.

  ## Options

    * `:layers` - layer indices to inspect (default: every layer)
    * `:positions` - token positions or selectors (default: `[:last]`)
    * `:capture` - capture points listed above (default: `[:hidden_state]`)
    * `:top_k_logits` - apply a logit lens to hidden states (default: `0`)
    * `:include_activations` - retain captured vectors in the report (default: `false`)

  Position selectors include `:last`, `:audio_begin`, `:audio_end`,
  `:first_audio`, `:last_audio`, an integer index, a token string, or
  `{:token, token_string, zero_based_occurrence}`.
  """
  @spec run(map(), map(), keyword()) :: {:ok, map()} | {:error, term()}
  def run(runtime, inputs, opts \\ []) do
    with {:ok, context} <- validate_context(runtime, inputs, opts),
         {:ok, positions} <- resolve_positions(context, Keyword.get(opts, :positions, [:last])),
         {:ok, probe_model, available, unavailable} <-
           build_probe_model(
             context.model,
             context.layers,
             context.captures,
             Enum.map(positions, & &1.index)
           ) do
      {_init_fun, predict_fun} = Axon.build(probe_model, build_opts(context.backend))
      captured = predict_fun.(context.params, inputs)

      {:ok,
       report(
         context,
         positions,
         captured,
         available,
         unavailable,
         opts
       )}
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  @doc "Like `run/3`, but raises on invalid input or probe execution failure."
  @spec run!(map(), map(), keyword()) :: map()
  def run!(runtime, inputs, opts \\ []) do
    case run(runtime, inputs, opts) do
      {:ok, report} -> report
      {:error, reason} -> raise ArgumentError, reason
    end
  end

  defp validate_context(runtime, inputs, opts) do
    with {:ok, model_info} <- fetch_map(runtime, :model_info),
         {:ok, model} <- fetch_map(model_info, :model),
         {:ok, params} <- fetch_map(model_info, :params),
         {:ok, spec} <- fetch_map(model_info, :spec),
         :ok <- validate_backend(Map.get(runtime, :backend)),
         :ok <- validate_model(model),
         :ok <- validate_inputs(inputs),
         {:ok, layers} <- validate_layers(Keyword.get(opts, :layers), spec),
         {:ok, captures} <- validate_captures(Keyword.get(opts, :capture, [:hidden_state])),
         {:ok, top_k} <- validate_top_k(Keyword.get(opts, :top_k_logits, 0), spec),
         :ok <- validate_logit_lens_capture(top_k, captures) do
      {:ok,
       %{
         model: model,
         params: params,
         spec: spec,
         inputs: inputs,
         tokenizer: Map.get(runtime, :tokenizer),
         backend: Map.get(runtime, :backend),
         model_name: Map.get(runtime, :model_name) || Map.get(runtime, :repo_id),
         layers: layers,
         captures: captures,
         top_k: top_k
       }}
    end
  end

  defp fetch_map(map, key) when is_map(map) do
    case Map.fetch(map, key) do
      {:ok, value} -> {:ok, value}
      :error -> {:error, "layer probe runtime is missing #{key}"}
    end
  end

  defp validate_backend({EXLA.Backend, opts}) when is_list(opts) do
    if Keyword.get(opts, :client) == :rocm do
      {:error,
       "layer probing is disabled on EXLA/ROCm because instrumented intermediate outputs " <>
         "currently crash the XLA autotuner; reload the runtime with backend: \"torchx:cpu\""}
    else
      :ok
    end
  end

  defp validate_backend(_backend), do: :ok

  defp validate_model(%Axon{}), do: :ok
  defp validate_model(other), do: {:error, "expected an Axon model, got: #{inspect(other)}"}

  defp validate_inputs(%{"input_ids" => %Nx.Tensor{} = ids} = inputs) do
    cond do
      Nx.rank(ids) != 2 ->
        {:error, "layer probe expects input_ids with shape {batch, sequence}"}

      Nx.axis_size(ids, 0) != 1 ->
        {:error, "layer probe currently supports batch size 1"}

      not match?(%Nx.Tensor{}, inputs["attention_mask"]) ->
        {:error, "layer probe inputs are missing attention_mask"}

      true ->
        :ok
    end
  end

  defp validate_inputs(_), do: {:error, "layer probe inputs are missing input_ids"}

  defp validate_layers(nil, spec), do: {:ok, Enum.to_list(0..(num_layers(spec) - 1))}

  defp validate_layers(layers, spec) when is_list(layers) and layers != [] do
    count = num_layers(spec)

    case Enum.find(layers, &(not (is_integer(&1) and &1 >= 0 and &1 < count))) do
      nil -> {:ok, Enum.uniq(layers)}
      invalid -> {:error, "expected layer index in 0..#{count - 1}, got: #{inspect(invalid)}"}
    end
  end

  defp validate_layers(other, _spec),
    do: {:error, "expected :layers to be a non-empty list, got: #{inspect(other)}"}

  defp validate_captures(captures) when is_list(captures) and captures != [] do
    case Enum.find(captures, &(&1 not in @captures)) do
      nil -> {:ok, Enum.uniq(captures)}
      invalid -> {:error, "unsupported layer probe capture: #{inspect(invalid)}"}
    end
  end

  defp validate_captures(other),
    do: {:error, "expected :capture to be a non-empty list, got: #{inspect(other)}"}

  defp validate_top_k(top_k, spec)
       when is_integer(top_k) and top_k >= 0 and top_k <= spec.vocab_size,
       do: {:ok, top_k}

  defp validate_top_k(top_k, _spec),
    do: {:error, "expected :top_k_logits to be a non-negative integer, got: #{inspect(top_k)}"}

  defp validate_logit_lens_capture(0, _captures), do: :ok

  defp validate_logit_lens_capture(_top_k, captures) do
    if :hidden_state in captures do
      :ok
    else
      {:error, ":top_k_logits requires :hidden_state in :capture"}
    end
  end

  defp resolve_positions(context, selectors) when is_list(selectors) and selectors != [] do
    ids = Nx.to_flat_list(context.inputs["input_ids"])
    attention_mask = Nx.to_flat_list(context.inputs["attention_mask"])

    Enum.reduce_while(selectors, {:ok, []}, fn selector, {:ok, positions} ->
      case resolve_position(selector, ids, attention_mask, context) do
        {:ok, position} -> {:cont, {:ok, [position | positions]}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, positions} -> {:ok, Enum.reverse(positions)}
      error -> error
    end
  end

  defp resolve_positions(_context, other),
    do: {:error, "expected :positions to be a non-empty list, got: #{inspect(other)}"}

  defp resolve_position(:last, ids, attention_mask, context) do
    index =
      attention_mask
      |> Enum.with_index()
      |> Enum.reduce(nil, fn
        {value, index}, _last when value != 0 -> index
        _, last -> last
      end)

    position(:last, index, ids, context)
  end

  defp resolve_position(:audio_begin, ids, _attention_mask, context) do
    find_token_position(:audio_begin, ids, Map.get(context.spec, :boa_token_id), 0, context)
  end

  defp resolve_position(:audio_end, ids, _attention_mask, context) do
    find_token_position(:audio_end, ids, Map.get(context.spec, :eoa_token_id), 0, context)
  end

  defp resolve_position(:first_audio, ids, _attention_mask, context) do
    find_token_position(:first_audio, ids, Map.get(context.spec, :audio_token_id), 0, context)
  end

  defp resolve_position(:last_audio, ids, _attention_mask, context) do
    token_id = Map.get(context.spec, :audio_token_id)

    occurrence =
      ids
      |> Enum.count(&(&1 == token_id))
      |> Kernel.-(1)

    find_token_position(:last_audio, ids, token_id, occurrence, context)
  end

  defp resolve_position(index, ids, _attention_mask, context) when is_integer(index) do
    position({:index, index}, index, ids, context)
  end

  defp resolve_position(token, ids, _attention_mask, context) when is_binary(token) do
    find_token_string_position(token, ids, 0, context)
  end

  defp resolve_position({:token, token, occurrence}, ids, _attention_mask, context)
       when is_binary(token) and is_integer(occurrence) and occurrence >= 0 do
    find_token_string_position(token, ids, occurrence, context)
  end

  defp resolve_position(selector, _ids, _attention_mask, _context) do
    {:error, "unsupported layer probe position selector: #{inspect(selector)}"}
  end

  defp find_token_string_position(token, _ids, _occurrence, %{tokenizer: nil}) do
    {:error, "cannot resolve token #{inspect(token)} without a tokenizer"}
  end

  defp find_token_string_position(token, ids, occurrence, context) do
    case Bumblebee.Tokenizer.token_to_id(context.tokenizer, token) do
      nil -> {:error, "tokenizer does not contain token #{inspect(token)}"}
      token_id -> find_token_position(token, ids, token_id, occurrence, context)
    end
  end

  defp find_token_position(label, ids, token_id, occurrence, context)
       when is_integer(token_id) and occurrence >= 0 do
    index =
      ids
      |> Enum.with_index()
      |> Enum.filter(fn {id, _index} -> id == token_id end)
      |> Enum.at(occurrence)
      |> case do
        nil -> nil
        {_id, index} -> index
      end

    position(label, index, ids, context)
  end

  defp find_token_position(label, _ids, token_id, _occurrence, _context) do
    {:error, "cannot resolve #{inspect(label)}: model spec has token id #{inspect(token_id)}"}
  end

  defp position(label, nil, _ids, _context) do
    {:error, "could not find layer probe position #{inspect(label)} in input_ids"}
  end

  defp position(label, index, ids, context) when index >= 0 and index < length(ids) do
    token_id = Enum.at(ids, index)

    {:ok,
     %{
       label: label,
       index: index,
       token_id: token_id,
       token: token_label(context.tokenizer, token_id)
     }}
  end

  defp position(label, index, ids, _context) do
    {:error,
     "layer probe position #{inspect(label)} resolved to #{index}, outside 0..#{length(ids) - 1}"}
  end

  defp build_probe_model(model, layers, captures, position_indices) do
    nodes = named_nodes(model)

    {outputs, available, unavailable} =
      Enum.reduce(layers, {%{}, [], []}, fn layer, acc ->
        Enum.reduce(captures, acc, fn capture, {outputs, available, unavailable} ->
          node_name = "decoder.blocks.#{layer}.#{Map.fetch!(@capture_nodes, capture)}"

          case nodes[node_name] do
            nil when capture == :per_layer_input ->
              {outputs, available, [{layer, capture} | unavailable]}

            nil ->
              raise ArgumentError, "model graph does not expose #{node_name}"

            node ->
              key = capture_key(layer, capture)

              output =
                %Axon{nodes: model.nodes, output: node.id}
                |> Axon.nx(
                  fn tensor ->
                    Nx.take(tensor, Nx.tensor(position_indices), axis: 1)
                  end,
                  name: "layer_probe.#{layer}.#{capture}"
                )

              {Map.put(outputs, key, output), [{layer, capture} | available], unavailable}
          end
        end)
      end)

    if map_size(outputs) == 0 do
      {:error, "none of the requested capture points are available"}
    else
      {:ok, Axon.container(outputs), Enum.reverse(available), Enum.reverse(unavailable)}
    end
  end

  defp named_nodes(model) do
    Axon.reduce_nodes(model, %{}, fn node, nodes ->
      case explicit_name(node) do
        nil -> nodes
        name -> Map.put(nodes, name, node)
      end
    end)
  end

  defp explicit_name(node) do
    try do
      case node.name.(node.op_name, %{}) do
        name when is_binary(name) -> name
        _ -> nil
      end
    rescue
      _ -> nil
    end
  end

  defp report(context, positions, captured, available, unavailable, opts) do
    host_captures = Map.new(captured, fn {key, tensor} -> {key, to_host(tensor)} end)
    layer_scalars = layer_scalars(context.params, context.layers)

    metrics =
      capture_metrics(
        context.layers,
        context.captures,
        positions,
        host_captures,
        layer_scalars
      )

    report = %{
      model: context.model_name,
      positions: positions,
      layers: layer_reports(context.spec, context.layers, metrics, layer_scalars),
      captures: context.captures,
      available: available,
      unavailable: unavailable,
      hidden_state_similarity: hidden_state_similarity(context.layers, positions, host_captures),
      logit_lens: logit_lens(context, positions, host_captures)
    }

    if Keyword.get(opts, :include_activations, false) do
      Map.put(report, :activations, host_captures)
    else
      report
    end
  end

  defp capture_metrics(layers, captures, positions, captured, layer_scalars) do
    Map.new(layers, fn layer ->
      hidden_norms = component_norms(captured[capture_key(layer, :hidden_state)])
      layer_scalar = layer_scalars[layer]

      components =
        Map.new(captures, fn component ->
          tensor = captured[capture_key(layer, component)]
          norms = component_norms(tensor)

          position_metrics =
            positions
            |> Enum.with_index()
            |> Enum.map(fn {position, position_offset} ->
              norm = at(norms, position_offset)
              hidden_norm = at(hidden_norms, position_offset)
              raw_ratio = safe_ratio(norm, hidden_norm)

              effective_ratio =
                if component == :hidden_state do
                  raw_ratio
                else
                  safe_ratio(scale_norm(norm, layer_scalar), hidden_norm)
                end

              %{
                position: position.label,
                index: position.index,
                norm: norm,
                rms: component_rms(tensor, position_offset),
                max_abs: component_max_abs(tensor, position_offset),
                hidden_norm_ratio: effective_ratio,
                pre_scalar_hidden_norm_ratio: raw_ratio
              }
            end)

          {component, position_metrics}
        end)

      {layer, components}
    end)
  end

  defp layer_reports(spec, layers, metrics, layer_scalars) do
    Enum.map(layers, fn layer ->
      %{
        index: layer,
        attention: layer_type(spec, layer),
        layer_scalar: layer_scalars[layer],
        metrics: metrics[layer]
      }
    end)
  end

  defp layer_scalars(params, layers) do
    Map.new(layers, fn layer ->
      scalar =
        get_in(params.data, ["decoder.blocks.#{layer}.layer_scalar", "layer_scalar"])

      value =
        case scalar do
          %Nx.Tensor{} -> scalar |> Nx.squeeze() |> Nx.as_type(:f32) |> Nx.to_number()
          nil -> 1.0
        end

      {layer, value}
    end)
  end

  defp hidden_state_similarity(layers, positions, captured) do
    layers
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.map(fn [left, right] ->
      left_tensor = captured[capture_key(left, :hidden_state)]
      right_tensor = captured[capture_key(right, :hidden_state)]

      %{
        from_layer: left,
        to_layer: right,
        positions:
          positions
          |> Enum.with_index()
          |> Enum.map(fn {position, offset} ->
            %{
              position: position.label,
              index: position.index,
              cosine: cosine_at(left_tensor, right_tensor, offset)
            }
          end)
      }
    end)
  end

  defp logit_lens(%{top_k: 0}, _positions, _captured), do: []

  defp logit_lens(context, positions, captured) do
    hidden_states =
      context.layers
      |> Enum.map(&captured[capture_key(&1, :hidden_state)])
      |> Nx.stack()
      |> Nx.backend_copy(runtime_backend(context.backend))

    params = context.params.data
    norm_weight = get_in(params, ["output_norm", "weight"])

    head_kernel =
      get_in(params, ["language_modeling_head.output", "kernel"]) ||
        get_in(params, ["embedder.token_embedding", "kernel"])

    if is_nil(norm_weight) or is_nil(head_kernel) do
      raise ArgumentError, "logit lens requires output_norm and language-model-head parameters"
    end

    {scores, token_ids} =
      run_logit_lens(
        hidden_states,
        norm_weight,
        head_kernel,
        context.spec.layer_norm_epsilon,
        Map.get(context.spec, :final_logit_softcapping),
        context.top_k,
        build_opts(context.backend)
      )

    score_values = scores |> to_host() |> Nx.to_list()
    id_values = token_ids |> to_host() |> Nx.to_list()

    context.layers
    |> Enum.with_index()
    |> Enum.map(fn {layer, layer_offset} ->
      %{
        layer: layer,
        positions:
          positions
          |> Enum.with_index()
          |> Enum.map(fn {position, position_offset} ->
            ids =
              get_in(id_values, [
                Access.at(layer_offset),
                Access.at(0),
                Access.at(position_offset)
              ])

            scores =
              get_in(score_values, [
                Access.at(layer_offset),
                Access.at(0),
                Access.at(position_offset)
              ])

            candidates =
              Enum.zip(ids, scores)
              |> Enum.map(fn {token_id, score} ->
                %{
                  token_id: token_id,
                  token: token_label(context.tokenizer, token_id),
                  score: score
                }
              end)

            %{position: position.label, index: position.index, candidates: candidates}
          end)
      }
    end)
  end

  defp run_logit_lens(hidden_states, norm_weight, head_kernel, epsilon, cap, top_k, opts) do
    fun =
      Nx.Defn.jit(
        fn hidden_states, norm_weight, head_kernel ->
          hidden_states = Nx.as_type(hidden_states, :f32)

          normalized =
            hidden_states
            |> Nx.multiply(
              hidden_states
              |> Nx.pow(2)
              |> Nx.mean(axes: [-1], keep_axes: true)
              |> Nx.add(epsilon)
              |> Nx.rsqrt()
            )
            |> Nx.multiply(Nx.as_type(norm_weight, :f32))
            |> Nx.as_type(Nx.type(head_kernel))

          # Keep the very large vocabulary kernel in its resident dtype. Casting
          # a 12B model's tied bf16 embedding table to f32 would add several GB
          # of temporary memory merely to inspect a few hidden states.
          logits = Nx.dot(normalized, [-1], head_kernel, [1]) |> Nx.as_type(:f32)

          logits =
            if cap do
              Nx.multiply(Nx.tanh(Nx.divide(logits, cap)), cap)
            else
              logits
            end

          Nx.top_k(logits, k: top_k)
        end,
        opts
      )

    fun.(hidden_states, norm_weight, head_kernel)
  end

  defp component_norms(nil), do: nil

  defp component_norms(tensor) do
    tensor
    |> Nx.as_type(:f32)
    |> Nx.pow(2)
    |> Nx.sum(axes: [-1])
    |> Nx.sqrt()
    |> Nx.to_flat_list()
  end

  defp component_rms(nil, _offset), do: nil

  defp component_rms(tensor, offset) do
    tensor
    |> vector_at(offset)
    |> Nx.as_type(:f32)
    |> Nx.pow(2)
    |> Nx.mean()
    |> Nx.sqrt()
    |> Nx.to_number()
  end

  defp component_max_abs(nil, _offset), do: nil

  defp component_max_abs(tensor, offset) do
    tensor
    |> vector_at(offset)
    |> Nx.as_type(:f32)
    |> Nx.abs()
    |> Nx.reduce_max()
    |> Nx.to_number()
  end

  defp cosine_at(nil, _right, _offset), do: nil
  defp cosine_at(_left, nil, _offset), do: nil

  defp cosine_at(left, right, offset) do
    left = left |> vector_at(offset) |> Nx.as_type(:f32)
    right = right |> vector_at(offset) |> Nx.as_type(:f32)
    denominator = Nx.to_number(Nx.multiply(Nx.LinAlg.norm(left), Nx.LinAlg.norm(right)))

    if denominator == 0 do
      nil
    else
      Nx.to_number(Nx.sum(Nx.multiply(left, right))) / denominator
    end
  end

  defp vector_at(tensor, offset), do: tensor[0][offset]

  defp safe_ratio(nil, _denominator), do: nil
  defp safe_ratio(_numerator, nil), do: nil
  defp safe_ratio(_numerator, 0), do: nil
  defp safe_ratio(numerator, denominator), do: numerator / denominator

  defp scale_norm(nil, _scalar), do: nil
  defp scale_norm(norm, scalar), do: norm * abs(scalar)

  defp at(nil, _offset), do: nil
  defp at(values, offset), do: Enum.at(values, offset)

  defp capture_key(layer, capture), do: "#{layer}:#{capture}"

  defp layer_type(spec, layer) do
    case Map.get(spec, :layer_types) do
      types when is_list(types) -> Enum.at(types, layer)
      _ -> nil
    end
  end

  defp num_layers(spec), do: Map.get(spec, :num_blocks) || Map.fetch!(spec, :num_hidden_layers)

  defp token_label(nil, token_id), do: Integer.to_string(token_id)

  defp token_label(tokenizer, token_id) do
    Bumblebee.Tokenizer.id_to_token(tokenizer, token_id) ||
      Bumblebee.Tokenizer.decode(tokenizer, [token_id])
  end

  defp to_host(%Nx.Tensor{} = tensor), do: Nx.backend_transfer(tensor, Nx.BinaryBackend)

  defp runtime_backend(nil), do: Nx.BinaryBackend
  defp runtime_backend(backend), do: backend

  defp build_opts(EXLA.Backend), do: [compiler: EXLA]

  defp build_opts({EXLA.Backend, backend_opts}),
    do: [compiler: EXLA] ++ Keyword.take(backend_opts, [:client, :device_id])

  defp build_opts(_backend), do: []
end
