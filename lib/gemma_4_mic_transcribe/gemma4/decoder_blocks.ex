defmodule Gemma4MicTranscribe.Gemma4.DecoderBlocks do
  @moduledoc """
  Extracts dense Gemma 4 decoder blocks for standalone prefill execution.

  The extracted block owns only that layer's parameters. Its input is the full
  hidden-state sequence produced by the preceding layer (or by the input
  embedding path for layer 0). Attention therefore still needs the sequence's
  position ids and attention mask; when omitted, contiguous positions and an
  all-visible mask are generated.

  This currently supports the unified dense Gemma 4 architecture. E4B has a
  different decoder block with shared KV projections and per-layer inputs.
  """

  alias Gemma4MicTranscribe.Gemma4Unified.Model

  defmodule Extracted do
    @moduledoc "A standalone decoder block and its isolated parameters."

    @enforce_keys [
      :id,
      :layer_index,
      :layer_type,
      :input_size,
      :parameter_count,
      :model,
      :params,
      :predict_fun,
      :backend
    ]
    defstruct @enforce_keys
  end

  defmodule Chain do
    @moduledoc "A contiguous sequence of standalone decoder blocks."

    @enforce_keys [
      :id,
      :layer_indices,
      :layer_types,
      :input_size,
      :parameter_count,
      :model,
      :params,
      :predict_fun,
      :backend
    ]
    defstruct @enforce_keys
  end

  defmodule Tail do
    @moduledoc "A final decoder-block chain with its output norm and vocabulary head."

    @enforce_keys [
      :id,
      :layer_indices,
      :layer_types,
      :input_size,
      :vocab_size,
      :parameter_count,
      :model,
      :params,
      :predict_fun,
      :backend,
      :tokenizer
    ]
    defstruct @enforce_keys
  end

  @doc "Extracts one decoder block from an already-loaded unified runtime."
  @spec extract(map(), non_neg_integer()) :: {:ok, Extracted.t()} | {:error, term()}
  def extract(runtime, layer_index) do
    with {:ok, model_info} <- fetch(runtime, :model_info),
         {:ok, spec} <- fetch(model_info, :spec),
         :ok <- validate_spec(spec),
         :ok <- validate_layer(layer_index, spec.num_blocks),
         {:ok, source_params} <- fetch(model_info, :params),
         {:ok, params} <- extract_params(source_params, layer_index) do
      model = Model.decoder_block_model(spec, layer_index)
      backend = Map.get(runtime, :backend)
      {_init_fun, predict_fun} = Axon.build(model, build_opts(backend))

      {:ok,
       %Extracted{
         id: "language_model.layer.#{layer_index}",
         layer_index: layer_index,
         layer_type: layer_type(spec, layer_index),
         input_size: spec.hidden_size,
         parameter_count: parameter_count(params.data),
         model: model,
         params: params,
         predict_fun: predict_fun,
         backend: backend
       }}
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  @doc "Like `extract/2`, but raises when extraction fails."
  def extract!(runtime, layer_index) do
    case extract(runtime, layer_index) do
      {:ok, block} -> block
      {:error, reason} -> raise ArgumentError, reason
    end
  end

  @doc "Extracts a contiguous ascending decoder-block chain."
  @spec extract_chain(map(), Enumerable.t()) :: {:ok, Chain.t()} | {:error, term()}
  def extract_chain(runtime, layer_indices) do
    layer_indices = Enum.to_list(layer_indices)

    with {:ok, model_info} <- fetch(runtime, :model_info),
         {:ok, spec} <- fetch(model_info, :spec),
         :ok <- validate_spec(spec),
         :ok <- validate_chain(layer_indices, spec.num_blocks),
         {:ok, source_params} <- fetch(model_info, :params),
         {:ok, params} <- extract_chain_params(source_params, layer_indices) do
      model = Model.decoder_block_chain_model(spec, layer_indices)
      backend = Map.get(runtime, :backend)
      {_init_fun, predict_fun} = Axon.build(model, build_opts(backend))

      {:ok,
       %Chain{
         id: "language_model.layers.#{hd(layer_indices)}-#{List.last(layer_indices)}",
         layer_indices: layer_indices,
         layer_types: Enum.map(layer_indices, &layer_type(spec, &1)),
         input_size: spec.hidden_size,
         parameter_count: parameter_count(params.data),
         model: model,
         params: params,
         predict_fun: predict_fun,
         backend: backend
       }}
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  @doc "Like `extract_chain/2`, but raises when extraction fails."
  def extract_chain!(runtime, layer_indices) do
    case extract_chain(runtime, layer_indices) do
      {:ok, chain} -> chain
      {:error, reason} -> raise ArgumentError, reason
    end
  end

  @doc "Extracts a final decoder-block chain plus the standalone vocabulary head."
  @spec extract_tail(map(), Enumerable.t()) :: {:ok, Tail.t()} | {:error, term()}
  def extract_tail(runtime, layer_indices) do
    layer_indices = Enum.to_list(layer_indices)

    with {:ok, model_info} <- fetch(runtime, :model_info),
         {:ok, spec} <- fetch(model_info, :spec),
         :ok <- validate_spec(spec),
         :ok <- validate_chain(layer_indices, spec.num_blocks),
         :ok <- validate_final_layer(layer_indices, spec.num_blocks),
         {:ok, source_params} <- fetch(model_info, :params),
         {:ok, params} <- extract_tail_params(source_params, layer_indices) do
      model = Model.decoder_tail_model(spec, layer_indices)
      backend = Map.get(runtime, :backend)
      {_init_fun, predict_fun} = Axon.build(model, build_opts(backend))

      {:ok,
       %Tail{
         id: "language_model.tail.#{hd(layer_indices)}-#{List.last(layer_indices)}",
         layer_indices: layer_indices,
         layer_types: Enum.map(layer_indices, &layer_type(spec, &1)),
         input_size: spec.hidden_size,
         vocab_size: spec.vocab_size,
         parameter_count: parameter_count(params.data),
         model: model,
         params: params,
         predict_fun: predict_fun,
         backend: backend,
         tokenizer: Map.get(runtime, :tokenizer)
       }}
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  @doc "Like `extract_tail/2`, but raises when extraction fails."
  def extract_tail!(runtime, layer_indices) do
    case extract_tail(runtime, layer_indices) do
      {:ok, tail} -> tail
      {:error, reason} -> raise ArgumentError, reason
    end
  end

  @doc "Runs an extracted block, chain, or vocabulary tail over a hidden-state sequence."
  @spec run(Extracted.t() | Chain.t() | Tail.t(), Nx.Tensor.t(), keyword()) ::
          {:ok, Nx.Tensor.t()} | {:error, term()}
  def run(component, hidden_state, opts \\ [])

  def run(%Extracted{} = block, %Nx.Tensor{} = hidden_state, opts) do
    run_model(block, hidden_state, opts)
  end

  def run(%Chain{} = chain, %Nx.Tensor{} = hidden_state, opts) do
    run_model(chain, hidden_state, opts)
  end

  def run(%Tail{} = tail, %Nx.Tensor{} = hidden_state, opts) do
    run_model(tail, hidden_state, opts)
  end

  defp run_model(component, hidden_state, opts) do
    hidden_state = copy_to_backend(hidden_state, component.backend)

    with :ok <- validate_hidden_state(hidden_state, component.input_size),
         {:ok, inputs} <- inputs(hidden_state, opts, component.backend) do
      {:ok, component.predict_fun.(component.params, inputs)}
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  @doc "Like `run/3`, but raises when standalone execution fails."
  def run!(component, hidden_state, opts \\ [])

  def run!(%Extracted{} = block, %Nx.Tensor{} = hidden_state, opts) do
    case run(block, hidden_state, opts) do
      {:ok, output} -> output
      {:error, reason} -> raise ArgumentError, reason
    end
  end

  def run!(%Chain{} = chain, %Nx.Tensor{} = hidden_state, opts) do
    case run(chain, hidden_state, opts) do
      {:ok, output} -> output
      {:error, reason} -> raise ArgumentError, reason
    end
  end

  def run!(%Tail{} = tail, %Nx.Tensor{} = hidden_state, opts) do
    case run(tail, hidden_state, opts) do
      {:ok, output} -> output
      {:error, reason} -> raise ArgumentError, reason
    end
  end

  @doc "Returns the highest-scoring next-token candidates from an extracted tail."
  def top_k(%Tail{} = tail, %Nx.Tensor{} = hidden_state, k, opts \\ []) do
    cond do
      not (is_integer(k) and k > 0 and k <= tail.vocab_size) ->
        {:error, "top-k must be an integer in 1..#{tail.vocab_size}, got: #{inspect(k)}"}

      Nx.axis_size(hidden_state, 0) != 1 ->
        {:error, "top-k token decoding currently supports batch size 1"}

      true ->
        with {:ok, logits} <- run(tail, hidden_state, opts) do
          {scores, token_ids} = Nx.top_k(logits, k: k)
          scores = scores |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()
          token_ids = token_ids |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()

          {:ok,
           Enum.zip(token_ids, scores)
           |> Enum.map(fn {token_id, score} ->
             %{token_id: token_id, token: token_label(tail.tokenizer, token_id), score: score}
           end)}
        end
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  @doc "Like `top_k/4`, but raises when standalone decoding fails."
  def top_k!(%Tail{} = tail, %Nx.Tensor{} = hidden_state, k, opts \\ []) do
    case top_k(tail, hidden_state, k, opts) do
      {:ok, candidates} -> candidates
      {:error, reason} -> raise ArgumentError, reason
    end
  end

  defp inputs(hidden_state, opts, configured_backend) do
    {batch_size, sequence_length, _hidden_size} = Nx.shape(hidden_state)
    backend = configured_backend || Nx.BinaryBackend

    position_ids =
      Keyword.get_lazy(opts, :position_ids, fn ->
        0..(sequence_length - 1)
        |> Enum.to_list()
        |> Nx.tensor(type: :s64, backend: backend)
        |> Nx.new_axis(0)
        |> Nx.broadcast({batch_size, sequence_length})
      end)

    attention_mask =
      Keyword.get_lazy(opts, :attention_mask, fn ->
        Nx.broadcast(Nx.tensor(1, type: :s64, backend: backend), {batch_size, sequence_length})
      end)

    cond do
      not match?(%Nx.Tensor{}, position_ids) ->
        {:error, ":position_ids must be an Nx tensor"}

      Nx.shape(position_ids) != {batch_size, sequence_length} ->
        {:error,
         ":position_ids must have shape #{inspect({batch_size, sequence_length})}, got: " <>
           inspect(Nx.shape(position_ids))}

      not match?(%Nx.Tensor{}, attention_mask) ->
        {:error, ":attention_mask must be an Nx tensor"}

      Nx.shape(attention_mask) != {batch_size, sequence_length} ->
        {:error,
         ":attention_mask must have shape #{inspect({batch_size, sequence_length})}, got: " <>
           inspect(Nx.shape(attention_mask))}

      true ->
        {:ok,
         %{
           "hidden_state" => hidden_state,
           "position_ids" => position_ids,
           "attention_mask" => attention_mask
         }}
    end
  end

  defp validate_hidden_state(hidden_state, input_size) do
    cond do
      Nx.rank(hidden_state) != 3 ->
        {:error, "hidden_state must have shape {batch, sequence, #{input_size}}"}

      Nx.axis_size(hidden_state, 2) != input_size ->
        {:error,
         "hidden_state last axis must be #{input_size}, got: #{Nx.axis_size(hidden_state, 2)}"}

      Nx.axis_size(hidden_state, 1) == 0 ->
        {:error, "hidden_state sequence must not be empty"}

      true ->
        :ok
    end
  end

  defp copy_to_backend(tensor, nil), do: tensor
  defp copy_to_backend(tensor, backend), do: Nx.backend_copy(tensor, backend)

  defp extract_params(%Axon.ModelState{data: data}, layer_index) do
    prefix = "decoder.blocks.#{layer_index}."
    data = Map.filter(data, fn {name, _value} -> String.starts_with?(name, prefix) end)

    if map_size(data) == 0 do
      {:error, "runtime contains no parameters for decoder layer #{layer_index}"}
    else
      {:ok, Axon.ModelState.new(data)}
    end
  end

  defp extract_params(other, _layer_index),
    do: {:error, "expected Axon model parameters, got: #{inspect(other)}"}

  defp extract_chain_params(%Axon.ModelState{data: data}, layer_indices) do
    prefixes = Enum.map(layer_indices, &"decoder.blocks.#{&1}.")

    missing_layers =
      Enum.reject(layer_indices, fn layer_index ->
        prefix = "decoder.blocks.#{layer_index}."
        Enum.any?(data, fn {name, _value} -> String.starts_with?(name, prefix) end)
      end)

    data =
      Map.filter(data, fn {name, _value} ->
        Enum.any?(prefixes, &String.starts_with?(name, &1))
      end)

    case missing_layers do
      [] -> {:ok, Axon.ModelState.new(data)}
      missing -> {:error, "runtime contains no parameters for decoder layers #{inspect(missing)}"}
    end
  end

  defp extract_chain_params(other, _layer_indices),
    do: {:error, "expected Axon model parameters, got: #{inspect(other)}"}

  defp extract_tail_params(%Axon.ModelState{data: data} = source_params, layer_indices) do
    with {:ok, block_params} <- extract_chain_params(source_params, layer_indices),
         {:ok, output_norm} <- required_params(data, "output_norm"),
         {:ok, head} <- output_head_params(data) do
      tail_data =
        block_params.data
        |> Map.put("output_norm", output_norm)
        |> Map.put("language_modeling_head.output", head)

      {:ok, Axon.ModelState.new(tail_data)}
    end
  end

  defp extract_tail_params(other, _layer_indices),
    do: {:error, "expected Axon model parameters, got: #{inspect(other)}"}

  defp required_params(data, name) do
    case Map.fetch(data, name) do
      {:ok, params} -> {:ok, params}
      :error -> {:error, "runtime contains no parameters for #{name}"}
    end
  end

  defp output_head_params(data) do
    case Map.get(data, "language_modeling_head.output") ||
           Map.get(data, "embedder.token_embedding") do
      nil -> {:error, "runtime contains no vocabulary-head parameters"}
      params -> {:ok, params}
    end
  end

  defp parameter_count(data) do
    Enum.reduce(data, 0, fn {_name, parameters}, count ->
      count + Enum.reduce(parameters, 0, fn {_name, tensor}, sum -> sum + Nx.size(tensor) end)
    end)
  end

  defp validate_spec(%Model{}), do: :ok

  defp validate_spec(spec),
    do:
      {:error,
       "standalone decoder blocks currently require Gemma4Unified.Model, got: " <>
         inspect(Map.get(spec, :__struct__, spec))}

  defp validate_layer(layer_index, count)
       when is_integer(layer_index) and layer_index >= 0 and layer_index < count,
       do: :ok

  defp validate_layer(layer_index, count),
    do: {:error, "expected layer index in 0..#{count - 1}, got: #{inspect(layer_index)}"}

  defp validate_chain([], _count), do: {:error, "decoder block chain must not be empty"}

  defp validate_chain(layer_indices, count) do
    with :ok <- validate_chain_layers(layer_indices, count) do
      first = hd(layer_indices)
      last = List.last(layer_indices)
      expected = if last >= first, do: Enum.to_list(first..last//1), else: []

      if layer_indices == expected do
        :ok
      else
        {:error, "decoder block chain must use contiguous ascending layer indices"}
      end
    end
  end

  defp validate_chain_layers(layer_indices, count) do
    case Enum.find(layer_indices, &(not (is_integer(&1) and &1 >= 0 and &1 < count))) do
      nil -> :ok
      invalid -> {:error, "expected layer index in 0..#{count - 1}, got: #{inspect(invalid)}"}
    end
  end

  defp validate_final_layer(layer_indices, count) do
    if List.last(layer_indices) == count - 1 do
      :ok
    else
      {:error, "decoder tail must end at final layer #{count - 1}"}
    end
  end

  defp layer_type(spec, layer_index) do
    spec.layer_types
    |> Kernel.||(default_layer_types(spec.num_blocks))
    |> Enum.fetch!(layer_index)
  end

  defp default_layer_types(count) do
    Enum.map(0..(count - 1), fn index ->
      if rem(index + 1, 6) == 0, do: :full_attention, else: :sliding_attention
    end)
  end

  defp token_label(nil, token_id), do: Integer.to_string(token_id)

  defp token_label(tokenizer, token_id) do
    Bumblebee.Tokenizer.id_to_token(tokenizer, token_id) ||
      Bumblebee.Tokenizer.decode(tokenizer, [token_id])
  end

  defp fetch(map, key) when is_map(map) do
    case Map.fetch(map, key) do
      {:ok, value} -> {:ok, value}
      :error -> {:error, "decoder block runtime is missing #{key}"}
    end
  end

  defp build_opts(EXLA.Backend), do: [compiler: EXLA]

  defp build_opts({EXLA.Backend, backend_opts}),
    do: [compiler: EXLA] ++ Keyword.take(backend_opts, [:client, :device_id])

  defp build_opts(_backend), do: []
end
