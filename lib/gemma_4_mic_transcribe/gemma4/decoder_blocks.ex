defmodule Gemma4MicTranscribe.Gemma4.DecoderBlocks do
  @moduledoc """
  Extracts one dense Gemma 4 decoder block for standalone prefill execution.

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

  @doc "Runs an extracted block over a complete hidden-state sequence."
  @spec run(Extracted.t(), Nx.Tensor.t(), keyword()) :: {:ok, Nx.Tensor.t()} | {:error, term()}
  def run(%Extracted{} = block, %Nx.Tensor{} = hidden_state, opts \\ []) do
    hidden_state = copy_to_backend(hidden_state, block.backend)

    with :ok <- validate_hidden_state(hidden_state, block.input_size),
         {:ok, inputs} <- inputs(hidden_state, opts, block.backend) do
      {:ok, block.predict_fun.(block.params, inputs)}
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  @doc "Like `run/3`, but raises when standalone execution fails."
  def run!(%Extracted{} = block, %Nx.Tensor{} = hidden_state, opts \\ []) do
    case run(block, hidden_state, opts) do
      {:ok, output} -> output
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
