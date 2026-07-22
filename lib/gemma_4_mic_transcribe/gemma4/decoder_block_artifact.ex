defmodule Gemma4MicTranscribe.Gemma4.DecoderBlockArtifact do
  @moduledoc """
  Saves and loads one standalone Gemma 4 decoder block.

  The artifact contains only the selected block's parameter tensors and the
  model metadata needed to rebuild that block's Axon graph. It does not contain
  embeddings, any other decoder layer, the output norm, or the vocabulary head.
  """

  alias Gemma4MicTranscribe.Gemma4.DecoderBlocks
  alias Gemma4MicTranscribe.Gemma4.DecoderBlocks.Extracted
  alias Gemma4MicTranscribe.Gemma4Unified.Model

  @version 1
  @manifest "manifest.etf"
  @parameters "parameters.safetensors"
  @verification "verification.safetensors"

  def save!(%Extracted{} = block, path, opts \\ []) do
    path = Path.expand(path)

    if File.exists?(path) do
      raise ArgumentError, "artifact path already exists: #{path}"
    end

    File.mkdir_p!(Path.dirname(path))
    temporary = path <> ".tmp-#{System.unique_integer([:positive])}"
    File.mkdir_p!(temporary)

    try do
      {tensors, parameter_paths} = flatten_parameters(block.params.data)
      Safetensors.write!(Path.join(temporary, @parameters), tensors)

      sequence_length = Keyword.get(opts, :verification_sequence_length, 8)
      verification = build_verification(block, sequence_length)
      Safetensors.write!(Path.join(temporary, @verification), verification)

      manifest = %{
        version: @version,
        kind: :decoder_block,
        id: block.id,
        layer_index: block.layer_index,
        layer_type: block.layer_type,
        input_size: block.input_size,
        parameter_count: block.parameter_count,
        spec: block.spec,
        parameter_paths: parameter_paths,
        verification: @verification,
        verification_sequence_length: sequence_length
      }

      File.write!(Path.join(temporary, @manifest), :erlang.term_to_binary(manifest))
      File.rename!(temporary, path)
      path
    rescue
      exception ->
        File.rm_rf(temporary)
        reraise exception, __STACKTRACE__
    end
  end

  def load!(path, backend) do
    path = Path.expand(path)
    manifest = read_manifest!(path)
    tensors = Safetensors.read!(Path.join(path, @parameters), lazy: true)

    params =
      manifest.parameter_paths
      |> Enum.map(fn {tensor_name, [node_name, parameter_name]} ->
        {node_name, parameter_name, load_tensor!(tensors, tensor_name, backend)}
      end)
      |> Enum.group_by(&elem(&1, 0), &{elem(&1, 1), elem(&1, 2)})
      |> Map.new(fn {node_name, parameters} -> {node_name, Map.new(parameters)} end)
      |> Axon.ModelState.new()

    model = Model.decoder_block_model(manifest.spec, manifest.layer_index)
    {_init_fun, predict_fun} = Axon.build(model, build_opts(backend))

    %Extracted{
      id: manifest.id,
      layer_index: manifest.layer_index,
      layer_type: manifest.layer_type,
      input_size: manifest.input_size,
      parameter_count: manifest.parameter_count,
      spec: manifest.spec,
      model: model,
      params: params,
      predict_fun: predict_fun,
      backend: backend
    }
  end

  def load_verification!(path, backend) do
    path = Path.expand(path)
    manifest = read_manifest!(path)

    path
    |> Path.join(manifest.verification)
    |> load_input!(backend)
  end

  def load_input!(path, backend) do
    tensors = Safetensors.read!(path, lazy: true)

    %{
      hidden_state: load_tensor!(tensors, "hidden_state", backend),
      position_ids: load_tensor!(tensors, "position_ids", backend),
      attention_mask: load_tensor!(tensors, "attention_mask", backend),
      expected_output: maybe_load_tensor(tensors, "expected_output", backend)
    }
  end

  def save_input!(path, hidden_state, opts \\ []) do
    path = Path.expand(path)

    if File.exists?(path) do
      raise ArgumentError, "decoder block input path already exists: #{path}"
    end

    {batch_size, sequence_length, _input_size} = Nx.shape(hidden_state)
    backend = tensor_backend(hidden_state)

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
        Nx.broadcast(
          Nx.tensor(1, type: :s64, backend: backend),
          {batch_size, sequence_length}
        )
      end)

    tensors = %{
      "hidden_state" => copy_to_binary(hidden_state),
      "position_ids" => copy_to_binary(position_ids),
      "attention_mask" => copy_to_binary(attention_mask)
    }

    tensors =
      case Keyword.get(opts, :expected_output) do
        nil -> tensors
        expected -> Map.put(tensors, "expected_output", copy_to_binary(expected))
      end

    File.mkdir_p!(Path.dirname(path))
    temporary = path <> ".tmp-#{System.unique_integer([:positive])}"

    try do
      Safetensors.write!(temporary, tensors)
      File.rename!(temporary, path)
      path
    rescue
      exception ->
        File.rm(temporary)
        reraise exception, __STACKTRACE__
    end
  end

  def verify!(%Extracted{} = block, input, opts \\ []) do
    output =
      DecoderBlocks.run!(block, input.hidden_state,
        position_ids: input.position_ids,
        attention_mask: input.attention_mask
      )

    case input.expected_output do
      nil ->
        %{output: output, verified: nil, max_abs_error: nil}

      expected ->
        difference = Nx.abs(Nx.subtract(output, expected))
        max_abs_error = difference |> Nx.reduce_max() |> scalar_number()

        verified =
          output
          |> Nx.all_close(expected,
            atol: Keyword.get(opts, :atol, 1.0e-2),
            rtol: Keyword.get(opts, :rtol, 1.0e-2)
          )
          |> scalar_number()
          |> Kernel.==(1)

        %{output: output, verified: verified, max_abs_error: max_abs_error}
    end
  end

  def read_manifest!(path) do
    manifest =
      path
      |> Path.join(@manifest)
      |> File.read!()
      |> :erlang.binary_to_term()

    cond do
      manifest.version != @version ->
        raise ArgumentError,
              "unsupported decoder block artifact version #{inspect(manifest.version)}"

      manifest.kind != :decoder_block ->
        raise ArgumentError, "artifact is not a decoder block"

      true ->
        manifest
    end
  end

  defp build_verification(block, sequence_length)
       when is_integer(sequence_length) and sequence_length > 0 do
    backend = block.backend || Nx.BinaryBackend
    element_count = sequence_length * block.input_size

    hidden_state =
      Nx.with_default_backend(backend, fn ->
        {1, sequence_length, block.input_size}
        |> Nx.iota(type: :f32)
        |> Nx.divide(element_count)
        |> Nx.as_type(:bf16)
      end)

    position_ids =
      0..(sequence_length - 1)
      |> Enum.to_list()
      |> Nx.tensor(type: :s64, backend: backend)
      |> Nx.new_axis(0)

    attention_mask =
      Nx.broadcast(Nx.tensor(1, type: :s64, backend: backend), {1, sequence_length})

    expected_output =
      DecoderBlocks.run!(block, hidden_state,
        position_ids: position_ids,
        attention_mask: attention_mask
      )

    %{
      "hidden_state" => hidden_state,
      "position_ids" => position_ids,
      "attention_mask" => attention_mask,
      "expected_output" => expected_output
    }
  end

  defp flatten_parameters(data) do
    data
    |> Enum.sort()
    |> Enum.flat_map(fn {node_name, parameters} ->
      parameters
      |> Enum.sort()
      |> Enum.map(fn {parameter_name, tensor} -> {node_name, parameter_name, tensor} end)
    end)
    |> Enum.with_index()
    |> Enum.reduce({%{}, %{}}, fn {{node_name, parameter_name, tensor}, index},
                                  {tensors, paths} ->
      tensor_name = "p#{index}"

      {
        Map.put(tensors, tensor_name, tensor),
        Map.put(paths, tensor_name, [node_name, parameter_name])
      }
    end)
  end

  defp load_tensor!(tensors, name, backend) do
    tensor =
      Nx.with_default_backend(Nx.BinaryBackend, fn -> Nx.to_tensor(Map.fetch!(tensors, name)) end)

    case backend do
      Nx.BinaryBackend -> tensor
      nil -> tensor
      backend -> Nx.backend_transfer(tensor, backend)
    end
  end

  defp maybe_load_tensor(tensors, name, backend) do
    case Map.fetch(tensors, name) do
      {:ok, _tensor} -> load_tensor!(tensors, name, backend)
      :error -> nil
    end
  end

  defp scalar_number(tensor) do
    tensor
    |> Nx.backend_transfer(Nx.BinaryBackend)
    |> Nx.to_number()
  end

  defp copy_to_binary(%Nx.Tensor{data: %Nx.BinaryBackend{}} = tensor), do: tensor
  defp copy_to_binary(tensor), do: Nx.backend_copy(tensor, Nx.BinaryBackend)

  defp tensor_backend(%Nx.Tensor{data: data}), do: data.__struct__

  defp build_opts(EXLA.Backend), do: [compiler: EXLA]

  defp build_opts({EXLA.Backend, backend_opts}),
    do: [compiler: EXLA] ++ Keyword.take(backend_opts, [:client, :device_id])

  defp build_opts(_backend), do: []
end
