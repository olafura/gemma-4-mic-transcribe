defmodule Gemma4MicTranscribe.Gemma4.ExpertArtifact do
  @moduledoc """
  Range-extracts one routed Gemma 4 MoE expert into a standalone artifact.

  Gemma 4 stores every routed expert for a layer in two tensors whose leading
  axis is the expert index. A single expert is contiguous within each tensor,
  so extraction downloads only the safetensors header and the selected slices.
  """

  alias Gemma4MicTranscribe.Gemma4.Experts

  @version 1
  @kind :gemma4_routed_expert
  @manifest "manifest.etf"
  @parameters "parameters.safetensors"
  @default_repo "google/gemma-4-26B-A4B-it"
  @bf_type_atom :bf

  @doc "Extracts one routed expert without downloading the complete checkpoint."
  def extract!(path, opts \\ []) do
    path = Path.expand(path)

    if File.exists?(path) do
      raise ArgumentError, "artifact path already exists: #{path}"
    end

    repo = Keyword.get(opts, :repo, @default_repo)
    revision = Keyword.get(opts, :revision, "main")
    layer = Keyword.get(opts, :layer, 0)
    expert = Keyword.get(opts, :expert, 0)
    fetch_json = Keyword.get(opts, :fetch_json, &fetch_json!/1)
    fetch_range = Keyword.get(opts, :fetch_range, &range_get!/3)
    base_url = "https://huggingface.co/#{repo}/resolve/#{revision}"

    config = fetch_json.("#{base_url}/config.json")
    index = fetch_json.("#{base_url}/model.safetensors.index.json")
    descriptor = descriptor!(config, layer, expert)

    sources = [
      gate_up: descriptor.weights.gate_up,
      down: descriptor.weights.down
    ]

    {source_tensors, _headers, downloaded_bytes} =
      Enum.reduce(sources, {%{}, %{}, 0}, fn {key, reference}, {tensors, headers, downloaded} ->
        shard = get_in(index, ["weight_map", reference.tensor])

        unless is_binary(shard) do
          raise ArgumentError, "checkpoint index is missing #{reference.tensor}"
        end

        shard_url = "#{base_url}/#{shard}"
        {header, headers, header_bytes} = fetch_header(shard_url, headers, fetch_range)

        {tensor, tensor_bytes} =
          fetch_expert_slice!(
            shard_url,
            header,
            reference,
            expert,
            fetch_range
          )

        source = %{
          tensor: reference.tensor,
          shard: shard,
          checkpoint_shape: reference.checkpoint_shape
        }

        {
          Map.put(tensors, key, {tensor, source}),
          headers,
          downloaded + header_bytes + tensor_bytes
        }
      end)

    {{gate, up}, gate_source} = split_gate_up!(source_tensors.gate_up, descriptor)
    {down, down_source} = source_tensors.down

    tensors = %{
      "gate.weight" => gate,
      "up.weight" => up,
      "down.weight" => down
    }

    temporary = path <> ".tmp-#{System.unique_integer([:positive])}"
    File.mkdir_p!(Path.dirname(path))
    File.mkdir_p!(temporary)

    try do
      parameters_path = Path.join(temporary, @parameters)
      Safetensors.write!(parameters_path, tensors)

      manifest = %{
        version: @version,
        kind: @kind,
        source_repo: repo,
        source_revision: revision,
        layer_index: layer,
        expert_index: expert,
        input_size: descriptor.input_size,
        intermediate_size: descriptor.intermediate_size,
        output_size: descriptor.output_size,
        activation: descriptor.activation,
        parameter_count: descriptor.parameter_count,
        parameter_type: safetensors_dtype!(Nx.type(gate)),
        parameter_file: @parameters,
        parameter_sha256: sha256_file(parameters_path),
        downloaded_bytes: downloaded_bytes,
        source_checkpoint_bytes: get_in(index, ["metadata", "total_size"]),
        source_tensors: %{gate_up: gate_source, down: down_source}
      }

      File.write!(Path.join(temporary, @manifest), :erlang.term_to_binary(manifest))
      File.rename!(temporary, path)
      Map.update!(manifest, :parameter_type, &nx_type!/1)
    rescue
      exception ->
        File.rm_rf(temporary)
        reraise exception, __STACKTRACE__
    end
  end

  @doc "Loads and validates the three standalone expert matrices."
  def load!(path, backend \\ Nx.BinaryBackend) do
    path = Path.expand(path)
    manifest = read_manifest!(path)

    unless manifest.version == @version and manifest.kind == @kind do
      raise ArgumentError, "unsupported Gemma 4 expert artifact"
    end

    parameters_path = Path.join(path, manifest.parameter_file)

    unless sha256_file(parameters_path) == manifest.parameter_sha256 do
      raise ArgumentError, "expert parameter checksum mismatch"
    end

    tensors = Safetensors.read!(parameters_path, lazy: true)

    params = %{
      gate:
        load_tensor!(
          tensors,
          "gate.weight",
          {manifest.intermediate_size, manifest.input_size},
          backend
        ),
      up:
        load_tensor!(
          tensors,
          "up.weight",
          {manifest.intermediate_size, manifest.input_size},
          backend
        ),
      down:
        load_tensor!(
          tensors,
          "down.weight",
          {manifest.output_size, manifest.intermediate_size},
          backend
        )
    }

    {manifest, params}
  end

  @doc "Reads an expert artifact's metadata without loading its matrices."
  def read_manifest!(path) do
    manifest =
      path
      |> Path.expand()
      |> Path.join(@manifest)
      |> File.read!()
      |> :erlang.binary_to_term([:safe])

    Map.update!(manifest, :parameter_type, &nx_type!/1)
  end

  defp descriptor!(config, layer, expert) do
    case Enum.find(
           Experts.list(config, include_shared: false),
           &(&1.layer_index == layer and &1.expert_index == expert)
         ) do
      nil ->
        raise ArgumentError,
              "no routed Gemma 4 expert at layer #{inspect(layer)}, index #{inspect(expert)}"

      descriptor ->
        descriptor
    end
  end

  defp fetch_header(url, headers, fetch_range) do
    case Map.fetch(headers, url) do
      {:ok, header} ->
        {header, headers, 0}

      :error ->
        <<header_length::little-unsigned-64>> = fetch_exact!(fetch_range, url, 0, 7)
        header = fetch_exact!(fetch_range, url, 8, 7 + header_length) |> Jason.decode!()
        cached = {header_length, header}
        {cached, Map.put(headers, url, cached), 8 + header_length}
    end
  end

  defp fetch_expert_slice!(
         url,
         {header_length, header},
         reference,
         expert,
         fetch_range
       ) do
    metadata = Map.fetch!(header, reference.tensor)
    expected_shape = Tuple.to_list(reference.checkpoint_shape)

    unless metadata["shape"] == expected_shape do
      raise ArgumentError,
            "#{reference.tensor} shape #{inspect(metadata["shape"])} does not match #{inspect(expected_shape)}"
    end

    type = safetensors_type!(metadata["dtype"])
    [data_start, data_end] = metadata["data_offsets"]
    expert_count = hd(metadata["shape"])
    tensor_bytes = data_end - data_start

    unless rem(tensor_bytes, expert_count) == 0 do
      raise ArgumentError, "#{reference.tensor} cannot be sliced evenly by expert"
    end

    slice_bytes = div(tensor_bytes, expert_count)
    absolute_start = 8 + header_length + data_start + expert * slice_bytes
    absolute_end = absolute_start + slice_bytes - 1
    binary = fetch_exact!(fetch_range, url, absolute_start, absolute_end)

    tensor =
      binary
      |> Nx.from_binary(type, backend: Nx.BinaryBackend)
      |> Nx.reshape(reference.slice.shape)

    {tensor, byte_size(binary)}
  end

  defp split_gate_up!({gate_up, source}, descriptor) do
    size = descriptor.intermediate_size

    gate = Nx.slice_along_axis(gate_up, 0, size, axis: 0)
    up = Nx.slice_along_axis(gate_up, size, size, axis: 0)
    {{gate, up}, source}
  end

  defp fetch_exact!(fetch_range, url, first, last) do
    body = fetch_range.(url, first, last)
    expected = last - first + 1

    unless byte_size(body) == expected do
      raise ArgumentError,
            "range #{first}-#{last} returned #{byte_size(body)} bytes, expected #{expected}"
    end

    body
  end

  defp fetch_json!(url) do
    response = Req.get!(url, redirect: true, max_redirects: 5, receive_timeout: 60_000)

    unless response.status == 200 do
      raise ArgumentError, "#{url} returned HTTP #{response.status}"
    end

    if is_map(response.body), do: response.body, else: Jason.decode!(response.body)
  end

  defp range_get!(url, first, last) do
    response =
      Req.get!(url,
        headers: [{"range", "bytes=#{first}-#{last}"}],
        redirect: true,
        max_redirects: 5,
        receive_timeout: 60_000
      )

    unless response.status == 206 do
      raise ArgumentError, "#{url} returned HTTP #{response.status}, expected 206"
    end

    response.body
  end

  defp safetensors_type!("BF16"), do: :bf16
  defp safetensors_type!("F32"), do: :f32

  defp safetensors_type!(dtype),
    do: raise(ArgumentError, "unsupported expert tensor dtype #{inspect(dtype)}")

  defp safetensors_dtype!({@bf_type_atom, 16}), do: "BF16"
  defp safetensors_dtype!({:f, 32}), do: "F32"

  defp safetensors_dtype!(type),
    do: raise(ArgumentError, "unsupported expert tensor type #{inspect(type)}")

  defp nx_type!("BF16"), do: {@bf_type_atom, 16}
  defp nx_type!("F32"), do: {:f, 32}
  defp nx_type!({@bf_type_atom, 16} = type), do: type
  defp nx_type!({:f, 32} = type), do: type

  defp nx_type!(type),
    do: raise(ArgumentError, "unsupported expert manifest type #{inspect(type)}")

  defp load_tensor!(tensors, name, expected_shape, backend) do
    tensor =
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        Nx.to_tensor(Map.fetch!(tensors, name))
      end)

    unless Nx.shape(tensor) == expected_shape do
      raise ArgumentError,
            "#{name} shape #{inspect(Nx.shape(tensor))}, expected #{inspect(expected_shape)}"
    end

    transfer(tensor, backend)
  end

  defp transfer(tensor, nil), do: tensor
  defp transfer(tensor, Nx.BinaryBackend), do: Nx.backend_copy(tensor, Nx.BinaryBackend)
  defp transfer(tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp sha256_file(path) do
    path
    |> File.read!()
    |> then(&:crypto.hash(:sha256, &1))
  end
end
