defmodule Gemma4MicTranscribe.HandoffProbe.Artifact do
  @moduledoc """
  Extracts and loads the 11-tensor Cactus E2B handoff probe artifact.

  Extraction uses HTTP byte ranges. It reads the safetensors header and only
  downloads the selected `handoff_probe.*` data region from the multi-gigabyte
  source shard.
  """

  @version 1
  @kind :gemma4_e2b_handoff_probe
  @manifest "manifest.etf"
  @parameters "parameters.safetensors"
  @prefix "handoff_probe."
  @default_repo "Cactus-Compute/gemma-4-e2b-it-hybrid"
  @default_shard "model-00003-of-00003.safetensors"

  @parameter_keys %{
    "handoff_probe.attn_query" => :attn_query,
    "handoff_probe.head.0.bias" => :head_0_bias,
    "handoff_probe.head.0.weight" => :head_0_weight,
    "handoff_probe.head.2.bias" => :head_2_bias,
    "handoff_probe.head.2.weight" => :head_2_weight,
    "handoff_probe.head.4.bias" => :head_4_bias,
    "handoff_probe.head.4.weight" => :head_4_weight,
    "handoff_probe.norm.bias" => :norm_bias,
    "handoff_probe.norm.weight" => :norm_weight,
    "handoff_probe.proj.bias" => :proj_bias,
    "handoff_probe.proj.weight" => :proj_weight
  }

  @expected_shapes %{
    attn_query: {32},
    head_0_bias: {128},
    head_0_weight: {128, 32},
    head_2_bias: {64},
    head_2_weight: {64, 128},
    head_4_bias: {1},
    head_4_weight: {1, 64},
    norm_bias: {1536},
    norm_weight: {1536},
    proj_bias: {32},
    proj_weight: {32, 1536}
  }

  @doc "Extracts the probe tensors into a new standalone artifact directory."
  def extract!(path, opts \\ []) do
    path = Path.expand(path)

    if File.exists?(path) do
      raise ArgumentError, "artifact path already exists: #{path}"
    end

    repo = Keyword.get(opts, :repo, @default_repo)
    revision = Keyword.get(opts, :revision, "main")
    shard = Keyword.get(opts, :shard, @default_shard)

    url =
      Keyword.get(
        opts,
        :url,
        "https://huggingface.co/#{repo}/resolve/#{revision}/#{shard}"
      )

    fetch_range = Keyword.get(opts, :fetch_range, &range_get!(url, &1, &2))
    {tensors, downloaded_bytes, source_bytes} = extract_tensors!(fetch_range)

    File.mkdir_p!(Path.dirname(path))
    temporary = path <> ".tmp-#{System.unique_integer([:positive])}"
    File.mkdir_p!(temporary)

    try do
      parameters_path = Path.join(temporary, @parameters)
      Safetensors.write!(parameters_path, tensors)

      manifest = %{
        version: @version,
        kind: @kind,
        source_repo: repo,
        source_revision: revision,
        source_shard: shard,
        source_url: url,
        source_bytes: source_bytes,
        downloaded_bytes: downloaded_bytes,
        parameter_file: @parameters,
        parameter_count: tensors |> Map.values() |> Enum.map(&Nx.size/1) |> Enum.sum(),
        parameter_sha256: sha256_file(parameters_path),
        probe_layer: 28,
        feature_size: 1536,
        projection_size: 32,
        max_tokens: 1024
      }

      File.write!(Path.join(temporary, @manifest), :erlang.term_to_binary(manifest))
      File.rename!(temporary, path)
      manifest
    rescue
      exception ->
        File.rm_rf(temporary)
        reraise exception, __STACKTRACE__
    end
  end

  @doc "Loads and validates an extracted probe artifact."
  def load!(path, backend) do
    path = Path.expand(path)
    manifest = read_manifest!(path)

    unless manifest.version == @version and manifest.kind == @kind do
      raise ArgumentError, "unsupported handoff probe artifact"
    end

    parameters_path = Path.join(path, manifest.parameter_file)

    unless sha256_file(parameters_path) == manifest.parameter_sha256 do
      raise ArgumentError, "handoff probe parameter checksum mismatch"
    end

    tensors = Safetensors.read!(parameters_path, lazy: true)

    params =
      Map.new(@parameter_keys, fn {tensor_name, parameter_name} ->
        tensor = load_tensor!(tensors, tensor_name, backend)
        expected_shape = Map.fetch!(@expected_shapes, parameter_name)

        unless Nx.shape(tensor) == expected_shape do
          raise ArgumentError,
                "invalid #{tensor_name} shape #{inspect(Nx.shape(tensor))}, expected #{inspect(expected_shape)}"
        end

        {parameter_name, tensor}
      end)

    {manifest, params}
  end

  @doc "Reads only the artifact metadata."
  def read_manifest!(path) do
    path
    |> Path.expand()
    |> Path.join(@manifest)
    |> File.read!()
    |> :erlang.binary_to_term([:safe])
  end

  defp extract_tensors!(fetch_range) do
    <<header_length::little-unsigned-64>> = fetch_exact!(fetch_range, 0, 7)
    header_end = 8 + header_length - 1
    header_binary = fetch_exact!(fetch_range, 8, header_end)
    header = Jason.decode!(header_binary)

    entries =
      header
      |> Enum.filter(fn {name, _metadata} -> String.starts_with?(name, @prefix) end)
      |> Map.new()

    missing = Map.keys(@parameter_keys) -- Map.keys(entries)

    if missing != [] do
      raise ArgumentError,
            "source checkpoint is missing probe tensors: #{Enum.join(missing, ", ")}"
    end

    offsets = Enum.flat_map(entries, fn {_name, metadata} -> metadata["data_offsets"] end)
    data_start = Enum.min(offsets)
    data_end = Enum.max(offsets)
    absolute_start = 8 + header_length + data_start
    absolute_end = 8 + header_length + data_end - 1
    data = fetch_exact!(fetch_range, absolute_start, absolute_end)

    tensors =
      Map.new(@parameter_keys, fn {name, _parameter_name} ->
        metadata = Map.fetch!(entries, name)

        unless metadata["dtype"] == "F32" do
          raise ArgumentError, "unsupported #{name} dtype #{inspect(metadata["dtype"])}"
        end

        [first, last] = metadata["data_offsets"]
        binary = binary_part(data, first - data_start, last - first)

        tensor =
          binary
          |> Nx.from_binary(:f32)
          |> Nx.reshape(List.to_tuple(metadata["shape"]))

        expected_shape = @expected_shapes[Map.fetch!(@parameter_keys, name)]

        unless Nx.shape(tensor) == expected_shape do
          raise ArgumentError,
                "invalid #{name} shape #{inspect(Nx.shape(tensor))}, expected #{inspect(expected_shape)}"
        end

        {name, tensor}
      end)

    downloaded_bytes = 8 + header_length + byte_size(data)
    source_bytes = absolute_end + 1
    {tensors, downloaded_bytes, source_bytes}
  end

  defp fetch_exact!(fetch_range, first, last) do
    body = fetch_range.(first, last)
    expected = last - first + 1

    unless byte_size(body) == expected do
      raise ArgumentError,
            "range #{first}-#{last} returned #{byte_size(body)} bytes, expected #{expected}"
    end

    body
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
      raise ArgumentError, "source server returned HTTP #{response.status}, expected 206"
    end

    response.body
  end

  defp load_tensor!(tensors, name, backend) do
    tensor =
      Nx.with_default_backend(Nx.BinaryBackend, fn -> Nx.to_tensor(Map.fetch!(tensors, name)) end)

    case backend do
      nil -> tensor
      Nx.BinaryBackend -> tensor
      backend -> Nx.backend_transfer(tensor, backend)
    end
  end

  defp sha256_file(path) do
    path
    |> File.read!()
    |> then(&:crypto.hash(:sha256, &1))
  end
end
