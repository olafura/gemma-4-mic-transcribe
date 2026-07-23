defmodule Gemma4MicTranscribe.Gemma4.MoeLayerArtifact do
  @moduledoc """
  Extracts one complete Gemma 4 MoE feed-forward layer as a byte-range artifact.

  The artifact contains the shared FFN, all routed expert banks, router, five
  feed-forward norms, and layer scalar. Attention is intentionally excluded.
  Source tensors are kept in their original contiguous shard representation,
  avoiding a multi-gigabyte decode/re-encode cycle during extraction.
  """

  @version 1
  @kind :gemma4_moe_feedforward_layer
  @manifest "manifest.etf"
  @parameters "parameters.bin"
  @default_repo "google/gemma-4-26B-A4B-it"
  @bf_type_atom :bf

  @doc "Range-extracts a complete MoE feed-forward shell for one decoder layer."
  def extract!(path, opts \\ []) do
    path = Path.expand(path)

    if File.exists?(path) do
      raise ArgumentError, "artifact path already exists: #{path}"
    end

    repo = Keyword.get(opts, :repo, @default_repo)
    revision = Keyword.get(opts, :revision, "main")
    layer = Keyword.get(opts, :layer, 0)
    fetch_json = Keyword.get(opts, :fetch_json, &fetch_json!/1)
    fetch_range = Keyword.get(opts, :fetch_range)
    base_url = "https://huggingface.co/#{repo}/resolve/#{revision}"

    config = fetch_json.("#{base_url}/config.json")
    index = fetch_json.("#{base_url}/model.safetensors.index.json")
    text_config = config["text_config"] || config
    specs = tensor_specs!(text_config, layer)

    shards =
      specs
      |> Map.values()
      |> Enum.map(fn spec -> get_in(index, ["weight_map", spec.source]) end)
      |> Enum.uniq()

    unless match?([shard] when is_binary(shard), shards) do
      raise ArgumentError,
            "MoE layer #{layer} tensors must be present in one checkpoint shard, got: #{inspect(shards)}"
    end

    [shard] = shards
    shard_url = "#{base_url}/#{shard}"
    {header_length, header} = fetch_header!(shard_url, fetch_range)
    tensors = validate_tensors!(specs, header)

    source_start =
      tensors
      |> Map.values()
      |> Enum.map(& &1.source_start)
      |> Enum.min()

    source_end =
      tensors
      |> Map.values()
      |> Enum.map(& &1.source_end)
      |> Enum.max()

    data_base = 8 + header_length
    absolute_start = data_base + source_start
    absolute_end = data_base + source_end - 1
    parameter_bytes = absolute_end - absolute_start + 1

    tensors =
      Map.new(tensors, fn {name, metadata} ->
        {name,
         metadata
         |> Map.put(:offset, metadata.source_start - source_start)
         |> Map.put(:byte_size, metadata.source_end - metadata.source_start)
         |> Map.drop([:source_start, :source_end])}
      end)

    temporary = path <> ".tmp-#{System.unique_integer([:positive])}"
    File.mkdir_p!(Path.dirname(path))
    File.mkdir_p!(temporary)

    try do
      parameters_path = Path.join(temporary, @parameters)

      copied =
        File.open!(parameters_path, [:write, :raw], fn file ->
          copy_range!(shard_url, absolute_start, absolute_end, file, fetch_range)
        end)

      unless copied == parameter_bytes do
        raise ArgumentError,
              "checkpoint range wrote #{copied} bytes, expected #{parameter_bytes}"
      end

      manifest = %{
        version: @version,
        kind: @kind,
        source_repo: repo,
        source_revision: revision,
        source_shard: shard,
        source_checkpoint_bytes: get_in(index, ["metadata", "total_size"]),
        source_range: {absolute_start, absolute_end},
        downloaded_bytes: 8 + header_length + copied,
        parameter_file: @parameters,
        parameter_bytes: copied,
        parameter_sha256: sha256_file(parameters_path),
        parameter_count: parameter_count(tensors),
        parameter_type: "BF16",
        layer_index: layer,
        hidden_size: positive_integer!(text_config, "hidden_size"),
        shared_intermediate_size: positive_integer!(text_config, "intermediate_size"),
        expert_intermediate_size: positive_integer!(text_config, "moe_intermediate_size"),
        num_experts: positive_integer!(text_config, "num_experts"),
        top_k_experts: positive_integer!(text_config, "top_k_experts"),
        rms_norm_eps: number!(text_config, "rms_norm_eps"),
        activation: text_config["hidden_activation"],
        tensors: tensors
      }

      File.write!(Path.join(temporary, @manifest), :erlang.term_to_binary(manifest))
      File.rename!(temporary, path)
      normalize_manifest(manifest)
    rescue
      exception ->
        File.rm_rf(temporary)
        reraise exception, __STACKTRACE__
    end
  end

  @doc "Loads the complete MoE shell onto the requested Nx backend."
  def load!(path, backend \\ Nx.BinaryBackend) do
    path = Path.expand(path)
    manifest = read_manifest!(path)

    unless manifest.version == @version and manifest.kind == @kind do
      raise ArgumentError, "unsupported Gemma 4 MoE layer artifact"
    end

    parameters_path = Path.join(path, manifest.parameter_file)

    unless sha256_file(parameters_path) == manifest.parameter_sha256 do
      raise ArgumentError, "MoE layer parameter checksum mismatch"
    end

    params =
      Map.new(manifest.tensors, fn {name, metadata} ->
        {String.to_existing_atom(name), load_tensor!(parameters_path, metadata, backend)}
      end)

    {manifest, params}
  end

  @doc "Reads the MoE artifact manifest without loading its matrices."
  def read_manifest!(path) do
    path
    |> Path.expand()
    |> Path.join(@manifest)
    |> File.read!()
    |> :erlang.binary_to_term([:safe])
    |> normalize_manifest()
  end

  defp tensor_specs!(config, layer) do
    num_layers = positive_integer!(config, "num_hidden_layers")

    unless is_integer(layer) and layer >= 0 and layer < num_layers do
      raise ArgumentError, "expected layer in 0..#{num_layers - 1}, got: #{inspect(layer)}"
    end

    hidden = positive_integer!(config, "hidden_size")
    shared = positive_integer!(config, "intermediate_size")
    expert = positive_integer!(config, "moe_intermediate_size")
    num_experts = positive_integer!(config, "num_experts")
    prefix = "model.language_model.layers.#{layer}"

    %{
      "experts_gate_up" =>
        spec("#{prefix}.experts.gate_up_proj", [num_experts, 2 * expert, hidden]),
      "experts_down" => spec("#{prefix}.experts.down_proj", [num_experts, hidden, expert]),
      "shared_gate" => spec("#{prefix}.mlp.gate_proj.weight", [shared, hidden]),
      "shared_up" => spec("#{prefix}.mlp.up_proj.weight", [shared, hidden]),
      "shared_down" => spec("#{prefix}.mlp.down_proj.weight", [hidden, shared]),
      "router_proj" => spec("#{prefix}.router.proj.weight", [num_experts, hidden]),
      "router_scale" => spec("#{prefix}.router.scale", [hidden]),
      "router_per_expert_scale" => spec("#{prefix}.router.per_expert_scale", [num_experts]),
      "norm_pre_shared" => spec("#{prefix}.pre_feedforward_layernorm.weight", [hidden]),
      "norm_post_shared" => spec("#{prefix}.post_feedforward_layernorm_1.weight", [hidden]),
      "norm_pre_experts" => spec("#{prefix}.pre_feedforward_layernorm_2.weight", [hidden]),
      "norm_post_experts" => spec("#{prefix}.post_feedforward_layernorm_2.weight", [hidden]),
      "norm_post_combined" => spec("#{prefix}.post_feedforward_layernorm.weight", [hidden]),
      "layer_scalar" => spec("#{prefix}.layer_scalar", [1])
    }
  end

  defp spec(source, shape), do: %{source: source, shape: shape}

  defp validate_tensors!(specs, header) do
    Map.new(specs, fn {name, spec} ->
      metadata = Map.fetch!(header, spec.source)

      unless metadata["dtype"] == "BF16" do
        raise ArgumentError,
              "#{spec.source} has dtype #{inspect(metadata["dtype"])}, expected BF16"
      end

      unless metadata["shape"] == spec.shape do
        raise ArgumentError,
              "#{spec.source} has shape #{inspect(metadata["shape"])}, expected #{inspect(spec.shape)}"
      end

      [source_start, source_end] = metadata["data_offsets"]

      {name,
       %{
         source: spec.source,
         dtype: "BF16",
         shape: spec.shape,
         source_start: source_start,
         source_end: source_end
       }}
    end)
  end

  defp fetch_header!(url, fetch_range) do
    <<header_length::little-unsigned-64>> = fetch_exact!(url, 0, 7, fetch_range)
    header = fetch_exact!(url, 8, 7 + header_length, fetch_range) |> Jason.decode!()
    {header_length, header}
  end

  defp fetch_exact!(url, first, last, nil), do: range_get!(url, first, last)

  defp fetch_exact!(url, first, last, fetch_range) do
    body = fetch_range.(url, first, last)
    expected = last - first + 1

    unless byte_size(body) == expected do
      raise ArgumentError,
            "range #{first}-#{last} returned #{byte_size(body)} bytes, expected #{expected}"
    end

    body
  end

  defp copy_range!(url, first, last, file, nil) do
    {:ok, start_position} = :file.position(file, :cur)

    response =
      Req.get!(url,
        headers: [{"range", "bytes=#{first}-#{last}"}],
        redirect: true,
        max_redirects: 5,
        receive_timeout: 300_000,
        into: fn {:data, data}, {request, response} ->
          if response.status == 206 do
            :ok = :file.write(file, data)
          end

          {:cont, {request, response}}
        end
      )

    unless response.status == 206 do
      raise ArgumentError, "#{url} returned HTTP #{response.status}, expected 206"
    end

    {:ok, end_position} = :file.position(file, :cur)
    end_position - start_position
  end

  defp copy_range!(url, first, last, file, fetch_range) do
    body = fetch_exact!(url, first, last, fetch_range)
    :ok = :file.write(file, body)
    byte_size(body)
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

  defp fetch_json!(url) do
    response = Req.get!(url, redirect: true, max_redirects: 5, receive_timeout: 60_000)

    unless response.status == 200 do
      raise ArgumentError, "#{url} returned HTTP #{response.status}"
    end

    if is_map(response.body), do: response.body, else: Jason.decode!(response.body)
  end

  defp load_tensor!(path, metadata, backend) do
    {:ok, file} = :file.open(String.to_charlist(path), [:read, :raw, :binary])

    binary =
      try do
        {:ok, binary} = :file.pread(file, metadata.offset, metadata.byte_size)
        binary
      after
        :ok = :file.close(file)
      end

    tensor =
      binary
      |> Nx.from_binary(:bf16, backend: Nx.BinaryBackend)
      |> Nx.reshape(List.to_tuple(metadata.shape))

    transfer(tensor, backend)
  end

  defp transfer(tensor, nil), do: tensor
  defp transfer(tensor, Nx.BinaryBackend), do: Nx.backend_copy(tensor, Nx.BinaryBackend)
  defp transfer(tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp parameter_count(tensors) do
    tensors
    |> Map.values()
    |> Enum.map(fn metadata -> Enum.product(metadata.shape) end)
    |> Enum.sum()
  end

  defp normalize_manifest(manifest) do
    Map.put(manifest, :parameter_type, nx_type!(manifest.parameter_type))
  end

  defp nx_type!("BF16"), do: {@bf_type_atom, 16}
  defp nx_type!({@bf_type_atom, 16} = type), do: type

  defp nx_type!(type),
    do: raise(ArgumentError, "unsupported MoE layer manifest type #{inspect(type)}")

  defp positive_integer!(map, key) do
    case map[key] do
      value when is_integer(value) and value > 0 -> value
      value -> raise ArgumentError, "expected positive #{key}, got: #{inspect(value)}"
    end
  end

  defp number!(map, key) do
    case map[key] do
      value when is_number(value) -> value
      value -> raise ArgumentError, "expected numeric #{key}, got: #{inspect(value)}"
    end
  end

  defp sha256_file(path) do
    path
    |> File.stream!(1_048_576, [])
    |> Enum.reduce(:crypto.hash_init(:sha256), &:crypto.hash_update(&2, &1))
    |> :crypto.hash_final()
  end
end
