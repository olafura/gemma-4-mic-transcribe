defmodule Gemma4MicTranscribe.Gemma4.ExpertCallerArtifact do
  @moduledoc """
  Extracts the layer-0 attention prefix that produces routed-expert inputs.

  Token embeddings are fetched separately by row. The companion MoE artifact
  supplies the router and `pre_feedforward_layernorm_2` weight.
  """

  @version 1
  @manifest "manifest.etf"
  @parameters "parameters.safetensors"
  @default_repo "google/gemma-4-26B-A4B-it"

  @doc "Extracts the layer-0 attention tensors needed by an expert caller."
  def extract!(path, opts \\ []) do
    path = Path.expand(path)

    if File.exists?(path) do
      raise ArgumentError, "artifact path already exists: #{path}"
    end

    repo = Keyword.get(opts, :repo, @default_repo)
    revision = Keyword.get(opts, :revision, "main")
    layer = Keyword.get(opts, :layer, 0)

    unless layer == 0 do
      raise ArgumentError, "expert caller extraction currently supports layer 0 only"
    end

    fetch_json = Keyword.get(opts, :fetch_json, &fetch_json!/1)
    fetch_range = Keyword.get(opts, :fetch_range, &range_get!/3)
    base_url = "https://huggingface.co/#{repo}/resolve/#{revision}"
    config = fetch_json.("#{base_url}/config.json")
    index = fetch_json.("#{base_url}/model.safetensors.index.json")
    text_config = config["text_config"] || config
    specs = tensor_specs!(text_config, layer)

    shards =
      specs
      |> Map.values()
      |> Enum.map(&get_in(index, ["weight_map", &1.source]))
      |> Enum.uniq()

    unless match?([shard] when is_binary(shard), shards) do
      raise ArgumentError, "layer-0 attention tensors must occupy one checkpoint shard"
    end

    [shard] = shards
    shard_url = "#{base_url}/#{shard}"
    {header_length, header} = fetch_header!(shard_url, fetch_range)
    data_base = 8 + header_length

    tensors =
      Map.new(specs, fn {name, spec} ->
        metadata = Map.fetch!(header, spec.source)

        unless metadata["dtype"] == "BF16" and metadata["shape"] == spec.shape do
          raise ArgumentError, "unexpected metadata for #{spec.source}"
        end

        [start_offset, end_offset] = metadata["data_offsets"]

        binary =
          fetch_exact!(
            shard_url,
            data_base + start_offset,
            data_base + end_offset - 1,
            fetch_range
          )

        tensor =
          binary
          |> Nx.from_binary(:bf16, backend: Nx.BinaryBackend)
          |> Nx.reshape(List.to_tuple(spec.shape))

        {name, tensor}
      end)

    temporary = path <> ".tmp-#{System.unique_integer([:positive])}"
    File.mkdir_p!(Path.dirname(path))
    File.mkdir_p!(temporary)

    try do
      parameters_path = Path.join(temporary, @parameters)
      Safetensors.write!(parameters_path, tensors)

      manifest = %{
        version: @version,
        kind: :gemma4_expert_caller,
        source_repo: repo,
        source_revision: revision,
        source_shard: shard,
        source_checkpoint_bytes: get_in(index, ["metadata", "total_size"]),
        layer_index: layer,
        hidden_size: positive_integer!(text_config, "hidden_size"),
        num_attention_heads: positive_integer!(text_config, "num_attention_heads"),
        num_key_value_heads: positive_integer!(text_config, "num_key_value_heads"),
        head_dim: positive_integer!(text_config, "head_dim"),
        rms_norm_eps: number!(text_config, "rms_norm_eps"),
        sliding_window: positive_integer!(text_config, "sliding_window"),
        rope_theta:
          text_config
          |> get_in(["rope_parameters", "sliding_attention", "rope_theta"])
          |> number_value!("sliding-attention rope_theta"),
        parameter_file: @parameters,
        parameter_bytes: File.stat!(parameters_path).size,
        parameter_count:
          tensors
          |> Map.values()
          |> Enum.map(&Nx.size/1)
          |> Enum.sum(),
        parameter_sha256: sha256_file(parameters_path)
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

  @doc "Loads an extracted attention prefix onto an Nx backend."
  def load!(path, backend \\ Nx.BinaryBackend) do
    path = Path.expand(path)
    manifest = read_manifest!(path)
    parameters_path = Path.join(path, manifest.parameter_file)

    unless manifest.version == @version and manifest.kind == :gemma4_expert_caller do
      raise ArgumentError, "unsupported expert caller artifact"
    end

    unless sha256_file(parameters_path) == manifest.parameter_sha256 do
      raise ArgumentError, "expert caller parameter checksum mismatch"
    end

    tensors = Safetensors.read!(parameters_path, lazy: true)

    params =
      Map.new(tensors, fn {name, tensor} ->
        tensor =
          Nx.with_default_backend(Nx.BinaryBackend, fn ->
            Nx.to_tensor(tensor)
          end)

        {parameter_name!(name), transfer(tensor, backend)}
      end)

    {manifest, params}
  end

  @doc "Reads the caller manifest without loading its tensors."
  def read_manifest!(path) do
    path
    |> Path.expand()
    |> Path.join(@manifest)
    |> File.read!()
    |> :erlang.binary_to_term([:safe])
  end

  defp tensor_specs!(config, layer) do
    hidden = positive_integer!(config, "hidden_size")
    heads = positive_integer!(config, "num_attention_heads")
    kv_heads = positive_integer!(config, "num_key_value_heads")
    head_dim = positive_integer!(config, "head_dim")
    prefix = "model.language_model.layers.#{layer}"

    %{
      "input_norm" => spec("#{prefix}.input_layernorm.weight", [hidden]),
      "post_attention_norm" => spec("#{prefix}.post_attention_layernorm.weight", [hidden]),
      "query" => spec("#{prefix}.self_attn.q_proj.weight", [heads * head_dim, hidden]),
      "key" => spec("#{prefix}.self_attn.k_proj.weight", [kv_heads * head_dim, hidden]),
      "value" => spec("#{prefix}.self_attn.v_proj.weight", [kv_heads * head_dim, hidden]),
      "output" => spec("#{prefix}.self_attn.o_proj.weight", [hidden, heads * head_dim]),
      "query_norm" => spec("#{prefix}.self_attn.q_norm.weight", [head_dim]),
      "key_norm" => spec("#{prefix}.self_attn.k_norm.weight", [head_dim])
    }
  end

  defp spec(source, shape), do: %{source: source, shape: shape}

  defp fetch_header!(url, fetch_range) do
    <<header_length::little-unsigned-64>> = fetch_exact!(url, 0, 7, fetch_range)
    header = fetch_exact!(url, 8, 7 + header_length, fetch_range) |> Jason.decode!()
    {header_length, header}
  end

  defp fetch_exact!(url, first, last, fetch_range) do
    body = fetch_range.(url, first, last)
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
        redirect_log_level: false,
        max_redirects: 5,
        receive_timeout: 120_000
      )

    unless response.status == 206 do
      raise ArgumentError, "#{url} returned HTTP #{response.status}, expected 206"
    end

    response.body
  end

  defp fetch_json!(url) do
    response =
      Req.get!(url,
        redirect: true,
        redirect_log_level: false,
        max_redirects: 5,
        receive_timeout: 60_000
      )

    unless response.status == 200 do
      raise ArgumentError, "#{url} returned HTTP #{response.status}"
    end

    if is_map(response.body), do: response.body, else: Jason.decode!(response.body)
  end

  defp positive_integer!(map, key) do
    case map[key] do
      value when is_integer(value) and value > 0 -> value
      value -> raise ArgumentError, "expected positive #{key}, got: #{inspect(value)}"
    end
  end

  defp number!(map, key), do: number_value!(map[key], key)

  defp number_value!(value, _name) when is_number(value), do: value

  defp number_value!(value, name),
    do: raise(ArgumentError, "expected numeric #{name}, got: #{inspect(value)}")

  defp parameter_name!("input_norm"), do: :input_norm
  defp parameter_name!("post_attention_norm"), do: :post_attention_norm
  defp parameter_name!("query"), do: :query
  defp parameter_name!("key"), do: :key
  defp parameter_name!("value"), do: :value
  defp parameter_name!("output"), do: :output
  defp parameter_name!("query_norm"), do: :query_norm
  defp parameter_name!("key_norm"), do: :key_norm

  defp parameter_name!(name),
    do: raise(ArgumentError, "unsupported expert caller parameter #{inspect(name)}")

  defp transfer(tensor, Nx.BinaryBackend), do: Nx.backend_copy(tensor, Nx.BinaryBackend)
  defp transfer(tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp sha256_file(path) do
    path
    |> File.stream!(1_048_576, [])
    |> Enum.reduce(:crypto.hash_init(:sha256), &:crypto.hash_update(&2, &1))
    |> :crypto.hash_final()
  end
end
