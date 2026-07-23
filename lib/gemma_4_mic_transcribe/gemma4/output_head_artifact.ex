defmodule Gemma4MicTranscribe.Gemma4.OutputHeadArtifact do
  @moduledoc """
  Extracts Gemma 4's final RMS norm and tied token-embedding output matrix.

  Each tensor is copied as its own checkpoint byte range. This avoids retaining
  the unrelated decoder tensors that sit between the embedding and final norm
  in the source shard.
  """

  @version 1
  @kind :gemma4_output_head
  @manifest "manifest.etf"
  @parameters "parameters.bin"
  @default_repo "google/gemma-4-26B-A4B-it"
  @bf_type_atom :bf

  @doc "Range-extracts the final norm and tied output embedding."
  def extract!(path, opts \\ []) do
    path = Path.expand(path)

    if File.exists?(path) do
      raise ArgumentError, "artifact path already exists: #{path}"
    end

    repo = Keyword.get(opts, :repo, @default_repo)
    revision = Keyword.get(opts, :revision, "main")
    fetch_json = Keyword.get(opts, :fetch_json, &fetch_json!/1)
    fetch_range = Keyword.get(opts, :fetch_range)
    base_url = "https://huggingface.co/#{repo}/resolve/#{revision}"

    config = fetch_json.("#{base_url}/config.json")
    index = fetch_json.("#{base_url}/model.safetensors.index.json")
    text_config = config["text_config"] || config
    hidden_size = positive_integer!(text_config, "hidden_size")
    vocab_size = positive_integer!(text_config, "vocab_size")

    specs = %{
      "embedding" => %{
        source: "model.language_model.embed_tokens.weight",
        shape: [vocab_size, hidden_size]
      },
      "norm" => %{source: "model.language_model.norm.weight", shape: [hidden_size]}
    }

    specs =
      Map.new(specs, fn {name, spec} ->
        shard = get_in(index, ["weight_map", spec.source])

        unless is_binary(shard) do
          raise ArgumentError, "#{spec.source} is missing from the checkpoint index"
        end

        {name, Map.put(spec, :shard, shard)}
      end)

    headers =
      specs
      |> Map.values()
      |> Enum.map(& &1.shard)
      |> Enum.uniq()
      |> Map.new(fn shard ->
        url = "#{base_url}/#{shard}"
        {header_length, header} = fetch_header!(url, fetch_range)
        {shard, %{url: url, header_length: header_length, header: header}}
      end)

    {ranges, parameter_bytes} =
      specs
      |> Enum.sort_by(fn {name, _spec} -> name end)
      |> Enum.map_reduce(0, fn {name, spec}, parameter_offset ->
        shard = Map.fetch!(headers, spec.shard)
        metadata = Map.fetch!(shard.header, spec.source)
        validate_tensor!(spec, metadata)
        [source_start, source_end] = metadata["data_offsets"]
        absolute_start = 8 + shard.header_length + source_start
        absolute_end = 8 + shard.header_length + source_end - 1
        byte_size = source_end - source_start

        range = %{
          name: name,
          source: spec.source,
          shard: spec.shard,
          url: shard.url,
          shape: spec.shape,
          absolute_start: absolute_start,
          absolute_end: absolute_end,
          parameter_offset: parameter_offset,
          byte_size: byte_size
        }

        {range, parameter_offset + byte_size}
      end)

    temporary = path <> ".tmp-#{System.unique_integer([:positive])}"
    File.mkdir_p!(Path.dirname(path))
    File.mkdir_p!(temporary)

    try do
      parameters_path = Path.join(temporary, @parameters)

      copied =
        File.open!(parameters_path, [:write, :raw], fn file ->
          Enum.reduce(ranges, 0, fn range, copied ->
            copied +
              copy_range!(
                range.url,
                range.absolute_start,
                range.absolute_end,
                file,
                fetch_range
              )
          end)
        end)

      unless copied == parameter_bytes do
        raise ArgumentError,
              "checkpoint ranges wrote #{copied} bytes, expected #{parameter_bytes}"
      end

      tensors =
        Map.new(ranges, fn range ->
          {range.name,
           %{
             source: range.source,
             dtype: "BF16",
             shape: range.shape,
             offset: range.parameter_offset,
             byte_size: range.byte_size
           }}
        end)

      manifest = %{
        version: @version,
        kind: @kind,
        source_repo: repo,
        source_revision: revision,
        source_checkpoint_bytes: get_in(index, ["metadata", "total_size"]),
        source_ranges:
          Enum.map(ranges, fn range ->
            %{
              shard: range.shard,
              range: {range.absolute_start, range.absolute_end},
              parameter_offset: range.parameter_offset
            }
          end),
        downloaded_bytes:
          copied +
            (headers
             |> Map.values()
             |> Enum.map(fn header -> 8 + header.header_length end)
             |> Enum.sum()),
        parameter_file: @parameters,
        parameter_bytes: copied,
        parameter_sha256: sha256_file(parameters_path),
        parameter_count: vocab_size * hidden_size + hidden_size,
        parameter_type: "BF16",
        hidden_size: hidden_size,
        vocab_size: vocab_size,
        rms_norm_eps: number!(text_config, "rms_norm_eps"),
        final_logit_softcapping: number!(text_config, "final_logit_softcapping"),
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

  @doc "Loads the output-head matrices, validating them by default."
  def load!(path, backend \\ Nx.BinaryBackend, opts \\ []) do
    path = Path.expand(path)
    manifest = read_manifest!(path)

    unless manifest.version == @version and manifest.kind == @kind do
      raise ArgumentError, "unsupported Gemma 4 output-head artifact"
    end

    parameters_path = Path.join(path, manifest.parameter_file)

    if Keyword.get(opts, :verify_checksum, true) and
         sha256_file(parameters_path) != manifest.parameter_sha256 do
      raise ArgumentError, "output-head parameter checksum mismatch"
    end

    params =
      Map.new(manifest.tensors, fn {name, metadata} ->
        {parameter_name!(name), load_tensor!(parameters_path, metadata, backend)}
      end)

    {manifest, params}
  end

  @doc "Reads the output-head manifest without loading its matrices."
  def read_manifest!(path) do
    path
    |> Path.expand()
    |> Path.join(@manifest)
    |> File.read!()
    |> :erlang.binary_to_term([:safe])
    |> normalize_manifest()
  end

  defp validate_tensor!(spec, metadata) do
    unless metadata["dtype"] == "BF16" do
      raise ArgumentError,
            "#{spec.source} has dtype #{inspect(metadata["dtype"])}, expected BF16"
    end

    unless metadata["shape"] == spec.shape do
      raise ArgumentError,
            "#{spec.source} has shape #{inspect(metadata["shape"])}, expected #{inspect(spec.shape)}"
    end
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
          if response.status == 206, do: :ok = :file.write(file, data)
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

  defp parameter_name!("embedding"), do: :embedding
  defp parameter_name!("norm"), do: :norm

  defp normalize_manifest(manifest) do
    Map.put(manifest, :parameter_type, nx_type!(manifest.parameter_type))
  end

  defp nx_type!("BF16"), do: {@bf_type_atom, 16}
  defp nx_type!({@bf_type_atom, 16} = type), do: type

  defp nx_type!(type),
    do: raise(ArgumentError, "unsupported output-head manifest type #{inspect(type)}")

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
