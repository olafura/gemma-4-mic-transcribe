defmodule Gemma4MicTranscribe.Gemma4.DecoderBlockArtifact do
  @moduledoc """
  Saves and loads one standalone Gemma 4 decoder block.

  The artifact contains only the selected block's parameter tensors and the
  model metadata needed to rebuild that block's Axon graph. It does not contain
  embeddings, any other decoder layer, the output norm, or the vocabulary head.
  """

  alias Gemma4MicTranscribe.Gemma4.DecoderBlocks
  alias Gemma4MicTranscribe.Gemma4.DecoderBlocks.Extracted
  alias Gemma4MicTranscribe.Gemma4.DecoderBlocks.Tail
  alias Gemma4MicTranscribe.Gemma4.DecoderPipeline
  alias Gemma4MicTranscribe.Gemma4.DecoderPipeline.Prefix
  alias Gemma4MicTranscribe.Gemma4Unified.Model
  alias Bumblebee.HuggingFace.Hub

  @version 1
  @manifest "manifest.etf"
  @parameters "parameters.safetensors"
  @verification "verification.safetensors"
  @tokenizer_files ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]

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

    if manifest.kind != :decoder_block do
      raise ArgumentError, "artifact is not a decoder block"
    end

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

  def save_tail!(%Tail{} = tail, path, opts \\ []) do
    path = Path.expand(path)

    if File.exists?(path) do
      raise ArgumentError, "artifact path already exists: #{path}"
    end

    File.mkdir_p!(Path.dirname(path))
    temporary = path <> ".tmp-#{System.unique_integer([:positive])}"
    File.mkdir_p!(temporary)

    try do
      {tensors, parameter_paths} = flatten_parameters(tail.params.data)
      Safetensors.write!(Path.join(temporary, @parameters), tensors)

      sequence_length = Keyword.get(opts, :verification_sequence_length, 8)
      verification = build_verification(tail, sequence_length)
      Safetensors.write!(Path.join(temporary, @verification), verification)
      maybe_copy_tokenizer!(temporary, Keyword.get(opts, :tokenizer_repository))

      manifest = %{
        version: @version,
        kind: :decoder_tail,
        id: tail.id,
        layer_indices: tail.layer_indices,
        layer_types: tail.layer_types,
        input_size: tail.input_size,
        vocab_size: tail.vocab_size,
        parameter_count: tail.parameter_count,
        spec: tail.spec,
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

  def load_tail!(path, backend) do
    path = Path.expand(path)
    manifest = read_manifest!(path)

    if manifest.kind != :decoder_tail do
      raise ArgumentError, "artifact is not a decoder tail"
    end

    tensors = Safetensors.read!(Path.join(path, @parameters), lazy: true)

    params =
      manifest.parameter_paths
      |> Enum.map(fn {tensor_name, [node_name, parameter_name]} ->
        {node_name, parameter_name, load_tensor!(tensors, tensor_name, backend)}
      end)
      |> Enum.group_by(&elem(&1, 0), &{elem(&1, 1), elem(&1, 2)})
      |> Map.new(fn {node_name, parameters} -> {node_name, Map.new(parameters)} end)
      |> Axon.ModelState.new()

    model = Model.decoder_tail_model(manifest.spec, manifest.layer_indices)
    {_init_fun, predict_fun} = Axon.build(model, build_opts(backend))

    tokenizer =
      if File.dir?(Path.join(path, "tokenizer")) do
        {:ok, tokenizer} =
          Bumblebee.load_tokenizer({:local, Path.join(path, "tokenizer")}, type: :gemma)

        tokenizer
      end

    %Tail{
      id: manifest.id,
      layer_indices: manifest.layer_indices,
      layer_types: manifest.layer_types,
      input_size: manifest.input_size,
      vocab_size: manifest.vocab_size,
      parameter_count: manifest.parameter_count,
      spec: manifest.spec,
      model: model,
      params: params,
      predict_fun: predict_fun,
      backend: backend,
      tokenizer: tokenizer
    }
  end

  def save_prefix!(%DecoderPipeline{} = pipeline, path, opts \\ []) do
    path = Path.expand(path)

    if File.exists?(path) do
      raise ArgumentError, "artifact path already exists: #{path}"
    end

    File.mkdir_p!(Path.dirname(path))
    temporary = path <> ".tmp-#{System.unique_integer([:positive])}"
    File.mkdir_p!(temporary)

    try do
      {tensors, parameter_paths} = flatten_parameters(pipeline.prefix.params.data)

      tensors =
        tensors
        |> Map.put("g0", pipeline.generation.suppression_mask)
        |> Map.put("g1", pipeline.generation.inside_channel_suppression_mask)
        |> Map.put("g2", pipeline.generation.content_suppression_mask)

      Safetensors.write!(Path.join(temporary, @parameters), tensors)
      maybe_copy_tokenizer!(temporary, Keyword.get(opts, :tokenizer_repository))

      manifest = %{
        version: @version,
        kind: :decoder_prefix,
        id: "language_model.prefix.0-#{pipeline.prefix.last_layer}",
        last_layer: pipeline.prefix.last_layer,
        parameter_count: pipeline.prefix.parameter_count,
        spec: pipeline.generation.spec,
        parameter_paths: parameter_paths,
        generation: %{
          channel_token_ids: pipeline.generation.channel_token_ids,
          eos_token_ids: pipeline.generation.eos_token_ids,
          no_repeat_ngram_size: pipeline.generation.no_repeat_ngram_size,
          suppression_mask: "g0",
          inside_channel_suppression_mask: "g1",
          content_suppression_mask: "g2"
        }
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

  def load_prefix!(path, backend) do
    path = Path.expand(path)
    manifest = read_manifest!(path)

    if manifest.kind != :decoder_prefix do
      raise ArgumentError, "artifact is not a decoder prefix"
    end

    tensors = Safetensors.read!(Path.join(path, @parameters), lazy: true)
    params = load_parameters!(tensors, manifest.parameter_paths, backend)
    model = Model.decoder_prefix_model(manifest.spec, manifest.last_layer)
    {_init_fun, predict_fun} = Axon.build(model, build_opts(backend))
    cached_model = Model.cached_decoder_prefix_model(manifest.spec, manifest.last_layer)
    {_init_fun, cached_predict_fun} = Axon.build(cached_model, build_opts(backend))

    tokenizer = maybe_load_tokenizer(path)
    generation = manifest.generation

    %{
      manifest: manifest,
      prefix: %Prefix{
        last_layer: manifest.last_layer,
        parameter_count: manifest.parameter_count,
        model: model,
        params: params,
        predict_fun: predict_fun,
        backend: backend
      },
      cached_model: cached_model,
      cached_predict_fun: cached_predict_fun,
      tokenizer: tokenizer,
      generation: %{
        spec: manifest.spec,
        suppression_mask: load_tensor!(tensors, generation.suppression_mask, backend),
        inside_channel_suppression_mask:
          load_tensor!(tensors, generation.inside_channel_suppression_mask, backend),
        content_suppression_mask:
          load_tensor!(tensors, generation.content_suppression_mask, backend),
        channel_token_ids: generation.channel_token_ids,
        eos_token_ids: generation.eos_token_ids,
        no_repeat_ngram_size: generation.no_repeat_ngram_size
      }
    }
  end

  def build_split_pipeline!(prefix_artifact, %Tail{} = tail, backend, opts \\ []) do
    spec = prefix_artifact.generation.spec
    bypass_layers = Keyword.get(opts, :bypass_layers, [])
    bypass_ffn_layers = Keyword.get(opts, :bypass_ffn_layers, [])
    bypass_phase = Keyword.get(opts, :bypass_phase, :all)
    fused_ffn = Keyword.get(opts, :fused_ffn, false)

    unless bypass_phase in [:all, :prefill, :decode] do
      raise ArgumentError, ":bypass_phase must be :all, :prefill, or :decode"
    end

    if fused_ffn and (bypass_layers != [] or bypass_ffn_layers != []) do
      raise ArgumentError, ":fused_ffn cannot be combined with decoder bypasses"
    end

    if tail.layer_indices !=
         Enum.to_list((prefix_artifact.prefix.last_layer + 1)..(spec.num_blocks - 1)) do
      raise ArgumentError, "prefix and tail layers are not contiguous"
    end

    cached_tail_model = Model.cached_decoder_tail_model(spec, tail.layer_indices)
    {_init_fun, cached_tail_predict_fun} = Axon.build(cached_tail_model, build_opts(backend))

    baseline_model = Bumblebee.build_model(spec)

    bypass_model =
      case {bypass_layers, bypass_ffn_layers} do
        {[], []} ->
          baseline_model

        {layers, ffn_layers} ->
          Model.cached_decoder_bypass_model(spec, layers, bypass_ffn_layers: ffn_layers)
      end

    {prefill_generation_model, generation_model} =
      if fused_ffn do
        fused_spec =
          spec
          |> Map.from_struct()
          |> Map.put(:fused_q4_ffn, true)
          |> then(&struct(Model, &1))

        {baseline_model, Bumblebee.build_model(fused_spec)}
      else
        case bypass_phase do
          :all -> {nil, bypass_model}
          :prefill -> {bypass_model, baseline_model}
          :decode -> {baseline_model, bypass_model}
        end
      end

    {_init_fun, generation_predict_fun} = Axon.build(generation_model, build_opts(backend))

    prefill_generation_predict_fun =
      if prefill_generation_model do
        {_init_fun, predict_fun} = Axon.build(prefill_generation_model, build_opts(backend))
        predict_fun
      end

    generation_params =
      prefix_artifact.prefix.params
      |> merge_model_states(tail.params)
      |> maybe_add_fused_ffn_params(spec, fused_ffn)

    %DecoderPipeline{
      prefix: prefix_artifact.prefix,
      tail: tail,
      input_context: %{
        backend: backend,
        tokenizer: prefix_artifact.tokenizer,
        e4b?: false,
        model_info: %{spec: spec}
      },
      generation: prefix_artifact.generation,
      cached_prefix_model: prefix_artifact.cached_model,
      cached_prefix_predict_fun: prefix_artifact.cached_predict_fun,
      cached_tail_model: cached_tail_model,
      cached_tail_predict_fun: cached_tail_predict_fun,
      generation_model: generation_model,
      generation_params: generation_params,
      generation_predict_fun: generation_predict_fun,
      prefill_generation_model: prefill_generation_model,
      prefill_generation_predict_fun: prefill_generation_predict_fun,
      parameter_count: prefix_artifact.prefix.parameter_count + tail.parameter_count
    }
  end

  def install_tail!(%DecoderPipeline{} = pipeline, %Tail{} = tail) do
    if tail.layer_indices != pipeline.tail.layer_indices do
      raise ArgumentError,
            "tail artifact layers #{inspect(tail.layer_indices)} do not match pipeline tail " <>
              inspect(pipeline.tail.layer_indices)
    end

    generation_params = merge_model_states(pipeline.prefix.params, tail.params)

    generation_params =
      if Enum.any?(Map.keys(pipeline.generation_params.data), &String.ends_with?(&1, ".gate_up")) do
        maybe_add_fused_ffn_params(generation_params, pipeline.generation.spec, true)
      else
        generation_params
      end

    %{
      pipeline
      | tail: tail,
        generation_params: generation_params,
        parameter_count: pipeline.prefix.parameter_count + tail.parameter_count
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

  def verify!(component, input, opts \\ []) do
    output =
      DecoderBlocks.run!(component, input.hidden_state,
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

      manifest.kind not in [:decoder_block, :decoder_tail, :decoder_prefix] ->
        raise ArgumentError, "artifact is not a decoder block, prefix, or tail"

      true ->
        manifest
    end
  end

  defp build_verification(component, sequence_length)
       when is_integer(sequence_length) and sequence_length > 0 do
    backend = component.backend || Nx.BinaryBackend
    element_count = sequence_length * component.input_size

    hidden_state =
      Nx.with_default_backend(backend, fn ->
        {1, sequence_length, component.input_size}
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
      DecoderBlocks.run!(component, hidden_state,
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

  defp load_parameters!(tensors, parameter_paths, backend) do
    parameter_paths
    |> Enum.map(fn {tensor_name, [node_name, parameter_name]} ->
      {node_name, parameter_name, load_tensor!(tensors, tensor_name, backend)}
    end)
    |> Enum.group_by(&elem(&1, 0), &{elem(&1, 1), elem(&1, 2)})
    |> Map.new(fn {node_name, parameters} -> {node_name, Map.new(parameters)} end)
    |> Axon.ModelState.new()
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

  defp merge_model_states(prefix, tail) do
    data =
      Map.merge(prefix.data, tail.data, fn _node_name, prefix_parameters, tail_parameters ->
        Map.merge(prefix_parameters, tail_parameters)
      end)

    Axon.ModelState.new(data)
  end

  defp maybe_add_fused_ffn_params(params, _spec, false), do: params

  defp maybe_add_fused_ffn_params(params, spec, true) do
    data =
      Enum.reduce(0..(spec.num_blocks - 1), params.data, fn layer, data ->
        prefix = "decoder.blocks.#{layer}.ffn"
        gate = Map.fetch!(data, "#{prefix}.gate")
        up = Map.fetch!(data, "#{prefix}.intermediate")

        fused = %{
          "gate_packed" => Map.fetch!(gate, "packed"),
          "gate_scales" => Map.fetch!(gate, "scales"),
          "up_packed" => Map.fetch!(up, "packed"),
          "up_scales" => Map.fetch!(up, "scales")
        }

        Map.put(data, "#{prefix}.gate_up", fused)
      end)

    Axon.ModelState.new(data)
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

  defp maybe_copy_tokenizer!(_path, nil), do: :ok

  defp maybe_copy_tokenizer!(path, repository_id) do
    tokenizer_dir = Path.join(path, "tokenizer")
    File.mkdir_p!(tokenizer_dir)
    cache_scope = repository_id |> String.replace("/", "--") |> String.replace(~r/[^\w-]/, "")

    Enum.each(@tokenizer_files, fn filename ->
      url = Hub.file_url(repository_id, filename, nil)

      case Hub.cached_download(url, cache_scope: cache_scope, offline: true) do
        {:ok, source} -> File.cp!(source, Path.join(tokenizer_dir, filename))
        {:error, _reason} when filename != "tokenizer.json" -> :ok
        {:error, reason} -> raise "could not copy tokenizer into artifact: #{reason}"
      end
    end)
  end

  defp maybe_load_tokenizer(path) do
    if File.dir?(Path.join(path, "tokenizer")) do
      {:ok, tokenizer} =
        Bumblebee.load_tokenizer({:local, Path.join(path, "tokenizer")}, type: :gemma)

      tokenizer
    end
  end

  defp build_opts(EXLA.Backend), do: [compiler: EXLA]

  defp build_opts({EXLA.Backend, backend_opts}),
    do: [compiler: EXLA] ++ Keyword.take(backend_opts, [:client, :device_id])

  defp build_opts(_backend), do: []
end
