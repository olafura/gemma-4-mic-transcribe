defmodule Gemma4MicTranscribe.Gemma4.DecoderPipelineArtifact do
  @moduledoc """
  Persists a recomposed decoder pipeline independently of its source runtime.

  Artifacts contain the model specification, generation configuration, local
  tokenizer files, and all parameter tensors. Compiled functions are rebuilt by
  the runner for its selected backend; they are deliberately not serialized.

  The versioned manifest uses Erlang external-term format. Only load artifacts
  from trusted sources.
  """

  alias Bumblebee.HuggingFace.Hub
  alias Gemma4MicTranscribe.Gemma4.DecoderPipeline
  alias Gemma4MicTranscribe.Gemma4Unified.Model

  @version 1
  @manifest "manifest.etf"
  @parameters "parameters.safetensors"
  @tokenizer_files ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]

  def save!(%DecoderPipeline{} = pipeline, path, opts \\ []) do
    path = Path.expand(path)

    if File.exists?(path) do
      raise ArgumentError, "artifact path already exists: #{path}"
    end

    parent = Path.dirname(path)
    File.mkdir_p!(parent)
    temporary = path <> ".tmp-#{System.unique_integer([:positive])}"
    File.mkdir_p!(temporary)

    try do
      {tensors, parameter_paths} = flatten_parameters(pipeline.generation_params.data)

      tensors =
        tensors
        |> Map.put("g0", pipeline.generation.suppression_mask)
        |> Map.put("g1", pipeline.generation.inside_channel_suppression_mask)
        |> Map.put("g2", pipeline.generation.content_suppression_mask)

      Safetensors.write!(Path.join(temporary, @parameters), tensors)

      manifest = %{
        version: @version,
        model_name: Keyword.get(opts, :model_name),
        spec: pipeline.generation.spec,
        tail_layers: pipeline.tail.layer_indices,
        parameter_paths: parameter_paths,
        generation: %{
          channel_token_ids: pipeline.generation.channel_token_ids,
          eos_token_ids: pipeline.generation.eos_token_ids,
          no_repeat_ngram_size: pipeline.generation.no_repeat_ngram_size,
          suppression_mask: "g0",
          inside_channel_suppression_mask: "g1",
          content_suppression_mask: "g2"
        },
        transplants: Keyword.get(opts, :transplants, []),
        blends: Keyword.get(opts, :blends, [])
      }

      File.write!(Path.join(temporary, @manifest), :erlang.term_to_binary(manifest))
      maybe_copy_tokenizer!(temporary, Keyword.get(opts, :tokenizer_repository))
      File.rename!(temporary, path)
      path
    rescue
      exception ->
        File.rm_rf(temporary)
        reraise exception, __STACKTRACE__
    end
  end

  def load!(path, backend, opts \\ []) do
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

    tokenizer =
      if Keyword.get(opts, :load_tokenizer, true) do
        {:ok, tokenizer} =
          Bumblebee.load_tokenizer({:local, Path.join(path, "tokenizer")}, type: :gemma)

        tokenizer
      end

    generation = manifest.generation

    %{
      manifest: manifest,
      spec: manifest.spec,
      params: params,
      tokenizer: tokenizer,
      suppression_mask: load_tensor!(tensors, generation.suppression_mask, backend),
      inside_channel_suppression_mask:
        load_tensor!(tensors, generation.inside_channel_suppression_mask, backend),
      content_suppression_mask:
        load_tensor!(tensors, generation.content_suppression_mask, backend),
      channel_token_ids: generation.channel_token_ids,
      eos_token_ids: generation.eos_token_ids,
      no_repeat_ngram_size: generation.no_repeat_ngram_size
    }
  end

  def build_pipeline!(artifact, backend) do
    model = Model.model(artifact.spec)
    {_init_fun, predict_fun} = Axon.build(model, build_opts(backend))

    runtime = %{
      model_name: artifact.manifest.model_name || "decoder-pipeline-artifact",
      backend: backend,
      tokenizer: artifact.tokenizer,
      suppression_mask: artifact.suppression_mask,
      inside_channel_suppression_mask: artifact.inside_channel_suppression_mask,
      content_suppression_mask: artifact.content_suppression_mask,
      channel_token_ids: artifact.channel_token_ids,
      generation_config: %{eos_token_id: artifact.eos_token_ids},
      no_repeat_ngram_size: artifact.no_repeat_ngram_size,
      predict_fun: predict_fun,
      model_info: %{model: model, params: artifact.params, spec: artifact.spec}
    }

    DecoderPipeline.extract!(runtime, artifact.manifest.tail_layers)
  end

  def read_manifest!(path) do
    manifest =
      path
      |> Path.join(@manifest)
      |> File.read!()
      |> :erlang.binary_to_term()

    if manifest.version == @version do
      manifest
    else
      raise ArgumentError,
            "unsupported decoder pipeline artifact version #{inspect(manifest.version)}"
    end
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

  defp build_opts(EXLA.Backend), do: [compiler: EXLA]

  defp build_opts({EXLA.Backend, backend_opts}),
    do: [compiler: EXLA] ++ Keyword.take(backend_opts, [:client, :device_id])

  defp build_opts(_backend), do: []
end
