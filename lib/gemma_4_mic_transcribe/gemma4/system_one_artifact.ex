defmodule Gemma4MicTranscribe.Gemma4.SystemOneArtifact do
  @moduledoc """
  The trained System One experts and routers on their own, as a directory of
  `manifest.etf` plus `parameters.safetensors` (the layout of
  `LanguageId.Artifact` and `Gemma4.DecoderBlockArtifact`).

  Nothing of Gemma is copied in: the artifact is the 71M new parameters and
  the handful of numbers that say where they attach (`layers`, `expert_size`,
  `gate_floor`). It is installed on a packed prefix and tail with
  `load_pipeline!/1`, which is also how the untrained and router-forced-closed
  baselines are run.
  """

  alias Gemma4MicTranscribe.Gemma4.DecoderBlockArtifact
  alias Gemma4MicTranscribe.Gemma4.SystemOne
  alias Gemma4MicTranscribe.Gemma4Unified.Runtime

  @version 1
  @manifest "manifest.etf"
  @parameters "parameters.safetensors"
  @closed_gate_floor 1.0

  defstruct [
    :layers,
    :hidden_size,
    :expert_size,
    :gate_floor,
    :activation,
    :params,
    :step,
    :meta
  ]

  @doc """
  An artifact of freshly initialised experts for `layers` of `spec`: zero
  `down_e` and a closed router, which adds exactly nothing to the model.
  """
  def new(spec, layers, opts \\ []) do
    %__MODULE__{
      layers: layers,
      hidden_size: Map.fetch!(spec, :hidden_size),
      expert_size: SystemOne.expert_size(spec),
      gate_floor: SystemOne.gate_floor(spec),
      activation: Map.get(spec, :activation, :gelu_approx_tanh),
      params: SystemOne.init_parameters(spec, layers),
      step: 0,
      meta: Keyword.get(opts, :meta, %{})
    }
  end

  @doc "An artifact from a trained model state, keeping only the expert nodes."
  def from_model_state(%__MODULE__{} = artifact, %Axon.ModelState{data: data}, opts \\ []) do
    %{
      artifact
      | params: Map.filter(data, fn {name, _parameters} -> SystemOne.parameter_node?(name) end),
        step: Keyword.get(opts, :step, artifact.step),
        meta: Map.merge(artifact.meta, Keyword.get(opts, :meta, %{}))
    }
  end

  @doc "Parameter count and byte size of the stored tensors."
  def size(%__MODULE__{params: params}) do
    tensors = params |> Map.values() |> Enum.flat_map(&Map.values/1)

    %{
      parameters: tensors |> Enum.map(&Nx.size/1) |> Enum.sum(),
      bytes: tensors |> Enum.map(&Nx.byte_size/1) |> Enum.sum()
    }
  end

  @doc """
  Writes the artifact to `path`.

  Options: `:type` casts every tensor on the way out (`{:bf, 16}` halves a
  deployment artifact), `:overwrite` replaces an existing directory, which is
  what a resumed run's checkpoint does, and `:keep` lists names inside an
  overwritten directory that survive it, because a run writes its checkpoints
  under the same directory as the artifact.
  """
  def save!(%__MODULE__{} = artifact, path, opts \\ []) do
    path = Path.expand(path)
    type = Keyword.get(opts, :type)
    keep = Keyword.get(opts, :keep, [])
    exists? = File.exists?(path)

    if exists? and not Keyword.get(opts, :overwrite, false) do
      raise ArgumentError, "artifact path already exists: #{path}"
    end

    File.mkdir_p!(Path.dirname(path))
    temporary = path <> ".tmp-#{System.unique_integer([:positive])}"
    File.mkdir_p!(temporary)

    try do
      artifact = %{artifact | params: cast(artifact.params, type)}
      {tensors, parameter_paths} = flatten_parameters(artifact.params)

      Safetensors.write!(Path.join(temporary, @parameters), tensors)

      manifest = %{
        version: @version,
        kind: :system_one_expert,
        layers: artifact.layers,
        hidden_size: artifact.hidden_size,
        expert_size: artifact.expert_size,
        gate_floor: artifact.gate_floor,
        activation: artifact.activation,
        step: artifact.step,
        parameter_paths: parameter_paths,
        size: size(artifact),
        meta: artifact.meta
      }

      File.write!(Path.join(temporary, @manifest), :erlang.term_to_binary(manifest))
    rescue
      exception ->
        File.rm_rf(temporary)
        reraise exception, __STACKTRACE__
    end

    # Only now, with the new directory complete, is the old one taken apart:
    # the kept names move across first so overwriting never loses them.
    if exists? do
      Enum.each(keep, fn name ->
        source = Path.join(path, name)
        if File.exists?(source), do: File.rename!(source, Path.join(temporary, name))
      end)

      File.rm_rf!(path)
    end

    File.rename!(temporary, path)
    path
  end

  @doc "Reads the manifest without loading any tensors."
  def manifest!(path) do
    manifest =
      path |> Path.expand() |> Path.join(@manifest) |> File.read!() |> :erlang.binary_to_term()

    if manifest.kind != :system_one_expert do
      raise ArgumentError, "artifact is not a System One expert"
    end

    manifest
  end

  @doc """
  Loads the artifact onto the binary backend, or `:backend` when given.
  `:type` casts every tensor, which is how the f32 training checkpoints are
  run at bf16 beside the packed model.
  """
  def load!(path, opts \\ []) do
    path = Path.expand(path)
    manifest = manifest!(path)
    tensors = Safetensors.read!(Path.join(path, @parameters), lazy: true)
    backend = Keyword.get(opts, :backend, Nx.BinaryBackend)
    type = Keyword.get(opts, :type)

    params =
      manifest.parameter_paths
      |> Enum.map(fn {tensor_name, [node_name, parameter_name]} ->
        tensor =
          Nx.with_default_backend(Nx.BinaryBackend, fn ->
            tensors |> Map.fetch!(tensor_name) |> Nx.to_tensor()
          end)

        {node_name, parameter_name, transfer(cast(tensor, type), backend)}
      end)
      |> Enum.group_by(&elem(&1, 0), &{elem(&1, 1), elem(&1, 2)})
      |> Map.new(fn {node_name, parameters} -> {node_name, Map.new(parameters)} end)

    %__MODULE__{
      layers: manifest.layers,
      hidden_size: manifest.hidden_size,
      expert_size: manifest.expert_size,
      gate_floor: manifest.gate_floor,
      activation: manifest.activation,
      params: params,
      step: manifest.step,
      meta: manifest.meta
    }
  end

  @doc """
  The base spec with the artifact's expert attached.

  `force_router_closed: true` raises the gate floor to 1.0, which clamps every
  gate to exactly 0: the expert subgraph is still built and still runs, and
  its contribution is exactly 0.0, so the run is the non-regression baseline
  rather than a different model. `gate_floor: f` overrides the floor the
  artifact was saved with, which is how an artifact trained before the floor
  was raised is evaluated at the new one.
  """
  def spec(base_spec, %__MODULE__{} = artifact, opts \\ []) do
    gate_floor =
      if Keyword.get(opts, :force_router_closed, false),
        do: @closed_gate_floor,
        else: Keyword.get(opts, :gate_floor) || artifact.gate_floor

    base_spec
    |> Map.put(:system_one_layers, artifact.layers)
    |> Map.put(:system_one_expert_size, artifact.expert_size)
    |> Map.put(:system_one_gate_floor, gate_floor)
  end

  @doc """
  Loads the packed prefix, the packed tail and an optional expert artifact
  into one `DecoderPipeline`.

  Options:

    * `:prefix_artifact`, `:tail_artifact` - artifact directories
    * `:expert` - a `%SystemOneArtifact{}`, a path, or `nil` for base Gemma
    * `:backend` - as `Runtime.resolve_backend/1`, default `"exla:rocm"`
    * `:force_router_closed` - see `spec/3`
    * `:logits_last_only` - see `build_pipeline!/4`
    * `:type` - cast the expert parameters, default `{:bf, 16}` to match the
      hidden states they are added to

  Returns `{pipeline, expert_artifact_or_nil}`.
  """
  def load_pipeline!(opts) do
    backend_name = Keyword.get(opts, :backend, "exla:rocm")
    {:ok, backend} = Runtime.resolve_backend(backend_name)

    prefix = DecoderBlockArtifact.load_prefix!(Keyword.fetch!(opts, :prefix_artifact), backend)
    tail = DecoderBlockArtifact.load_tail!(Keyword.fetch!(opts, :tail_artifact), backend)

    build_pipeline!(prefix, tail, backend, opts)
  end

  @doc """
  Installs an expert on an already-loaded prefix and tail, `load_pipeline!/1`
  without the loading. Two pipelines built from the same prefix and tail share
  their weight tensors, which is how `regress` runs a router-closed pipeline
  against the bare one without a second copy of the model.

  `logits_last_only: false` keeps the composed generation model's logits at
  every prompt position instead of only the last one. A right-padded prompt
  continues at `:logits_index`, which needs that position to still be there.
  """
  def build_pipeline!(prefix, tail, backend, opts \\ []) do
    artifact = expert_artifact(opts, backend)
    spec = pipeline_spec(prefix.generation.spec, artifact, opts)
    prefix = put_in(prefix.generation.spec, spec)

    tail =
      case artifact do
        nil ->
          %{tail | spec: spec}

        artifact ->
          model =
            Gemma4MicTranscribe.Gemma4Unified.Model.decoder_tail_model(spec, tail.layer_indices)

          {_init_fun, predict_fun} = Axon.build(model, build_opts(backend))

          %{
            tail
            | spec: spec,
              params: merge_parameters(tail.params, artifact.params),
              model: model,
              predict_fun: predict_fun
          }
      end

    {DecoderBlockArtifact.build_split_pipeline!(prefix, tail, backend), artifact}
  end

  defp pipeline_spec(base_spec, artifact, opts) do
    spec = if artifact, do: spec(base_spec, artifact, opts), else: base_spec

    case Keyword.get(opts, :logits_last_only) do
      nil -> spec
      value -> Map.put(spec, :logits_last_only, value)
    end
  end

  defp expert_artifact(opts, backend) do
    type = Keyword.get(opts, :type, {:bf, 16})

    case Keyword.get(opts, :expert) do
      nil -> nil
      %__MODULE__{} = artifact -> %{artifact | params: cast(artifact.params, type)}
      path when is_binary(path) -> load!(path, backend: backend, type: type)
    end
  end

  @doc """
  A function reading the routers' gates over a tail-boundary hidden state, or
  `nil` when the tail carries no expert.

  Axon cannot name an intermediate output, so the gates are read by running
  the tail's decoder blocks again with the gate nodes as the output. It is one
  extra prompt-length pass, which is what an evaluation run pays to report the
  mean gate beside a reply. The gates come back floor-clamped, as inference
  applies them.
  """
  def gate_probe(tail) do
    layers = SystemOne.layers(tail.spec)

    if layers == [] do
      nil
    else
      model =
        tail.spec
        |> Gemma4MicTranscribe.Gemma4Unified.Model.decoder_block_chain_model(tail.layer_indices)
        |> SystemOne.gate_nodes(layers)
        |> Map.new(fn {layer_index, node} -> {"#{layer_index}", node} end)
        |> Axon.container()

      {_init_fun, predict_fun} = Axon.build(model, build_opts(tail.backend))

      fn inputs -> predict_fun.(tail.params, inputs) end
    end
  end

  @doc """
  Adds the expert parameters to a model state or plain parameter map. The
  expert nodes are new names, so nothing of Gemma is overwritten.
  """
  def merge_parameters(%Axon.ModelState{data: data} = params, expert_params) do
    %{params | data: merge_parameters(data, expert_params)}
  end

  def merge_parameters(data, expert_params) when is_map(data) do
    Map.merge(data, expert_params, fn _node, existing, added -> Map.merge(existing, added) end)
  end

  defp cast(params, nil), do: params

  defp cast(%Nx.Tensor{} = tensor, type), do: Nx.as_type(tensor, type)

  defp cast(params, type) when is_map(params),
    do: Map.new(params, fn {name, value} -> {name, cast(value, type)} end)

  defp transfer(tensor, Nx.BinaryBackend), do: tensor
  defp transfer(tensor, nil), do: tensor
  defp transfer(tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp build_opts(EXLA.Backend), do: [compiler: EXLA]

  defp build_opts({EXLA.Backend, backend_opts}),
    do: [compiler: EXLA] ++ Keyword.take(backend_opts, [:client, :device_id])

  defp build_opts(_backend), do: []

  defp flatten_parameters(params) do
    params
    |> Enum.sort()
    |> Enum.flat_map(fn {node_name, parameters} ->
      parameters
      |> Enum.sort()
      |> Enum.map(fn {parameter_name, tensor} -> {[node_name, parameter_name], tensor} end)
    end)
    |> Enum.with_index()
    |> Enum.reduce({%{}, %{}}, fn {{path, tensor}, index}, {tensors, paths} ->
      tensor_name = "p#{index}"

      {
        Map.put(tensors, tensor_name, Nx.backend_copy(tensor, Nx.BinaryBackend)),
        Map.put(paths, tensor_name, path)
      }
    end)
  end
end
