defmodule Gemma4MicTranscribe.LanguageId.Artifact do
  @moduledoc """
  A self-contained spoken-language detector: the front of the Gemma 4 audio
  tower (subsampling stack plus the first `depth` conformer blocks), a
  trained `LanguageId.Head`, and the language list.

  The artifact directory holds `manifest.etf` (encoder spec, depth,
  languages, parameter paths, sizes) and `parameters.safetensors` with the
  tower tensors and the head tensors. Loading it never touches the original
  checkpoint.
  """

  alias Gemma4MicTranscribe.LanguageId.Encoder
  alias Gemma4MicTranscribe.LanguageId.Head
  alias Gemma4MicTranscribe.LanguageId.Runtime

  @version 1
  @manifest "manifest.etf"
  @parameters "parameters.safetensors"
  @sample_rate 16_000

  defstruct [:spec, :depth, :params, :head, :languages, :seconds, :meta]

  @doc """
  Builds an artifact from a loaded runtime (any depth at or above `depth`)
  and a head trained on that depth's pooled features. Only the tower layers
  the truncated encoder needs are kept.
  """
  def build(%Runtime{} = runtime, depth, %Head{} = head, opts \\ []) do
    if depth > Encoder.depth(runtime.spec) do
      raise ArgumentError, "runtime only has #{Encoder.depth(runtime.spec)} blocks, cannot export depth #{depth}"
    end

    spec = Bumblebee.configure(runtime.spec, depth: depth, capture_all_depths: false)
    keep = fn name -> tower_layer?(name, depth) end

    params = %{
      runtime.params
      | data: runtime.params.data |> Enum.filter(fn {name, _} -> keep.(name) end) |> Map.new(),
        state: runtime.params.state |> Enum.filter(fn {name, _} -> keep.(name) end) |> Map.new()
    }

    %__MODULE__{
      spec: spec,
      depth: depth,
      params: params,
      head: head,
      languages: head.languages,
      seconds: runtime.seconds,
      meta: Keyword.get(opts, :meta, %{})
    }
  end

  defp tower_layer?("audio_encoder.subsample" <> _rest, _depth), do: true

  defp tower_layer?("audio_encoder.blocks." <> rest, depth) do
    case Integer.parse(rest) do
      {index, _} -> index < depth
      :error -> false
    end
  end

  defp tower_layer?(_name, _depth), do: false

  @doc "Tower and head parameter counts and byte sizes."
  def size(%__MODULE__{} = artifact) do
    tower = Runtime.size(%Runtime{params: artifact.params})
    head = artifact.head |> Head.to_tensors() |> Map.values()

    %{
      tower: tower,
      head: %{
        parameters: head |> Enum.map(&Nx.size/1) |> Enum.sum(),
        bytes: head |> Enum.map(&Nx.byte_size/1) |> Enum.sum()
      },
      total: %{
        parameters: tower.parameters + (head |> Enum.map(&Nx.size/1) |> Enum.sum()),
        bytes: tower.bytes + (head |> Enum.map(&Nx.byte_size/1) |> Enum.sum())
      }
    }
  end

  def save!(%__MODULE__{} = artifact, path) do
    path = Path.expand(path)

    if File.exists?(path) do
      raise ArgumentError, "artifact path already exists: #{path}"
    end

    File.mkdir_p!(Path.dirname(path))
    temporary = path <> ".tmp-#{System.unique_integer([:positive])}"
    File.mkdir_p!(temporary)

    try do
      {tensors, parameter_paths} = flatten_parameters(artifact.params)

      tensors =
        tensors
        |> Map.merge(Head.to_tensors(artifact.head))
        |> Map.new(fn {name, tensor} -> {name, Nx.backend_copy(tensor, Nx.BinaryBackend)} end)

      Safetensors.write!(Path.join(temporary, @parameters), tensors)

      manifest = %{
        version: @version,
        kind: :language_id,
        spec: artifact.spec,
        depth: artifact.depth,
        languages: artifact.languages,
        seconds: artifact.seconds,
        parameter_paths: parameter_paths,
        size: size(artifact),
        meta: artifact.meta
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

  @doc "Reads the manifest without loading any tensors."
  def manifest!(path) do
    manifest = path |> Path.expand() |> Path.join(@manifest) |> File.read!() |> :erlang.binary_to_term()

    if manifest.kind != :language_id do
      raise ArgumentError, "artifact is not a language detector"
    end

    manifest
  end

  @doc "Loads the artifact's tensors onto the binary backend."
  def load!(path) do
    path = Path.expand(path)
    manifest = manifest!(path)
    tensors = Safetensors.read!(Path.join(path, @parameters), lazy: true)

    read = fn name ->
      Nx.with_default_backend(Nx.BinaryBackend, fn -> Nx.to_tensor(Map.fetch!(tensors, name)) end)
    end

    {data, state} =
      Enum.reduce(manifest.parameter_paths, {%{}, %{}}, fn
        {tensor_name, [:data, node_name, parameter_name]}, {data, state} ->
          {put_in(data, [Access.key(node_name, %{}), parameter_name], read.(tensor_name)), state}

        {tensor_name, [:state, node_name, parameter_name]}, {data, state} ->
          {data, put_in(state, [Access.key(node_name, %{}), parameter_name], read.(tensor_name))}
      end)

    head = tensors |> Map.take(Map.keys(Head.to_tensors(%Head{}))) |> Map.new(fn {k, _} -> {k, read.(k)} end)

    %__MODULE__{
      spec: manifest.spec,
      depth: manifest.depth,
      params: %{Axon.ModelState.new(data) | state: state},
      head: Head.from_tensors(head, manifest.languages),
      languages: manifest.languages,
      seconds: manifest.seconds,
      meta: manifest.meta
    }
  end

  @doc """
  Builds a runtime for the artifact's truncated tower on `backend` (see
  `Runtime.backend!/1`). Returns `{runtime, artifact}` with the head kept
  alongside for `detect/3`.
  """
  def runtime(%__MODULE__{} = artifact, opts \\ []) do
    Runtime.load(
      spec: artifact.spec,
      params: artifact.params,
      depth: artifact.depth,
      backend: Keyword.get(opts, :backend, "torchx:cpu"),
      seconds: artifact.seconds
    )
  end

  @doc """
  Ranks languages for mono 16 kHz f32 `samples`. Returns a list of
  `%{language: code, probability: p}` sorted by probability.

  With `candidates: [codes]` the ranking is restricted to those languages
  and renormalised over them (see `restrict/2`), for callers that know
  which languages can occur.
  """
  def detect(%__MODULE__{} = artifact, %Runtime{} = runtime, samples, opts \\ []) when is_list(samples) do
    prepared = Runtime.prepare(runtime, samples)
    features = runtime |> Runtime.encode([prepared]) |> Map.fetch!(Encoder.depth_key(artifact.depth))

    artifact.head
    |> Head.log_probs(features)
    |> Nx.exp()
    |> Nx.to_flat_list()
    |> Enum.zip(artifact.languages)
    |> Enum.map(fn {probability, language} -> %{language: language, probability: probability} end)
    |> Enum.sort_by(& &1.probability, :desc)
    |> restrict(Keyword.get(opts, :candidates))
  end

  @doc """
  Restricts a ranking to `candidates` and renormalises the probabilities
  over them, which is the softmax over the candidate logits alone. `nil`
  or a set naming none of the ranked languages leaves the ranking as is.
  """
  def restrict(ranked, nil), do: ranked

  def restrict(ranked, candidates) do
    allowed = MapSet.new(candidates)
    kept = Enum.filter(ranked, &MapSet.member?(allowed, &1.language))
    total = kept |> Enum.map(& &1.probability) |> Enum.sum()

    cond do
      kept == [] -> ranked
      total <= 0.0 -> Enum.map(kept, &%{&1 | probability: 1.0 / length(kept)})
      true -> Enum.map(kept, &%{&1 | probability: &1.probability / total})
    end
  end

  @doc "Seconds of audio the detector listens to."
  def window_samples(%__MODULE__{seconds: seconds}), do: seconds * @sample_rate

  defp flatten_parameters(%Axon.ModelState{data: data, state: state}) do
    [{:data, data}, {:state, state}]
    |> Enum.flat_map(fn {kind, map} ->
      map
      |> Enum.sort()
      |> Enum.flat_map(fn {node_name, parameters} ->
        parameters
        |> Enum.sort()
        |> Enum.map(fn {parameter_name, tensor} -> {[kind, node_name, parameter_name], tensor} end)
      end)
    end)
    |> Enum.with_index()
    |> Enum.reduce({%{}, %{}}, fn {{path, tensor}, index}, {tensors, paths} ->
      tensor_name = "p#{index}"
      {Map.put(tensors, tensor_name, tensor), Map.put(paths, tensor_name, path)}
    end)
  end
end
