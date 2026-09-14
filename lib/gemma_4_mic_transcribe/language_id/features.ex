defmodule Gemma4MicTranscribe.LanguageId.Features do
  @moduledoc """
  Pooled encoder features for labeled clips, saved so classifier heads can be
  trained and compared without re-running the audio tower.

  A feature set is a directory with `features.safetensors` (one `{n, hidden}`
  f32 tensor per captured depth plus an `{n}` label tensor) and a
  `manifest.json` naming the languages, clip keys, and extraction settings.
  """

  alias Gemma4MicTranscribe.LanguageId.Corpus
  alias Gemma4MicTranscribe.LanguageId.Encoder
  alias Gemma4MicTranscribe.LanguageId.Runtime

  @features_file "features.safetensors"
  @manifest_file "manifest.json"

  defstruct [:depths, :labels, :languages, :keys, :meta]

  @type t :: %__MODULE__{
          depths: %{String.t() => Nx.Tensor.t()},
          labels: Nx.Tensor.t(),
          languages: [String.t()],
          keys: [String.t()],
          meta: map()
        }

  @doc """
  Decodes and encodes `clips` (maps with `:path`, `:language`, `:key`) with
  the loaded runtime. Labels index into the sorted language list given by
  `:languages`, defaulting to the languages present in `clips`.

  Options: `:batch_size` (default 16), `:concurrency` for decoding, and
  `:progress` — a function called with the number of clips done so far.
  """
  def extract(%Runtime{} = runtime, clips, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 16)
    concurrency = Keyword.get(opts, :concurrency, max(div(System.schedulers_online(), 2), 1))
    progress = Keyword.get(opts, :progress, fn _done -> :ok end)
    languages = Keyword.get(opts, :languages) || clips |> Enum.map(& &1.language) |> Enum.uniq() |> Enum.sort()
    index = languages |> Enum.with_index() |> Map.new()

    {depths, done} =
      clips
      |> Task.async_stream(
        fn clip ->
          samples = Corpus.decode!(clip.path, runtime.seconds)
          Runtime.prepare(runtime, samples)
        end,
        max_concurrency: concurrency,
        ordered: true,
        timeout: :infinity
      )
      |> Stream.map(fn {:ok, prepared} -> prepared end)
      |> Stream.chunk_every(batch_size)
      |> Enum.reduce({[], 0}, fn chunk, {acc, done} ->
        encoded = Runtime.encode_all(runtime, chunk, batch_size)
        done = done + length(chunk)
        progress.(done)
        {[encoded | acc], done}
      end)

    ^done = length(clips)

    depths =
      depths
      |> Enum.reverse()
      |> Enum.reduce(fn batch, acc ->
        Map.merge(acc, batch, fn _key, left, right -> Nx.concatenate([left, right]) end)
      end)

    labels =
      clips
      |> Enum.map(&Map.fetch!(index, &1.language))
      |> Nx.tensor(type: :s64, backend: Nx.BinaryBackend)

    %__MODULE__{
      depths: depths,
      labels: labels,
      languages: languages,
      keys: Enum.map(clips, & &1.key),
      meta: %{
        seconds: runtime.seconds,
        frames: runtime.frames,
        tokens: runtime.tokens,
        hidden_size: runtime.spec.audio_hidden_size,
        pooling: runtime.spec.pooling,
        tower_parameters: Runtime.layer_sizes(runtime),
        parameter_bytes: parameter_bytes(runtime),
        depths: depths |> Map.keys() |> Enum.sort_by(&depth_index/1)
      }
    }
  end

  defp parameter_bytes(%Runtime{params: params}) do
    {_class, bits} =
      params.data
      |> Enum.flat_map(fn {_name, layer} -> Map.values(layer) end)
      |> List.first()
      |> Nx.type()

    div(bits, 8)
  end

  @doc "Integer depth of a `depth_n` key."
  def depth_index(key) do
    key |> String.replace_prefix("depth_", "") |> String.to_integer()
  end

  @doc "Depth keys in ascending order."
  def depth_keys(%__MODULE__{depths: depths}) do
    depths |> Map.keys() |> Enum.sort_by(&depth_index/1)
  end

  def count(%__MODULE__{labels: labels}), do: Nx.axis_size(labels, 0)

  @doc "Clips per language, keyed by language code."
  def counts(%__MODULE__{} = features) do
    features.labels
    |> Nx.to_flat_list()
    |> Enum.frequencies()
    |> Map.new(fn {index, count} -> {Enum.at(features.languages, index), count} end)
  end

  def save!(%__MODULE__{} = features, path, extra_meta \\ %{}) do
    path = Path.expand(path)
    File.mkdir_p!(path)

    tensors =
      features.depths
      |> Map.new(fn {key, tensor} -> {key, Nx.backend_copy(tensor, Nx.BinaryBackend)} end)
      |> Map.put("labels", Nx.backend_copy(features.labels, Nx.BinaryBackend))

    Safetensors.write!(Path.join(path, @features_file), tensors)

    manifest = %{
      version: 1,
      kind: "language_id_features",
      languages: features.languages,
      keys: features.keys,
      meta: Map.merge(features.meta, extra_meta)
    }

    File.write!(Path.join(path, @manifest_file), Jason.encode!(manifest, pretty: true))
    path
  end

  def load!(path) do
    path = Path.expand(path)
    manifest = path |> Path.join(@manifest_file) |> File.read!() |> Jason.decode!()
    {labels, depths} = path |> Path.join(@features_file) |> Safetensors.read!() |> Map.pop!("labels")

    %__MODULE__{
      depths: depths,
      labels: labels,
      languages: manifest["languages"],
      keys: manifest["keys"],
      meta: manifest["meta"]
    }
  end

  @doc "Features of one depth, as `{x, labels}`."
  def depth(%__MODULE__{} = features, depth) when is_integer(depth) do
    depth(features, Encoder.depth_key(depth))
  end

  def depth(%__MODULE__{} = features, key) when is_binary(key) do
    {Map.fetch!(features.depths, key), features.labels}
  end
end
