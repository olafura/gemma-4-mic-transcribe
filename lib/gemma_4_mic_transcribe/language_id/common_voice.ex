defmodule Gemma4MicTranscribe.LanguageId.CommonVoice do
  @moduledoc """
  Samples full-sentence clips from Common Voice parquet shards, the layout of
  `fsicoli/common_voice_17_0` mirrored into a Hugging Face bucket.

  The root holds one directory per language. A split is either a flat set of
  shards (`de/test-00000-of-00008.parquet`) or nested under chunk-range
  directories (`ha/test/chunk-range-.../test-00000-of-00008.parquet`). Each
  row carries the MP3 bytes in the `audio` struct column, so clips are read
  from the shard and decoded through ffmpeg without unpacking the dataset.

  Only the columns needed to pick clips are read from every shard; the audio
  column is read once per shard that actually contributed clips, which keeps
  a run over a network-mounted bucket to a few hundred megabytes per language.
  """

  alias Explorer.DataFrame, as: DF
  alias Explorer.Series
  alias Gemma4MicTranscribe.LanguageId.Corpus

  @meta_columns ["path", "client_id", "locale", "sentence"]

  @doc "Sorted language directories present under the root."
  def languages(root) do
    root
    |> File.ls!()
    |> Enum.filter(&File.dir?(Path.join(root, &1)))
    |> Enum.sort()
  end

  @doc "Parquet shards of one language and split, flat or nested, sorted."
  def shards(root, language, split) do
    flat = Path.wildcard(Path.join([root, language, "#{split}-*.parquet"]))
    nested = Path.wildcard(Path.join([root, language, split, "**", "*.parquet"]))
    Enum.sort(flat ++ nested)
  end

  @doc """
  Deterministically samples up to `per_language` clips of `split` from every
  language directory (or the given `:languages`).

  Options: `:seed` (default 42), `:languages`, and `:shards`, the number of
  shards read per language (default 1, chosen from the seed; a shard that
  does not parse is skipped with a warning). Clips are
  ordered by a hash of the seed and clip key, as the single-word corpus is.
  Each clip is a map with `:key`, `:language` (the row's locale, falling back
  to the directory name), `:directory`, `:client_id`, `:sentence`, `:path`
  (the MP3 file name) and `:bytes`.
  """
  def sample(root, split, per_language, opts \\ []) do
    seed = Keyword.get(opts, :seed, 42)
    shard_count = Keyword.get(opts, :shards, 1)
    languages = Keyword.get(opts, :languages) || languages(root)

    Enum.flat_map(languages, fn language ->
      chosen =
        root
        |> shards(language, split)
        |> pick_shards(language, seed)
        |> readable_shards(shard_count)
        |> Enum.flat_map(fn {shard, rows} ->
          Enum.with_index(rows, fn row, index -> Map.merge(row, %{"shard" => shard, "index" => index}) end)
        end)
        |> Enum.map(fn row ->
          %{
            key: language <> "/" <> row["path"],
            language: blank_to(row["locale"], language),
            directory: language,
            client_id: row["client_id"],
            sentence: row["sentence"],
            path: row["path"],
            shard: row["shard"],
            index: row["index"]
          }
        end)
        |> Enum.sort_by(fn clip -> :crypto.hash(:sha256, "#{seed}:#{clip.key}") end)
        |> Enum.take(per_language)

      audio =
        chosen
        |> Enum.group_by(& &1.shard, & &1.index)
        |> Map.new(fn {shard, indices} ->
          bytes =
            shard
            |> DF.from_parquet!(columns: ["audio"])
            |> DF.slice(indices)
            |> DF.pull("audio")
            |> Series.to_list()
            |> Enum.map(& &1["bytes"])

          {shard, Enum.zip(indices, bytes) |> Map.new()}
        end)

      Enum.map(chosen, fn clip ->
        clip
        |> Map.put(:bytes, audio[clip.shard][clip.index])
        |> Map.drop([:shard, :index])
      end)
    end)
  end

  @doc """
  Decodes a sampled clip to mono 16 kHz f32 samples cut to `seconds`, with
  the same onset trimming as `Corpus.decode!/3`.
  """
  def decode!(%{bytes: bytes, path: path}, seconds, opts \\ []) do
    file =
      Path.join(
        System.tmp_dir!(),
        "language-id-cv-#{System.unique_integer([:positive])}#{Path.extname(path)}"
      )

    File.write!(file, bytes)

    try do
      Corpus.decode!(file, seconds, opts)
    after
      File.rm(file)
    end
  end

  # A seeded order of shards so different seeds see different speakers
  # while the same seed always reads the same files.
  defp pick_shards(shards, language, seed) do
    Enum.sort_by(shards, fn shard -> :crypto.hash(:sha256, "#{seed}:#{language}:#{Path.basename(shard)}") end)
  end

  # The first `count` shards in seeded order whose metadata reads. A shard
  # that fails to parse (a truncated upload, typically) is reported on
  # stderr and the next one in the order takes its place, so one bad file
  # does not stop a run over a whole bucket.
  defp readable_shards(shards, count) do
    shards
    |> Stream.map(fn shard ->
      case DF.from_parquet(shard, columns: @meta_columns) do
        {:ok, frame} ->
          {shard, DF.to_rows(frame)}

        {:error, error} ->
          IO.puts(:stderr, "skipping unreadable shard #{shard}: #{Exception.message(error)}")
          nil
      end
    end)
    |> Stream.reject(&is_nil/1)
    |> Enum.take(count)
  end

  defp blank_to(nil, default), do: default
  defp blank_to("", default), do: default
  defp blank_to(value, _default), do: value
end
