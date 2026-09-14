defmodule Gemma4MicTranscribe.LanguageId.Corpus do
  @moduledoc """
  Samples and decodes labeled clips from a Common Voice single-word corpus.

  The corpus root holds one directory per language code with `train.tsv`,
  `dev.tsv`, `test.tsv`, and a `clips/` directory of MP3 files. Sampling is
  seeded and per language, so train and test sets are reproducible and no
  language can dominate by size.
  """

  alias Gemma4MicTranscribe.Audio

  @sample_rate 16_000

  @doc "Sorted language codes present in the corpus."
  def languages(corpus) do
    corpus
    |> File.ls!()
    |> Enum.filter(&File.dir?(Path.join(corpus, &1)))
    |> Enum.sort()
  end

  @doc """
  Deterministically samples up to `per_language` clips of `split` from every
  language (or the given `:languages`).
  """
  def sample(corpus, split, per_language, opts \\ []) do
    seed = Keyword.get(opts, :seed, 42)
    languages = Keyword.get(opts, :languages)

    corpus
    |> load_cases(split, languages, per_language, seed)
    |> Enum.map(fn sample ->
      %{key: sample.key, language: sample.language, path: sample.path, split: split}
    end)
  end

  @doc """
  Seeded per-language sample of transcription cases: every clip listed in
  `<language>/<split>.tsv` whose MP3 exists, ordered by a hash of the seed and
  clip key so the same seed always picks the same clips.
  """
  def load_cases(corpus, split, languages, per_language, seed \\ 42) do
    languages = languages || languages(corpus)

    Enum.flat_map(languages, fn language ->
      corpus
      |> cases(language, split)
      |> Enum.sort_by(fn sample -> :crypto.hash(:sha256, "#{seed}:#{sample.key}") end)
      |> Enum.take(per_language)
    end)
  end

  @doc "Every clip of one language and split with its expected transcript."
  def cases(corpus, language, split) do
    tsv = Path.join([corpus, language, split <> ".tsv"])

    if File.regular?(tsv) do
      [header | rows] = tsv |> File.read!() |> String.split("\n", trim: true)
      columns = header |> String.split("\t") |> Enum.with_index() |> Map.new()
      path_index = Map.fetch!(columns, "path")
      sentence_index = Map.fetch!(columns, "sentence")

      Enum.flat_map(rows, fn row ->
        fields = String.split(row, "\t")
        relative_path = Enum.at(fields, path_index)
        expected = Enum.at(fields, sentence_index)
        path = Path.join([corpus, language, "clips", relative_path || ""])

        if relative_path && expected && File.regular?(path) do
          [
            %{
              key: language <> "/" <> relative_path,
              language: language,
              relative_path: relative_path,
              path: path,
              expected: expected
            }
          ]
        else
          []
        end
      end)
    else
      []
    end
  end

  @doc """
  Decodes a clip to mono 16 kHz f32 samples, cut to `seconds`. Returns the
  real samples without padding; the runtime pads to its fixed length.

  Common Voice clips open with roughly a second of silence before the word,
  so with `trim: true` (the default) the window starts `lead` seconds before
  the first frame whose energy exceeds `threshold` times the clip's peak.
  A live microphone pipeline would gate on voice activity the same way, so
  short windows see speech rather than the recording's leading silence.
  """
  def decode!(path, seconds, opts \\ []) do
    trim = Keyword.get(opts, :trim, true)
    lead = Keyword.get(opts, :lead, 0.1)
    threshold = Keyword.get(opts, :threshold, 0.1)
    limit = if trim, do: seconds + Keyword.get(opts, :search, 3.0), else: seconds
    duration = :erlang.float_to_binary(limit / 1, decimals: 3)

    args = [
      "-v",
      "error",
      "-i",
      path,
      "-t",
      duration,
      "-f",
      "f32le",
      "-ac",
      "1",
      "-ar",
      Integer.to_string(@sample_rate),
      "pipe:1"
    ]

    case System.cmd("ffmpeg", args, stderr_to_stdout: true) do
      {audio, 0} ->
        audio =
          if trim,
            do: trim_onset(audio, lead: lead, threshold: threshold),
            else: audio

        audio
        |> binary_part(0, min(byte_size(audio), round(seconds * @sample_rate) * 4))
        |> Audio.binary_to_f32_samples()

      {message, status} ->
        raise "ffmpeg failed for #{path} (#{status}): #{message}"
    end
  end

  @frame 160

  @doc """
  Drops leading silence from f32le audio: everything before `lead` seconds
  ahead of the first 30 ms whose 10 ms frames all have RMS above `threshold`
  times the loudest frame. Audio with no frame above the floor is returned unchanged.
  """
  def trim_onset(audio, opts \\ []) when is_binary(audio) do
    lead = Keyword.get(opts, :lead, 0.1)
    threshold = Keyword.get(opts, :threshold, 0.1)

    energies =
      for <<frame::binary-size(@frame * 4) <- audio>> do
        for(<<sample::little-float-32 <- frame>>, reduce: 0.0, do: (acc -> acc + sample * sample))
      end

    peak = Enum.max(energies, fn -> 0.0 end)

    floor = max(peak * threshold * threshold, 1.0e-8)

    # Three consecutive loud frames (30 ms) so a click at the start of the
    # recording does not count as the onset.
    onset =
      energies
      |> Enum.chunk_every(3, 1, :discard)
      |> Enum.find_index(fn frames -> Enum.all?(frames, &(&1 > floor)) end)

    case onset do
      nil ->
        audio

      index ->
        start = max(index * @frame - round(lead * @sample_rate), 0) * 4
        binary_part(audio, start, byte_size(audio) - start)
    end
  end
end
