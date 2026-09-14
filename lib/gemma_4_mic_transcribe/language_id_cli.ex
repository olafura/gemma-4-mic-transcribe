defmodule Gemma4MicTranscribe.LanguageIdCLI do
  @moduledoc false

  alias Gemma4MicTranscribe.LanguageId.Artifact
  alias Gemma4MicTranscribe.LanguageId.CommonVoice
  alias Gemma4MicTranscribe.LanguageId.Corpus
  alias Gemma4MicTranscribe.LanguageId.Features
  alias Gemma4MicTranscribe.LanguageId.Finetune
  alias Gemma4MicTranscribe.LanguageId.Head
  alias Gemma4MicTranscribe.LanguageId.Reference
  alias Gemma4MicTranscribe.LanguageId.Runtime
  alias Gemma4MicTranscribe.LanguageId.Server

  @commands ["extract", "sweep", "export", "detect", "compare", "inputs", "finetune", "validate", "serve"]

  @switches [
    corpus: :string,
    split: :string,
    per_language: :integer,
    seed: :integer,
    seconds: :integer,
    batch_size: :integer,
    train: :string,
    test: :string,
    depth: :integer,
    depths: :string,
    steps: :integer,
    learning_rate: :float,
    weight_decay: :float,
    artifact: :string,
    output: :string,
    input: :string,
    top_k: :integer,
    backend: :string,
    model_name: :string,
    pooling: :string,
    whisper_model: :string,
    whisper_cli: :string,
    reference: :string,
    inputs_train: :string,
    inputs_test: :string,
    epochs: :integer,
    max_grad_norm: :float,
    freeze: :string,
    shards: :integer,
    languages: :string,
    candidates: :string,
    port: :integer,
    help: :boolean
  ]

  def main(argv) do
    {:ok, _started} = Application.ensure_all_started(:gemma_4_mic_transcribe)

    case parse(argv) do
      {:ok, :extract, opts} -> extract!(opts)
      {:ok, :sweep, opts} -> sweep!(opts)
      {:ok, :export, opts} -> export!(opts)
      {:ok, :detect, opts} -> detect!(opts)
      {:ok, :compare, opts} -> compare!(opts)
      {:ok, :inputs, opts} -> inputs!(opts)
      {:ok, :finetune, opts} -> finetune!(opts)
      {:ok, :validate, opts} -> validate!(opts)
      {:ok, :serve, opts} -> serve!(opts)
      {:help, usage} -> IO.puts(usage)
      {:error, message} -> abort(message)
    end
  end

  def parse([command | argv]) when command in @commands do
    mode = String.to_existing_atom(command)

    case OptionParser.parse(argv, strict: @switches, aliases: [h: :help]) do
      {opts, [], []} -> parse_options(mode, opts)
      {_opts, args, []} -> {:error, "unexpected arguments: #{Enum.join(args, " ")}"}
      {_opts, _args, invalid} -> {:error, "invalid options: #{inspect(invalid)}"}
    end
  end

  def parse(["--help"]), do: {:help, usage()}
  def parse([]), do: {:help, usage()}
  def parse(_argv), do: {:error, "expected one of: #{Enum.join(@commands, ", ")}"}

  defp parse_options(mode, opts) do
    if Keyword.get(opts, :help, false) do
      {:help, usage()}
    else
      parse_values(mode, opts)
    end
  end

  defp parse_values(mode, opts) do
    values = %{
      corpus: Keyword.get(opts, :corpus, default_corpus(mode)),
      split: Keyword.get(opts, :split, default_split(mode)),
      per_language: Keyword.get(opts, :per_language, 200),
      seed: Keyword.get(opts, :seed, 42),
      seconds: Keyword.get(opts, :seconds, 4),
      batch_size: Keyword.get(opts, :batch_size, 16),
      train: opts[:train],
      test: opts[:test],
      depth: opts[:depth],
      depths: parse_depths(opts[:depths]),
      steps: Keyword.get(opts, :steps, 400),
      learning_rate: Keyword.get(opts, :learning_rate, default_learning_rate(mode)),
      weight_decay: Keyword.get(opts, :weight_decay, 0.01),
      artifact: opts[:artifact],
      output: opts[:output],
      input: opts[:input],
      top_k: Keyword.get(opts, :top_k, 5),
      backend: Keyword.get(opts, :backend, default_backend(mode)),
      model_name: Keyword.get(opts, :model_name, "google/gemma-4-E2B-it"),
      pooling: Keyword.get(opts, :pooling, "mean"),
      whisper_model: opts[:whisper_model],
      whisper_cli: opts[:whisper_cli],
      reference: opts[:reference],
      inputs_train: opts[:inputs_train],
      inputs_test: opts[:inputs_test],
      epochs: Keyword.get(opts, :epochs, 3),
      max_grad_norm: Keyword.get(opts, :max_grad_norm, 1.0),
      freeze: opts[:freeze] |> to_string() |> String.split(",", trim: true),
      shards: Keyword.get(opts, :shards, 1),
      languages: opts[:languages] && String.split(opts[:languages], ",", trim: true),
      candidates: opts[:candidates] && String.split(opts[:candidates], ",", trim: true),
      port: Keyword.get(opts, :port, 7860)
    }

    with :ok <- required(mode, values),
         :ok <- positive(values.per_language, "--per-language"),
         :ok <- positive(values.seconds, "--seconds"),
         :ok <- positive(values.batch_size, "--batch-size"),
         :ok <- positive(values.steps, "--steps"),
         :ok <- positive(values.epochs, "--epochs"),
         :ok <- positive(values.top_k, "--top-k"),
         :ok <- positive(values.shards, "--shards"),
         :ok <- positive(values.port, "--port"),
         :ok <- positive_number(values.learning_rate, "--learning-rate"),
         :ok <- non_negative_number(values.weight_decay, "--weight-decay"),
         :ok <- valid_depth(values.depth),
         :ok <- valid_depths(values.depths),
         :ok <- valid_split(values.split),
         :ok <- valid_pooling(values.pooling) do
      {:ok, mode, values}
    end
  end

  defp default_backend(:detect), do: "torchx:cpu"
  defp default_backend(:compare), do: "torchx:cpu"
  defp default_backend(:validate), do: "torchx:cpu"
  defp default_backend(:serve), do: "torchx:cpu"
  defp default_backend(_mode), do: "exla:rocm"

  # validate reads the bucket a Hugging Face job or Space mounts at /data.
  defp default_corpus(:validate), do: "/data/common_voice"
  defp default_corpus(_mode), do: "~/Downloads/cv-corpus-7.0-singleword"

  defp default_split(:validate), do: "test"
  defp default_split(_mode), do: "train"

  defp default_learning_rate(:finetune), do: 2.0e-5
  defp default_learning_rate(_mode), do: 0.01

  defp parse_depths(nil), do: nil

  defp parse_depths(text) do
    text
    |> String.split(",", trim: true)
    |> Enum.map(fn item ->
      case Integer.parse(String.trim(item)) do
        {depth, ""} -> depth
        _other -> :invalid
      end
    end)
  end

  defp required(:extract, %{output: nil}), do: {:error, "extract requires --output"}
  defp required(:sweep, %{train: nil}), do: {:error, "sweep requires --train"}
  defp required(:sweep, %{test: nil}), do: {:error, "sweep requires --test"}
  defp required(:export, %{train: nil}), do: {:error, "export requires --train"}
  defp required(:export, %{depth: nil}), do: {:error, "export requires --depth"}
  defp required(:export, %{artifact: nil}), do: {:error, "export requires --artifact"}
  defp required(:detect, %{artifact: nil}), do: {:error, "detect requires --artifact"}
  defp required(:inputs, %{output: nil}), do: {:error, "inputs requires --output"}
  defp required(:finetune, %{inputs_train: nil}), do: {:error, "finetune requires --inputs-train"}
  defp required(:finetune, %{train: nil}), do: {:error, "finetune requires --train"}
  defp required(:finetune, %{depth: nil}), do: {:error, "finetune requires --depth"}
  defp required(:finetune, %{artifact: nil}), do: {:error, "finetune requires --artifact"}
  defp required(:detect, %{input: nil}), do: {:error, "detect requires --input"}
  defp required(:compare, %{artifact: nil}), do: {:error, "compare requires --artifact"}
  defp required(:compare, %{whisper_model: nil, reference: nil}),
    do: {:error, "compare requires --whisper-model or --reference"}
  defp required(:validate, %{artifact: nil}), do: {:error, "validate requires --artifact"}
  defp required(:serve, %{artifact: nil}), do: {:error, "serve requires --artifact"}
  defp required(_mode, _values), do: :ok

  defp positive(value, _flag) when is_integer(value) and value > 0, do: :ok
  defp positive(_value, flag), do: {:error, "#{flag} must be a positive integer"}

  defp positive_number(value, _flag) when is_number(value) and value > 0, do: :ok
  defp positive_number(_value, flag), do: {:error, "#{flag} must be positive"}

  defp non_negative_number(value, _flag) when is_number(value) and value >= 0, do: :ok
  defp non_negative_number(_value, flag), do: {:error, "#{flag} must not be negative"}

  defp valid_depth(nil), do: :ok
  defp valid_depth(depth) when is_integer(depth) and depth >= 0, do: :ok
  defp valid_depth(_depth), do: {:error, "--depth must be a non-negative integer"}

  defp valid_depths(nil), do: :ok

  defp valid_depths(depths) do
    if Enum.all?(depths, &(is_integer(&1) and &1 >= 0)),
      do: :ok,
      else: {:error, "--depths must be a comma-separated list of non-negative integers"}
  end

  defp pooling_atom("mean"), do: :mean
  defp pooling_atom("mean_std"), do: :mean_std

  defp valid_pooling(pooling) when pooling in ["mean", "mean_std"], do: :ok
  defp valid_pooling(pooling), do: {:error, "unknown pooling #{inspect(pooling)}; expected mean or mean_std"}

  defp valid_split(split) when split in ["train", "dev", "test"], do: :ok
  defp valid_split(split), do: {:error, "unknown split #{inspect(split)}; expected train, dev, or test"}

  defp extract!(opts) do
    corpus = Path.expand(opts.corpus)
    languages = Corpus.languages(corpus)
    clips = Corpus.sample_any(corpus, opts.split, opts.per_language, seed: opts.seed, shards: opts.shards)

    IO.puts(
      "extracting #{length(clips)} #{opts.split} clips across #{length(languages)} languages " <>
        "(up to #{opts.per_language} per language, seed #{opts.seed})"
    )

    started = System.monotonic_time(:millisecond)

    runtime =
      Runtime.load(
        repo: {:hf, opts.model_name},
        backend: opts.backend,
        capture_all_depths: true,
        pooling: pooling_atom(opts.pooling),
        seconds: opts.seconds
      )

    IO.puts("loaded #{opts.model_name} audio tower: #{describe_size(Runtime.size(runtime))}")

    features =
      Features.extract(runtime, clips,
        languages: languages,
        batch_size: opts.batch_size,
        progress: fn done ->
          if rem(done, opts.batch_size * 50) == 0 do
            IO.puts("  #{done}/#{length(clips)} clips, #{elapsed(started)}")
          end
        end
      )

    path =
      Features.save!(features, opts.output, %{
        split: opts.split,
        per_language: opts.per_language,
        seed: opts.seed,
        model_name: opts.model_name,
        pooling: opts.pooling
      })

    IO.puts("saved #{Features.count(features)} feature rows to #{path} in #{elapsed(started)}")
  end

  defp sweep!(opts) do
    train = Features.load_all!(opts.train)
    test = opts.test |> Features.load_all!() |> Features.relabel(train.languages)

    if train.languages != test.languages do
      abort("train and test feature sets list different languages")
    end

    depths = opts.depths || Enum.map(Features.depth_keys(train), &Features.depth_index/1)
    spec_size = size_per_depth(train)

    IO.puts(
      "train=#{Features.count(train)} test=#{Features.count(test)} languages=#{length(train.languages)} " <>
        "steps=#{opts.steps} lr=#{opts.learning_rate} wd=#{opts.weight_decay}"
    )

    IO.puts(String.pad_trailing("depth", 7) <> String.pad_leading("tower MB", 10) <> String.pad_leading("train", 8) <> String.pad_leading("test", 8) <> String.pad_leading("macro", 8) <> "  weakest languages")

    for depth <- depths do
      {x, y} = Features.depth(train, depth)
      {xt, yt} = Features.depth(test, depth)
      head = train_head(x, y, train.languages, opts)
      train_eval = Head.evaluate(head, x, y)
      test_eval = Head.evaluate(head, xt, yt)

      weakest =
        test_eval.per_language
        |> Enum.sort_by(fn {_language, result} -> result.accuracy end)
        |> Enum.take(3)
        |> Enum.map_join(", ", fn {language, result} -> "#{language} #{percent(result.accuracy)}" end)

      IO.puts(
        String.pad_trailing("#{depth}", 7) <>
          String.pad_leading(Float.to_string(Float.round(spec_size.(depth) / 1.0e6, 1)), 10) <>
          String.pad_leading(percent(train_eval.accuracy), 8) <>
          String.pad_leading(percent(test_eval.accuracy), 8) <>
          String.pad_leading(percent(test_eval.macro_accuracy), 8) <> "  " <> weakest
      )
    end
  end

  defp export!(opts) do
    train = Features.load_all!(opts.train)
    {x, y} = Features.depth(train, opts.depth)
    started = System.monotonic_time(:millisecond)
    head = train_head(x, y, train.languages, opts)
    IO.puts("trained depth #{opts.depth} head on #{Features.count(train)} clips (#{length(train.languages)} languages) in #{elapsed(started)}")

    if opts.test do
      test = opts.test |> Features.load_all!() |> Features.relabel(train.languages)
      {xt, yt} = Features.depth(test, opts.depth)
      evaluation = Head.evaluate(head, xt, yt)
      IO.puts("test accuracy #{percent(evaluation.accuracy)} (macro #{percent(evaluation.macro_accuracy)}) on #{evaluation.samples} clips")
    end

    runtime =
      Runtime.load(
        repo: {:hf, opts.model_name},
        backend: "torchx:cpu",
        depth: opts.depth,
        pooling: pooling_atom(train.meta["pooling"] || "mean"),
        seconds: train.meta["seconds"] || opts.seconds
      )

    artifact =
      Artifact.build(runtime, opts.depth, head,
        meta: %{model_name: opts.model_name, train: Path.expand(opts.train), train_clips: Features.count(train)}
      )

    path = Artifact.save!(artifact, opts.artifact)
    size = Artifact.size(artifact)

    IO.puts("saved #{path}")
    IO.puts("  tower: #{describe_size(size.tower)}")
    IO.puts("  head:  #{describe_size(size.head)}")
    IO.puts("  total: #{describe_size(size.total)}")
  end

  defp detect!(opts) do
    started = System.monotonic_time(:millisecond)
    artifact = Artifact.load!(opts.artifact)
    runtime = Artifact.runtime(artifact, backend: opts.backend)
    IO.puts("loaded detector (depth #{artifact.depth}, #{length(artifact.languages)} languages) in #{elapsed(started)}")

    samples = Corpus.decode!(Path.expand(opts.input), artifact.seconds)
    started = System.monotonic_time(:millisecond)
    ranked = Artifact.detect(artifact, runtime, samples, candidates: candidates(opts, artifact))
    IO.puts("detected in #{elapsed(started)}")

    ranked
    |> Enum.take(opts.top_k)
    |> Enum.each(fn %{language: language, probability: probability} ->
      IO.puts("  #{String.pad_trailing(language, 8)} #{percent(probability)}")
    end)
  end

  # Runs the Gemma detector, Whisper's own language detection, and the
  # Whisper-transcript-into-XLM-RoBERTa chain over the same seeded test clips,
  # restricted to the languages the text detector can name.
  defp compare!(opts) do
    artifact = Artifact.load!(opts.artifact)
    runtime = Artifact.runtime(artifact, backend: opts.backend)
    text = Reference.text_detector()
    corpus = Path.expand(opts.corpus)

    shared =
      artifact.languages
      |> Enum.filter(&(Reference.text_label(&1) in text.languages))

    clips = Corpus.sample_any(corpus, opts.split, opts.per_language, seed: opts.seed, languages: shared, shards: opts.shards)
    reference = opts.reference && load_reference!(opts.reference, clips)

    IO.puts(
      "comparing #{length(clips)} #{opts.split} clips across #{length(shared)} shared languages " <>
        "(#{Enum.join(shared, ", ")}), #{artifact.seconds} s window" <>
        if(reference, do: ", whisper answers from #{opts.reference}", else: "")
    )

    whisper_model = if reference, do: reference.whisper_model, else: opts.whisper_model
    whisper_opts = [model: whisper_model && Path.expand(whisper_model), binary: opts.whisper_cli]
    shared_set = MapSet.new(shared)

    # warm every path once so compile time stays out of the latency columns
    warm = Corpus.decode!(hd(clips).path, artifact.seconds)
    Artifact.detect(artifact, runtime, warm)
    Reference.detect_text(text, "warm up")

    rows =
      clips
      |> Enum.with_index(1)
      |> Enum.map(fn {clip, index} ->
        samples = Corpus.decode!(clip.path, artifact.seconds)
        wav = Path.join(System.tmp_dir!(), "language-id-compare-#{System.unique_integer([:positive])}.wav")
        write_wav!(wav, samples)

        started = System.monotonic_time(:millisecond)
        ranked = Artifact.detect(artifact, runtime, samples)
        gemma_ms = System.monotonic_time(:millisecond) - started
        gemma_shared = ranked |> Enum.filter(&MapSet.member?(shared_set, &1.language)) |> hd()

        whisper_columns =
          if reference do
            File.rm(wav)
            Map.fetch!(reference.rows, clip.key)
          else
            whisper = Reference.whisper(wav, whisper_opts)
            File.rm(wav)

            chained =
              if whisper.text == "",
                do: %{ranked: [], ms: 0},
                else: Reference.detect_text(text, whisper.text)

            %{
              whisper: whisper.language,
              whisper_text: whisper.text,
              whisper_ms: whisper.ms,
              chained: chained.ranked |> List.first() |> then(&(&1 && &1.language)),
              chained_ms: whisper.ms + chained.ms
            }
          end

        expected = Reference.text_label(clip.language)

        row =
          Map.merge(whisper_columns, %{
            key: clip.key,
            language: clip.language,
            expected: expected,
            gemma: Reference.text_label(hd(ranked).language),
            gemma_shared: Reference.text_label(gemma_shared.language),
            gemma_ms: gemma_ms
          })

        if rem(index, 25) == 0, do: IO.puts("  #{index}/#{length(clips)}")
        row
      end)

    if opts.output do
      write_output!(opts.output, Jason.encode!(%{artifact: Path.expand(opts.artifact), whisper_model: whisper_model, rows: rows}, pretty: true))
    end

    print_comparison(rows)

    if reference do
      previous = Enum.count(clips, &(Map.fetch!(reference.rows, &1.key).gemma_shared == Reference.text_label(&1.language)))
      current = Enum.count(rows, &(&1.gemma_shared == &1.expected))
      IO.puts("")
      IO.puts("shared-language accuracy: reference #{percent(previous / length(rows))}, this artifact #{percent(current / length(rows))}")
    end
  end

  # Loads a compare JSON so its Whisper and XLM-RoBERTa answers can be reused
  # for a new artifact over the same clips. Every sampled clip must be present.
  defp load_reference!(path, clips) do
    %{"rows" => rows} = decoded = path |> Path.expand() |> File.read!() |> Jason.decode!()

    rows =
      Map.new(rows, fn row ->
        {row["key"],
         %{
           gemma_shared: row["gemma_shared"],
           whisper: row["whisper"],
           whisper_text: row["whisper_text"],
           whisper_ms: row["whisper_ms"],
           chained: row["chained"],
           chained_ms: row["chained_ms"]
         }}
      end)

    missing = Enum.reject(clips, &Map.has_key?(rows, &1.key))

    if missing != [] do
      abort("#{length(missing)} sampled clips are not in #{path}; pass the same --split, --per-language and --seed")
    end

    %{rows: rows, whisper_model: decoded["whisper_model"]}
  end

  defp print_comparison(rows) do
    systems = [
      {"gemma (34-way)", :gemma, :gemma_ms},
      {"gemma (shared)", :gemma_shared, :gemma_ms},
      {"whisper detect", :whisper, :whisper_ms},
      {"whisper -> xlm-r", :chained, :chained_ms}
    ]

    IO.puts("")
    IO.puts(String.pad_trailing("system", 18) <> String.pad_leading("accuracy", 10) <> String.pad_leading("macro", 8) <> String.pad_leading("p50 ms", 8) <> String.pad_leading("p90 ms", 8))

    for {name, field, ms_field} <- systems do
      hits = Enum.count(rows, &(Map.fetch!(&1, field) == &1.expected))

      macro =
        rows
        |> Enum.group_by(& &1.expected)
        |> Enum.map(fn {_language, group} -> Enum.count(group, &(Map.fetch!(&1, field) == &1.expected)) / length(group) end)
        |> then(&(Enum.sum(&1) / length(&1)))

      latencies = rows |> Enum.map(&Map.fetch!(&1, ms_field)) |> Enum.sort()

      IO.puts(
        String.pad_trailing(name, 18) <>
          String.pad_leading(percent(hits / length(rows)), 10) <>
          String.pad_leading(percent(macro), 8) <>
          String.pad_leading("#{percentile(latencies, 0.5)}", 8) <>
          String.pad_leading("#{percentile(latencies, 0.9)}", 8)
      )
    end

    IO.puts("")
    IO.puts(String.pad_trailing("language", 10) <> String.pad_leading("n", 5) <> String.pad_leading("gemma", 9) <> String.pad_leading("shared", 9) <> String.pad_leading("whisper", 9) <> String.pad_leading("chain", 9))

    rows
    |> Enum.group_by(& &1.expected)
    |> Enum.sort()
    |> Enum.each(fn {language, group} ->
      accuracy = fn field -> percent(Enum.count(group, &(Map.fetch!(&1, field) == language)) / length(group)) end

      IO.puts(
        String.pad_trailing(language, 10) <>
          String.pad_leading("#{length(group)}", 5) <>
          String.pad_leading(accuracy.(:gemma), 9) <>
          String.pad_leading(accuracy.(:gemma_shared), 9) <>
          String.pad_leading(accuracy.(:whisper), 9) <>
          String.pad_leading(accuracy.(:chained), 9)
      )
    end)
  end

  defp percentile([], _fraction), do: 0

  defp percentile(sorted, fraction) do
    index = min(round(fraction * (length(sorted) - 1)), length(sorted) - 1)
    Enum.at(sorted, index)
  end

  defp write_wav!(path, samples) do
    pcm = for sample <- samples, into: <<>>, do: <<round(max(min(sample, 1.0), -1.0) * 32767)::little-signed-16>>
    data_size = byte_size(pcm)

    header =
      <<"RIFF", 36 + data_size::little-32, "WAVE", "fmt ", 16::little-32, 1::little-16, 1::little-16,
        16_000::little-32, 32_000::little-32, 2::little-16, 16::little-16, "data", data_size::little-32>>

    File.write!(path, header <> pcm)
  end

  # Scores a saved detector on full-sentence Common Voice clips read straight
  # from parquet shards (fixie-ai/common_voice_17_0 mirrored into a bucket).
  # Languages the detector was never trained on cannot be right or wrong, so
  # they are reported by what the detector calls them instead.
  defp validate!(opts) do
    corpus = Path.expand(opts.corpus)
    File.dir?(corpus) || abort("corpus #{corpus} is not a directory; pass --corpus")

    started = System.monotonic_time(:millisecond)
    artifact = Artifact.load!(opts.artifact)
    runtime = Artifact.runtime(artifact, backend: opts.backend)
    IO.puts("loaded detector (depth #{artifact.depth}, #{length(artifact.languages)} languages) in #{elapsed(started)}")

    languages = opts.languages || CommonVoice.languages(corpus)
    known = MapSet.new(artifact.languages)

    # --candidates known: the languages of the corpus the detector knows,
    # i.e. the answer set a caller who knows the corpus would pass
    candidates =
      case opts.candidates do
        ["known"] -> Enum.filter(languages, &MapSet.member?(known, &1))
        _other -> candidates(opts, artifact)
      end

    candidates && IO.puts("answers restricted to #{length(candidates)} candidates: #{Enum.join(candidates, ", ")}")

    started = System.monotonic_time(:millisecond)

    clips =
      CommonVoice.sample(corpus, opts.split, opts.per_language,
        seed: opts.seed,
        languages: languages,
        shards: opts.shards
      )

    clips == [] && abort("no #{opts.split} shards under #{corpus}")

    IO.puts(
      "sampled #{length(clips)} #{opts.split} clips from #{length(languages)} language directories " <>
        "(#{opts.shards} shard(s) each) in #{elapsed(started)}, #{artifact.seconds} s window"
    )

    # warm the compiled path so the first clip's latency is not compile time
    Artifact.detect(artifact, runtime, CommonVoice.decode!(hd(clips), artifact.seconds))

    rows =
      clips
      |> Enum.with_index(1)
      |> Enum.map(fn {clip, index} ->
        samples = CommonVoice.decode!(clip, artifact.seconds)
        started = System.monotonic_time(:millisecond)
        full = Artifact.detect(artifact, runtime, samples)
        ms = System.monotonic_time(:millisecond) - started
        ranked = Artifact.restrict(full, candidates)
        top = Enum.map(ranked, & &1.language)

        if rem(index, 50) == 0, do: IO.puts("  #{index}/#{length(clips)}")

        %{
          key: clip.key,
          language: clip.language,
          directory: clip.directory,
          client_id: clip.client_id,
          sentence: clip.sentence,
          known: MapSet.member?(known, clip.language),
          predicted: hd(top),
          top3: Enum.take(top, 3),
          probability: hd(ranked).probability,
          # the whole distribution, so any candidate set can be scored later
          # from the JSON without running the detector again
          probabilities: Map.new(full, &{&1.language, &1.probability}),
          ms: ms
        }
      end)

    summary = summarize_validation(rows)
    # the table first: an hour of scoring must not be lost to a write error
    print_validation(summary)

    if opts.output do
      write_output!(
        opts.output,
        Jason.encode!(
          %{
            artifact: Path.expand(opts.artifact),
            corpus: corpus,
            split: opts.split,
            seconds: artifact.seconds,
            per_language: opts.per_language,
            seed: opts.seed,
            shards: opts.shards,
            candidates: candidates,
            languages: summary.languages,
            summary: summary.overall,
            rows: rows
          },
          pretty: true
        )
      )
    end
  end

  # Creates the parent directory, which a bucket mounted into a job may lack.
  defp write_output!(output, contents) do
    path = Path.expand(output)
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, contents)
  end

  @doc false
  def summarize_validation(rows) do
    languages =
      rows
      |> Enum.group_by(& &1.language)
      |> Enum.map(fn {language, clips} ->
        predicted =
          clips
          |> Enum.frequencies_by(& &1.predicted)
          |> Enum.sort_by(fn {code, count} -> {-count, code} end)
          |> Enum.take(3)
          |> Enum.map(fn {code, count} -> %{language: code, count: count} end)

        known = hd(clips).known

        %{
          language: language,
          known: known,
          clips: length(clips),
          speakers: clips |> Enum.map(& &1.client_id) |> Enum.uniq() |> length(),
          top1: if(known, do: Enum.count(clips, &(&1.predicted == &1.language)) / length(clips)),
          top3: if(known, do: Enum.count(clips, &(&1.language in &1.top3)) / length(clips)),
          predicted: predicted
        }
      end)
      |> Enum.sort_by(&{!&1.known, &1.language})

    known_rows = Enum.filter(rows, & &1.known)
    latencies = rows |> Enum.map(& &1.ms) |> Enum.sort()

    overall = %{
      known_languages: Enum.count(languages, & &1.known),
      known_clips: length(known_rows),
      top1: safe_ratio(Enum.count(known_rows, &(&1.predicted == &1.language)), length(known_rows)),
      top3: safe_ratio(Enum.count(known_rows, &(&1.language in &1.top3)), length(known_rows)),
      unknown_languages: Enum.count(languages, &(not &1.known)),
      unknown_clips: length(rows) - length(known_rows),
      p50_ms: percentile(latencies, 0.5),
      p95_ms: percentile(latencies, 0.95)
    }

    %{languages: languages, overall: overall}
  end

  defp safe_ratio(_numerator, 0), do: nil
  defp safe_ratio(numerator, denominator), do: numerator / denominator

  defp print_validation(%{languages: languages, overall: overall}) do
    IO.puts("")
    IO.puts("  language  clips  speakers  top-1  top-3  predicted")

    Enum.each(languages, fn row ->
      predicted = Enum.map_join(row.predicted, ", ", &"#{&1.language} #{&1.count}")

      IO.puts(
        "  " <>
          String.pad_trailing(row.language, 9) <>
          String.pad_leading(Integer.to_string(row.clips), 6) <>
          String.pad_leading(Integer.to_string(row.speakers), 10) <>
          String.pad_leading(maybe_percent(row.top1), 7) <>
          String.pad_leading(maybe_percent(row.top3), 7) <>
          "  " <> predicted
      )
    end)

    IO.puts("")

    if overall.known_clips > 0 do
      IO.puts(
        "languages the detector knows: #{overall.known_languages}, #{overall.known_clips} clips, " <>
          "top-1 #{percent(overall.top1)}, top-3 #{percent(overall.top3)}"
      )
    end

    if overall.unknown_clips > 0 do
      IO.puts(
        "languages outside the detector: #{overall.unknown_languages}, #{overall.unknown_clips} clips " <>
          "(see the predicted column)"
      )
    end

    IO.puts("detect latency per clip: p50 #{overall.p50_ms} ms, p95 #{overall.p95_ms} ms")
  end

  defp maybe_percent(nil), do: "-"
  defp maybe_percent(value), do: percent(value)

  defp serve!(opts) do
    started = System.monotonic_time(:millisecond)
    artifact = Artifact.load!(opts.artifact)
    runtime = Artifact.runtime(artifact, backend: opts.backend)
    IO.puts("loaded detector (depth #{artifact.depth}, #{length(artifact.languages)} languages) in #{elapsed(started)}")

    started = System.monotonic_time(:millisecond)
    Artifact.detect(artifact, runtime, List.duplicate(0.0, 1_600))
    IO.puts("warmed up in #{elapsed(started)}; listening on http://0.0.0.0:#{opts.port}")
    Server.run!(artifact, runtime, opts.port, candidates: candidates(opts, artifact))
  end

  # --candidates restricts answers to languages the caller knows can occur;
  # names the detector does not know are dropped with a warning, and a set
  # with nothing left is an error rather than a silent no-op.
  defp candidates(%{candidates: nil}, _artifact), do: nil

  defp candidates(%{candidates: list}, artifact) do
    known = MapSet.new(artifact.languages)
    {kept, unknown} = Enum.split_with(list, &MapSet.member?(known, &1))
    unknown != [] && IO.puts(:stderr, "ignoring --candidates the detector does not know: #{Enum.join(unknown, ", ")}")
    kept == [] && abort("none of --candidates is a language of the detector")
    kept
  end

  defp inputs!(opts) do
    corpus = Path.expand(opts.corpus)
    languages = Corpus.languages(corpus)
    clips = Corpus.sample_any(corpus, opts.split, opts.per_language, seed: opts.seed, shards: opts.shards)
    {:ok, spec} = Bumblebee.load_spec({:hf, opts.model_name}, module: Gemma4MicTranscribe.LanguageId.Encoder, architecture: :audio_encoder)

    IO.puts("preparing #{length(clips)} #{opts.split} clips across #{length(languages)} languages, #{opts.seconds} s window")
    started = System.monotonic_time(:millisecond)
    inputs = Finetune.prepare_inputs(spec, clips, opts.seconds, languages: languages)
    path = Finetune.save_inputs!(inputs, opts.output)
    IO.puts("saved #{Finetune.Inputs.count(inputs)} clips (#{inspect(Nx.shape(inputs.features))} mel) to #{path} in #{elapsed(started)}")
  end

  defp finetune!(opts) do
    train_inputs = Finetune.load_inputs!(opts.inputs_train)
    test_inputs = if opts.inputs_test, do: Finetune.load_inputs!(opts.inputs_test)
    features = Features.load_all!(opts.train)

    if features.languages != train_inputs.languages do
      abort("--train features and --inputs-train were sampled over different language sets")
    end

    {x, y} = Features.depth(features, opts.depth)
    head = Head.train(x, y, features.languages, steps: opts.steps, learning_rate: 0.01, weight_decay: opts.weight_decay)
    pooling = pooling_atom(features.meta["pooling"] || "mean")
    seconds = train_inputs.seconds

    IO.puts(
      "fine-tuning depth #{opts.depth} on #{Finetune.Inputs.count(train_inputs)} clips " <>
        "(#{opts.epochs} epochs, batch #{opts.batch_size}, lr #{opts.learning_rate}, " <>
        "frozen: #{if opts.freeze == [], do: "none", else: Enum.join(opts.freeze, ", ")})"
    )

    runtime =
      Runtime.load(
        repo: {:hf, opts.model_name},
        backend: opts.backend,
        depth: opts.depth,
        pooling: pooling,
        seconds: seconds,
        type: {:f, 32}
      )

    started = System.monotonic_time(:millisecond)

    result =
      Finetune.train(train_inputs, opts.depth,
        runtime: runtime,
        head: head,
        test: test_inputs,
        epochs: opts.epochs,
        batch_size: opts.batch_size,
        learning_rate: opts.learning_rate,
        freeze: opts.freeze,
        max_grad_norm: if(opts.max_grad_norm > 0, do: opts.max_grad_norm),
        seed: opts.seed,
        compiler_opts: compiler_opts(opts.backend),
        log: fn progress ->
          loss = if progress.loss, do: " loss #{format_loss(progress.loss)}", else: ""
          accuracy = if progress.accuracy, do: " test #{percent(progress.accuracy)}", else: ""
          skipped = if progress.skipped > 0, do: " skipped #{progress.skipped} steps", else: ""
          IO.puts("  epoch #{progress.epoch}#{loss}#{accuracy}#{skipped} (#{elapsed(started)})")
        end
      )

    if result.evaluation do
      IO.puts("test accuracy #{percent(result.evaluation.accuracy)} (macro #{percent(result.evaluation.macro_accuracy)}) on #{result.evaluation.samples} clips")
    end

    artifact =
      Finetune.to_artifact(runtime, result.model_state, opts.depth, train_inputs.languages, seconds,
        meta: %{
          model_name: opts.model_name,
          finetuned: true,
          epochs: opts.epochs,
          learning_rate: opts.learning_rate,
          train_clips: Finetune.Inputs.count(train_inputs),
          test_accuracy: result.evaluation && result.evaluation.accuracy
        }
      )

    path = Artifact.save!(artifact, opts.artifact)
    size = Artifact.size(artifact)
    IO.puts("saved #{path}")
    IO.puts("  total: #{describe_size(size.total)}")
  end

  defp format_loss(loss) when is_float(loss), do: Float.round(loss, 4)
  defp format_loss(other), do: inspect(other)

  defp compiler_opts(backend) do
    case Runtime.backend!(backend) do
      {EXLA.Backend, backend_opts} -> [compiler: EXLA] ++ Keyword.take(backend_opts, [:client])
      _other -> []
    end
  end

  defp train_head(x, y, languages, opts) do
    Head.train(x, y, languages,
      steps: opts.steps,
      learning_rate: opts.learning_rate,
      weight_decay: opts.weight_decay
    )
  end

  defp size_per_depth(%Features{meta: meta}) do
    case meta do
      %{"tower_parameters" => %{"subsample" => subsample, "blocks" => blocks}, "parameter_bytes" => bytes} ->
        fn depth -> Runtime.parameters_at_depth(%{subsample: subsample, blocks: blocks}, depth) * bytes end

      _other ->
        fn _depth -> 0 end
    end
  end

  defp describe_size(%{parameters: parameters, bytes: bytes}) do
    "#{parameters} parameters, #{Float.round(bytes / 1.0e6, 1)} MB"
  end

  defp percent(value), do: "#{Float.round(value * 100, 1)}%"

  defp elapsed(started), do: "#{System.monotonic_time(:millisecond) - started} ms"

  defp abort(message) do
    IO.puts(:stderr, "error: #{message}")
    System.halt(1)
  end

  defp usage do
    """
    Usage:
      language_id extract --output DIR [--split train|dev|test] [--per-language N] [options]
      language_id sweep --train DIR[,DIR] --test DIR[,DIR] [--depths 0,1,2] [options]
      language_id export --train DIR[,DIR] --depth N --artifact DIR [--test DIR[,DIR]] [options]
      language_id detect --artifact DIR --input AUDIO [--top-k N] [--candidates LIST] [--backend NAME]
      language_id compare --artifact DIR (--whisper-model GGML | --reference JSON) [--per-language N] [--output JSON]
      language_id inputs --output DIR [--split train|dev|test] [--per-language N] [--seconds N]
      language_id finetune --inputs-train DIR --train FEATURES --depth N --artifact DIR [--inputs-test DIR] [options]
      language_id validate --artifact DIR [--corpus DIR] [--split test] [--per-language N] [--shards N] [--candidates LIST|known] [--output JSON]
      language_id serve --artifact DIR [--port 7860] [--candidates LIST] [--backend NAME]

    extract runs the Gemma 4 audio tower over Common Voice single-word clips and
    saves one pooled feature vector per conformer depth. sweep trains a softmax
    head per depth and reports test accuracy so the shallowest useful depth can
    be picked. export trains the head for one depth and saves a self-contained
    detector: subsampling stack, the first N conformer blocks, head, languages.
    detect classifies one audio file with a saved detector. compare runs the
    detector, whisper.cpp language detection, and Whisper transcripts fed to
    papluca/xlm-roberta-base-language-detection over the same test clips.
    inputs caches mel features for end-to-end training; finetune trains the
    truncated tower and head together (Axon.Loop, Adam) starting from the
    pretrained tower and a logistic head fitted on --train features, then
    exports the result as a detector. validate scores a detector on
    full-sentence Common Voice parquet shards (fixie-ai/common_voice_17_0, one
    directory per language, as mounted from a Hugging Face bucket) and reports
    per-language accuracy; serve answers POST /detect with the ranking for the
    audio file in the request body.

    Options:
      --corpus PATH          Common Voice single-word corpus (default ~/Downloads/cv-corpus-7.0-singleword)
                             or a parquet corpus root as validate reads (sentence clips, onset window);
                             validate defaults to /data/common_voice
      --split NAME           train, dev, or test (default train; validate: test)
      --per-language N       clips sampled per language (default 200)
      --seed N               sampling seed (default 42)
      --seconds N            fixed clip window in seconds (default 4)
      --batch-size N         encoder batch size (default 16)
      --depth N              conformer blocks kept in the exported detector
      --depths LIST          comma-separated depths to sweep (default all)
      --steps N              head training steps (default 400)
      --learning-rate F      Adam learning rate (default 0.01)
      --weight-decay F       L2 penalty on the head kernel (default 0.01)
      --top-k N              languages to print for detect (default 5)
      --backend NAME         torchx:cpu, exla:host, exla:cuda, exla:rocm (extract default exla:rocm, detect default torchx:cpu)
      --model-name NAME      Hugging Face repo of the Gemma 4 checkpoint (default google/gemma-4-E2B-it)
      --pooling MODE         mean or mean_std frame pooling for extract (default mean)
      --whisper-model PATH   ggml Whisper model for compare
      --whisper-cli PATH     whisper-cli binary for compare (default $WHISPER_CLI or whisper-cli)
      --reference JSON       compare: reuse Whisper and XLM-RoBERTa answers from an earlier --output
      --output PATH          extract/inputs: output directory; compare/validate: JSON with every row
      --inputs-train DIR     cached mel inputs for finetune (from inputs)
      --inputs-test DIR      cached mel inputs evaluated after every epoch
      --epochs N             finetune epochs (default 3)
      --max-grad-norm X      finetune global gradient-norm clip (default 1.0, 0 disables)
      --freeze LIST          finetune layer-name prefixes to keep fixed, e.g. audio_encoder.subsample
                             (finetune --learning-rate defaults to 2.0e-5)
      --shards N             validate: parquet shards read per language (default 1)
      --languages LIST       validate: comma-separated language directories (default all)
      --candidates LIST      detect/serve/validate: only answer from these languages, renormalised;
                             validate also takes "known" (the corpus languages the detector has)
      --port N               serve: HTTP port (default 7860)
    """
  end
end

defmodule Gemma4MicTranscribe.LanguageIdCLI.Escript do
  @moduledoc false

  # The NIF-backed applications (torchx, explorer) cannot live inside the
  # escript archive, so it loads them from the build that produced it; the
  # build directory depends on MIX_ENV and MIX_TARGET at build time.
  @build_dir Path.relative_to_cwd(Mix.Project.build_path())

  def main(argv) do
    root =
      :escript.script_name()
      |> List.to_string()
      |> Path.expand()
      |> Path.dirname()

    root
    |> Path.join("#{@build_dir}/lib/*/ebin")
    |> Path.wildcard()
    |> Enum.each(&Code.prepend_path/1)

    Gemma4MicTranscribe.LanguageIdCLI.main(argv)
  end
end
