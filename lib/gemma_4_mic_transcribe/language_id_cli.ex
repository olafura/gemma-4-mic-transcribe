defmodule Gemma4MicTranscribe.LanguageIdCLI do
  @moduledoc false

  alias Gemma4MicTranscribe.LanguageId.Artifact
  alias Gemma4MicTranscribe.LanguageId.Corpus
  alias Gemma4MicTranscribe.LanguageId.Features
  alias Gemma4MicTranscribe.LanguageId.Head
  alias Gemma4MicTranscribe.LanguageId.Runtime

  @commands ["extract", "sweep", "export", "detect"]

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
    help: :boolean
  ]

  def main(argv) do
    {:ok, _started} = Application.ensure_all_started(:gemma_4_mic_transcribe)

    case parse(argv) do
      {:ok, :extract, opts} -> extract!(opts)
      {:ok, :sweep, opts} -> sweep!(opts)
      {:ok, :export, opts} -> export!(opts)
      {:ok, :detect, opts} -> detect!(opts)
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
      corpus: Keyword.get(opts, :corpus, "~/Downloads/cv-corpus-7.0-singleword"),
      split: Keyword.get(opts, :split, "train"),
      per_language: Keyword.get(opts, :per_language, 200),
      seed: Keyword.get(opts, :seed, 42),
      seconds: Keyword.get(opts, :seconds, 4),
      batch_size: Keyword.get(opts, :batch_size, 16),
      train: opts[:train],
      test: opts[:test],
      depth: opts[:depth],
      depths: parse_depths(opts[:depths]),
      steps: Keyword.get(opts, :steps, 400),
      learning_rate: Keyword.get(opts, :learning_rate, 0.01),
      weight_decay: Keyword.get(opts, :weight_decay, 0.01),
      artifact: opts[:artifact],
      output: opts[:output],
      input: opts[:input],
      top_k: Keyword.get(opts, :top_k, 5),
      backend: Keyword.get(opts, :backend, default_backend(mode)),
      model_name: Keyword.get(opts, :model_name, "google/gemma-4-E2B-it"),
      pooling: Keyword.get(opts, :pooling, "mean")
    }

    with :ok <- required(mode, values),
         :ok <- positive(values.per_language, "--per-language"),
         :ok <- positive(values.seconds, "--seconds"),
         :ok <- positive(values.batch_size, "--batch-size"),
         :ok <- positive(values.steps, "--steps"),
         :ok <- positive(values.top_k, "--top-k"),
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
  defp default_backend(_mode), do: "exla:rocm"

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
  defp required(:detect, %{input: nil}), do: {:error, "detect requires --input"}
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
    clips = Corpus.sample(corpus, opts.split, opts.per_language, seed: opts.seed)

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
    train = Features.load!(opts.train)
    test = Features.load!(opts.test)

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
    train = Features.load!(opts.train)
    {x, y} = Features.depth(train, opts.depth)
    started = System.monotonic_time(:millisecond)
    head = train_head(x, y, train.languages, opts)
    IO.puts("trained depth #{opts.depth} head on #{Features.count(train)} clips in #{elapsed(started)}")

    if opts.test do
      test = Features.load!(opts.test)
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
    ranked = Artifact.detect(artifact, runtime, samples)
    IO.puts("detected in #{elapsed(started)}")

    ranked
    |> Enum.take(opts.top_k)
    |> Enum.each(fn %{language: language, probability: probability} ->
      IO.puts("  #{String.pad_trailing(language, 8)} #{percent(probability)}")
    end)
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
      language_id sweep --train DIR --test DIR [--depths 0,1,2] [options]
      language_id export --train DIR --depth N --artifact DIR [--test DIR] [options]
      language_id detect --artifact DIR --input AUDIO [--top-k N] [--backend NAME]

    extract runs the Gemma 4 audio tower over Common Voice single-word clips and
    saves one pooled feature vector per conformer depth. sweep trains a softmax
    head per depth and reports test accuracy so the shallowest useful depth can
    be picked. export trains the head for one depth and saves a self-contained
    detector: subsampling stack, the first N conformer blocks, head, languages.
    detect classifies one audio file with a saved detector.

    Options:
      --corpus PATH          Common Voice single-word corpus (default ~/Downloads/cv-corpus-7.0-singleword)
      --split NAME           train, dev, or test (default train)
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
      --output PATH          extract: output directory
    """
  end
end

defmodule Gemma4MicTranscribe.LanguageIdCLI.Escript do
  @moduledoc false

  def main(argv) do
    root =
      :escript.script_name()
      |> List.to_string()
      |> Path.expand()
      |> Path.dirname()

    mix_env = System.get_env("MIX_ENV", "dev")

    root
    |> Path.join("_build/#{mix_env}/lib/*/ebin")
    |> Path.wildcard()
    |> Enum.each(&Code.prepend_path/1)

    Gemma4MicTranscribe.LanguageIdCLI.main(argv)
  end
end
