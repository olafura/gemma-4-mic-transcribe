defmodule Gemma4MicTranscribe.SingleWordBenchmark do
  @moduledoc false

  alias Gemma4MicTranscribe.Audio
  alias Gemma4MicTranscribe.Gemma4.DecoderBlockArtifact
  alias Gemma4MicTranscribe.Gemma4.DecoderPipeline
  alias Gemma4MicTranscribe.Gemma4Unified.Input
  alias Gemma4MicTranscribe.Gemma4Unified.Runtime

  @sample_rate 16_000

  @switches [
    corpus: :string,
    prefix_artifact: :string,
    tail_artifact: :string,
    backend: :string,
    split: :string,
    languages: :string,
    per_language: :integer,
    seed: :integer,
    seconds: :float,
    max_new_tokens: :integer,
    execution: :string,
    prompt: :string,
    output: :string,
    baseline: :string,
    help: :boolean
  ]

  def main(argv) do
    {:ok, _started} = Application.ensure_all_started(:gemma_4_mic_transcribe)

    case parse(argv) do
      {:ok, opts} -> run!(opts)
      {:help, usage} -> IO.puts(usage)
      {:error, message} -> abort(message)
    end
  end

  def parse(argv) do
    case OptionParser.parse(argv, strict: @switches, aliases: [h: :help]) do
      {opts, [], []} -> parse_options(opts)
      {_opts, args, []} -> {:error, "unexpected arguments: #{Enum.join(args, " ")}"}
      {_opts, _args, invalid} -> {:error, "invalid options: #{inspect(invalid)}"}
    end
  end

  def load_cases(corpus, split, languages, per_language, seed \\ 42) do
    languages = languages || corpus_languages(corpus)

    languages
    |> Enum.flat_map(fn language ->
      corpus
      |> language_cases(language, split)
      |> Enum.sort_by(fn sample -> :crypto.hash(:sha256, "#{seed}:#{sample.key}") end)
      |> Enum.take(per_language)
    end)
  end

  def normalize(text) do
    text
    |> String.normalize(:nfc)
    |> String.downcase()
    |> String.replace(~r/[\p{P}\p{S}\s]+/u, "")
  end

  def summarize(cases) do
    exact_count = Enum.count(cases, & &1.exact)
    reference_characters = Enum.sum(Enum.map(cases, &grapheme_length(&1.normalized_expected)))
    edit_distance = Enum.sum(Enum.map(cases, & &1.edit_distance))
    latencies = cases |> Enum.map(& &1.elapsed_ms) |> Enum.sort()

    %{
      samples: length(cases),
      exact_count: exact_count,
      exact_accuracy: ratio(exact_count, length(cases)),
      character_error_rate: ratio(edit_distance, reference_characters),
      mean_latency_ms: mean(latencies),
      p50_latency_ms: percentile(latencies, 0.50),
      p95_latency_ms: percentile(latencies, 0.95),
      languages: language_summaries(cases)
    }
  end

  defp parse_options(opts) do
    if Keyword.get(opts, :help, false) do
      {:help, usage()}
    else
      values = %{
        corpus: opts[:corpus],
        prefix_artifact: opts[:prefix_artifact],
        tail_artifact: opts[:tail_artifact],
        backend: Keyword.get(opts, :backend, "exla:rocm"),
        split: Keyword.get(opts, :split, "test"),
        languages: parse_languages(opts[:languages]),
        per_language: Keyword.get(opts, :per_language, 1),
        seed: Keyword.get(opts, :seed, 42),
        seconds: Keyword.get(opts, :seconds, 3.0),
        max_new_tokens: Keyword.get(opts, :max_new_tokens, 8),
        execution: parse_execution(Keyword.get(opts, :execution, "composed")),
        prompt:
          Keyword.get(
            opts,
            :prompt,
            "Transcribe the single spoken word exactly as spoken. Use the speaker's language and native writing system. Spell out numbers as words. Return only the word."
          ),
        output: opts[:output],
        baseline: opts[:baseline]
      }

      with :ok <- required_directory(values.corpus, "--corpus"),
           :ok <- required_directory(values.prefix_artifact, "--prefix-artifact"),
           :ok <- required_directory(values.tail_artifact, "--tail-artifact"),
           :ok <- required_path(values.output, "--output"),
           :ok <- optional_file(values.baseline, "--baseline"),
           :ok <- positive_integer(values.per_language, "--per-language"),
           :ok <- non_negative_integer(values.seed, "--seed"),
           :ok <- positive_integer(values.max_new_tokens, "--max-new-tokens"),
           :ok <- positive_number(values.seconds, "--seconds"),
           :ok <- valid_execution(values.execution),
           :ok <- ffmpeg_available() do
        {:ok, values}
      end
    end
  end

  defp run!(opts) do
    cases = load_cases(opts.corpus, opts.split, opts.languages, opts.per_language, opts.seed)

    if cases == [] do
      abort("no labeled clips found for split #{inspect(opts.split)}")
    end

    {:ok, backend} = Runtime.resolve_backend(opts.backend)

    prefix =
      timed_load(
        "prefix_artifact_load",
        fn -> DecoderBlockArtifact.load_prefix!(opts.prefix_artifact, backend) end
      )

    tail =
      timed_load(
        "tail_artifact_load",
        fn -> DecoderBlockArtifact.load_tail!(opts.tail_artifact, backend) end
      )

    pipeline = DecoderBlockArtifact.build_split_pipeline!(prefix, tail, backend)

    IO.puts(
      Jason.encode!(%{
        event: "single_word_ready",
        samples: length(cases),
        languages: cases |> Enum.map(& &1.language) |> Enum.uniq() |> length(),
        execution: opts.execution,
        seconds: opts.seconds
      })
    )

    first_input = cases |> hd() |> build_input(opts.seconds, opts.prompt)

    case DecoderPipeline.generate(pipeline, first_input,
           max_new_tokens: opts.max_new_tokens,
           execution: opts.execution
         ) do
      {:ok, _output} -> :ok
      {:error, reason} -> abort("warmup failed: #{reason}")
    end

    results = Enum.map(cases, &run_case!(pipeline, &1, opts))
    summary = summarize(results)
    comparison = compare_baseline(results, summary, opts.baseline)

    report = %{
      version: 1,
      corpus: Path.expand(opts.corpus),
      split: opts.split,
      per_language: opts.per_language,
      seed: opts.seed,
      seconds: opts.seconds,
      max_new_tokens: opts.max_new_tokens,
      execution: opts.execution,
      prompt: opts.prompt,
      prefix_artifact: Path.expand(opts.prefix_artifact),
      tail_artifact: Path.expand(opts.tail_artifact),
      summary: summary,
      comparison: comparison,
      cases: results
    }

    File.mkdir_p!(Path.dirname(Path.expand(opts.output)))
    File.write!(opts.output, Jason.encode!(report, pretty: true) <> "\n")

    IO.puts(
      Jason.encode!(%{
        event: "single_word_summary",
        summary: summary,
        comparison: comparison,
        output: Path.expand(opts.output)
      })
    )
  end

  defp run_case!(pipeline, sample, opts) do
    input = build_input(sample, opts.seconds, opts.prompt)

    {elapsed_us, result} =
      :timer.tc(fn ->
        DecoderPipeline.generate(pipeline, input,
          max_new_tokens: opts.max_new_tokens,
          execution: opts.execution
        )
      end)

    output =
      case result do
        {:ok, output} -> output
        {:error, reason} -> abort("generation failed for #{sample.key}: #{reason}")
      end

    normalized_expected = normalize(sample.expected)
    normalized_actual = normalize(output.text)
    distance = edit_distance(normalized_expected, normalized_actual)

    result = %{
      key: sample.key,
      language: sample.language,
      path: sample.relative_path,
      expected: sample.expected,
      actual: output.text,
      normalized_expected: normalized_expected,
      normalized_actual: normalized_actual,
      exact: normalized_expected == normalized_actual,
      edit_distance: distance,
      elapsed_ms: div(elapsed_us, 1_000),
      token_ids: output.token_ids
    }

    IO.puts(Jason.encode!(Map.put(result, :event, "single_word_case")))
    result
  end

  defp build_input(sample, seconds, prompt) do
    samples = decode_fixed_audio!(sample.path, seconds)
    Input.build(samples, prompt: prompt)
  end

  defp decode_fixed_audio!(path, seconds) do
    duration = :erlang.float_to_binary(seconds / 1, decimals: 3)

    args = [
      "-v",
      "error",
      "-i",
      path,
      "-af",
      "apad",
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
      {audio, 0} -> Audio.binary_to_f32_samples(audio)
      {message, status} -> abort("ffmpeg failed for #{path} (#{status}): #{message}")
    end
  end

  defp corpus_languages(corpus) do
    corpus
    |> File.ls!()
    |> Enum.filter(&File.dir?(Path.join(corpus, &1)))
    |> Enum.sort()
  end

  defp language_cases(corpus, language, split) do
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

  defp compare_baseline(_results, _summary, nil), do: nil

  defp compare_baseline(results, summary, path) do
    baseline = path |> File.read!() |> Jason.decode!()
    baseline_cases = Map.new(baseline["cases"], &{&1["key"], &1})

    paired =
      Enum.flat_map(results, fn result ->
        case Map.fetch(baseline_cases, result.key) do
          {:ok, baseline_case} -> [{baseline_case, result}]
          :error -> []
        end
      end)

    changed =
      Enum.count(paired, fn {before, candidate} ->
        before["normalized_actual"] != candidate.normalized_actual
      end)

    lost =
      Enum.count(paired, fn {before, candidate} -> before["exact"] && not candidate.exact end)

    gained =
      Enum.count(paired, fn {before, candidate} -> not before["exact"] && candidate.exact end)

    baseline_summary = baseline["summary"]

    %{
      paired_samples: length(paired),
      changed_outputs: changed,
      lost_exact_matches: lost,
      gained_exact_matches: gained,
      exact_accuracy_delta: summary.exact_accuracy - baseline_summary["exact_accuracy"],
      character_error_rate_delta:
        summary.character_error_rate - baseline_summary["character_error_rate"],
      mean_latency_delta_ms: summary.mean_latency_ms - baseline_summary["mean_latency_ms"],
      speedup: ratio(baseline_summary["mean_latency_ms"], summary.mean_latency_ms)
    }
  end

  defp language_summaries(cases) do
    cases
    |> Enum.group_by(& &1.language)
    |> Map.new(fn {language, language_cases} ->
      exact = Enum.count(language_cases, & &1.exact)

      {language,
       %{
         samples: length(language_cases),
         exact_count: exact,
         exact_accuracy: ratio(exact, length(language_cases))
       }}
    end)
  end

  defp edit_distance(expected, actual) do
    left = String.graphemes(expected)
    right = String.graphemes(actual)
    initial = Enum.to_list(0..length(right))

    left
    |> Enum.with_index(1)
    |> Enum.reduce(initial, fn {left_character, row_index}, previous ->
      right
      |> Enum.with_index(1)
      |> Enum.reduce([row_index], fn {right_character, column_index}, current_reversed ->
        insertion = hd(current_reversed) + 1
        deletion = Enum.at(previous, column_index) + 1

        substitution =
          Enum.at(previous, column_index - 1) +
            if(left_character == right_character, do: 0, else: 1)

        [min(insertion, min(deletion, substitution)) | current_reversed]
      end)
      |> Enum.reverse()
    end)
    |> List.last()
  end

  defp grapheme_length(text), do: text |> String.graphemes() |> length()
  defp ratio(_numerator, 0), do: 0.0
  defp ratio(numerator, denominator), do: numerator / denominator
  defp mean([]), do: 0.0
  defp mean(values), do: Enum.sum(values) / length(values)
  defp percentile([], _quantile), do: 0

  defp percentile(values, quantile) do
    Enum.at(values, ceil(quantile * length(values)) - 1)
  end

  defp timed_load(event, fun) do
    {elapsed_us, value} = :timer.tc(fun)
    IO.puts(Jason.encode!(%{event: event, elapsed_ms: div(elapsed_us, 1_000)}))
    value
  end

  defp parse_languages(nil), do: nil
  defp parse_languages(value), do: String.split(value, ",", trim: true)
  defp parse_execution("composed"), do: :composed
  defp parse_execution("split"), do: :split
  defp parse_execution(value), do: {:invalid, value}

  defp required_directory(nil, option), do: {:error, "#{option} PATH is required"}

  defp required_directory(path, _option) when is_binary(path) and path != "" do
    if File.dir?(path), do: :ok, else: {:error, "directory does not exist: #{path}"}
  end

  defp required_path(nil, option), do: {:error, "#{option} PATH is required"}
  defp required_path("", option), do: {:error, "#{option} PATH is required"}
  defp required_path(_path, _option), do: :ok

  defp optional_file(nil, _option), do: :ok

  defp optional_file(path, _option) do
    if File.regular?(path), do: :ok, else: {:error, "baseline file does not exist: #{path}"}
  end

  defp positive_integer(value, _option) when is_integer(value) and value > 0, do: :ok
  defp positive_integer(_value, option), do: {:error, "#{option} must be a positive integer"}
  defp non_negative_integer(value, _option) when is_integer(value) and value >= 0, do: :ok
  defp non_negative_integer(_value, option), do: {:error, "#{option} must be non-negative"}
  defp positive_number(value, _option) when is_number(value) and value > 0, do: :ok
  defp positive_number(_value, option), do: {:error, "#{option} must be positive"}
  defp valid_execution(execution) when execution in [:composed, :split], do: :ok

  defp valid_execution({:invalid, value}),
    do: {:error, "--execution must be composed or split, got: #{inspect(value)}"}

  defp ffmpeg_available do
    case System.find_executable("ffmpeg") do
      nil -> {:error, "ffmpeg is required to decode corpus MP3 files"}
      _path -> :ok
    end
  end

  defp abort(message) do
    IO.puts(:stderr, "error: #{message}")
    System.halt(1)
  end

  defp usage do
    """
    Usage: single_word_bench [options]

      --corpus PATH             Common Voice single-word corpus root
      --prefix-artifact PATH    independently saved decoder prefix
      --tail-artifact PATH      independently saved decoder tail
      --output PATH             JSON result and future comparison baseline
      --baseline PATH           optional prior JSON result to compare
      --backend BACKEND         default exla:rocm
      --split NAME              TSV split, default test
      --languages LIST          comma-separated language codes; default all
      --per-language COUNT      deterministic samples per language, default 1
      --seed INTEGER            repeatable random sample seed, default 42
      --seconds SECONDS         pad/truncate every clip to this shape, default 3
      --max-new-tokens COUNT    default 8
      --execution MODE          composed (default) or split
      --prompt TEXT             fixed transcription instruction for every clip
    """
  end
end

defmodule Gemma4MicTranscribe.SingleWordBenchmark.Escript do
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

    Gemma4MicTranscribe.SingleWordBenchmark.main(argv)
  end
end
