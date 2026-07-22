defmodule Gemma4MicTranscribe.SingleWordBenchmarkTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.SingleWordBenchmark

  test "loads a deterministic balanced sample from Common Voice TSV files" do
    root =
      Path.join(System.tmp_dir!(), "single-word-corpus-#{System.unique_integer([:positive])}")

    on_exit(fn -> File.rm_rf(root) end)

    for language <- ["en", "ja"] do
      File.mkdir_p!(Path.join([root, language, "clips"]))

      rows =
        case language do
          "en" -> [{"b.mp3", "yes"}, {"a.mp3", "no"}]
          "ja" -> [{"c.mp3", "はい"}]
        end

      Enum.each(rows, fn {path, _sentence} ->
        File.write!(Path.join([root, language, "clips", path]), "fixture")
      end)

      body =
        ["client_id\tpath\tsentence"] ++
          Enum.map(rows, fn {path, sentence} -> "client\t#{path}\t#{sentence}" end)

      File.write!(Path.join([root, language, "test.tsv"]), Enum.join(body, "\n") <> "\n")
    end

    cases = SingleWordBenchmark.load_cases(root, "test", nil, 1)

    assert Enum.map(cases, & &1.language) == ["en", "ja"]
    assert Enum.all?(cases, &File.regular?(&1.path))
    assert SingleWordBenchmark.load_cases(root, "test", nil, 1) == cases
  end

  test "normalizes transcript formatting without changing letters" do
    assert SingleWordBenchmark.normalize("  HéLLo! ") == "héllo"
    assert SingleWordBenchmark.normalize("你好。") == "你好"
  end

  test "summarizes exact match, character errors, and latency percentiles" do
    cases = [
      %{
        language: "en",
        exact: true,
        normalized_expected: "yes",
        edit_distance: 0,
        elapsed_ms: 10
      },
      %{
        language: "en",
        exact: false,
        normalized_expected: "no",
        edit_distance: 1,
        elapsed_ms: 30
      }
    ]

    summary = SingleWordBenchmark.summarize(cases)

    assert summary.samples == 2
    assert summary.exact_accuracy == 0.5
    assert summary.character_error_rate == 0.2
    assert summary.mean_latency_ms == 20.0
    assert summary.p50_latency_ms == 10
    assert summary.p95_latency_ms == 30
  end

  test "accepts a catalog model without extracted artifacts" do
    root =
      Path.join(System.tmp_dir!(), "single-word-corpus-#{System.unique_integer([:positive])}")

    File.mkdir_p!(root)
    on_exit(fn -> File.rm_rf(root) end)

    assert {:ok, opts} =
             SingleWordBenchmark.parse([
               "--corpus",
               root,
               "--model-name",
               "gemma4-e4b",
               "--output",
               Path.join(root, "result.json")
             ])

    assert opts.model_name == "gemma4-e4b"
    assert opts.prefix_artifact == nil
    assert opts.tail_artifact == nil
  end
end
