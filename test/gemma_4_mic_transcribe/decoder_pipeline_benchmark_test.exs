defmodule Gemma4MicTranscribe.DecoderPipelineBenchmarkTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.DecoderPipelineBenchmark

  test "parses a reproducible GPU XLA benchmark" do
    assert {:ok, opts} =
             DecoderPipelineBenchmark.parse([
               "--wav",
               "journal1.wav",
               "--backend",
               "exla:rocm",
               "--tail-start",
               "45",
               "--transplant",
               "44:45,41:47",
               "--runs",
               "3"
             ])

    assert opts.backend == "exla:rocm"
    assert opts.tail_start == 45
    assert opts.transplants == [%{source: 44, target: 45}, %{source: 41, target: 47}]
    assert opts.runs == 3
    assert opts.max_new_tokens == 32
    assert opts.min_new_tokens == 0
    assert opts.seconds == 5.0
  end

  test "rejects malformed layer transplants" do
    assert {:error, message} =
             DecoderPipelineBenchmark.parse([
               "--wav",
               "journal1.wav",
               "--transplant",
               "45-46"
             ])

    assert message =~ "expected SOURCE:TARGET"
  end

  test "rejects invalid run counts" do
    assert {:error, "--runs must be positive"} =
             DecoderPipelineBenchmark.parse(["--wav", "journal1.wav", "--runs", "0"])
  end

  test "prints dedicated benchmark help" do
    assert {:help, usage} = DecoderPipelineBenchmark.parse(["--help"])
    assert usage =~ "decoder_pipeline_bench"
    assert usage =~ "--backend"
  end

  test "parses separate artifact extraction" do
    path = Path.join(System.tmp_dir!(), "new-decoder-artifact")
    File.rm_rf(path)

    assert {:ok, opts} =
             DecoderPipelineBenchmark.parse([
               "extract",
               "--artifact",
               path,
               "--wav",
               "journal1.wav",
               "--transplant",
               "44:45"
             ])

    assert opts.artifact == path
    assert opts.backend == "torchx:cpu"
    assert opts.transplants == [%{source: 44, target: 45}]
  end

  test "requires an existing artifact for separate execution" do
    assert {:error, message} =
             DecoderPipelineBenchmark.parse([
               "run",
               "--artifact",
               "/definitely/missing/decoder-artifact",
               "--wav",
               "journal1.wav"
             ])

    assert message =~ "artifact directory does not exist"
  end
end
