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
               "--runs",
               "3"
             ])

    assert opts.backend == "exla:rocm"
    assert opts.tail_start == 45
    assert opts.runs == 3
    assert opts.max_new_tokens == 3
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
end
