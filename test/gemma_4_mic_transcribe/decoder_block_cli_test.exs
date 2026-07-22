defmodule Gemma4MicTranscribe.DecoderBlockCLITest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.DecoderBlockCLI

  test "parses block extraction independently from block execution" do
    assert {:ok, :extract, extract} =
             DecoderBlockCLI.parse([
               "extract",
               "--artifact",
               "artifacts/layer-45",
               "--layer",
               "45",
               "--sequence-length",
               "4"
             ])

    assert extract.backend == "torchx:cpu"
    assert extract.layer == 45
    assert extract.sequence_length == 4

    path = Path.join(System.tmp_dir!(), "decoder-block-cli-#{System.unique_integer([:positive])}")
    File.mkdir_p!(path)
    on_exit(fn -> File.rm_rf(path) end)

    assert {:ok, :run, run} =
             DecoderBlockCLI.parse([
               "run",
               "--artifact",
               path,
               "--backend",
               "exla:rocm",
               "--output",
               Path.join(path, "output.safetensors")
             ])

    assert run.backend == "exla:rocm"
    assert run.runs == 2
    assert run.output == Path.join(path, "output.safetensors")
  end

  test "requires an artifact path" do
    assert {:error, "--artifact PATH is required"} = DecoderBlockCLI.parse(["extract"])
  end

  test "parses tail extraction and real prefix capture" do
    assert {:ok, :extract_tail, tail} =
             DecoderBlockCLI.parse([
               "extract-tail",
               "--artifact",
               "artifacts/tail-45-47",
               "--tail-start",
               "45"
             ])

    assert tail.backend == "torchx:cpu"
    assert tail.tail_start == 45

    pipeline = Path.join(System.tmp_dir!(), "pipeline-cli-#{System.unique_integer([:positive])}")
    File.mkdir_p!(pipeline)
    output = Path.join(pipeline, "prefix.safetensors")
    on_exit(fn -> File.rm_rf(pipeline) end)

    assert {:ok, :capture_prefix, capture} =
             DecoderBlockCLI.parse([
               "capture-prefix",
               "--pipeline-artifact",
               pipeline,
               "--output",
               output,
               "--wav",
               "journal1.wav"
             ])

    assert capture.output == output
    assert capture.seconds == 5.0
  end
end
