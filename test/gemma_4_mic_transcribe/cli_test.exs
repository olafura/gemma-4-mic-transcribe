defmodule Gemma4MicTranscribe.CLITest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.CLI
  alias Gemma4MicTranscribe.CLI.RunConfig

  test "parses defaults" do
    assert {:ok, %RunConfig{} = config} = CLI.parse([])
    assert config.window_seconds == 5.0
    assert config.stride_seconds == 2.5
    assert config.sample_rate == 16_000
    assert config.skip_windows == 0
    assert config.model_name == "google/gemma-4-12B-it"
    assert config.backend == "torchx"
    assert config.speech_gate
    assert config.min_speech_seconds == 0.25
  end

  test "parses list models" do
    assert {:list_models, output} = CLI.parse(["--list-models"])
    assert output =~ "google/gemma-4-12B-it"
    assert output =~ "qat-q4_0-gguf"
  end

  test "parses debug logging flag" do
    assert {:ok, %RunConfig{} = config} = CLI.parse(["--debug"])
    assert config.debug
  end

  test "parses speech gate controls" do
    assert {:ok, %RunConfig{} = config} =
             CLI.parse([
               "--no-speech-gate",
               "--min-speech-seconds",
               "0.5",
               "--speech-threshold",
               "0.02",
               "--speech-min-active-ratio",
               "0.4",
               "--speech-max-zero-crossing-rate",
               "0.25"
             ])

    refute config.speech_gate
    assert config.min_speech_seconds == 0.5
    assert config.speech_threshold == 0.02
    assert config.speech_min_active_ratio == 0.4
    assert config.speech_max_zero_crossing_rate == 0.25
  end

  test "parses explicit Torchx CUDA backend" do
    assert {:ok, %RunConfig{} = config} = CLI.parse(["--backend", "torchx:cuda"])
    assert config.backend == "torchx:cuda"
  end

  test "parses explicit EXLA ROCm backend" do
    assert {:ok, %RunConfig{} = config} = CLI.parse(["--backend", "exla:rocm"])
    assert config.backend == "exla:rocm"
  end

  test "rejects removed litert options" do
    assert {:error, message} = CLI.parse(["--backend", "gpu", "--audio-backend", "gpu"])
    assert message =~ "--audio-backend"
  end

  test "validates positive window" do
    assert {:error, message} = CLI.parse(["--window-seconds", "0"])
    assert message =~ "--window-seconds"
  end
end
