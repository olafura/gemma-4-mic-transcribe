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
    assert config.system_message_source == :none
    assert config.speech_gate
    assert config.min_speech_seconds == 0.25
  end

  test "parses list models" do
    assert {:list_models, output} = CLI.parse(["--list-models"])
    assert output =~ "google/gemma-4-12B-it"
    assert output =~ "qat-q4_0-gguf"
    assert output =~ "qat-w4a16-ct"
    assert output =~ "compressed_tensors"
    assert output =~ "compressed-tensors unpacking"
    assert output =~ "llama.cpp"
  end

  test "parses debug logging flag" do
    assert {:ok, %RunConfig{} = config} = CLI.parse(["--debug"])
    assert config.debug
  end

  test "parses system message file" do
    path = Path.join(System.tmp_dir!(), "gemma-system-message-#{System.unique_integer()}.txt")
    on_exit(fn -> File.rm(path) end)

    File.write!(path, "  Transcribe only.  \n")

    assert {:ok, %RunConfig{} = config} = CLI.parse(["--system-message-file", path])
    assert config.system_message == "Transcribe only."
    assert config.system_message_source == {:system_message_file, Path.expand(path)}
  end

  test "tracks inline system message source" do
    assert {:ok, %RunConfig{} = config} = CLI.parse(["--system-message", "Transcribe only."])
    assert config.system_message == "Transcribe only."
    assert config.system_message_source == :system_message
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

  test "parses streaming controls" do
    assert {:ok, %RunConfig{} = config} =
             CLI.parse([
               "--stream-wav",
               "--output",
               "jsonl",
               "--chunk-ms",
               "50",
               "--speech-start-ms",
               "80",
               "--speech-end-silence-ms",
               "300",
               "--min-utterance-ms",
               "200",
               "--max-utterance-ms",
               "5000",
               "--partial-interval-ms",
               "750",
               "--no-partials",
               "--tts-text",
               "hello",
               "--tts-timestamp-ms",
               "1200"
             ])

    assert config.stream_wav
    refute config.realtime
    assert config.output == "jsonl"
    assert config.chunk_ms == 50.0
    assert config.speech_start_ms == 80.0
    assert config.speech_end_silence_ms == 300.0
    assert config.min_utterance_ms == 200.0
    assert config.max_utterance_ms == 5000.0
    assert config.partial_interval_ms == 750.0
    refute config.partials
    assert config.tts_text == "hello"
    assert config.tts_timestamp_ms == 1200.0
  end

  test "parses realtime streaming benchmark flag" do
    assert {:ok, %RunConfig{realtime: true}} = CLI.parse(["--stream-wav", "--realtime"])
  end

  test "parses fused FFN decode flag" do
    assert {:ok, %RunConfig{fused_ffn: true}} = CLI.parse(["--fused-ffn"])
  end

  test "parses transcript self-review flag" do
    assert {:ok, %RunConfig{self_review: true}} = CLI.parse(["--self-review"])
  end

  test "parses E4B cascade controls" do
    assert {:ok, %RunConfig{} = config} =
             CLI.parse([
               "--e4b-cascade",
               "--cascade-min-chars-per-second",
               "1.5",
               "--cascade-min-logit-margin",
               "0.125"
             ])

    assert config.e4b_cascade
    assert config.cascade_min_chars_per_second == 1.5
    assert config.cascade_min_logit_margin == 0.125
  end

  test "rejects incremental prefill with the initial E4B cascade" do
    assert {:error, message} = CLI.parse(["--e4b-cascade", "--incremental-prefill"])
    assert message =~ "does not support"
  end

  test "requires packed weights for fused FFN decode" do
    assert {:error, message} = CLI.parse(["--fused-ffn", "--weights", "bf16"])
    assert message =~ "packed or hybrid"
  end

  test "validates output format" do
    assert {:error, message} = CLI.parse(["--output", "xml"])
    assert message =~ "--output"
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
