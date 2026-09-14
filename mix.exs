defmodule Gemma4MicTranscribe.MixProject do
  use Mix.Project

  def project do
    [
      app: :gemma_4_mic_transcribe,
      version: "0.1.0",
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.target()),
      elixirc_options: elixirc_options(Mix.target()),
      escript: escript(),
      deps: deps()
    ]
  end

  # MIX_TARGET=language_id builds only the spoken-language detector: the
  # audio tower, the LanguageId modules and their CLI, on Torchx CPU. It is
  # what the Dockerfile compiles, so the image needs neither the vendored
  # EXLA/XLA build nor the WebRTC and Boombox stack. MIX_TARGET=language_id_cuda
  # is the same slice plus EXLA against the precompiled cuda12 XLA archive,
  # for running the detector on an Nvidia GPU (--backend exla:cuda).
  defp elixirc_paths(target) when target in [:language_id, :language_id_cuda] do
    [
      "lib/gemma_4_mic_transcribe/audio.ex",
      "lib/gemma_4_mic_transcribe/gemma4_e4b",
      "lib/gemma_4_mic_transcribe/language_id",
      "lib/gemma_4_mic_transcribe/language_id_cli.ex",
      "lib/gemma_4_mic_transcribe/rocm_preflight.ex"
    ]
  end

  defp elixirc_paths(_target), do: ["lib"]

  # Boombox is only reached from the transcription pipeline; the language-ID
  # build leaves it out, so its remote calls are expected to be undefined.
  defp elixirc_options(target) when target in [:language_id, :language_id_cuda],
    do: [no_warn_undefined: [Boombox]]
  defp elixirc_options(_target), do: []

  defp escript do
    {main_module, name} =
      case System.get_env("GEMMA4_ESCRIPT") do
        "decoder_block" ->
          {Gemma4MicTranscribe.DecoderBlockCLI.Escript, "decoder_block"}

        "single_word_bench" ->
          {Gemma4MicTranscribe.SingleWordBenchmark.Escript, "single_word_bench"}

        "handoff_probe" ->
          {Gemma4MicTranscribe.HandoffProbeCLI.Escript, "handoff_probe"}

        "expert" ->
          {Gemma4MicTranscribe.ExpertCLI.Escript, "expert_tool"}

        "language_id" ->
          {Gemma4MicTranscribe.LanguageIdCLI.Escript, "language_id"}

        _other ->
          {Gemma4MicTranscribe.DecoderPipelineBenchmark.Escript, "decoder_pipeline_bench"}
      end

    [
      main_module: main_module,
      name: name,
      app: nil,
      shebang:
        "#! /usr/bin/env -S XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer= --xla_gpu_enable_triton_gemm=false' escript\n"
    ]
  end

  def application do
    [
      extra_applications: [:logger, :ssl, :inets, :eex, :runtime_tools]
    ]
  end

  defp deps do
    [
      {:boombox, "~> 0.2.11", runtime: false, targets: [:host]},
      {:bumblebee, "~> 0.7.0"},
      {:ex_webrtc, "~> 0.15.0", runtime: false, targets: [:host]},
      {:explorer, "~> 0.12.0"},
      # override: ratio 4.0.1 (via membrane) declares decimal ~> 2.0 while
      # numbers 5.2.5 and explorer need ~> 3.x; ratio only pattern-matches the
      # unchanged %Decimal{} struct fields, so 3.x is fine
      {:decimal, "~> 3.1", override: true},
      {:exla, path: "vendor/exla", override: true, runtime: false, targets: [:host, :language_id_cuda]},
      {:ex_libsrt, path: "vendor/ex_libsrt", override: true, targets: [:host]},
      {:jason, "~> 1.4"},
      # override: bumblebee 0.7.0 (latest) pins nx ~> 0.12.0, but nx 0.13
      # works with it and is required by exla/torchx 0.13
      {:nx, "~> 0.13.0", override: true},
      # override: ex_hls/membrane_webrtc_plugin pin req 0.5.x, but the fixes
      # for CVE-2026-49755 and the multipart injection advisory are 0.6-only
      {:req, "~> 0.6.3", override: true},
      {:torchx, "~> 0.13.0"},
      {:xla, path: "vendor/xla", override: true, runtime: false, targets: [:host, :language_id_cuda]}
    ]
  end
end
