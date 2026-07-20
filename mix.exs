defmodule Gemma4MicTranscribe.MixProject do
  use Mix.Project

  def project do
    [
      app: :gemma_4_mic_transcribe,
      version: "0.1.0",
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger, :ssl, :inets, :eex, :runtime_tools]
    ]
  end

  defp deps do
    [
      {:boombox, "~> 0.2.11", runtime: false},
      {:bumblebee, "~> 0.7.0"},
      {:ex_webrtc, "~> 0.15.0", runtime: false},
      {:exla, path: "vendor/exla", override: true, runtime: false},
      {:ex_libsrt, path: "vendor/ex_libsrt", override: true},
      {:jason, "~> 1.4"},
      # override: bumblebee 0.7.0 (latest) pins nx ~> 0.12.0, but nx 0.13
      # works with it and is required by exla/torchx 0.13
      {:nx, "~> 0.13.0", override: true},
      # override: ex_hls/membrane_webrtc_plugin pin req 0.5.x, but the fixes
      # for CVE-2026-49755 and the multipart injection advisory are 0.6-only
      {:req, "~> 0.6.3", override: true},
      {:torchx, "~> 0.13.0"},
      {:xla, path: "vendor/xla", override: true, runtime: false}
    ]
  end
end
