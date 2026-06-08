defmodule Gemma4MicTranscribe.MixProject do
  use Mix.Project

  def project do
    [
      app: :gemma_4_mic_transcribe,
      version: "0.1.0",
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      escript: [main_module: Gemma4MicTranscribe]
    ]
  end

  def application do
    [
      extra_applications: [:logger, :ssl, :inets, :eex]
    ]
  end

  defp deps do
    [
      {:boombox, "~> 0.2.11", runtime: false},
      {:bumblebee, "~> 0.7.0"},
      {:ex_libsrt, path: "vendor/ex_libsrt", override: true},
      {:jason, "~> 1.4"},
      {:nx, "~> 0.12.0"},
      {:torchx, "~> 0.12.0"}
    ]
  end
end
