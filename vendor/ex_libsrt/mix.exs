defmodule ExLibSRT.MixProject do
  use Mix.Project

  def project do
    [
      app: :ex_libsrt,
      version: "0.1.7-stub",
      elixir: "~> 1.19",
      deps: []
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end
end
