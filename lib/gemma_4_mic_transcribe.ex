defmodule Gemma4MicTranscribe do
  @moduledoc false

  def main(argv) do
    argv
    |> Gemma4MicTranscribe.CLI.main()
    |> System.halt()
  end
end
