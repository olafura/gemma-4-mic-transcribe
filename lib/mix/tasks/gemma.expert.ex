defmodule Mix.Tasks.Gemma.Expert do
  @shortdoc "Runs the extracted Gemma 4 expert tool with application paths"

  @moduledoc """
  Runs the expert artifact tool as a started Mix application.

  This is the native-backend entrypoint. Unlike an escript archive, a started
  application gives EXLA a real application directory, so its NIF resolves
  through `:code.priv_dir(:exla)` as intended.
  """

  use Mix.Task

  @requirements ["app.start"]

  @impl Mix.Task
  def run(argv) do
    case Gemma4MicTranscribe.ExpertCLI.main(argv) do
      0 -> :ok
      status -> Mix.raise("expert tool exited with status #{status}")
    end
  end
end
