defmodule Mix.Tasks.Gemma.SystemOne do
  @shortdoc "Caches, trains, generates and regression-tests the System One expert"

  @moduledoc """
  Runs the System One tool as a started Mix application.

  Like `mix gemma.expert` this is the native-backend entrypoint: a started
  application gives EXLA a real application directory, so its NIF resolves
  through `:code.priv_dir(:exla)` instead of an escript's archive.

      mix gemma.system_one cache --input data/system-one/seed-pairs.jsonl \\
        --output data/system-one/prefix-cache --last-prompt-token-only

      mix gemma.system_one train --cache data/system-one/full-cache \\
        --output artifacts/system-one-expert

      mix gemma.system_one generate --input data/system-one/heldout.jsonl \\
        --output data/system-one/heldout-replies.jsonl \\
        --expert artifacts/system-one-expert

      mix gemma.system_one regress --expert artifacts/system-one-expert
  """

  use Mix.Task

  @requirements ["app.start"]

  @impl Mix.Task
  def run(argv) do
    case Gemma4MicTranscribe.SystemOneCLI.main(argv) do
      0 -> :ok
      status -> Mix.raise("system one tool exited with status #{status}")
    end
  end
end
