defmodule Gemma4MicTranscribe.HandoffProbeCLI do
  @moduledoc false

  alias Gemma4MicTranscribe.HandoffProbe.Artifact

  @switches [artifact: :string, repo: :string, revision: :string, help: :boolean]

  def main(argv) do
    case parse(argv) do
      {:help, text} ->
        IO.puts(text)
        0

      {:extract, opts} ->
        manifest =
          Artifact.extract!(opts.artifact,
            repo: opts.repo,
            revision: opts.revision
          )

        IO.puts(
          Jason.encode!(%{
            event: "handoff_probe_extracted",
            artifact: Path.expand(opts.artifact),
            source_repo: manifest.source_repo,
            probe_layer: manifest.probe_layer,
            parameter_count: manifest.parameter_count,
            downloaded_bytes: manifest.downloaded_bytes
          })
        )

        0

      {:inspect, artifact} ->
        manifest = Artifact.read_manifest!(artifact)

        IO.puts(
          Jason.encode!(%{
            event: "handoff_probe_artifact",
            artifact: Path.expand(artifact),
            source_repo: manifest.source_repo,
            probe_layer: manifest.probe_layer,
            feature_size: manifest.feature_size,
            projection_size: manifest.projection_size,
            max_tokens: manifest.max_tokens,
            parameter_count: manifest.parameter_count,
            downloaded_bytes: manifest.downloaded_bytes
          })
        )

        0

      {:error, reason} ->
        IO.puts(:stderr, "error: #{reason}")
        1
    end
  rescue
    exception ->
      IO.puts(:stderr, "error: #{Exception.message(exception)}")
      1
  end

  def parse(["extract" | argv]) do
    with {:ok, opts} <- parse_options(argv) do
      cond do
        opts[:help] ->
          {:help, usage()}

        is_binary(opts[:artifact]) ->
          {:extract,
           %{
             artifact: opts[:artifact],
             repo: opts[:repo] || "Cactus-Compute/gemma-4-e2b-it-hybrid",
             revision: opts[:revision] || "main"
           }}

        true ->
          {:error, "--artifact PATH is required"}
      end
    end
  end

  def parse(["inspect" | argv]) do
    with {:ok, opts} <- parse_options(argv) do
      cond do
        opts[:help] -> {:help, usage()}
        is_binary(opts[:artifact]) -> {:inspect, opts[:artifact]}
        true -> {:error, "--artifact PATH is required"}
      end
    end
  end

  def parse(["--help"]), do: {:help, usage()}
  def parse(["-h"]), do: {:help, usage()}
  def parse([]), do: {:help, usage()}
  def parse([command | _]), do: {:error, "unknown command #{inspect(command)}"}

  defp parse_options(argv) do
    case OptionParser.parse(argv, strict: @switches, aliases: [h: :help]) do
      {opts, [], []} -> {:ok, opts}
      {_opts, args, []} -> {:error, "unexpected arguments: #{Enum.join(args, " ")}"}
      {_opts, _args, invalid} -> {:error, "invalid options: #{inspect(invalid)}"}
    end
  end

  defp usage do
    """
    Usage:
      handoff_probe extract --artifact PATH [options]
      handoff_probe inspect --artifact PATH

    extract downloads only the Cactus handoff_probe.* safetensors byte range
    and creates a standalone artifact. It does not download or save the base
    Gemma model.

      --artifact PATH     New artifact directory
      --repo REPOSITORY   Default Cactus-Compute/gemma-4-e2b-it-hybrid
      --revision REVISION Default main
    """
  end

  defmodule Escript do
    @moduledoc false

    def main(argv) do
      {:ok, _} = Application.ensure_all_started(:gemma_4_mic_transcribe)
      System.halt(Gemma4MicTranscribe.HandoffProbeCLI.main(argv))
    end
  end
end
