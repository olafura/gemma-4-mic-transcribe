defmodule Gemma4MicTranscribe.ExpertCLI do
  @moduledoc false

  alias Gemma4MicTranscribe.Gemma4.ExpertArtifact
  alias Gemma4MicTranscribe.Gemma4.ExtractedExpert
  alias Gemma4MicTranscribe.Gemma4Unified.Runtime

  @switches [
    artifact: :string,
    repo: :string,
    revision: :string,
    layer: :integer,
    expert: :integer,
    backend: :string,
    tokens: :integer,
    runs: :integer,
    input_value: :float,
    help: :boolean
  ]

  def main(argv) do
    case parse(argv) do
      {:help, text} ->
        IO.puts(text)
        0

      {:extract, opts} ->
        manifest =
          ExpertArtifact.extract!(opts.artifact,
            repo: opts.repo,
            revision: opts.revision,
            layer: opts.layer,
            expert: opts.expert
          )

        print_manifest("expert_extracted", opts.artifact, manifest)
        0

      {:inspect, artifact} ->
        print_manifest("expert_artifact", artifact, ExpertArtifact.read_manifest!(artifact))
        0

      {:run, opts} ->
        {:ok, backend} = Runtime.resolve_backend(opts.backend)
        expert = ExtractedExpert.load!(opts.artifact, backend)

        input =
          Nx.broadcast(
            Nx.tensor(opts.input_value, type: :f32),
            {opts.tokens, expert.manifest.input_size}
          )

        Enum.each(1..3, fn _ -> :ok = ExtractedExpert.warmup(expert, opts.tokens) end)

        {times, output} =
          Enum.map_reduce(1..opts.runs, nil, fn _run, _last_output ->
            started_at = System.monotonic_time(:microsecond)

            output =
              expert
              |> ExtractedExpert.run(input)
              |> Nx.backend_copy(Nx.BinaryBackend)

            elapsed = System.monotonic_time(:microsecond) - started_at
            {elapsed, output}
          end)

        output_f32 = Nx.as_type(output, :f32)
        sorted_times = Enum.sort(times)

        IO.puts(
          Jason.encode!(%{
            event: "expert_run",
            artifact: Path.expand(opts.artifact),
            backend: opts.backend,
            layer: expert.manifest.layer_index,
            expert: expert.manifest.expert_index,
            tokens: opts.tokens,
            runs: opts.runs,
            mean_us: Enum.sum(times) / length(times),
            min_us: Enum.min(times),
            median_us: percentile(sorted_times, 0.5),
            p95_us: percentile(sorted_times, 0.95),
            output_shape: Tuple.to_list(Nx.shape(output)),
            output_mean_abs: output_f32 |> Nx.abs() |> Nx.mean() |> Nx.to_number(),
            output_max_abs: output_f32 |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
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
    with {:ok, opts} <- parse_options(argv),
         :ok <- require_artifact(opts),
         :ok <- non_negative(opts[:layer] || 0, "--layer"),
         :ok <- non_negative(opts[:expert] || 0, "--expert") do
      if opts[:help] do
        {:help, usage()}
      else
        {:extract,
         %{
           artifact: opts[:artifact],
           repo: opts[:repo] || "google/gemma-4-26B-A4B-it",
           revision: opts[:revision] || "main",
           layer: opts[:layer] || 0,
           expert: opts[:expert] || 0
         }}
      end
    end
  end

  def parse(["inspect" | argv]) do
    with {:ok, opts} <- parse_options(argv),
         :ok <- require_artifact(opts) do
      if opts[:help], do: {:help, usage()}, else: {:inspect, opts[:artifact]}
    end
  end

  def parse(["run" | argv]) do
    with {:ok, opts} <- parse_options(argv),
         :ok <- require_artifact(opts),
         :ok <- positive(opts[:tokens] || 1, "--tokens"),
         :ok <- positive(opts[:runs] || 3, "--runs") do
      if opts[:help] do
        {:help, usage()}
      else
        {:run,
         %{
           artifact: opts[:artifact],
           backend: opts[:backend] || "exla:rocm",
           tokens: opts[:tokens] || 1,
           runs: opts[:runs] || 3,
           input_value: opts[:input_value] || 0.01
         }}
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

  defp require_artifact(opts) do
    if is_binary(opts[:artifact]), do: :ok, else: {:error, "--artifact PATH is required"}
  end

  defp positive(value, _name) when is_integer(value) and value > 0, do: :ok
  defp positive(_value, name), do: {:error, "#{name} must be positive"}
  defp non_negative(value, _name) when is_integer(value) and value >= 0, do: :ok
  defp non_negative(_value, name), do: {:error, "#{name} must be non-negative"}

  defp percentile(sorted_values, quantile) do
    index = ceil((length(sorted_values) - 1) * quantile)
    Enum.at(sorted_values, index)
  end

  defp print_manifest(event, artifact, manifest) do
    IO.puts(
      Jason.encode!(%{
        event: event,
        artifact: Path.expand(artifact),
        source_repo: manifest.source_repo,
        layer: manifest.layer_index,
        expert: manifest.expert_index,
        input_size: manifest.input_size,
        intermediate_size: manifest.intermediate_size,
        parameter_count: manifest.parameter_count,
        parameter_bytes: div(manifest.parameter_count * elem(manifest.parameter_type, 1), 8),
        downloaded_bytes: manifest.downloaded_bytes,
        source_checkpoint_bytes: manifest.source_checkpoint_bytes
      })
    )
  end

  defp usage do
    """
    Usage:
      expert_tool extract --artifact PATH [options]
      expert_tool inspect --artifact PATH
      expert_tool run --artifact PATH [options]

    extract range-downloads one routed expert from Gemma 4 26B-A4B. It does
    not download or save the complete checkpoint.

      --artifact PATH       New or existing standalone artifact directory
      --repo REPOSITORY     Default google/gemma-4-26B-A4B-it
      --revision REVISION   Default main
      --layer INDEX         MoE layer, default 0
      --expert INDEX        Routed expert, default 0

    run executes only the extracted gated FFN over synthetic hidden states.
    Router weighting, the shared expert, norms, and residual are not included.

      --backend BACKEND     Default exla:rocm
      --tokens N            Input rows, default 1
      --runs N              Timed runs after warmup, default 3
      --input-value FLOAT   Value in every input cell, default 0.01
    """
  end

  defmodule Escript do
    @moduledoc false

    def main(["run" | _argv]) do
      IO.puts(
        :stderr,
        "error: native expert execution requires real application priv paths; " <>
          "use `mix gemma.expert run ...`"
      )

      System.halt(1)
    end

    def main(argv) do
      {:ok, _} = Application.ensure_all_started(:gemma_4_mic_transcribe)
      System.halt(Gemma4MicTranscribe.ExpertCLI.main(argv))
    end
  end
end
