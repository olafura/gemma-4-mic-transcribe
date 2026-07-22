defmodule Gemma4MicTranscribe.DecoderPipelineBenchmark do
  @moduledoc false

  alias Gemma4MicTranscribe.Audio
  alias Gemma4MicTranscribe.Config
  alias Gemma4MicTranscribe.Gemma4.DecoderPipeline
  alias Gemma4MicTranscribe.Gemma4.DecoderPipelineArtifact
  alias Gemma4MicTranscribe.Gemma4Unified.Input
  alias Gemma4MicTranscribe.Gemma4Unified.Runtime

  @switches [
    wav: :string,
    backend: :string,
    model_name: :string,
    param_type: :string,
    tail_start: :integer,
    max_new_tokens: :integer,
    min_new_tokens: :integer,
    transplant: :string,
    runs: :integer,
    sample_rate: :integer,
    seconds: :float,
    debug: :boolean,
    help: :boolean
  ]

  @artifact_switches @switches ++ [artifact: :string]

  def main(argv) do
    {:ok, _started} = Application.ensure_all_started(:gemma_4_mic_transcribe)

    case argv do
      ["extract" | argv] -> artifact_main(:extract, argv)
      ["run" | argv] -> artifact_main(:run, argv)
      argv -> benchmark_main(argv)
    end
  end

  defp benchmark_main(argv) do
    case parse(argv) do
      {:ok, opts} -> run!(opts)
      {:help, usage} -> IO.puts(usage)
      {:error, message} -> abort(message)
    end
  end

  defp artifact_main(mode, argv) do
    case parse_artifact(mode, argv) do
      {:ok, opts} when mode == :extract -> extract_artifact!(opts)
      {:ok, opts} when mode == :run -> run_artifact!(opts)
      {:help, usage} -> IO.puts(usage)
      {:error, message} -> abort(message)
    end
  end

  def parse(["extract" | argv]), do: parse_artifact(:extract, argv)
  def parse(["run" | argv]), do: parse_artifact(:run, argv)

  def parse(argv) do
    case OptionParser.parse(argv, strict: @switches, aliases: [h: :help]) do
      {opts, [], []} -> parse_options(opts, [])
      {_opts, args, []} -> {:error, "unexpected arguments: #{Enum.join(args, " ")}"}
      {_opts, _args, invalid} -> {:error, "invalid options: #{inspect(invalid)}"}
    end
  end

  defp parse_artifact(mode, argv) do
    case OptionParser.parse(argv, strict: @artifact_switches, aliases: [h: :help]) do
      {opts, [], []} ->
        cond do
          Keyword.get(opts, :help, false) ->
            {:help, artifact_usage(mode)}

          is_nil(opts[:artifact]) ->
            {:error, "--artifact PATH is required"}

          mode == :run and not File.dir?(opts[:artifact]) ->
            {:error, "artifact directory does not exist: #{opts[:artifact]}"}

          true ->
            defaults = if mode == :extract, do: [backend: "torchx:cpu"], else: []

            with {:ok, values} <- parse_options(Keyword.delete(opts, :artifact), defaults) do
              {:ok, Map.put(values, :artifact, opts[:artifact])}
            end
        end

      {_opts, args, []} ->
        {:error, "unexpected arguments: #{Enum.join(args, " ")}"}

      {_opts, _args, invalid} ->
        {:error, "invalid options: #{inspect(invalid)}"}
    end
  end

  defp parse_options(opts, defaults) do
    if Keyword.get(opts, :help, false) do
      {:help, usage()}
    else
      with {:ok, transplants} <- parse_transplants(Keyword.get(opts, :transplant)),
           values = %{
             wav: Keyword.get(opts, :wav, "journal1.wav"),
             backend: Keyword.get(opts, :backend, defaults[:backend] || "exla:rocm"),
             model_name: Keyword.get(opts, :model_name, "google/gemma-4-12B-it"),
             param_type: Keyword.get(opts, :param_type, "bf16"),
             tail_start: Keyword.get(opts, :tail_start, 45),
             max_new_tokens: Keyword.get(opts, :max_new_tokens, 3),
             min_new_tokens: Keyword.get(opts, :min_new_tokens, 3),
             transplants: transplants,
             runs: Keyword.get(opts, :runs, 2),
             sample_rate: Keyword.get(opts, :sample_rate, 16_000),
             seconds: Keyword.get(opts, :seconds, 2.0),
             debug: Keyword.get(opts, :debug, false)
           },
           :ok <- positive(values.runs, "--runs"),
           :ok <- positive(values.sample_rate, "--sample-rate"),
           :ok <- positive(values.seconds, "--seconds"),
           :ok <- positive(values.max_new_tokens, "--max-new-tokens"),
           :ok <- non_negative(values.min_new_tokens, "--min-new-tokens"),
           :ok <- non_negative(values.tail_start, "--tail-start"),
           :ok <- regular_file(values.wav) do
        {:ok, values}
      end
    end
  end

  def run!(opts) do
    samples =
      opts.wav
      |> Audio.read_wav_samples!(opts.sample_rate)
      |> Enum.take(round(opts.sample_rate * opts.seconds))

    input = Input.build(samples, prompt: Config.default_prompt())

    runtime =
      timed!("load", fn ->
        Runtime.load(
          model_name: opts.model_name,
          backend: opts.backend,
          param_type: opts.param_type,
          max_response_tokens: opts.max_new_tokens,
          debug: opts.debug
        )
      end)

    last_layer = runtime.model_info.spec.num_blocks - 1

    unless opts.tail_start in 1..last_layer do
      abort("--tail-start must be in 1..#{last_layer}")
    end

    pipeline = DecoderPipeline.extract!(runtime, opts.tail_start..last_layer)

    variants =
      case opts.transplants do
        [] ->
          [{"baseline", pipeline}]

        transplants ->
          frankenstein =
            Enum.reduce(transplants, pipeline, fn %{source: source, target: target}, pipeline ->
              DecoderPipeline.transplant_layer!(pipeline, source, target)
            end)

          [{"baseline", pipeline}, {"frankenstein", frankenstein}]
      end

    IO.puts(
      Jason.encode!(%{
        event: "ready",
        backend: opts.backend,
        model: opts.model_name,
        tail_layers: [opts.tail_start, last_layer],
        transplants: opts.transplants,
        samples: length(samples),
        runs: opts.runs
      })
    )

    Enum.each(variants, fn {variant, variant_pipeline} ->
      run_variant(variant_pipeline, variant, input, opts)
    end)
  end

  defp extract_artifact!(opts) do
    runtime =
      timed!("load", fn ->
        Runtime.load(
          model_name: opts.model_name,
          backend: opts.backend,
          param_type: opts.param_type,
          max_response_tokens: opts.max_new_tokens,
          debug: opts.debug
        )
      end)

    last_layer = runtime.model_info.spec.num_blocks - 1
    pipeline = DecoderPipeline.extract!(runtime, opts.tail_start..last_layer)

    pipeline =
      Enum.reduce(opts.transplants, pipeline, fn %{source: source, target: target}, pipeline ->
        DecoderPipeline.transplant_layer!(pipeline, source, target)
      end)

    artifact =
      timed!("artifact_save", fn ->
        {:ok,
         DecoderPipelineArtifact.save!(pipeline, opts.artifact,
           model_name: runtime.repo_id,
           tokenizer_repository: runtime.repo_id,
           transplants: opts.transplants
         )}
      end)

    IO.puts(
      Jason.encode!(%{
        event: "artifact",
        path: artifact,
        model: runtime.repo_id,
        tail_layers: [opts.tail_start, last_layer],
        transplants: opts.transplants
      })
    )
  end

  defp run_artifact!(opts) do
    {:ok, backend} = Runtime.resolve_backend(opts.backend)

    artifact =
      timed!("artifact_load", fn ->
        {:ok, DecoderPipelineArtifact.load!(opts.artifact, backend)}
      end)

    pipeline = DecoderPipelineArtifact.build_pipeline!(artifact, backend)

    samples =
      opts.wav
      |> Audio.read_wav_samples!(opts.sample_rate)
      |> Enum.take(round(opts.sample_rate * opts.seconds))

    input = Input.build(samples, prompt: Config.default_prompt())

    IO.puts(
      Jason.encode!(%{
        event: "ready",
        artifact: Path.expand(opts.artifact),
        backend: opts.backend,
        samples: length(samples),
        runs: opts.runs,
        transplants: artifact.manifest.transplants
      })
    )

    run_variant(pipeline, "artifact", input, opts)
  end

  defp run_variant(pipeline, variant, input, opts) do
    Enum.each(1..opts.runs, fn run ->
      {elapsed_us, result} =
        :timer.tc(fn ->
          DecoderPipeline.generate(pipeline, input,
            max_new_tokens: opts.max_new_tokens,
            min_new_tokens: opts.min_new_tokens
          )
        end)

      case result do
        {:ok, output} ->
          IO.puts(
            Jason.encode!(%{
              event: "generation",
              variant: variant,
              run: run,
              cold: variant in ["baseline", "artifact"] and run == 1,
              elapsed_ms: div(elapsed_us, 1_000),
              text: output.text,
              token_ids: output.token_ids
            })
          )

        {:error, reason} ->
          abort("generation run #{run} failed: #{reason}")
      end
    end)
  end

  defp parse_transplants(nil), do: {:ok, []}

  defp parse_transplants(value) do
    value
    |> String.split(",", trim: true)
    |> Enum.reduce_while({:ok, []}, fn transplant, {:ok, acc} ->
      case String.split(transplant, ":", parts: 2) do
        [source, target] ->
          with {source, ""} <- Integer.parse(source),
               {target, ""} <- Integer.parse(target),
               true <- source >= 0 and target >= 0 and source != target do
            {:cont, {:ok, [%{source: source, target: target} | acc]}}
          else
            _ ->
              {:halt,
               {:error, "invalid --transplant #{inspect(transplant)}; expected SOURCE:TARGET"}}
          end

        _other ->
          {:halt, {:error, "invalid --transplant #{inspect(transplant)}; expected SOURCE:TARGET"}}
      end
    end)
    |> case do
      {:ok, transplants} -> {:ok, Enum.reverse(transplants)}
      error -> error
    end
  end

  defp timed!(event, fun) do
    {elapsed_us, result} = :timer.tc(fun)
    IO.puts(Jason.encode!(%{event: event, elapsed_ms: div(elapsed_us, 1_000)}))

    case result do
      {:ok, value} -> value
      {:error, reason} -> abort("#{event} failed: #{reason}")
    end
  end

  defp positive(value, _name) when is_number(value) and value > 0, do: :ok
  defp positive(_value, name), do: {:error, "#{name} must be positive"}

  defp non_negative(value, _name) when is_integer(value) and value >= 0, do: :ok
  defp non_negative(_value, name), do: {:error, "#{name} must be a non-negative integer"}

  defp regular_file(path) do
    if File.regular?(path), do: :ok, else: {:error, "--wav is not a file: #{path}"}
  end

  defp abort(message) do
    IO.puts(:stderr, "error: #{message}")
    System.halt(1)
  end

  defp usage do
    """
    Usage: decoder_pipeline_bench [options]

      --wav PATH                 input PCM WAV, default journal1.wav
      --backend BACKEND          default exla:rocm
      --model-name MODEL         default google/gemma-4-12B-it
      --param-type TYPE          bf16, f16, or f32; default bf16
      --tail-start LAYER         first extracted tail layer, default 45
      --max-new-tokens COUNT     generated-token limit, default 3
      --min-new-tokens COUNT     force generation through this step, default 3
      --transplant SOURCE:TARGET copy a compatible source layer into a target slot;
                                 comma-separate multiple transplants
      --runs COUNT               first run is cold, later runs are warm; default 2
      --sample-rate HZ           default 16000
      --seconds SECONDS          leading audio duration, default 2.0
      --debug                    enable runtime progress logging
    """
  end

  defp artifact_usage(:extract) do
    """
    Usage: decoder_pipeline_bench extract --artifact PATH [options]

    Loads the source checkpoint, applies --transplant mutations, and writes a
    self-contained artifact. The default extraction backend is torchx:cpu.

      --artifact PATH            new output directory
      --model-name MODEL         source model, default google/gemma-4-12B-it
      --tail-start LAYER         first independently extracted tail layer
      --transplant SOURCE:TARGET optional compatible layer transplant
      --backend BACKEND          extraction backend, default torchx:cpu
    """
  end

  defp artifact_usage(:run) do
    """
    Usage: decoder_pipeline_bench run --artifact PATH [options]

    Loads only the saved artifact, compiles it for the selected backend, and
    generates from the WAV input. The source checkpoint is not loaded.

      --artifact PATH            artifact directory created by extract
      --wav PATH                 input PCM WAV, default journal1.wav
      --backend BACKEND          execution backend, default exla:rocm
      --runs COUNT               first run is cold, later runs are warm
      --max-new-tokens COUNT     generated-token limit
    """
  end
end

defmodule Gemma4MicTranscribe.DecoderPipelineBenchmark.Escript do
  @moduledoc false

  def main(argv) do
    root =
      :escript.script_name()
      |> List.to_string()
      |> Path.expand()
      |> Path.dirname()

    mix_env = System.get_env("MIX_ENV", "dev")

    root
    |> Path.join("_build/#{mix_env}/lib/*/ebin")
    |> Path.wildcard()
    |> Enum.each(&Code.prepend_path/1)

    Gemma4MicTranscribe.DecoderPipelineBenchmark.main(argv)
  end
end
