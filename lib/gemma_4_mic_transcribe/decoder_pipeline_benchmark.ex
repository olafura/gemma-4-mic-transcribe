defmodule Gemma4MicTranscribe.DecoderPipelineBenchmark do
  @moduledoc false

  alias Gemma4MicTranscribe.Audio
  alias Gemma4MicTranscribe.Config
  alias Gemma4MicTranscribe.Gemma4.DecoderPipeline
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
    runs: :integer,
    sample_rate: :integer,
    seconds: :float,
    debug: :boolean,
    help: :boolean
  ]

  def main(argv) do
    {:ok, _started} = Application.ensure_all_started(:gemma_4_mic_transcribe)

    case parse(argv) do
      {:ok, opts} -> run!(opts)
      {:help, usage} -> IO.puts(usage)
      {:error, message} -> abort(message)
    end
  end

  def parse(argv) do
    case OptionParser.parse(argv, strict: @switches, aliases: [h: :help]) do
      {opts, [], []} -> parse_options(opts)
      {_opts, args, []} -> {:error, "unexpected arguments: #{Enum.join(args, " ")}"}
      {_opts, _args, invalid} -> {:error, "invalid options: #{inspect(invalid)}"}
    end
  end

  defp parse_options(opts) do
    if Keyword.get(opts, :help, false) do
      {:help, usage()}
    else
      values = %{
        wav: Keyword.get(opts, :wav, "journal1.wav"),
        backend: Keyword.get(opts, :backend, "exla:rocm"),
        model_name: Keyword.get(opts, :model_name, "google/gemma-4-12B-it"),
        param_type: Keyword.get(opts, :param_type, "bf16"),
        tail_start: Keyword.get(opts, :tail_start, 45),
        max_new_tokens: Keyword.get(opts, :max_new_tokens, 3),
        min_new_tokens: Keyword.get(opts, :min_new_tokens, 3),
        runs: Keyword.get(opts, :runs, 2),
        sample_rate: Keyword.get(opts, :sample_rate, 16_000),
        seconds: Keyword.get(opts, :seconds, 2.0),
        debug: Keyword.get(opts, :debug, false)
      }

      with :ok <- positive(values.runs, "--runs"),
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

    IO.puts(
      Jason.encode!(%{
        event: "ready",
        backend: opts.backend,
        model: opts.model_name,
        tail_layers: [opts.tail_start, last_layer],
        samples: length(samples),
        runs: opts.runs
      })
    )

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
              run: run,
              cold: run == 1,
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
      --runs COUNT               first run is cold, later runs are warm; default 2
      --sample-rate HZ           default 16000
      --seconds SECONDS          leading audio duration, default 2.0
      --debug                    enable runtime progress logging
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
