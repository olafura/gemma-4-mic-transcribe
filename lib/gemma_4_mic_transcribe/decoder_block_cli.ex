defmodule Gemma4MicTranscribe.DecoderBlockCLI do
  @moduledoc false

  alias Gemma4MicTranscribe.Gemma4.DecoderBlockArtifact
  alias Gemma4MicTranscribe.Gemma4.DecoderBlocks
  alias Gemma4MicTranscribe.Gemma4Unified.Runtime

  @switches [
    artifact: :string,
    input: :string,
    output: :string,
    layer: :integer,
    backend: :string,
    model_name: :string,
    param_type: :string,
    sequence_length: :integer,
    runs: :integer,
    debug: :boolean,
    help: :boolean
  ]

  def main(argv) do
    {:ok, _started} = Application.ensure_all_started(:gemma_4_mic_transcribe)

    case parse(argv) do
      {:ok, :extract, opts} -> extract!(opts)
      {:ok, :run, opts} -> run!(opts)
      {:help, usage} -> IO.puts(usage)
      {:error, message} -> abort(message)
    end
  end

  def parse([command | argv]) when command in ["extract", "run"] do
    mode = String.to_existing_atom(command)

    case OptionParser.parse(argv, strict: @switches, aliases: [h: :help]) do
      {opts, [], []} -> parse_options(mode, opts)
      {_opts, args, []} -> {:error, "unexpected arguments: #{Enum.join(args, " ")}"}
      {_opts, _args, invalid} -> {:error, "invalid options: #{inspect(invalid)}"}
    end
  end

  def parse(["--help"]), do: {:help, usage()}
  def parse([]), do: {:help, usage()}
  def parse(_argv), do: {:error, "expected extract or run subcommand"}

  defp parse_options(mode, opts) do
    if Keyword.get(opts, :help, false) do
      {:help, usage()}
    else
      parse_values(mode, opts)
    end
  end

  defp parse_values(mode, opts) do
    defaults =
      case mode do
        :extract -> [backend: "torchx:cpu"]
        :run -> [backend: "exla:rocm"]
      end

    values = %{
      artifact: opts[:artifact],
      input: opts[:input],
      output: opts[:output],
      layer: Keyword.get(opts, :layer, 45),
      backend: Keyword.get(opts, :backend, defaults[:backend]),
      model_name: Keyword.get(opts, :model_name, "google/gemma-4-12B-it"),
      param_type: Keyword.get(opts, :param_type, "bf16"),
      sequence_length: Keyword.get(opts, :sequence_length, 8),
      runs: Keyword.get(opts, :runs, 2),
      debug: Keyword.get(opts, :debug, false)
    }

    with :ok <- required(values.artifact, "--artifact PATH is required"),
         :ok <- positive(values.sequence_length, "--sequence-length"),
         :ok <- positive(values.runs, "--runs"),
         :ok <- valid_layer(values.layer),
         :ok <- valid_run_paths(mode, values) do
      {:ok, mode, values}
    end
  end

  defp extract!(opts) do
    runtime =
      timed!("model_load", fn ->
        Runtime.load(
          model_name: opts.model_name,
          backend: opts.backend,
          param_type: opts.param_type,
          max_response_tokens: 1,
          debug: opts.debug
        )
      end)

    block = DecoderBlocks.extract!(runtime, opts.layer)

    artifact =
      timed!("artifact_save", fn ->
        {:ok,
         DecoderBlockArtifact.save!(block, opts.artifact,
           verification_sequence_length: opts.sequence_length
         )}
      end)

    parameter_bytes = File.stat!(Path.join(artifact, "parameters.safetensors")).size

    IO.puts(
      Jason.encode!(%{
        event: "artifact",
        artifact: artifact,
        layer: block.layer_index,
        layer_type: block.layer_type,
        input_size: block.input_size,
        parameter_count: block.parameter_count,
        parameter_bytes: parameter_bytes,
        verification_sequence_length: opts.sequence_length
      })
    )
  end

  defp run!(opts) do
    {:ok, backend} = Runtime.resolve_backend(opts.backend)

    block =
      timed!("artifact_load", fn ->
        {:ok, DecoderBlockArtifact.load!(opts.artifact, backend)}
      end)

    input =
      case opts.input do
        nil -> DecoderBlockArtifact.load_verification!(opts.artifact, backend)
        path -> DecoderBlockArtifact.load_input!(path, backend)
      end

    IO.puts(
      Jason.encode!(%{
        event: "ready",
        artifact: Path.expand(opts.artifact),
        backend: opts.backend,
        layer: block.layer_index,
        layer_type: block.layer_type,
        parameter_count: block.parameter_count,
        input_shape: Tuple.to_list(Nx.shape(input.hidden_state)),
        runs: opts.runs
      })
    )

    last_result =
      Enum.reduce(1..opts.runs, nil, fn run, _previous ->
        {elapsed_us, result} = :timer.tc(fn -> DecoderBlockArtifact.verify!(block, input) end)

        IO.puts(
          Jason.encode!(%{
            event: "block_run",
            run: run,
            cold: run == 1,
            elapsed_ms: div(elapsed_us, 1_000),
            output_shape: Tuple.to_list(Nx.shape(result.output)),
            verified: result.verified,
            max_abs_error: result.max_abs_error
          })
        )

        result
      end)

    if opts.output do
      output =
        DecoderBlockArtifact.save_input!(opts.output, last_result.output,
          position_ids: input.position_ids,
          attention_mask: input.attention_mask
        )

      IO.puts(Jason.encode!(%{event: "output", path: output}))
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

  defp required(nil, message), do: {:error, message}
  defp required(_value, _message), do: :ok

  defp positive(value, _name) when is_integer(value) and value > 0, do: :ok
  defp positive(_value, name), do: {:error, "#{name} must be positive"}

  defp valid_layer(layer) when is_integer(layer) and layer >= 0, do: :ok
  defp valid_layer(_layer), do: {:error, "--layer must be a non-negative integer"}

  defp valid_run_paths(:extract, _opts), do: :ok

  defp valid_run_paths(:run, opts) do
    cond do
      not File.dir?(opts.artifact) ->
        {:error, "artifact directory does not exist: #{opts.artifact}"}

      opts.input && not File.regular?(opts.input) ->
        {:error, "input safetensors file does not exist: #{opts.input}"}

      opts.output && File.exists?(opts.output) ->
        {:error, "output path already exists: #{opts.output}"}

      true ->
        :ok
    end
  end

  defp abort(message) do
    IO.puts(:stderr, "error: #{message}")
    System.halt(1)
  end

  defp usage do
    """
    Usage:
      decoder_block extract --artifact PATH [options]
      decoder_block run --artifact PATH [options]

    extract loads the source checkpoint and writes only one decoder block.
      --layer INDEX              layer to extract, default 45
      --model-name MODEL         default google/gemma-4-12B-it
      --backend BACKEND          extraction backend, default torchx:cpu
      --param-type TYPE          bf16, f16, or f32; default bf16
      --sequence-length COUNT    verification fixture length, default 8

    run loads only the block artifact and a hidden-state safetensors input.
      --backend BACKEND          execution backend, default exla:rocm
      --input PATH               optional external input; defaults to verification fixture
      --output PATH              save output as the next block's input
      --runs COUNT               first run is cold, default 2
    """
  end
end

defmodule Gemma4MicTranscribe.DecoderBlockCLI.Escript do
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

    Gemma4MicTranscribe.DecoderBlockCLI.main(argv)
  end
end
