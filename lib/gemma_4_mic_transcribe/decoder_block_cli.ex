defmodule Gemma4MicTranscribe.DecoderBlockCLI do
  @moduledoc false

  alias Gemma4MicTranscribe.Gemma4.DecoderBlockArtifact
  alias Gemma4MicTranscribe.Gemma4.DecoderBlocks
  alias Gemma4MicTranscribe.Gemma4.DecoderPipeline
  alias Gemma4MicTranscribe.Gemma4.DecoderPipelineArtifact
  alias Gemma4MicTranscribe.Audio
  alias Gemma4MicTranscribe.Config
  alias Gemma4MicTranscribe.Gemma4Unified.Input
  alias Gemma4MicTranscribe.Gemma4Unified.Runtime

  @switches [
    artifact: :string,
    prefix_artifact: :string,
    pipeline_artifact: :string,
    input: :string,
    output: :string,
    wav: :string,
    seconds: :float,
    tail_start: :integer,
    top_k: :integer,
    max_new_tokens: :integer,
    min_new_tokens: :integer,
    execution: :string,
    bypass_ffn_layers: :string,
    bypass_phase: :string,
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
      {:ok, :extract_tail, opts} -> extract_tail!(opts)
      {:ok, :extract_prefix, opts} -> extract_prefix!(opts)
      {:ok, :run_tail, opts} -> run_tail!(opts)
      {:ok, :capture_prefix, opts} -> capture_prefix!(opts)
      {:ok, :run_split, opts} -> run_split!(opts)
      {:help, usage} -> IO.puts(usage)
      {:error, message} -> abort(message)
    end
  end

  def parse([command | argv])
      when command in [
             "extract",
             "run",
             "extract-prefix",
             "extract-tail",
             "run-tail",
             "capture-prefix",
             "run-split"
           ] do
    mode = command |> String.replace("-", "_") |> String.to_existing_atom()

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
        mode when mode in [:extract, :extract_prefix, :extract_tail] ->
          [backend: "torchx:cpu"]

        _other ->
          [backend: "exla:rocm"]
      end

    values = %{
      artifact: opts[:artifact],
      prefix_artifact: opts[:prefix_artifact],
      pipeline_artifact: opts[:pipeline_artifact],
      input: opts[:input],
      output: opts[:output],
      wav: Keyword.get(opts, :wav, "journal1.wav"),
      seconds: Keyword.get(opts, :seconds, 5.0),
      tail_start: Keyword.get(opts, :tail_start, 45),
      top_k: Keyword.get(opts, :top_k, 10),
      max_new_tokens: Keyword.get(opts, :max_new_tokens, 32),
      min_new_tokens: Keyword.get(opts, :min_new_tokens, 0),
      execution: parse_execution(Keyword.get(opts, :execution, "composed")),
      bypass_ffn_layers: parse_layers(Keyword.get(opts, :bypass_ffn_layers)),
      bypass_phase: parse_bypass_phase(Keyword.get(opts, :bypass_phase, "all")),
      layer: Keyword.get(opts, :layer, 45),
      backend: Keyword.get(opts, :backend, defaults[:backend]),
      model_name: Keyword.get(opts, :model_name, "google/gemma-4-12B-it"),
      param_type: Keyword.get(opts, :param_type, "bf16"),
      sequence_length: Keyword.get(opts, :sequence_length, 8),
      runs: Keyword.get(opts, :runs, 2),
      debug: Keyword.get(opts, :debug, false)
    }

    with :ok <- required_paths(mode, values),
         :ok <- positive_number(values.seconds, "--seconds"),
         :ok <- positive(values.top_k, "--top-k"),
         :ok <- positive(values.max_new_tokens, "--max-new-tokens"),
         :ok <- non_negative(values.min_new_tokens, "--min-new-tokens"),
         :ok <- valid_token_limits(values.min_new_tokens, values.max_new_tokens),
         :ok <- positive(values.sequence_length, "--sequence-length"),
         :ok <- positive(values.runs, "--runs"),
         :ok <- valid_execution(values.execution),
         :ok <- valid_layers(values.bypass_ffn_layers),
         :ok <- valid_bypass_phase(values.bypass_phase),
         :ok <- valid_bypass_execution(values.bypass_ffn_layers, values.execution),
         :ok <- valid_layer(values.layer),
         :ok <- valid_run_paths(mode, values) do
      {:ok, mode, values}
    end
  end

  defp extract_tail!(opts) do
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

    last_layer = runtime.model_info.spec.num_blocks - 1
    tail = DecoderBlocks.extract_tail!(runtime, opts.tail_start..last_layer)

    artifact =
      timed!("artifact_save", fn ->
        {:ok,
         DecoderBlockArtifact.save_tail!(tail, opts.artifact,
           tokenizer_repository: runtime.repo_id,
           verification_sequence_length: opts.sequence_length
         )}
      end)

    parameter_bytes = File.stat!(Path.join(artifact, "parameters.safetensors")).size

    IO.puts(
      Jason.encode!(%{
        event: "tail_artifact",
        artifact: artifact,
        layers: tail.layer_indices,
        input_size: tail.input_size,
        vocab_size: tail.vocab_size,
        parameter_count: tail.parameter_count,
        parameter_bytes: parameter_bytes
      })
    )
  end

  defp extract_prefix!(opts) do
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

    last_layer = runtime.model_info.spec.num_blocks - 1
    pipeline = DecoderPipeline.extract!(runtime, opts.tail_start..last_layer)

    artifact =
      timed!("artifact_save", fn ->
        {:ok,
         DecoderBlockArtifact.save_prefix!(pipeline, opts.artifact,
           tokenizer_repository: runtime.repo_id
         )}
      end)

    parameter_bytes = File.stat!(Path.join(artifact, "parameters.safetensors")).size

    IO.puts(
      Jason.encode!(%{
        event: "prefix_artifact",
        artifact: artifact,
        last_layer: pipeline.prefix.last_layer,
        parameter_count: pipeline.prefix.parameter_count,
        parameter_bytes: parameter_bytes
      })
    )
  end

  defp capture_prefix!(opts) do
    {:ok, backend} = Runtime.resolve_backend(opts.backend)

    artifact =
      timed!("pipeline_artifact_load", fn ->
        {:ok, DecoderPipelineArtifact.load!(opts.pipeline_artifact, backend)}
      end)

    pipeline = DecoderPipelineArtifact.build_pipeline!(artifact, backend)

    if pipeline.prefix.last_layer != opts.tail_start - 1 do
      abort(
        "pipeline prefix ends at layer #{pipeline.prefix.last_layer}, " <>
          "but --tail-start #{opts.tail_start} requires layer #{opts.tail_start - 1}"
      )
    end

    samples =
      opts.wav
      |> Audio.read_wav_samples!(16_000)
      |> Enum.take(round(16_000 * opts.seconds))

    input = Input.build(samples, prompt: Config.default_prompt())

    prepared =
      timed!("input_prepare", fn -> Runtime.prepare_input(pipeline.input_context, input) end)

    hidden_state =
      timed!("prefix_run", fn -> DecoderPipeline.run_prefix(pipeline.prefix, prepared) end)

    output =
      DecoderBlockArtifact.save_input!(opts.output, hidden_state,
        position_ids: prepared["position_ids"],
        attention_mask: prepared["attention_mask"]
      )

    IO.puts(
      Jason.encode!(%{
        event: "prefix_output",
        path: output,
        last_layer: pipeline.prefix.last_layer,
        shape: Tuple.to_list(Nx.shape(hidden_state))
      })
    )
  end

  defp run_tail!(opts) do
    {:ok, backend} = Runtime.resolve_backend(opts.backend)

    tail =
      timed!("artifact_load", fn ->
        {:ok, DecoderBlockArtifact.load_tail!(opts.artifact, backend)}
      end)

    input =
      case opts.input do
        nil -> DecoderBlockArtifact.load_verification!(opts.artifact, backend)
        path -> DecoderBlockArtifact.load_input!(path, backend)
      end

    IO.puts(
      Jason.encode!(%{
        event: "tail_ready",
        artifact: Path.expand(opts.artifact),
        backend: opts.backend,
        layers: tail.layer_indices,
        parameter_count: tail.parameter_count,
        input_shape: Tuple.to_list(Nx.shape(input.hidden_state)),
        runs: opts.runs
      })
    )

    Enum.each(1..opts.runs, fn run ->
      {elapsed_us, result} = :timer.tc(fn -> DecoderBlockArtifact.verify!(tail, input) end)
      candidates = top_candidates(result.output, tail.tokenizer, opts.top_k)

      IO.puts(
        Jason.encode!(%{
          event: "tail_run",
          run: run,
          cold: run == 1,
          elapsed_ms: div(elapsed_us, 1_000),
          verified: result.verified,
          max_abs_error: result.max_abs_error,
          candidates: candidates
        })
      )
    end)
  end

  defp run_split!(opts) do
    {:ok, backend} = Runtime.resolve_backend(opts.backend)

    pipeline =
      case opts.prefix_artifact do
        nil ->
          timed!("pipeline_artifact_load", fn ->
            {:ok, load_prefix_pipeline!(opts.pipeline_artifact, backend)}
          end)

        path ->
          prefix =
            timed!("prefix_artifact_load", fn ->
              {:ok, DecoderBlockArtifact.load_prefix!(path, backend)}
            end)

          %{prefix_artifact: prefix}
      end

    :erlang.garbage_collect(self())

    tail =
      timed!("tail_artifact_load", fn ->
        {:ok, DecoderBlockArtifact.load_tail!(opts.artifact, backend)}
      end)

    pipeline =
      case pipeline do
        %{prefix_artifact: prefix} ->
          DecoderBlockArtifact.build_split_pipeline!(prefix, tail, backend,
            bypass_ffn_layers: opts.bypass_ffn_layers,
            bypass_phase: opts.bypass_phase
          )

        %DecoderPipeline{} = pipeline ->
          if tail.layer_indices != pipeline.tail.layer_indices do
            abort(
              "tail artifact layers #{inspect(tail.layer_indices)} do not match pipeline tail " <>
                inspect(pipeline.tail.layer_indices)
            )
          end

          DecoderBlockArtifact.install_tail!(pipeline, tail)
      end

    samples =
      opts.wav
      |> Audio.read_wav_samples!(16_000)
      |> Enum.take(round(16_000 * opts.seconds))

    input = Input.build(samples, prompt: Config.default_prompt())

    IO.puts(
      Jason.encode!(%{
        event: "split_ready",
        prefix_artifact: opts.prefix_artifact && Path.expand(opts.prefix_artifact),
        pipeline_artifact: opts.pipeline_artifact && Path.expand(opts.pipeline_artifact),
        tail_artifact: Path.expand(opts.artifact),
        tail_layers: tail.layer_indices,
        backend: opts.backend,
        execution: opts.execution,
        bypass_ffn_layers: opts.bypass_ffn_layers,
        bypass_phase: opts.bypass_phase,
        samples: length(samples),
        runs: opts.runs
      })
    )

    Enum.each(1..opts.runs, fn run ->
      {elapsed_us, result} =
        :timer.tc(fn ->
          DecoderPipeline.generate(pipeline, input,
            max_new_tokens: opts.max_new_tokens,
            min_new_tokens: opts.min_new_tokens,
            execution: opts.execution
          )
        end)

      case result do
        {:ok, output} ->
          IO.puts(
            Jason.encode!(%{
              event: "split_generation",
              run: run,
              cold: run == 1,
              elapsed_ms: div(elapsed_us, 1_000),
              text: output.text,
              token_ids: output.token_ids
            })
          )

        {:error, reason} ->
          abort("split generation failed: #{reason}")
      end
    end)
  end

  defp load_prefix_pipeline!(path, backend) do
    source = DecoderPipelineArtifact.load!(path, backend)
    pipeline = DecoderPipelineArtifact.build_pipeline!(source, backend)
    empty_params = Axon.ModelState.new(%{})

    %{
      pipeline
      | tail: %{pipeline.tail | params: empty_params},
        generation_model: nil,
        generation_params: empty_params,
        generation_predict_fun: nil
    }
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

  defp required_paths(:capture_prefix, values) do
    with :ok <- required(values.pipeline_artifact, "--pipeline-artifact PATH is required"),
         :ok <- required(values.output, "--output PATH is required") do
      :ok
    end
  end

  defp required_paths(:run_split, values) do
    cond do
      is_nil(values.prefix_artifact) and is_nil(values.pipeline_artifact) ->
        {:error, "--prefix-artifact PATH or --pipeline-artifact PATH is required"}

      values.prefix_artifact && values.pipeline_artifact ->
        {:error, "choose either --prefix-artifact or --pipeline-artifact"}

      true ->
        required(values.artifact, "--artifact PATH is required")
    end
  end

  defp required_paths(_mode, values), do: required(values.artifact, "--artifact PATH is required")

  defp positive(value, _name) when is_integer(value) and value > 0, do: :ok
  defp positive(_value, name), do: {:error, "#{name} must be positive"}

  defp non_negative(value, _name) when is_integer(value) and value >= 0, do: :ok
  defp non_negative(_value, name), do: {:error, "#{name} must be non-negative"}

  defp valid_token_limits(minimum, maximum) when minimum <= maximum, do: :ok

  defp valid_token_limits(_minimum, _maximum),
    do: {:error, "--min-new-tokens must not exceed --max-new-tokens"}

  defp positive_number(value, _name) when is_number(value) and value > 0, do: :ok
  defp positive_number(_value, name), do: {:error, "#{name} must be positive"}

  defp valid_layer(layer) when is_integer(layer) and layer >= 0, do: :ok
  defp valid_layer(_layer), do: {:error, "--layer must be a non-negative integer"}

  defp parse_execution("composed"), do: :composed
  defp parse_execution("split"), do: :split
  defp parse_execution(value), do: {:invalid, value}

  defp parse_layers(nil), do: []

  defp parse_layers(value) do
    value
    |> String.split(",", trim: true)
    |> Enum.map(fn layer ->
      case Integer.parse(layer) do
        {layer, ""} -> layer
        _other -> {:invalid, layer}
      end
    end)
  end

  defp parse_bypass_phase("all"), do: :all
  defp parse_bypass_phase("prefill"), do: :prefill
  defp parse_bypass_phase("decode"), do: :decode
  defp parse_bypass_phase(value), do: {:invalid, value}

  defp valid_execution(execution) when execution in [:composed, :split], do: :ok

  defp valid_execution({:invalid, value}),
    do: {:error, "--execution must be composed or split, got: #{inspect(value)}"}

  defp valid_layers(layers) do
    if Enum.all?(layers, &is_integer/1) and Enum.all?(layers, &(&1 in 0..47)) do
      :ok
    else
      {:error, "--bypass-ffn-layers must contain comma-separated indices in 0..47"}
    end
  end

  defp valid_bypass_phase(phase) when phase in [:all, :prefill, :decode], do: :ok

  defp valid_bypass_phase({:invalid, value}),
    do: {:error, "--bypass-phase must be all, prefill, or decode, got: #{inspect(value)}"}

  defp valid_bypass_execution([], _execution), do: :ok
  defp valid_bypass_execution(_layers, :composed), do: :ok

  defp valid_bypass_execution(_layers, :split),
    do: {:error, "--bypass-ffn-layers requires --execution composed"}

  defp valid_run_paths(:extract, _opts), do: :ok
  defp valid_run_paths(:extract_prefix, _opts), do: :ok
  defp valid_run_paths(:extract_tail, _opts), do: :ok

  defp valid_run_paths(:capture_prefix, opts) do
    cond do
      not File.dir?(opts.pipeline_artifact) ->
        {:error, "pipeline artifact directory does not exist: #{opts.pipeline_artifact}"}

      not File.regular?(opts.wav) ->
        {:error, "WAV input does not exist: #{opts.wav}"}

      File.exists?(opts.output) ->
        {:error, "output path already exists: #{opts.output}"}

      true ->
        :ok
    end
  end

  defp valid_run_paths(:run_tail, opts), do: valid_run_paths(:run, %{opts | output: nil})

  defp valid_run_paths(:run_split, opts) do
    cond do
      opts.bypass_ffn_layers != [] and is_nil(opts.prefix_artifact) ->
        {:error, "--bypass-ffn-layers requires --prefix-artifact"}

      opts.prefix_artifact && not File.dir?(opts.prefix_artifact) ->
        {:error, "prefix artifact directory does not exist: #{opts.prefix_artifact}"}

      opts.pipeline_artifact && not File.dir?(opts.pipeline_artifact) ->
        {:error, "pipeline artifact directory does not exist: #{opts.pipeline_artifact}"}

      not File.dir?(opts.artifact) ->
        {:error, "tail artifact directory does not exist: #{opts.artifact}"}

      not File.regular?(opts.wav) ->
        {:error, "WAV input does not exist: #{opts.wav}"}

      true ->
        :ok
    end
  end

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

  defp top_candidates(logits, tokenizer, k) do
    {scores, token_ids} = Nx.top_k(logits, k: k)
    scores = scores |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()
    token_ids = token_ids |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()

    Enum.zip(token_ids, scores)
    |> Enum.map(fn {token_id, score} ->
      %{
        token_id: token_id,
        token: Bumblebee.Tokenizer.decode(tokenizer, [token_id]),
        score: score
      }
    end)
  end

  defp usage do
    """
    Usage:
      decoder_block extract --artifact PATH [options]
      decoder_block run --artifact PATH [options]
      decoder_block extract-prefix --artifact PATH [options]
      decoder_block extract-tail --artifact PATH [options]
      decoder_block run-tail --artifact PATH --input PATH [options]
      decoder_block capture-prefix --pipeline-artifact PATH --output PATH [options]
      decoder_block run-split --prefix-artifact PATH --artifact TAIL [options]

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

    extract-tail persists the final decoder layers, output norm, and vocabulary
    head. run-tail turns a captured prefix hidden state into token candidates.
      --tail-start INDEX         first tail layer, default 45
      --top-k COUNT              candidates returned by run-tail, default 10

    extract-prefix persists embeddings, audio projection, and all layers before
    --tail-start. Together, independently persisted prefix and tail artifacts
    are sufficient for run-split generation.

    capture-prefix runs layers 0 through tail-start-1 from a complete pipeline
    artifact and saves the real audio-conditioned boundary hidden state.
      --pipeline-artifact PATH   complete pipeline artifact used only for capture
      --wav PATH                 PCM WAV input, default journal1.wav
      --seconds SECONDS          leading audio duration, default 5.0

    run-split performs cache-aware generation with independently loaded prefix
    and tail artifacts. --pipeline-artifact remains as a compatibility fallback.
      --prefix-artifact PATH     artifact created by extract-prefix
      --max-new-tokens COUNT     generated-token limit, default 32
      --min-new-tokens COUNT     force this many decode steps, default 0
      --execution MODE           composed (default) fuses the loaded artifacts;
                                 split keeps the observable XLA boundary
      --bypass-ffn-layers LIST   retain attention but remove selected FFNs
      --bypass-phase PHASE       apply bypass to all, prefill, or decode; default all
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
