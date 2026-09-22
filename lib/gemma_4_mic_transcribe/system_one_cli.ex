defmodule Gemma4MicTranscribe.SystemOneCLI do
  @moduledoc """
  Commands for the System One expert, see `docs/system-one-expert-plan.md`.

  `cache` renders a JSONL of items with the fixed System One template, runs
  the packed prefix (layers 0 to 44) over each one, and stores the layer-45
  input. Only the prefix is loaded: it ends in a terminal output, so it avoids
  the ROCm autotuner crash on intermediate outputs, and nothing downstream of
  layer 44 has to fit beside the job.

  `train` fits the experts on such a cache, `generate` answers a JSONL of eval
  items with the packed pipeline plus an expert artifact, and `regress` is the
  non-regression gate: the router forced closed must still be base Gemma.
  """

  alias Gemma4MicTranscribe.Audio
  alias Gemma4MicTranscribe.Config
  alias Gemma4MicTranscribe.Gemma4.DecoderBlockArtifact
  alias Gemma4MicTranscribe.Gemma4.DecoderPipeline
  alias Gemma4MicTranscribe.Gemma4.SystemOne
  alias Gemma4MicTranscribe.Gemma4.SystemOne.Prompt
  alias Gemma4MicTranscribe.Gemma4.SystemOne.Trainer
  alias Gemma4MicTranscribe.Gemma4.SystemOneArtifact
  alias Gemma4MicTranscribe.Gemma4Unified.Input
  alias Gemma4MicTranscribe.Gemma4Unified.Runtime
  alias Gemma4MicTranscribe.Gemma4Unified.Transcript

  @default_buckets [64, 128, 256, 384]
  @default_prefix_artifact "artifacts/gemma4-12b-packed-prefix-0-44"
  @default_tail_artifact "artifacts/gemma4-12b-packed-tail-45-47"
  @default_wav "journal1.wav"
  @sample_rate 16_000
  @manifest "manifest.json"
  @rows "rows"
  @version 1

  # The plan's non-regression ids, from the bf16 split run recorded in the
  # README ("Splitting raw-audio inference"). A packed W4A16 run of the same
  # audio decodes differently, which is why `regress` also compares the expert
  # pipeline against the same pipeline without it.
  @reference_token_ids [712, 81686, 3124, 8178, 586, 5756, 506, 5597, 2214]

  @cache_switches [
    input: :string,
    output: :string,
    prefix_artifact: :string,
    backend: :string,
    buckets: :string,
    system_message: :string,
    thought_channel: :boolean,
    last_prompt_token_only: :boolean,
    verify_padding: :integer,
    limit: :integer,
    help: :boolean
  ]

  @train_switches [
    cache: :string,
    output: :string,
    tail_artifact: :string,
    layers: :string,
    backend: :string,
    stage_backend: :string,
    batch_size: :integer,
    epochs: :integer,
    max_response_tokens: :integer,
    expert_lr: :float,
    router_lr: :float,
    max_grad_norm: :float,
    kl_weight: :float,
    gate_open_weight: :float,
    gate_closed_weight: :float,
    gate_floor: :float,
    lead_tokens: :integer,
    lead_weight: :float,
    gate_mode: :string,
    init_from: :string,
    checkpoint_every: :integer,
    log_every: :integer,
    max_steps: :integer,
    limit: :integer,
    seed: :integer,
    head_type: :string,
    resume: :boolean,
    dry_run: :boolean,
    help: :boolean
  ]

  @generate_switches [
    input: :string,
    output: :string,
    expert: :string,
    prefix_artifact: :string,
    tail_artifact: :string,
    backend: :string,
    buckets: :string,
    system_message: :string,
    thought_channel: :boolean,
    max_new_tokens: :integer,
    force_router_closed: :boolean,
    gate_floor: :float,
    gate_probe: :boolean,
    limit: :integer,
    help: :boolean
  ]

  @regress_switches [
    expert: :string,
    prefix_artifact: :string,
    tail_artifact: :string,
    backend: :string,
    wav: :string,
    seconds: :float,
    max_new_tokens: :integer,
    gate_floor: :float,
    expected: :string,
    compare_base: :boolean,
    strict_reference: :boolean,
    help: :boolean
  ]

  def main(argv) do
    case parse(argv) do
      {:ok, :cache, opts} ->
        cache!(opts)
        0

      {:ok, :train, opts} ->
        train!(opts)
        0

      {:ok, :generate, opts} ->
        generate!(opts)
        0

      {:ok, :regress, opts} ->
        regress!(opts)

      {:help, usage} ->
        IO.puts(usage)
        0

      {:error, message} ->
        abort(message)
    end
  end

  def parse(["cache" | argv]), do: parse_command(:cache, argv, @cache_switches)
  def parse(["train" | argv]), do: parse_command(:train, argv, @train_switches)
  def parse(["generate" | argv]), do: parse_command(:generate, argv, @generate_switches)
  def parse(["regress" | argv]), do: parse_command(:regress, argv, @regress_switches)

  def parse(["--help"]), do: {:help, usage()}
  def parse(["-h"]), do: {:help, usage()}
  def parse([]), do: {:help, usage()}

  def parse([command | _argv]),
    do: {:error, "unknown subcommand #{command}, expected cache, train, generate or regress"}

  defp parse_command(command, argv, switches) do
    case OptionParser.parse(argv, strict: switches, aliases: [h: :help]) do
      {opts, [], []} -> parse_options(command, opts)
      {_opts, args, []} -> {:error, "unexpected arguments: #{Enum.join(args, " ")}"}
      {_opts, _args, invalid} -> {:error, "invalid options: #{inspect(invalid)}"}
    end
  end

  defp parse_options(command, opts) do
    if Keyword.get(opts, :help, false) do
      {:help, usage()}
    else
      parse_values(command, opts)
    end
  end

  defp parse_values(:train, opts) do
    with {:ok, layers} <- parse_layers(Keyword.get(opts, :layers)),
         {:ok, head_type} <- parse_head_type(Keyword.get(opts, :head_type)),
         {:ok, gate_mode} <- parse_gate_mode(Keyword.get(opts, :gate_mode)),
         :ok <- required(opts[:cache], "--cache PATH is required"),
         :ok <- required(opts[:output], "--output PATH is required") do
      defaults = Trainer.defaults()

      {:ok, :train,
       %{
         cache: opts[:cache],
         output: opts[:output],
         tail_artifact: Keyword.get(opts, :tail_artifact, @default_tail_artifact),
         layers: layers,
         backend: Keyword.get(opts, :backend, "exla:rocm"),
         stage_backend: Keyword.get(opts, :stage_backend, "torchx:cpu"),
         batch_size: Keyword.get(opts, :batch_size, defaults.batch_size),
         epochs: Keyword.get(opts, :epochs, defaults.epochs),
         max_response_tokens:
           Keyword.get(opts, :max_response_tokens, defaults.max_response_tokens),
         expert_learning_rate: Keyword.get(opts, :expert_lr, defaults.expert_learning_rate),
         router_learning_rate: Keyword.get(opts, :router_lr, defaults.router_learning_rate),
         max_grad_norm: Keyword.get(opts, :max_grad_norm, defaults.max_grad_norm),
         kl_weight: Keyword.get(opts, :kl_weight, defaults.kl_weight),
         gate_open_weight: Keyword.get(opts, :gate_open_weight, defaults.gate_open_weight),
         gate_closed_weight: Keyword.get(opts, :gate_closed_weight, defaults.gate_closed_weight),
         gate_floor: Keyword.get(opts, :gate_floor, defaults.gate_floor),
         lead_tokens: Keyword.get(opts, :lead_tokens, defaults.lead_tokens),
         lead_weight: Keyword.get(opts, :lead_weight, defaults.lead_weight),
         gate_mode: gate_mode,
         init_from: Keyword.get(opts, :init_from),
         checkpoint_every: Keyword.get(opts, :checkpoint_every, defaults.checkpoint_every),
         log_every: Keyword.get(opts, :log_every, defaults.log_every),
         max_steps: Keyword.get(opts, :max_steps),
         limit: Keyword.get(opts, :limit),
         seed: Keyword.get(opts, :seed, defaults.seed),
         head_type: head_type,
         resume: Keyword.get(opts, :resume, true),
         dry_run: Keyword.get(opts, :dry_run, false)
       }}
    end
  end

  defp parse_values(:generate, opts) do
    with {:ok, buckets} <- parse_buckets(Keyword.get(opts, :buckets)),
         :ok <- required(opts[:input], "--input PATH is required"),
         :ok <- required(opts[:output], "--output PATH is required") do
      {:ok, :generate,
       %{
         input: opts[:input],
         output: opts[:output],
         expert: Keyword.get(opts, :expert),
         prefix_artifact: Keyword.get(opts, :prefix_artifact, @default_prefix_artifact),
         tail_artifact: Keyword.get(opts, :tail_artifact, @default_tail_artifact),
         backend: Keyword.get(opts, :backend, "exla:rocm"),
         buckets: buckets,
         system_message: Keyword.get(opts, :system_message),
         thought_channel: Keyword.get(opts, :thought_channel, true),
         max_new_tokens: Keyword.get(opts, :max_new_tokens, 64),
         force_router_closed: Keyword.get(opts, :force_router_closed, false),
         gate_floor: Keyword.get(opts, :gate_floor),
         gate_probe: Keyword.get(opts, :gate_probe, true),
         limit: Keyword.get(opts, :limit)
       }}
    end
  end

  defp parse_values(:regress, opts) do
    with {:ok, expected} <- parse_token_ids(Keyword.get(opts, :expected)) do
      {:ok, :regress,
       %{
         expert: Keyword.get(opts, :expert),
         prefix_artifact: Keyword.get(opts, :prefix_artifact, @default_prefix_artifact),
         tail_artifact: Keyword.get(opts, :tail_artifact, @default_tail_artifact),
         backend: Keyword.get(opts, :backend, "exla:rocm"),
         wav: Keyword.get(opts, :wav, @default_wav),
         seconds: Keyword.get(opts, :seconds, 5.0),
         max_new_tokens: Keyword.get(opts, :max_new_tokens, 32),
         gate_floor: Keyword.get(opts, :gate_floor),
         expected: expected,
         compare_base: Keyword.get(opts, :compare_base, true),
         strict_reference: Keyword.get(opts, :strict_reference, false)
       }}
    end
  end

  defp parse_values(:cache, opts) do
    with {:ok, buckets} <- parse_buckets(Keyword.get(opts, :buckets)),
         :ok <- required(opts[:input], "--input PATH is required"),
         :ok <- required(opts[:output], "--output PATH is required") do
      {:ok, :cache,
       %{
         input: opts[:input],
         output: opts[:output],
         prefix_artifact: Keyword.get(opts, :prefix_artifact, @default_prefix_artifact),
         backend: Keyword.get(opts, :backend, "exla:rocm"),
         buckets: buckets,
         system_message: Keyword.get(opts, :system_message),
         thought_channel: Keyword.get(opts, :thought_channel, true),
         last_prompt_token_only: Keyword.get(opts, :last_prompt_token_only, false),
         verify_padding: Keyword.get(opts, :verify_padding),
         limit: Keyword.get(opts, :limit)
       }}
    end
  end

  defp parse_head_type(nil), do: {:ok, Trainer.defaults().head_type}
  defp parse_head_type("bf16"), do: {:ok, {:bf, 16}}
  defp parse_head_type("f32"), do: {:ok, {:f, 32}}

  defp parse_head_type(value),
    do: {:error, "--head-type must be bf16 or f32, got: #{value}"}

  defp parse_gate_mode(nil), do: {:ok, Trainer.defaults().gate_mode}
  defp parse_gate_mode("response"), do: {:ok, :response}
  defp parse_gate_mode("classifier"), do: {:ok, :classifier}

  defp parse_gate_mode(value),
    do: {:error, "--gate-mode must be response or classifier, got: #{value}"}

  defp parse_layers(nil), do: {:ok, Trainer.defaults().layers}

  defp parse_layers(value) do
    case parse_integers(value) do
      {:ok, layers} -> {:ok, layers}
      :error -> {:error, "--layers must be a comma-separated list of indices, got: #{value}"}
    end
  end

  defp parse_token_ids(nil), do: {:ok, @reference_token_ids}

  defp parse_token_ids(value) do
    case parse_integers(value) do
      {:ok, token_ids} -> {:ok, token_ids}
      :error -> {:error, "--expected must be a comma-separated list of token ids, got: #{value}"}
    end
  end

  defp parse_integers(value) do
    parsed =
      value |> String.split(",", trim: true) |> Enum.map(&Integer.parse(String.trim(&1)))

    if parsed != [] and Enum.all?(parsed, &match?({number, ""} when number >= 0, &1)) do
      {:ok, Enum.map(parsed, &elem(&1, 0))}
    else
      :error
    end
  end

  defp parse_buckets(nil), do: {:ok, @default_buckets}

  defp parse_buckets(value) do
    buckets =
      value
      |> String.split(",", trim: true)
      |> Enum.map(&Integer.parse(String.trim(&1)))

    if Enum.all?(buckets, &match?({length, ""} when length > 0, &1)) do
      {:ok, buckets |> Enum.map(&elem(&1, 0)) |> Enum.uniq() |> Enum.sort()}
    else
      {:error, "--buckets must be a comma-separated list of positive integers, got: #{value}"}
    end
  end

  @doc """
  The smallest bucket a prompt of `length` tokens fits in, given ascending
  `buckets`.

  Every distinct sequence length is a separate XLA executable. Padding to a
  handful of lengths compiles a handful of programs instead of one per row,
  which is most of the wall time of a small cache run.
  """
  def bucket(length, buckets) when is_integer(length) and length > 0 do
    case Enum.find(buckets, &(&1 >= length)) do
      nil ->
        {:error,
         "prompt is #{length} tokens, longer than the largest bucket #{List.last(buckets)}"}

      bucket ->
        {:ok, bucket}
    end
  end

  defp cache!(opts) do
    output = Path.expand(opts.output)

    if File.exists?(output) do
      abort("output path already exists: #{output}")
    end

    rows = read_rows!(opts.input, opts.limit)
    {:ok, backend} = Runtime.resolve_backend(opts.backend)

    artifact =
      timed!("prefix_artifact_load", fn ->
        {:ok, DecoderBlockArtifact.load_prefix!(opts.prefix_artifact, backend)}
      end)

    tokenizer = artifact.tokenizer || abort("prefix artifact has no tokenizer")
    spec = artifact.generation.spec

    IO.puts(
      Jason.encode!(%{
        event: "cache_ready",
        prefix_artifact: Path.expand(opts.prefix_artifact),
        last_layer: artifact.prefix.last_layer,
        backend: opts.backend,
        hidden_size: spec.hidden_size,
        rows: length(rows),
        buckets: opts.buckets,
        last_prompt_token_only: opts.last_prompt_token_only
      })
    )

    File.mkdir_p!(Path.join(output, @rows))

    entries =
      rows
      |> Enum.with_index()
      |> Enum.map(fn {row, index} ->
        cache_row!(row, index, artifact, tokenizer, spec, backend, output, opts)
      end)

    manifest = %{
      version: @version,
      kind: "system_one_prefix_cache",
      prefix_artifact: Path.expand(opts.prefix_artifact),
      last_layer: artifact.prefix.last_layer,
      hidden_size: spec.hidden_size,
      dtype: "f16",
      buckets: opts.buckets,
      last_prompt_token_only: opts.last_prompt_token_only,
      thought_channel: opts.thought_channel,
      system_message: opts.system_message,
      input: Path.expand(opts.input),
      rows: entries
    }

    File.write!(Path.join(output, @manifest), Jason.encode!(manifest, pretty: true))

    IO.puts(
      Jason.encode!(%{
        event: "cache_written",
        path: output,
        rows: length(entries),
        bytes: Enum.reduce(entries, 0, &(&1.bytes + &2))
      })
    )
  end

  defp cache_row!(row, index, artifact, tokenizer, spec, backend, output, opts) do
    id = Map.get(row, "id", "row-#{index}")

    input =
      Input.build_text(Prompt.render(row),
        system_message: system_message(row, opts),
        thought_channel: opts.thought_channel,
        response: Map.get(row, "target")
      )

    token_ids = tokenize!(tokenizer, input.prompt)
    head_ids = tokenize!(tokenizer, input.prompt_without_response)
    response_range = Input.response_range(input, token_ids, head_ids)
    length = length(token_ids)

    # `response_range/3` splits on token counts, so it is only the response
    # boundary if the two tokenizations agree up to it. They can disagree when
    # the prompt does not end on a special token (`--no-thought-channel` ends
    # it on a newline, which the tokenizer may merge with the response).
    if response_range && not List.starts_with?(token_ids, head_ids) do
      abort("#{id}: the forced response retokenizes the prompt, so its boundary is not a token")
    end

    bucket =
      case bucket(length, opts.buckets) do
        {:ok, bucket} -> bucket
        {:error, reason} -> abort("#{id}: #{reason}")
      end

    # The probe only ever reads the last prompt position, which is the token
    # before the teacher-forced response when there is one.
    last_prompt_index = if response_range, do: response_range.start - 1, else: length - 1

    {start, kept_length} =
      if opts.last_prompt_token_only, do: {last_prompt_index, 1}, else: {0, length}

    run = fn pad_length ->
      run_row!(id, artifact, input, spec, backend, token_ids, pad_length, start, kept_length)
    end

    {elapsed_us, hidden_state} = :timer.tc(fn -> run.(bucket) end)
    padding_check = verify_padding!(id, run, token_ids, opts.verify_padding, hidden_state)

    file = Path.join(@rows, "#{String.pad_leading("#{index}", 6, "0")}.safetensors")

    Safetensors.write!(Path.join(output, file), %{
      "hidden_state" => Nx.as_type(hidden_state, {:f, 16}),
      "input_ids" => Nx.tensor(token_ids, type: :s64, backend: Nx.BinaryBackend),
      "response_range" => response_range_tensor(response_range, length)
    })

    entry = %{
      id: id,
      file: file,
      # The trainer treats a `replay` row as base-model behaviour to preserve
      # and every other row as one the expert should answer.
      kind: Map.get(row, "kind", "system_one"),
      decidable: Map.get(row, "decidable"),
      length: length,
      bucket: bucket,
      last_prompt_index: last_prompt_index,
      response: response_range,
      bytes: File.stat!(Path.join(output, file)).size
    }

    IO.puts(
      Jason.encode!(
        entry
        |> Map.merge(%{event: "cache_row", index: index, elapsed_ms: div(elapsed_us, 1_000)})
        |> Map.merge(stats(hidden_state))
        |> Map.merge(padding_check)
      )
    )

    entry
  end

  defp run_row!(id, artifact, input, spec, backend, token_ids, pad_length, start, kept_length) do
    prepared = prepared_inputs(input, spec, backend, token_ids, pad_length)

    hidden_state =
      case DecoderPipeline.run_prefix(artifact.prefix, prepared) do
        {:ok, hidden_state} -> hidden_state
        {:error, reason} -> abort("#{id}: prefix run failed: #{reason}")
      end

    kept = host_slice(hidden_state, start, kept_length, spec)

    Nx.backend_deallocate(hidden_state)
    Nx.backend_deallocate(prepared)
    kept
  end

  # The prefix output comes back on the XLA client, where eager ops have
  # segfaulted this box, so the whole thing is copied to the host before
  # anything is sliced or cast. A padded prefix output is at most 5.6 MB.
  defp host_slice(hidden_state, start, length, spec) do
    hidden_state
    |> Nx.backend_copy(Nx.BinaryBackend)
    |> Nx.slice([0, start, 0], [1, length, spec.hidden_size])
    |> Nx.as_type({:f, 32})
  end

  # The cache is stored f16, 7.5 kB per token at 3840 wide, so a full-sequence
  # cache is the one thing here big enough to care about. f16 tops out at
  # 65504 and Gemma hidden states carry large outlier channels, so `max_abs`
  # is reported beside the finiteness count: it is what says whether the cast
  # is lossy in the way that matters.
  defp stats(slice) do
    non_finite =
      slice
      |> Nx.is_infinity()
      |> Nx.logical_or(Nx.is_nan(slice))
      |> Nx.sum()
      |> Nx.to_number()

    %{
      non_finite: non_finite,
      max_abs: slice |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    }
  end

  # Re-runs the row padded to a second length and reports how far the kept
  # slice moved. Right padding is masked out of attention, so a correct run
  # gives zero; anything else is an attention-masking bug.
  defp verify_padding!(_id, _run, _token_ids, nil, _kept), do: %{}

  defp verify_padding!(id, run, token_ids, pad_length, kept) do
    if pad_length < length(token_ids) do
      abort(
        "#{id}: --verify-padding #{pad_length} is shorter than the #{length(token_ids)}-token prompt"
      )
    end

    difference =
      kept |> Nx.subtract(run.(pad_length)) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    %{verify_padding: pad_length, padding_max_abs_diff: difference}
  end

  defp response_range_tensor(nil, length),
    do: Nx.tensor([length, 0], type: :s64, backend: Nx.BinaryBackend)

  defp response_range_tensor(range, _length),
    do: Nx.tensor([range.start, range.length], type: :s64, backend: Nx.BinaryBackend)

  # Padding is right-appended pad tokens masked out of attention: the real
  # positions never attend to them, so a bucketed prompt has the same prefix
  # output as an exact-length one.
  defp prepared_inputs(input, spec, backend, token_ids, bucket) do
    length = length(token_ids)
    padding = bucket - length

    Nx.with_default_backend(backend, fn ->
      %{
        "input_ids" =>
          Nx.tensor([token_ids ++ List.duplicate(spec.pad_token_id, padding)], type: :s64),
        "attention_mask" =>
          Nx.tensor([List.duplicate(1, length) ++ List.duplicate(0, padding)], type: :s64),
        "position_ids" => Nx.tensor([Enum.to_list(0..(bucket - 1))], type: :s64),
        "input_features" => Nx.backend_copy(Nx.new_axis(input.audio.input_features, 0), backend),
        "input_features_mask" =>
          Nx.backend_copy(Nx.new_axis(input.audio.attention_mask, 0), backend)
      }
    end)
  end

  defp train!(opts) do
    opts |> Map.to_list() |> Trainer.train()
  end

  defp generate!(opts) do
    output = Path.expand(opts.output)

    if File.exists?(output) do
      abort("output path already exists: #{output}")
    end

    rows = read_rows!(opts.input, opts.limit)

    {pipeline, artifact} =
      timed!("generate_pipeline_load", fn ->
        {:ok,
         SystemOneArtifact.load_pipeline!(
           prefix_artifact: opts.prefix_artifact,
           tail_artifact: opts.tail_artifact,
           expert: opts.expert,
           backend: opts.backend,
           force_router_closed: opts.force_router_closed,
           gate_floor: opts.gate_floor,
           logits_last_only: false
         )}
      end)

    tokenizer =
      pipeline.tail.tokenizer || pipeline.input_context.tokenizer ||
        abort("neither artifact carries a tokenizer")

    spec = pipeline.generation.spec
    probe = if opts.gate_probe, do: SystemOneArtifact.gate_probe(pipeline.tail)

    IO.puts(
      Jason.encode!(%{
        event: "generate_ready",
        rows: length(rows),
        expert: opts.expert,
        layers: SystemOne.layers(spec),
        gate_floor: SystemOne.gate_floor(spec),
        force_router_closed: opts.force_router_closed,
        gate_probe: not is_nil(probe),
        expert_step: artifact && artifact.step,
        buckets: opts.buckets,
        max_new_tokens: opts.max_new_tokens
      })
    )

    results =
      rows
      |> Enum.with_index()
      |> Enum.map(fn {row, index} ->
        generate_row!(row, index, pipeline, tokenizer, spec, probe, opts)
      end)

    File.mkdir_p!(Path.dirname(output))
    File.write!(output, Enum.map_join(results, &(Jason.encode!(&1) <> "\n")))

    IO.puts(Jason.encode!(%{event: "generate_written", path: output, rows: length(results)}))
  end

  defp generate_row!(row, index, pipeline, tokenizer, spec, probe, opts) do
    id = Map.get(row, "id", "row-#{index}")
    backend = pipeline.prefix.backend

    input =
      Input.build_text(Prompt.render(row),
        system_message: system_message(row, opts),
        thought_channel: opts.thought_channel
      )

    token_ids = tokenize!(tokenizer, input.prompt)
    prompt_length = length(token_ids)

    bucket =
      case bucket(prompt_length, opts.buckets) do
        {:ok, bucket} -> bucket
        {:error, reason} -> abort("#{id}: #{reason}")
      end

    prepared = prepared_inputs(input, spec, backend, token_ids, bucket)

    # The prompt is right-padded to the bucket, so generation continues from
    # the last real position instead of the last one in the tensor. The padded
    # positions are masked out of attention and their cache entries stay
    # masked for the decode steps.
    {elapsed_us, generated} =
      :timer.tc(fn ->
        DecoderPipeline.generate_prepared(pipeline, prepared,
          max_new_tokens: opts.max_new_tokens,
          thought_channel: opts.thought_channel,
          logits_index: prompt_length - 1
        )
      end)

    generated =
      case generated do
        {:ok, token_ids} -> token_ids
        {:error, reason} -> abort("#{id}: generation failed: #{reason}")
      end

    gate = gate_value(probe, pipeline, prepared, prompt_length)
    Nx.backend_deallocate(prepared)

    result =
      Map.merge(row, %{
        "id" => id,
        "reply" => Transcript.decode(tokenizer, generated),
        "token_ids" => generated,
        "tokens" => length(generated),
        "ms" => div(elapsed_us, 1_000),
        "gate" => gate && gate.mean,
        "gate_open_fraction" => gate && gate.open_fraction,
        "gate_last" => gate && gate.last,
        "prompt_tokens" => prompt_length,
        "bucket" => bucket
      })

    IO.puts(
      Jason.encode!(%{
        event: "generate_row",
        index: index,
        id: id,
        tokens: result["tokens"],
        ms: result["ms"],
        gate: result["gate"],
        gate_open_fraction: result["gate_open_fraction"],
        gate_last: result["gate_last"]
      })
    )

    result
  end

  # One more prefix pass and one gate-only pass over the tail blocks, which is
  # what reporting a gate beside a reply costs; `--no-gate-probe` skips both.
  defp gate_value(nil, _pipeline, _prepared, _prompt_length), do: nil

  defp gate_value(probe, pipeline, prepared, prompt_length) do
    hidden_state =
      case DecoderPipeline.run_prefix(pipeline.prefix, prepared) do
        {:ok, hidden_state} -> hidden_state
        {:error, reason} -> abort("gate probe failed: #{reason}")
      end

    gates =
      probe.(%{
        "hidden_state" => hidden_state,
        "position_ids" => prepared["position_ids"],
        "attention_mask" => prepared["attention_mask"]
      })

    Nx.backend_deallocate(hidden_state)
    mean_gate(gates, prompt_length)
  end

  # The probe runs in inference mode, so a gate under the floor is already
  # clamped to exactly 0. `open_fraction` is therefore the share of prompt
  # positions where the router actually routes, which is the number the plan's
  # replay gate is stated in; the mean is what the Brier score is read off.
  # Both cover prompt positions only, not the generated continuation. `last`
  # is the gate at the final prompt position, the one whose hidden state
  # predicts the first reply token: it is the router's decision on the
  # request, where the mean over the prompt dilutes it by the prompt length.
  defp mean_gate(gates, prompt_length) do
    per_layer =
      Enum.map(gates, fn {_layer, gate} ->
        values =
          gate
          |> Nx.backend_copy(Nx.BinaryBackend)
          |> Nx.reshape({:auto})
          |> Nx.slice([0], [prompt_length])

        {Nx.to_number(Nx.mean(values)), values |> Nx.greater(0.0) |> Nx.mean() |> Nx.to_number(),
         Nx.to_number(values[prompt_length - 1])}
      end)

    case per_layer do
      [] ->
        nil

      per_layer ->
        count = length(per_layer)

        %{
          mean: per_layer |> Enum.map(&elem(&1, 0)) |> Enum.sum() |> Kernel./(count),
          open_fraction: per_layer |> Enum.map(&elem(&1, 1)) |> Enum.sum() |> Kernel./(count),
          last: per_layer |> Enum.map(&elem(&1, 2)) |> Enum.sum() |> Kernel./(count)
        }
    end
  end

  defp regress!(opts) do
    {:ok, backend} = Runtime.resolve_backend(opts.backend)

    prefix =
      timed!("prefix_artifact_load", fn ->
        {:ok, DecoderBlockArtifact.load_prefix!(opts.prefix_artifact, backend)}
      end)

    tail =
      timed!("tail_artifact_load", fn ->
        {:ok, DecoderBlockArtifact.load_tail!(opts.tail_artifact, backend)}
      end)

    samples =
      opts.wav
      |> Path.expand()
      |> Audio.read_wav_samples!(@sample_rate)
      |> Enum.take(round(@sample_rate * opts.seconds))

    input = Input.build(samples, prompt: Config.default_prompt())

    # How the reference ids were produced, so the comparison is like for like:
    # `decoder_pipeline_benchmark.ex`'s artifact run, the first 5 seconds of
    # journal1.wav at 16 kHz, the default prompt, greedy, 32 new tokens. The
    # ids in the plan are from the bf16 split pipeline; a packed W4A16 run of
    # the same audio decodes differently, which is why the gate that decides
    # the exit code is the expert pipeline against the same pipeline with no
    # expert installed.
    IO.puts(
      Jason.encode!(%{
        event: "regress_configured_as",
        wav: Path.expand(opts.wav),
        sample_rate: @sample_rate,
        seconds: opts.seconds,
        samples: length(samples),
        prompt: Config.default_prompt(),
        thought_channel: true,
        max_new_tokens: opts.max_new_tokens,
        min_new_tokens: 0,
        execution: :composed,
        prefix_artifact: Path.expand(opts.prefix_artifact),
        tail_artifact: Path.expand(opts.tail_artifact),
        expert: opts.expert,
        gate_floor: opts.gate_floor || 1.0,
        reference_token_ids: opts.expected,
        reference_run: "bf16 split pipeline, README 'Splitting raw-audio inference'"
      })
    )

    # Without `--gate-floor` the expert is installed with every gate clamped
    # to 0, which is the gate that says the expert subgraph itself changes
    # nothing. With a floor the router is left to decide, so the same run
    # asks the harder question: on audio the expert was never trained for,
    # does it stay shut at the floor inference actually uses?
    {closed_pipeline, _artifact} =
      SystemOneArtifact.build_pipeline!(prefix, tail, backend,
        expert: opts.expert,
        force_router_closed: is_nil(opts.gate_floor),
        gate_floor: opts.gate_floor
      )

    run_name = if opts.gate_floor, do: "router_at_floor", else: "router_closed"
    closed = regress_run!(run_name, closed_pipeline, input, opts)

    base =
      if opts.compare_base do
        {base_pipeline, nil} = SystemOneArtifact.build_pipeline!(prefix, tail, backend)
        regress_run!("base", base_pipeline, input, opts)
      end

    report(closed, base, opts)
  end

  defp regress_run!(name, pipeline, input, opts) do
    {elapsed_us, result} =
      :timer.tc(fn ->
        DecoderPipeline.generate(pipeline, input,
          max_new_tokens: opts.max_new_tokens,
          min_new_tokens: 0
        )
      end)

    case result do
      {:ok, %{token_ids: token_ids, text: text}} ->
        entry = %{run: name, token_ids: token_ids, text: text, ms: div(elapsed_us, 1_000)}
        IO.puts(Jason.encode!(Map.put(entry, :event, "regress_run")))
        entry

      {:error, reason} ->
        abort("#{name} run failed: #{reason}")
    end
  end

  defp report(closed, base, opts) do
    reference_prefix = Enum.take(closed.token_ids, length(opts.expected))
    reference_match = reference_prefix == opts.expected
    base_match = is_nil(base) or closed.token_ids == base.token_ids
    passed = base_match and (reference_match or not opts.strict_reference)

    IO.puts(
      Jason.encode!(%{
        event: "regress_result",
        passed: passed,
        base_compared: not is_nil(base),
        base_match: base_match,
        reference_match: reference_match,
        reference_prefix: reference_prefix,
        expected: opts.expected,
        strict_reference: opts.strict_reference,
        token_ids: closed.token_ids,
        text: closed.text
      })
    )

    if passed, do: 0, else: 1
  end

  # A replay row may carry its own system turn, because the prompt it replays
  # was written with one. `--system-message` is the fallback for every other
  # row, so the two baselines can be run over the same file.
  defp system_message(row, opts) do
    case Map.get(row, "system") do
      system when is_binary(system) -> system
      _ -> opts.system_message
    end
  end

  defp tokenize!(tokenizer, text) do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      tokenizer
      |> Bumblebee.apply_tokenizer([text])
      |> Map.fetch!("input_ids")
      |> Nx.to_flat_list()
    end)
  end

  defp read_rows!(path, limit) do
    stream =
      path
      |> Path.expand()
      |> File.stream!()
      |> Stream.map(&String.trim/1)
      |> Stream.reject(&(&1 == ""))
      |> Stream.map(&Jason.decode!/1)

    case limit do
      nil -> Enum.to_list(stream)
      limit -> Enum.take(stream, limit)
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

  defp abort(message) do
    IO.puts(:stderr, "error: #{message}")
    System.halt(1)
  end

  defp usage do
    defaults = Trainer.defaults()

    """
    usage: mix gemma.system_one cache|train|generate|regress [options]

    cache: run the packed prefix over a JSONL of items and store layer 45's input

      --input PATH               JSONL of items (id, state, question, options, target), or
                                 replay rows (id, prompt, target, optional system,
                                 kind: replay), required
      --output PATH              Cache directory to create, required
      --prefix-artifact PATH     Packed prefix artifact, default #{@default_prefix_artifact}
      --backend NAME             Nx backend, default exla:rocm
      --buckets LIST             Padded sequence lengths, default #{Enum.join(@default_buckets, ",")}
      --system-message TEXT      System turn prepended to every item, default none
      --no-thought-channel       End the prompt at the model turn instead of an empty thought channel
      --last-prompt-token-only   Store only the hidden state at the final prompt position
      --verify-padding N         Re-run every row padded to N and report how far the kept
                                 slice moves; right padding is masked out, so a correct
                                 run reports 0
      --limit N                  Cache only the first N rows

    train: fit the experts and routers on a full-sequence cache

      --cache PATH               Cache directory from `cache`, required
      --output PATH              Expert artifact directory to write, required
      --tail-artifact PATH       Packed tail artifact, default #{@default_tail_artifact}
      --layers LIST              Layers to attach experts to, default #{Enum.join(defaults.layers, ",")}
      --backend NAME             Nx backend of the training step, default exla:rocm
      --stage-backend NAME       Backend the cache and dequantization stage on, default torchx:cpu
      --batch-size N             Rows per step, default #{defaults.batch_size}
      --epochs N                 Passes over the cache, default #{defaults.epochs}
      --max-response-tokens N    Response positions the head is applied to, default #{defaults.max_response_tokens}
      --expert-lr F              Expert learning rate, default #{defaults.expert_learning_rate}
      --router-lr F              Router learning rate, default #{defaults.router_learning_rate}
      --max-grad-norm F          Global gradient norm clip, default #{defaults.max_grad_norm}
      --kl-weight F              Replay KL weight, default #{defaults.kl_weight}
      --gate-open-weight F       Weight of the open half of the router BCE, default #{defaults.gate_open_weight}
      --gate-closed-weight F     Weight of the closed half of the router BCE, default #{defaults.gate_closed_weight}
      --gate-floor F             Inference gate floor recorded in the artifact, default #{defaults.gate_floor}
      --lead-tokens N            Response tokens treated as the decision, default #{defaults.lead_tokens}
      --lead-weight F            Cross-entropy weight on those tokens, default #{defaults.lead_weight}
      --gate-mode response|classifier
                                 response: the gate opens on every System One reply and the
                                 expert answers or asks; classifier: it opens only on
                                 underspecified items and base Gemma answers the decidable
                                 ones, default #{defaults.gate_mode}
      --init-from PATH           Start from another expert artifact's weights, with a
                                 fresh optimizer state and this run's gate floor
      --checkpoint-every N       Steps between checkpoints, default #{defaults.checkpoint_every}
      --log-every N              Steps between progress lines, default #{defaults.log_every}
      --max-steps N              Stop after N steps
      --limit N                  Train on the first N cached rows only
      --seed N                   Shuffle seed, default #{defaults.seed}
      --head-type bf16|f32       Type of the 262k vocabulary projection, default f32. bf16
                                 is strictly worse here: ROCm has no bf16 GEMM for this
                                 shape with the autotuner and Triton off, so XLA converts
                                 the kernel back to f32 for the dot and the step holds
                                 both types, 3.75 GiB more rather than less
      --no-resume                Ignore the checkpoints under --output
      --dry-run                  Build the graph, print the plan, touch no device

    generate: answer a JSONL of eval items, the input of scripts/system_one/scorecard.py

      --input PATH               JSONL of items, or rows with a bare `prompt`, required
      --output PATH              JSONL to write (the same rows plus reply, tokens, ms, gate), required
      --expert PATH              Expert artifact, default none (base Gemma)
      --prefix-artifact PATH     Packed prefix artifact, default #{@default_prefix_artifact}
      --tail-artifact PATH       Packed tail artifact, default #{@default_tail_artifact}
      --backend NAME             Nx backend, default exla:rocm
      --buckets LIST             Padded prompt lengths, default #{Enum.join(@default_buckets, ",")}
      --system-message TEXT      System turn prepended to every item, default none
      --no-thought-channel       End the prompt at the model turn
      --max-new-tokens N         Generation budget per item, default 64
      --force-router-closed      Clamp every gate to 0, the base-model baseline
      --gate-floor F             Override the floor the expert artifact was saved with
      --no-gate-probe            Skip the extra pass that reports the mean gate and the
                                 fraction of prompt positions the router opens on

    regress: the non-regression gate, the router forced closed must still be base Gemma

      --expert PATH              Expert artifact to install with its router closed
      --prefix-artifact PATH     Packed prefix artifact, default #{@default_prefix_artifact}
      --tail-artifact PATH       Packed tail artifact, default #{@default_tail_artifact}
      --backend NAME             Nx backend, default exla:rocm
      --wav PATH                 Audio to transcribe, default #{@default_wav}
      --seconds F                Window from the start of the file, default 5.0
      --max-new-tokens N         Generation budget, default 32
      --gate-floor F             Let the router decide at this floor instead of clamping
                                 every gate to 0, so the gate is a real inference run
      --expected LIST            Reference token ids, default the plan's bf16 ids
      --no-compare-base          Skip the run without an expert, which is the real gate
      --strict-reference         Also fail when the reference ids do not match
    """
  end
end
