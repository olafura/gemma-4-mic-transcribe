defmodule Gemma4MicTranscribe.ExpertCLI do
  @moduledoc false

  alias Gemma4MicTranscribe.Gemma4.ExpertArtifact
  alias Gemma4MicTranscribe.Gemma4.ExpertCaller
  alias Gemma4MicTranscribe.Gemma4.ExpertCallerArtifact
  alias Gemma4MicTranscribe.Gemma4.ExtractedDecoderLayer
  alias Gemma4MicTranscribe.Gemma4.ExtractedExpert
  alias Gemma4MicTranscribe.Gemma4.ExtractedMoeLayer
  alias Gemma4MicTranscribe.Gemma4.ExtractedOutputHead
  alias Gemma4MicTranscribe.Gemma4.MathExpertProfiler
  alias Gemma4MicTranscribe.Gemma4.MoeLayerArtifact
  alias Gemma4MicTranscribe.Gemma4.OutputHeadArtifact
  alias Gemma4MicTranscribe.Gemma4Unified.Runtime

  @switches [
    artifact: :string,
    artifact_prefix: :string,
    caller_artifact: :string,
    expert_artifact: :string,
    head_artifact: :string,
    next_artifact: :keep,
    next_caller_artifact: :keep,
    repo: :string,
    revision: :string,
    layer: :integer,
    last_layer: :integer,
    expert: :integer,
    backend: :string,
    tokens: :integer,
    runs: :integer,
    limit: :integer,
    input_value: :float,
    expert_scale: :float,
    text: :string,
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

      {:extract_layer, opts} ->
        manifest =
          MoeLayerArtifact.extract!(opts.artifact,
            repo: opts.repo,
            revision: opts.revision,
            layer: opts.layer
          )

        print_layer_manifest("moe_layer_extracted", opts.artifact, manifest)
        0

      {:inspect_layer, artifact} ->
        print_layer_manifest(
          "moe_layer_artifact",
          artifact,
          MoeLayerArtifact.read_manifest!(artifact)
        )

        0

      {:extract_head, opts} ->
        manifest =
          OutputHeadArtifact.extract!(opts.artifact,
            repo: opts.repo,
            revision: opts.revision
          )

        IO.puts(
          Jason.encode!(%{
            event: "output_head_extracted",
            artifact: Path.expand(opts.artifact),
            parameter_count: manifest.parameter_count,
            parameter_bytes: manifest.parameter_bytes,
            vocab_size: manifest.vocab_size,
            hidden_size: manifest.hidden_size
          })
        )

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

      {:run_layer, opts} ->
        {:ok, backend} = Runtime.resolve_backend(opts.backend)
        layer = ExtractedMoeLayer.load!(opts.artifact, backend)

        input =
          Nx.broadcast(
            Nx.tensor(opts.input_value, type: :f32),
            {opts.tokens, layer.manifest.hidden_size}
          )

        Enum.each(1..3, fn _ -> :ok = ExtractedMoeLayer.warmup(layer, opts.tokens) end)

        {times, result} =
          Enum.map_reduce(1..opts.runs, nil, fn _run, _last_result ->
            started_at = System.monotonic_time(:microsecond)

            result =
              layer
              |> ExtractedMoeLayer.run(input)
              |> Nx.backend_copy(Nx.BinaryBackend)

            elapsed = System.monotonic_time(:microsecond) - started_at
            {elapsed, result}
          end)

        output_f32 = Nx.as_type(result.output, :f32)
        sorted_times = Enum.sort(times)

        IO.puts(
          Jason.encode!(%{
            event: "moe_layer_run",
            artifact: Path.expand(opts.artifact),
            backend: opts.backend,
            layer: layer.manifest.layer_index,
            experts: layer.manifest.num_experts,
            top_k: layer.manifest.top_k_experts,
            tokens: opts.tokens,
            runs: opts.runs,
            mean_us: Enum.sum(times) / length(times),
            min_us: Enum.min(times),
            median_us: percentile(sorted_times, 0.5),
            p95_us: percentile(sorted_times, 0.95),
            selected_experts: Nx.to_list(result.top_k_indices),
            selected_weights: Nx.to_list(result.top_k_weights),
            output_shape: Tuple.to_list(Nx.shape(result.output)),
            output_mean_abs: output_f32 |> Nx.abs() |> Nx.mean() |> Nx.to_number(),
            output_max_abs: output_f32 |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
          })
        )

        0

      {:profile_math, opts} ->
        {:ok, backend} = Runtime.resolve_backend(opts.backend)

        report =
          MathExpertProfiler.profile!(opts.artifact, backend: backend)
          |> Map.update!(:experts, &Enum.take(&1, opts.limit))

        IO.puts(Jason.encode!(report))
        0

      {:extract_caller, opts} ->
        manifest =
          ExpertCallerArtifact.extract!(opts.artifact,
            repo: opts.repo,
            revision: opts.revision,
            layer: opts.layer
          )

        IO.puts(
          Jason.encode!(%{
            event: "expert_caller_extracted",
            artifact: Path.expand(opts.artifact),
            layer: manifest.layer_index,
            parameter_count: manifest.parameter_count,
            parameter_bytes: manifest.parameter_bytes
          })
        )

        0

      {:call_chain, opts} ->
        {:ok, backend} = Runtime.resolve_backend(opts.backend)

        layer_0 =
          ExpertCaller.load_layer!(
            opts.caller_artifact,
            opts.artifact,
            opts.expert_artifact,
            backend
          )

        {embedding_data, first_device} =
          ExpertCaller.call_layer_device!(layer_0, opts.text, expert_scale: opts.expert_scale)

        first_layer_index = layer_0.manifest.layer_index
        expert_index = layer_0.expert.manifest.expert_index

        first_report =
          decoder_layer_report(
            first_layer_index,
            first_device,
            %{output: first_device.baseline_output}
          )

        unload_first_layer(layer_0)

        {final_output, baseline_final_output, layer_reports} =
          run_decoder_chain!(
            opts.layers,
            backend,
            first_layer_index + 1,
            first_device.output,
            first_device.baseline_output,
            [first_report]
          )

        {predictions, baseline_predictions} =
          output_predictions(
            opts[:head_artifact],
            backend,
            final_output,
            baseline_final_output,
            embedding_data.tokenizer
          )

        IO.puts(
          Jason.encode!(%{
            event: "decoder_layer_chain_called",
            layers: Enum.map(layer_reports, & &1.layer),
            text: opts.text,
            expert_override: expert_index,
            expert_scale: opts.expert_scale,
            override_route_count:
              first_device.override_route_count
              |> Nx.backend_copy(Nx.BinaryBackend)
              |> Nx.to_number(),
            tokens:
              embedding_data.tokens
              |> Enum.with_index()
              |> Enum.map(fn {token, position} ->
                %{position: position, id: token.id, token: token.token}
              end),
            layer_reports: layer_reports,
            final_output_shape: Tuple.to_list(Nx.shape(final_output)),
            predictions: predictions,
            baseline_predictions: baseline_predictions
          })
        )

        0

      {:call_expert, opts} ->
        {:ok, backend} = Runtime.resolve_backend(opts.backend)

        caller =
          ExpertCaller.load!(
            opts.caller_artifact,
            opts.artifact,
            opts.expert_artifact,
            backend
          )

        report = ExpertCaller.call_text!(caller, opts.text)
        output = report.expert_outputs

        IO.puts(
          Jason.encode!(%{
            event: "expert_called",
            expert: report.expert,
            text: report.text,
            tokens:
              report.tokens
              |> Enum.with_index()
              |> Enum.map(fn {token, position} ->
                %{position: position, id: token.id, token: token.token}
              end),
            selected_calls: report.selected_calls,
            expert_input_shape: Tuple.to_list(Nx.shape(report.expert_inputs)),
            expert_output_shape: if(output, do: Tuple.to_list(Nx.shape(output))),
            expert_output_mean_abs:
              if(output,
                do: output |> Nx.as_type(:f32) |> Nx.abs() |> Nx.mean() |> Nx.to_number()
              )
          })
        )

        0

      {:call_layer, opts} ->
        {:ok, backend} = Runtime.resolve_backend(opts.backend)

        caller =
          ExpertCaller.load_layer!(
            opts.caller_artifact,
            opts.artifact,
            opts.expert_artifact,
            backend
          )

        report =
          ExpertCaller.call_layer_text!(caller, opts.text, expert_scale: opts.expert_scale)

        output_f32 = Nx.as_type(report.layer_output, :f32)
        baseline_f32 = Nx.as_type(report.baseline_layer_output, :f32)

        IO.puts(
          Jason.encode!(%{
            event: "layer_called_with_expert_override",
            expert: report.expert,
            expert_scale: report.expert_scale,
            text: report.text,
            tokens:
              report.tokens
              |> Enum.with_index()
              |> Enum.map(fn {token, position} ->
                %{position: position, id: token.id, token: token.token}
              end),
            selected_calls: report.selected_calls,
            override_route_count: report.override_route_count,
            expert_input_shape: Tuple.to_list(Nx.shape(report.expert_inputs)),
            expert_output_shape: Tuple.to_list(Nx.shape(report.expert_outputs)),
            layer_output_shape: Tuple.to_list(Nx.shape(report.layer_output)),
            layer_output_mean_abs: output_f32 |> Nx.abs() |> Nx.mean() |> Nx.to_number(),
            baseline_output_mean_abs: baseline_f32 |> Nx.abs() |> Nx.mean() |> Nx.to_number(),
            layer_output_delta_mean_abs: report.layer_output_delta_mean_abs,
            layer_output_delta_max_abs: report.layer_output_delta_max_abs
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

  def parse(["extract-layer" | argv]) do
    with {:ok, opts} <- parse_options(argv),
         :ok <- require_artifact(opts),
         :ok <- non_negative(opts[:layer] || 0, "--layer") do
      if opts[:help] do
        {:help, usage()}
      else
        {:extract_layer,
         %{
           artifact: opts[:artifact],
           repo: opts[:repo] || "google/gemma-4-26B-A4B-it",
           revision: opts[:revision] || "main",
           layer: opts[:layer] || 0
         }}
      end
    end
  end

  def parse(["inspect-layer" | argv]) do
    with {:ok, opts} <- parse_options(argv),
         :ok <- require_artifact(opts) do
      if opts[:help], do: {:help, usage()}, else: {:inspect_layer, opts[:artifact]}
    end
  end

  def parse(["extract-head" | argv]) do
    with {:ok, opts} <- parse_options(argv),
         :ok <- require_artifact(opts) do
      if opts[:help] do
        {:help, usage()}
      else
        {:extract_head,
         %{
           artifact: opts[:artifact],
           repo: opts[:repo] || "google/gemma-4-26B-A4B-it",
           revision: opts[:revision] || "main"
         }}
      end
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

  def parse(["run-layer" | argv]) do
    with {:ok, opts} <- parse_options(argv),
         :ok <- require_artifact(opts),
         :ok <- positive(opts[:tokens] || 1, "--tokens"),
         :ok <- positive(opts[:runs] || 3, "--runs") do
      if opts[:help] do
        {:help, usage()}
      else
        {:run_layer,
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

  def parse(["profile-math" | argv]) do
    with {:ok, opts} <- parse_options(argv),
         :ok <- require_artifact(opts),
         :ok <- positive(opts[:limit] || 10, "--limit") do
      if opts[:help] do
        {:help, usage()}
      else
        {:profile_math,
         %{
           artifact: opts[:artifact],
           backend: opts[:backend] || "exla:rocm",
           limit: opts[:limit] || 10
         }}
      end
    end
  end

  def parse(["extract-caller" | argv]) do
    with {:ok, opts} <- parse_options(argv),
         :ok <- require_artifact(opts),
         :ok <- non_negative(opts[:layer] || 0, "--layer") do
      if opts[:help] do
        {:help, usage()}
      else
        {:extract_caller,
         %{
           artifact: opts[:artifact],
           repo: opts[:repo] || "google/gemma-4-26B-A4B-it",
           revision: opts[:revision] || "main",
           layer: opts[:layer] || 0
         }}
      end
    end
  end

  def parse(["call-expert" | argv]) do
    with {:ok, opts} <- parse_options(argv),
         :ok <- require_artifact(opts),
         :ok <- require_string(opts, :caller_artifact, "--caller-artifact PATH"),
         :ok <- require_string(opts, :expert_artifact, "--expert-artifact PATH"),
         :ok <- require_string(opts, :text, "--text TEXT") do
      if opts[:help] do
        {:help, usage()}
      else
        {:call_expert,
         %{
           artifact: opts[:artifact],
           caller_artifact: opts[:caller_artifact],
           expert_artifact: opts[:expert_artifact],
           text: opts[:text],
           backend: opts[:backend] || "exla:rocm"
         }}
      end
    end
  end

  def parse(["call-layer" | argv]) do
    with {:ok, opts} <- parse_options(argv),
         :ok <- require_artifact(opts),
         :ok <- require_string(opts, :caller_artifact, "--caller-artifact PATH"),
         :ok <- require_string(opts, :expert_artifact, "--expert-artifact PATH"),
         :ok <- require_string(opts, :text, "--text TEXT") do
      if opts[:help] do
        {:help, usage()}
      else
        {:call_layer,
         %{
           artifact: opts[:artifact],
           caller_artifact: opts[:caller_artifact],
           expert_artifact: opts[:expert_artifact],
           text: opts[:text],
           expert_scale: opts[:expert_scale] || 1.0,
           backend: opts[:backend] || "exla:rocm"
         }}
      end
    end
  end

  def parse(["call-chain" | argv]) do
    with {:ok, opts} <- parse_options(argv),
         :ok <- require_artifact(opts),
         :ok <- require_string(opts, :caller_artifact, "--caller-artifact PATH"),
         :ok <- require_string(opts, :expert_artifact, "--expert-artifact PATH"),
         {:ok, layers} <- parse_layer_pairs(opts),
         :ok <- require_string(opts, :text, "--text TEXT") do
      if opts[:help] do
        {:help, usage()}
      else
        {:call_chain,
         %{
           artifact: opts[:artifact],
           caller_artifact: opts[:caller_artifact],
           expert_artifact: opts[:expert_artifact],
           head_artifact: opts[:head_artifact],
           layers: layers,
           text: opts[:text],
           expert_scale: opts[:expert_scale] || 1.0,
           backend: opts[:backend] || "exla:rocm"
         }}
      end
    end
  end

  def parse(["call-prefix" | argv]) do
    with {:ok, opts} <- parse_options(argv),
         :ok <- require_string(opts, :artifact_prefix, "--artifact-prefix PATH"),
         :ok <- require_string(opts, :expert_artifact, "--expert-artifact PATH"),
         :ok <- require_string(opts, :text, "--text TEXT"),
         :ok <- decoder_layer_index(opts[:last_layer] || 29, "--last-layer") do
      if opts[:help] do
        {:help, usage()}
      else
        last_layer = opts[:last_layer] || 29
        prefix = opts[:artifact_prefix]

        {:call_chain,
         %{
           artifact: layer_artifact(prefix, 0, "moe"),
           caller_artifact: layer_artifact(prefix, 0, "caller"),
           expert_artifact: opts[:expert_artifact],
           head_artifact: opts[:head_artifact],
           layers:
             Enum.map(1..last_layer, fn layer ->
               %{
                 caller_artifact: layer_artifact(prefix, layer, "caller"),
                 moe_artifact: layer_artifact(prefix, layer, "moe")
               }
             end),
           text: opts[:text],
           expert_scale: opts[:expert_scale] || 1.0,
           backend: opts[:backend] || "exla:rocm"
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

  defp require_string(opts, key, label) do
    if is_binary(opts[key]), do: :ok, else: {:error, "#{label} is required"}
  end

  defp parse_layer_pairs(opts) do
    moe_artifacts = Keyword.get_values(opts, :next_artifact)
    caller_artifacts = Keyword.get_values(opts, :next_caller_artifact)

    cond do
      moe_artifacts == [] ->
        {:error, "at least one --next-artifact PATH is required"}

      caller_artifacts == [] ->
        {:error, "at least one --next-caller-artifact PATH is required"}

      length(moe_artifacts) != length(caller_artifacts) ->
        {:error, "--next-artifact and --next-caller-artifact counts must match"}

      true ->
        {:ok,
         Enum.zip_with(caller_artifacts, moe_artifacts, fn caller, moe ->
           %{caller_artifact: caller, moe_artifact: moe}
         end)}
    end
  end

  defp run_decoder_chain!(
         [],
         _backend,
         _expected_layer,
         output,
         baseline_output,
         reports
       ) do
    {output, baseline_output, Enum.reverse(reports)}
  end

  defp run_decoder_chain!(
         [artifact | rest],
         backend,
         expected_layer,
         input,
         baseline_input,
         reports
       ) do
    layer =
      ExtractedDecoderLayer.load!(
        artifact.caller_artifact,
        artifact.moe_artifact,
        backend
      )

    unless layer.manifest.layer_index == expected_layer do
      raise ArgumentError,
            "expected decoder layer #{expected_layer}, got #{layer.manifest.layer_index}"
    end

    result = ExtractedDecoderLayer.run(layer, input)
    baseline_result = ExtractedDecoderLayer.run(layer, baseline_input)
    report = decoder_layer_report(expected_layer, result, baseline_result)
    unload_decoder_layer(layer)

    run_decoder_chain!(
      rest,
      backend,
      expected_layer + 1,
      result.output,
      baseline_result.output,
      [report | reports]
    )
  end

  defp unload_first_layer(layer) do
    Nx.backend_deallocate(layer.attention_params)
    Nx.backend_deallocate(layer.moe_params)
    Nx.backend_deallocate(layer.expert.params)
  end

  defp unload_decoder_layer(layer) do
    Nx.backend_deallocate(layer.attention_params)
    Nx.backend_deallocate(layer.moe_params)
  end

  defp decoder_layer_report(layer, result, baseline_result) do
    output = Nx.backend_copy(result.output, Nx.BinaryBackend)
    baseline_output = Nx.backend_copy(baseline_result.output, Nx.BinaryBackend)

    delta =
      output
      |> Nx.as_type(:f32)
      |> Nx.subtract(Nx.as_type(baseline_output, :f32))
      |> Nx.abs()

    %{
      layer: layer,
      output_shape: Tuple.to_list(Nx.shape(output)),
      output_mean_abs: output |> Nx.as_type(:f32) |> Nx.abs() |> Nx.mean() |> Nx.to_number(),
      output_delta_mean_abs: delta |> Nx.mean() |> Nx.to_number(),
      output_delta_max_abs: delta |> Nx.reduce_max() |> Nx.to_number(),
      selected_experts: result.top_k_indices |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_list(),
      selected_weights: result.top_k_weights |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_list()
    }
  end

  defp output_predictions(nil, _backend, _output, _baseline_output, _tokenizer),
    do: {nil, nil}

  defp output_predictions(artifact, backend, output, baseline_output, tokenizer) do
    head = ExtractedOutputHead.load!(artifact, backend)
    predictions = ExtractedOutputHead.run(head, output) |> prediction_rows(tokenizer)
    baseline = ExtractedOutputHead.run(head, baseline_output) |> prediction_rows(tokenizer)
    {predictions, baseline}
  end

  defp prediction_rows(result, tokenizer) do
    ids =
      result.top_k_indices
      |> Nx.backend_copy(Nx.BinaryBackend)
      |> Nx.to_flat_list()

    values =
      result.top_k_values
      |> Nx.backend_copy(Nx.BinaryBackend)
      |> Nx.to_flat_list()

    Enum.zip_with(ids, values, fn id, logit ->
      %{
        id: id,
        token:
          Bumblebee.Tokenizer.id_to_token(tokenizer, id) ||
            Bumblebee.Tokenizer.decode(tokenizer, [id]),
        logit: logit
      }
    end)
  end

  defp positive(value, _name) when is_integer(value) and value > 0, do: :ok
  defp positive(_value, name), do: {:error, "#{name} must be positive"}
  defp non_negative(value, _name) when is_integer(value) and value >= 0, do: :ok
  defp non_negative(_value, name), do: {:error, "#{name} must be non-negative"}

  defp decoder_layer_index(value, _name) when is_integer(value) and value in 1..29, do: :ok

  defp decoder_layer_index(_value, name),
    do: {:error, "#{name} must be an integer from 1 through 29"}

  defp layer_artifact(prefix, layer, kind), do: "#{prefix}-layer#{layer}-#{kind}"

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

  defp print_layer_manifest(event, artifact, manifest) do
    IO.puts(
      Jason.encode!(%{
        event: event,
        artifact: Path.expand(artifact),
        source_repo: manifest.source_repo,
        layer: manifest.layer_index,
        hidden_size: manifest.hidden_size,
        shared_intermediate_size: manifest.shared_intermediate_size,
        expert_intermediate_size: manifest.expert_intermediate_size,
        experts: manifest.num_experts,
        top_k: manifest.top_k_experts,
        parameter_count: manifest.parameter_count,
        parameter_bytes: manifest.parameter_bytes,
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
      expert_tool extract-layer --artifact PATH [options]
      expert_tool inspect-layer --artifact PATH
      expert_tool extract-head --artifact PATH [options]
      expert_tool run-layer --artifact PATH [options]
      expert_tool profile-math --artifact PATH [options]
      expert_tool extract-caller --artifact PATH [--layer INDEX]
      expert_tool call-expert --artifact MOE_PATH --caller-artifact PATH
                              --expert-artifact PATH --text TEXT [options]
      expert_tool call-layer --artifact MOE_PATH --caller-artifact PATH
                             --expert-artifact PATH --text TEXT [options]
      expert_tool call-chain --artifact MOE_PATH --caller-artifact PATH
                             --expert-artifact PATH --next-artifact MOE_PATH
                             --next-caller-artifact PATH
                             [--next-artifact ... --next-caller-artifact ...]
                             --text TEXT [options]
      expert_tool call-prefix --artifact-prefix PATH --expert-artifact PATH
                              --text TEXT [--last-layer INDEX]
                              [--head-artifact PATH] [options]

    extract range-downloads one routed expert from Gemma 4 26B-A4B. It does
    not download or save the complete checkpoint.

      --artifact PATH       New or existing standalone artifact directory
      --repo REPOSITORY     Default google/gemma-4-26B-A4B-it
      --revision REVISION   Default main
      --layer INDEX         MoE layer, default 0
      --expert INDEX        Routed expert, default 0

    run executes only the extracted gated FFN over synthetic hidden states.
    Router weighting, the shared expert, norms, and residual are not included.

    extract-layer range-downloads a complete language-model MoE feed-forward
    layer: all 128 routed expert banks, the router, shared FFN, norms, and layer
    scalar. run-layer executes that standalone shell and reports which experts
    each synthetic residual-state row selected. Attention and token embedding
    are not included. The 26B-A4B checkpoint has no audio encoder.

    profile-math compares real token embeddings for curated math and control
    corpora against a layer-0 router. It is a fast candidate search, not a
    substitute for capturing post-attention activations from the full model.

    extract-caller saves one decoder layer's attention prefix. call-expert tokenizes
    text, reconstructs the post-attention residual and routing decision, then
    sends the exact pre_feedforward_layernorm_2 rows selected by the router to
    the standalone expert.

    call-layer additionally loads the complete MoE shell, substitutes the
    standalone expert output wherever its index is routed, and produces the
    complete layer-0 output. It also returns the unchanged baseline for an
    exact comparison. --expert-scale changes the inserted output for controlled
    replacement and ablation experiments.

    call-chain feeds that complete layer-0 output through any contiguous list
    of later decoder layers without applying embedding scaling again. Repeat
    --next-artifact and --next-caller-artifact as an ordered pair for every
    layer. Later layer parameters load sequentially while hidden states remain
    on the GPU.

    call-prefix derives that ordered list from artifacts named
    PREFIX-layer0-moe, PREFIX-layer0-caller, and so on. It runs through layer
    29 by default; --last-layer can stop earlier for prefix experiments.
    extract-head saves the final norm and tied vocabulary projection separately.
    Pass it as --head-artifact on call-prefix to report the top next-token
    predictions for both the modified and unchanged paths.

      --backend BACKEND     Default exla:rocm
      --expert-scale FLOAT  Standalone expert output multiplier, default 1.0
      --tokens N            Input rows, default 1
      --runs N              Timed runs after warmup, default 3
      --limit N             Profile result count, default 10
      --input-value FLOAT   Value in every input cell, default 0.01
    """
  end

  defmodule Escript do
    @moduledoc false

    def main([command | _argv])
        when command in [
               "run",
               "run-layer",
               "profile-math",
               "call-expert",
               "call-layer",
               "call-chain",
               "call-prefix"
             ] do
      IO.puts(
        :stderr,
        "error: native execution requires real application priv paths; " <>
          "use `mix gemma.expert #{command} ...`"
      )

      System.halt(1)
    end

    def main(argv) do
      {:ok, _} = Application.ensure_all_started(:gemma_4_mic_transcribe)
      System.halt(Gemma4MicTranscribe.ExpertCLI.main(argv))
    end
  end
end
