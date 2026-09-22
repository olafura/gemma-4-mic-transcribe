defmodule Gemma4MicTranscribe.Gemma4.SystemOne.Trainer do
  @moduledoc """
  Trains the System One experts and routers on a cached layer-45 input, see
  `docs/system-one-expert-plan.md` section 4.

  The prefix is a constant for a teacher-forced sequence, so the cache holds
  its output and training only ever runs the three tail layers. The tail is
  dequantized to f32 once (the packed `Q4Gemv` custom call has no gradient)
  and frozen; the only trainable tensors are the roughly 71M expert and router
  parameters.

  Everything is fixed-shape so XLA compiles one training step: rows are padded
  to the cache's bucket, and the head is applied to a fixed number of gathered
  response positions instead of all 256, because the vocabulary is 262k wide
  and a full logit tensor at batch 8 would be 2 GB.

  Loss, per batch:

    * cross-entropy on the response tokens of every row, with the first
      `:lead_tokens` of a System One reply weighted `:lead_weight` times;
    * where the cache stores the base model's top-k logits, the KL to them;
    * a binary cross-entropy on the router, open on a System One row from the
      position that predicts the first response token to the end of the reply
      and closed everywhere else, including every position of a replay row.
      The two directions are averaged separately, because a batch has an
      order of magnitude more closed positions than open ones.

  Checkpoints are `SystemOneArtifact` directories holding the trained
  parameters only; a resumed run restarts Adam's moments, which is visible as
  a small bump in the loss right after a resume.
  """

  import Nx.Defn, only: [defnp: 2]

  alias Gemma4MicTranscribe.Gemma4.SystemOne
  alias Gemma4MicTranscribe.Gemma4.SystemOneArtifact
  alias Gemma4MicTranscribe.Gemma4Unified.CompressedTensors
  alias Gemma4MicTranscribe.Gemma4Unified.Model
  alias Gemma4MicTranscribe.Gemma4Unified.Q4Gemv
  alias Gemma4MicTranscribe.LanguageId.Finetune

  @manifest "manifest.json"
  @checkpoints "checkpoints"

  @defaults %{
    layers: [45, 46, 47],
    batch_size: 8,
    max_response_tokens: 48,
    epochs: 2,
    expert_learning_rate: 1.0e-4,
    router_learning_rate: 1.0e-3,
    max_grad_norm: 1.0,
    kl_weight: 1.0,
    gate_open_weight: 0.5,
    gate_closed_weight: 1.0,
    # The first tokens of a reply are the decision - "Sure," against "Which
    # calendar?" - and everything after them is committed to it. They are a
    # handful of the 48 response slots, so the mean cross-entropy barely
    # notices them; this weights them so it does.
    lead_tokens: 4,
    lead_weight: 3.0,
    # Round 1 trained at a floor of 0.05, which is under where the gate loss
    # settles, so the router was open on 28% of replay positions. A newly
    # trained artifact records 0.5 instead: the gate is supervised as a binary
    # decision, so half is the decision boundary rather than a tuned number.
    gate_floor: 0.5,
    # `:response` opens the gate on every System One reply and the expert
    # learns to answer or ask. Rounds 1 and 2 showed it then asks whenever the
    # answer is hard to produce, because a question is the cheaper reply under
    # cross-entropy. `:classifier` opens it only on underspecified items and
    # trains it shut on decidable ones with no cross-entropy there, so the
    # router is a plain "is something missing" decision and base Gemma keeps
    # answering the decidable items itself.
    gate_mode: :response,
    checkpoint_every: 200,
    log_every: 10,
    seed: 42,
    # bf16 halves the vocabulary projection on paper and costs more here: the
    # ROCm GEMM has no bf16 kernel for this shape with the autotuner and Triton
    # off, so XLA converts the kernel back to f32 for the dot and the step then
    # holds both types. Measured: 3.75 GiB of bf16 buffers on top of the f32
    # ones it was meant to replace.
    head_type: {:f, 32}
  }

  @doc "Default option values, so the CLI and the tests agree on them."
  def defaults, do: @defaults

  ## Cache

  @doc """
  Reads a cache directory's manifest. `:kinds` restricts the rows to the given
  kinds, `:limit` to the first N.
  """
  def load_cache!(path, opts \\ []) do
    path = Path.expand(path)
    manifest = path |> Path.join(@manifest) |> File.read!() |> Jason.decode!()

    if manifest["kind"] != "system_one_prefix_cache" do
      raise ArgumentError, "#{path} is not a System One prefix cache"
    end

    if manifest["last_prompt_token_only"] do
      raise ArgumentError, "training needs a full-sequence cache, this one is last-token only"
    end

    rows =
      manifest
      |> Map.fetch!("rows")
      |> Enum.map(&entry/1)
      |> filter_kinds(Keyword.get(opts, :kinds))
      |> take(Keyword.get(opts, :limit))

    %{
      path: path,
      hidden_size: manifest["hidden_size"],
      sequence_length: manifest["buckets"] |> List.wrap() |> Enum.max(),
      thought_channel: manifest["thought_channel"],
      system_message: manifest["system_message"],
      rows: rows
    }
  end

  defp entry(row) do
    response = row["response"] || %{"start" => 0, "length" => 0}

    %{
      id: row["id"],
      file: row["file"],
      length: row["length"],
      kind: row["kind"] || "system_one",
      decidable: row["decidable"],
      response_start: response["start"],
      response_length: response["length"]
    }
  end

  defp filter_kinds(rows, nil), do: rows
  defp filter_kinds(rows, kinds), do: Enum.filter(rows, &(&1.kind in kinds))

  defp take(rows, nil), do: rows
  defp take(rows, limit), do: Enum.take(rows, limit)

  @doc """
  Reads one cached row onto `backend`.

  The optional `base_top_k_values` / `base_top_k_token_ids` tensors are the
  base model's top-k logits at the response positions, for the replay KL; a
  row without them is trained on cross-entropy alone.
  """
  def read_row!(cache, entry, backend) do
    tensors = cache.path |> Path.join(entry.file) |> Safetensors.read!()

    %{
      id: entry.id,
      kind: entry.kind,
      decidable: entry.decidable,
      length: entry.length,
      response_start: entry.response_start,
      response_length: entry.response_length,
      hidden_state: tensors |> Map.fetch!("hidden_state") |> Nx.squeeze(axes: [0]) |> to(backend),
      input_ids: tensors |> Map.fetch!("input_ids") |> to(backend),
      base_values: tensors |> Map.get("base_top_k_values") |> to(backend),
      base_ids: tensors |> Map.get("base_top_k_token_ids") |> to(backend)
    }
  end

  defp to(nil, _backend), do: nil
  defp to(tensor, nil), do: tensor
  defp to(tensor, backend), do: Nx.backend_transfer(tensor, backend)

  @doc """
  Turns rows into one fixed-shape `{inputs, targets}` batch.

  `label_positions` is where the head reads: the token at response position
  `r` is predicted by the hidden state one position before it. Padded label
  slots point at position 0 and are masked out of every term.
  """
  def batch(rows, opts) do
    config = %{
      sequence_length: Keyword.fetch!(opts, :sequence_length),
      responses: Keyword.fetch!(opts, :max_response_tokens),
      hidden_size: Keyword.fetch!(opts, :hidden_size),
      top_k: Keyword.get(opts, :top_k, 1),
      type: Keyword.get(opts, :type, {:f, 32}),
      lead_tokens: Keyword.get(opts, :lead_tokens, @defaults.lead_tokens),
      lead_weight: Keyword.get(opts, :lead_weight, @defaults.lead_weight),
      gate_mode: Keyword.get(opts, :gate_mode, @defaults.gate_mode)
    }

    built = Enum.map(rows, &build_row(&1, config))

    inputs = %{
      "hidden_state" => stack(built, :hidden_state),
      "attention_mask" => stack(built, :attention_mask),
      "position_ids" => stack(built, :position_ids),
      "label_positions" => stack(built, :label_positions)
    }

    targets = %{
      labels: stack(built, :labels),
      label_mask: stack(built, :label_mask),
      label_weight: stack(built, :label_weight),
      gate_open_mask: stack(built, :gate_open_mask),
      gate_closed_mask: stack(built, :gate_closed_mask),
      system_one_mask: stack(built, :system_one_mask),
      replay_mask: stack(built, :replay_mask),
      base_values: stack(built, :base_values),
      base_ids: stack(built, :base_ids),
      base_mask: stack(built, :base_mask)
    }

    {inputs, targets}
  end

  defp stack(built, key), do: built |> Enum.map(&Map.fetch!(&1, key)) |> Nx.stack()

  defp build_row(row, config) do
    %{sequence_length: sequence_length, responses: responses, type: type} = config

    if row.length > sequence_length do
      raise ArgumentError,
            "#{row.id} is #{row.length} tokens, longer than the cache bucket #{sequence_length}"
    end

    backend = backend_of(row.hidden_state)
    length = row.length
    kept = min(row.response_length, responses)
    top_k = config.top_k
    system_one? = row.kind != "replay"

    # A decidable item under `:classifier` is one the router has to stay shut
    # on: base Gemma answers it, so the expert gets no cross-entropy for it and
    # every position is a closed-gate target, as on a replay row.
    expert_row? = system_one? and not (config.gate_mode == :classifier and decidable?(row))

    hidden_state =
      row.hidden_state
      |> Nx.slice([0, 0], [length, config.hidden_size])
      |> Nx.as_type(type)
      |> pad_rows(sequence_length - length)

    token_mask = mask(length, sequence_length, type, backend)
    none = zeros({sequence_length}, type, backend)

    # The rule the router is trained on is "open only while producing the
    # response to a System One prompt". The hidden state at `response_start -
    # 1` is the one that predicts the first response token, so it is the first
    # position that has to be open; everything before it is a prompt the
    # router sees on ordinary requests too, and has to stay shut on.
    open_from = max(row.response_start - 1, 0)

    open_mask =
      if expert_row?,
        do: range_mask(open_from, length, sequence_length, type, backend),
        else: none

    closed_mask =
      if expert_row?,
        do: mask(open_from, sequence_length, type, backend),
        else: token_mask

    # The response token at slot `r` is the one the hidden state at
    # `response_start + r - 1` has to predict.
    positions =
      shifted_positions(row.response_start, kept, responses, sequence_length, backend)

    labels =
      row.input_ids
      |> Nx.as_type(:s64)
      |> gather_slots(row.response_start, kept, responses, backend)

    label_mask = mask(kept, responses, type, backend)

    %{
      hidden_state: hidden_state,
      attention_mask: mask(length, sequence_length, :s64, backend),
      position_ids: Nx.iota({sequence_length}, type: :s64, backend: backend),
      label_positions: positions,
      labels: labels,
      label_mask: label_mask,
      label_weight: label_weight(label_mask, system_one?, expert_row?, config, backend),
      gate_open_mask: open_mask,
      gate_closed_mask: closed_mask,
      system_one_mask: if(system_one?, do: token_mask, else: none),
      replay_mask: if(system_one?, do: none, else: token_mask),
      base_values: base_slots(row.base_values, kept, responses, top_k, type, backend),
      base_ids: base_slots(row.base_ids, kept, responses, top_k, :s64, backend),
      base_mask: base_mask(row.base_values, kept, responses, type, backend)
    }
  end

  # The cross-entropy is a mean over these weights, so the lead tokens of a
  # System One reply count `lead_weight` times as much as the rest of it. A
  # replay row has no decision to make, so its response is weighted flat, and
  # a decidable row under `:classifier` is not the expert's to answer at all.
  defp label_weight(label_mask, false = _system_one?, _expert_row?, _config, _backend),
    do: label_mask

  defp label_weight(label_mask, true, false = _expert_row?, _config, _backend),
    do: Nx.multiply(label_mask, 0.0)

  defp label_weight(label_mask, true, true, config, backend) do
    type = Nx.type(label_mask)
    lead = mask(config.lead_tokens, config.responses, type, backend)

    lead
    |> Nx.multiply(config.lead_weight - 1.0)
    |> Nx.add(1.0)
    |> Nx.multiply(label_mask)
  end

  # The cache manifest carries `decidable` for rows cached since round 3; older
  # caches only have the id, whose `-d` / `-u` suffix is the same twin label.
  defp decidable?(%{decidable: decidable}) when is_boolean(decidable), do: decidable

  defp decidable?(%{id: id}) do
    cond do
      String.ends_with?(id, "-d") ->
        true

      String.ends_with?(id, "-u") ->
        false

      true ->
        raise ArgumentError, "#{id}: gate mode :classifier needs a decidable flag on the row"
    end
  end

  defp base_mask(nil, _kept, responses, type, backend), do: zeros({responses}, type, backend)

  defp base_mask(_tensor, kept, responses, type, backend),
    do: mask(kept, responses, type, backend)

  defp backend_of(%Nx.Tensor{data: %module{}}), do: module

  defp pad_rows(tensor, 0), do: tensor

  defp pad_rows(tensor, padding) do
    Nx.pad(tensor, 0.0, [{0, padding, 0}, {0, 0, 0}])
  end

  defp zeros(shape, type, backend),
    do: Nx.broadcast(Nx.tensor(0, type: type, backend: backend), shape)

  defp mask(count, length, type, backend) do
    Nx.iota({length}, type: :s64, backend: backend)
    |> Nx.less(count)
    |> Nx.as_type(type)
  end

  # `from <= i < to`, as a mask over `length`.
  defp range_mask(from, to, length, type, backend) do
    iota = Nx.iota({length}, type: :s64, backend: backend)

    iota
    |> Nx.greater_equal(from)
    |> Nx.logical_and(Nx.less(iota, to))
    |> Nx.as_type(type)
  end

  # Padded slots keep a valid index; `label_mask` is what drops them.
  defp shifted_positions(start, _kept, responses, sequence_length, backend) do
    Nx.iota({responses}, type: :s64, backend: backend)
    |> Nx.add(start - 1)
    |> Nx.min(sequence_length - 1)
    |> Nx.max(0)
  end

  defp gather_slots(input_ids, start, kept, responses, backend) do
    if kept == 0 do
      zeros({responses}, :s64, backend)
    else
      input_ids
      |> Nx.slice([start], [kept])
      |> Nx.pad(0, [{0, responses - kept, 0}])
    end
  end

  defp base_slots(nil, _kept, responses, top_k, type, backend),
    do: zeros({responses, top_k}, type, backend)

  defp base_slots(_tensor, 0, responses, top_k, type, backend),
    do: zeros({responses, top_k}, type, backend)

  defp base_slots(tensor, kept, responses, top_k, type, backend) do
    tensor
    |> Nx.as_type(type)
    |> Nx.slice([0, 0], [kept, top_k])
    |> Nx.pad(Nx.tensor(0, type: type, backend: backend), [{0, responses - kept, 0}, {0, 0, 0}])
  end

  @doc """
  An infinite-per-epoch stream of batches: full batches only, reshuffled every
  epoch with a seeded index shuffle, all epochs in one stream.

  `Axon.Loop.run` re-enumerates its data once per epoch, so the loop is run
  with `epochs: 1` over every epoch's batches; handing it a multi-epoch stream
  and `epochs: n` would train n squared epochs.
  """
  def batches(cache, opts) do
    batch_size = Keyword.fetch!(opts, :batch_size)
    epochs = Keyword.fetch!(opts, :epochs)
    seed = Keyword.fetch!(opts, :seed)
    backend = Keyword.get(opts, :stage_backend)
    rows = cache.rows
    count = length(rows)
    full = div(count, batch_size)

    Stream.flat_map(0..(epochs - 1), fn epoch ->
      order = shuffle(count, seed, epoch)

      order
      |> Enum.take(full * batch_size)
      |> Enum.chunk_every(batch_size)
      |> Stream.map(fn indices ->
        indices
        |> Enum.map(fn index -> read_row!(cache, Enum.fetch!(rows, index), backend) end)
        |> batch(opts)
      end)
    end)
  end

  @doc "A deterministic shuffle of `0..count-1` that does not touch the process RNG."
  def shuffle(count, seed, epoch) do
    0..(count - 1)
    |> Enum.map(fn index -> {:erlang.phash2({seed, epoch, index}), index} end)
    |> Enum.sort()
    |> Enum.map(&elem(&1, 1))
  end

  @doc "How many batches one epoch has, and how many rows are dropped to keep them full."
  def batch_count(cache, batch_size) do
    count = length(cache.rows)
    %{batches: div(count, batch_size), dropped: rem(count, batch_size)}
  end

  ## Model

  @doc """
  The spec the trainer runs: the tail's spec with the experts attached and the
  packed linears turned off, because gradients cannot flow through the int4
  custom call.
  """
  def training_spec(spec, layers, opts \\ []) do
    spec
    |> Map.put(:packed_linear, false)
    |> Map.put(:hybrid_linear, false)
    |> Map.put(:fused_q4_ffn, false)
    |> Map.put(:system_one_layers, layers)
    |> Map.put(
      :system_one_expert_size,
      Keyword.get(opts, :expert_size, SystemOne.expert_size(spec))
    )
    |> Map.put(:system_one_gate_floor, Keyword.get(opts, :gate_floor) || @defaults.gate_floor)
  end

  @doc """
  The tail over cached hidden states, with logits at the gathered response
  positions and every layer's raw gate as a second output.
  """
  def model(spec, layers, opts \\ []) do
    blocks = Model.decoder_block_chain_model(spec, layers)

    logits =
      Model.gathered_output_logits(blocks, positions_input(), spec, head_type: head_type(opts))

    Axon.container(%{logits: logits, gate_logits: router_logit_nodes(blocks, layers)})
  end

  @doc """
  The same tail with the gathered hidden state in place of the logits.

  This is the model the training step differentiates. The vocabulary
  projection is 262144 x 3840 and XLA keeps a buffer per layout its GEMMs
  want, so the head runs as its own program (`head_terms/4`) over this
  model's output and the tail's own graph never holds it.
  """
  def hidden_model(spec, layers) do
    blocks = Model.decoder_block_chain_model(spec, layers)
    hidden = Model.gathered_hidden_state(blocks, positions_input())

    Axon.container(%{hidden: hidden, gate_logits: router_logit_nodes(blocks, layers)})
  end

  @doc """
  The PRNG state a `mode: :train` build expects, as parameter-map entries.

  Bumblebee's attention adds an `Axon.dropout/2` node per block, and in train
  mode every one of them reads a `"key"` state parameter. The tail artifact
  holds weights only, and the trainer never calls the model's `init` (it would
  allocate, and hand back, a second copy of the frozen tail), so the keys are
  built here instead. Gemma's dropout rate is 0.0, so the layer is the
  identity and the key is only ever split, never used.
  """
  def dropout_state(model, seed) do
    {names, _counts} =
      Axon.reduce_nodes(model, {[], %{}}, fn %Axon.Node{op: op, name: name}, {names, counts} ->
        resolved = name.(op, counts)
        counts = Map.update(counts, op, 1, &(&1 + 1))
        {if(op == :dropout, do: [resolved | names], else: names), counts}
      end)

    Map.new(names, fn name -> {name, %{"key" => Nx.Random.key(seed)}} end)
  end

  defp positions_input, do: Axon.input("label_positions", shape: {nil, nil})

  # The gate itself is read off the main graph, where it multiplies the
  # expert; what the loss needs is the logit under it, see
  # `SystemOne.router_logit_nodes/2`.
  defp router_logit_nodes(blocks, layers) do
    blocks
    |> SystemOne.router_logit_nodes(layers)
    |> Map.new(fn {layer_index, node} -> {"#{layer_index}", node} end)
  end

  @doc """
  Dequantizes packed parameters to plain dense kernels.

  `Q4Gemv.dequantize/3` is the artifact's layout (`CompressedTensors` is the
  raw checkpoint's and would give the transposed matrix). It is ordinary Nx,
  so it runs wherever the tensors are: `:stage_backend` keeps that off the
  ROCm client, where eager ops have segfaulted this box, and `:backend` is
  where the finished kernels end up.
  """
  def dequantize(params, opts \\ []) do
    group_size = Keyword.get(opts, :group_size, CompressedTensors.quant_group_size())
    stage = Keyword.get(opts, :stage_backend)
    backend = Keyword.get(opts, :backend)
    type = Keyword.get(opts, :type, {:f, 32})
    head_type = head_type(opts) || type

    data(params)
    |> Map.new(fn {node_name, parameters} ->
      node_type = if head_node?(node_name), do: head_type, else: type
      {node_name, dequantize_node(parameters, group_size, stage, backend, node_type)}
    end)
  end

  @doc "True for the vocabulary projection, which trains at its own type."
  def head_node?(name), do: String.starts_with?(name, "language_modeling_head.")

  defp head_type(opts), do: Keyword.get(opts, :head_type, @defaults.head_type)

  defp data(%Axon.ModelState{data: data}), do: data
  defp data(data) when is_map(data), do: data

  defp dequantize_node(
         %{"packed" => packed, "scales" => scales},
         group_size,
         stage,
         backend,
         type
       ) do
    kernel =
      packed
      |> to(stage)
      |> Q4Gemv.dequantize(to(scales, stage), group_size)
      |> Nx.as_type(type)
      |> to(backend)

    %{"kernel" => kernel}
  end

  defp dequantize_node(parameters, _group_size, stage, backend, type) do
    Map.new(parameters, fn {name, tensor} ->
      {name, tensor |> to(stage) |> Nx.as_type(type) |> to(backend)}
    end)
  end

  ## Loss

  @doc """
  The loss components of one batch, as a map of scalars. `total/2` weights
  them into the training objective and each one is also logged as a metric.
  """
  def components(targets, outputs) do
    log_probs = log_probs(outputs.logits)

    targets
    |> head_components(log_probs)
    |> Map.merge(gate_components(targets, outputs.gate_logits))
  end

  @doc "The terms that depend on the vocabulary head: the ones `head_terms/4` differentiates."
  def head_components(targets, log_probs) do
    %{
      ce: cross_entropy(log_probs, targets.labels, targets.label_weight),
      kl: kl(log_probs, targets)
    }
  end

  @doc """
  The terms that depend on the gates alone, from the raw `{batch, sequence, 1}`
  router logit of every System One layer.

  `gate_open` and `gate_closed` are the two halves of one binary
  cross-entropy, averaged separately: a row has a few dozen open-target
  positions against a couple of hundred closed-target ones, and a single mean
  over both would be the closed half with a rounding error on it. `total/2`
  puts them back together with their weights.

  The `false_` metrics are the ones that matter at inference, where the gate
  is a yes-or-no against the floor rather than a number: the fraction of
  positions on the wrong side of it. A gate of 0.5 is a logit of 0, so they
  are read straight off the logit's sign. `gate_false_open` covers every
  closed-target position, which mixes replay rows with the prompt prefix of
  System One rows; `gate_false_open_replay` is the replay rows alone, the
  regression the router is there to avoid.
  """
  def gate_components(targets, gate_logits) do
    logits = Map.new(gate_logits, fn {name, logit} -> {name, Nx.squeeze(logit, axes: [-1])} end)

    open = targets.gate_open_mask
    closed = targets.gate_closed_mask

    %{
      gate_open:
        mean_over_gates(logits, &masked_mean(Nx.multiply(softplus(Nx.negate(&1)), open), open)),
      gate_closed:
        mean_over_gates(logits, &masked_mean(Nx.multiply(softplus(&1), closed), closed)),
      gate_mean_system_one:
        mean_over_gates(logits, &weighted_mean(Nx.sigmoid(&1), targets.system_one_mask)),
      gate_mean_replay:
        mean_over_gates(logits, &weighted_mean(Nx.sigmoid(&1), targets.replay_mask)),
      gate_false_open: mean_over_gates(logits, &weighted_mean(open_side(&1), closed)),
      gate_false_open_replay:
        mean_over_gates(logits, &weighted_mean(open_side(&1), targets.replay_mask)),
      gate_false_closed: mean_over_gates(logits, &weighted_mean(closed_side(&1), open))
    }
  end

  defp open_side(logit), do: logit |> Nx.greater_equal(0.0) |> Nx.as_type(Nx.type(logit))

  defp closed_side(logit), do: logit |> Nx.less(0.0) |> Nx.as_type(Nx.type(logit))

  @doc "Log-softmax of the gathered logits."
  def log_probs(logits) do
    Nx.subtract(logits, Nx.logsumexp(logits, axes: [-1], keep_axes: true))
  end

  @doc "The weighted objective; `weights` carries `:kl`, `:gate_open` and `:gate_closed`."
  def total(components, weights) do
    components.ce
    |> Nx.add(Nx.multiply(components.kl, weights.kl))
    |> Nx.add(Nx.multiply(components.gate_open, weights.gate_open))
    |> Nx.add(Nx.multiply(components.gate_closed, weights.gate_closed))
  end

  # `weights` is `label_mask` scaled by the lead-token weight, so this is a
  # weighted mean: padded slots weigh 0 and the first tokens of a System One
  # reply weigh `lead_weight`.
  defp cross_entropy(log_probs, labels, weights) do
    labels
    |> then(&Nx.take_along_axis(log_probs, Nx.new_axis(&1, -1), axis: -1))
    |> Nx.squeeze(axes: [-1])
    |> Nx.multiply(weights)
    |> masked_mean(weights)
    |> Nx.negate()
  end

  # KL(base || model) over the stored top-k, renormalised over those k, and
  # only on the rows that carry base logits.
  defp kl(log_probs, targets) do
    base_log_probs =
      Nx.subtract(
        targets.base_values,
        Nx.logsumexp(targets.base_values, axes: [-1], keep_axes: true)
      )

    base_probs = Nx.exp(base_log_probs)
    model_log_probs = Nx.take_along_axis(log_probs, targets.base_ids, axis: -1)

    base_probs
    |> Nx.multiply(Nx.subtract(base_log_probs, model_log_probs))
    |> Nx.sum(axes: [-1])
    |> Nx.multiply(targets.base_mask)
    |> masked_mean(targets.base_mask)
  end

  # `log(1 + exp(x))` without the overflow: `softplus(-z)` is `-log sigmoid(z)`
  # and `softplus(z)` is `-log(1 - sigmoid(z))`, the two halves of a binary
  # cross-entropy on the router.
  defp softplus(x) do
    x
    |> Nx.abs()
    |> Nx.negate()
    |> Nx.exp()
    |> Nx.log1p()
    |> Nx.add(Nx.max(x, 0.0))
  end

  defp weighted_mean(gate, mask) do
    gate |> Nx.multiply(mask) |> masked_mean(mask)
  end

  defp masked_mean(weighted, mask) do
    Nx.divide(Nx.sum(weighted), Nx.max(Nx.sum(mask), 1.0))
  end

  defp mean_over_gates(gates, _fun) when map_size(gates) == 0, do: Nx.tensor(0.0)

  defp mean_over_gates(gates, fun) do
    values = gates |> Map.values() |> Enum.map(fun)
    values |> Enum.reduce(&Nx.add/2) |> Nx.divide(length(values))
  end

  ## Step

  @doc """
  The tail's forward pass: the hidden state at the gathered response positions
  and the gate terms, which need no head and no gradient.

  `predict` is `Axon.build(hidden_model(spec, layers), mode: :train)`'s second
  element. `frozen` and `expert` are the two halves of the parameter map; they
  are separate arguments so the frozen half can stay on the device across
  steps while only the expert half is returned by `tail_gradients/7`.
  """
  def tail_forward(predict, frozen, expert, inputs, targets) do
    %{prediction: outputs} = predict.(model_state(Map.merge(frozen, expert)), inputs)
    {outputs.hidden, gate_components(targets, outputs.gate_logits)}
  end

  @doc """
  The head's loss terms and `dloss/dhidden`, as `{terms, cotangent}`.

  `head` holds the final norm's weight and the vocabulary kernel in both the
  layouts the two GEMMs want. The forward is `hidden . kernel_t` and the
  backward `dlogits . kernel`, each contracting the last axis of the left
  against the first of the right, so XLA has no transpose to materialise and
  the 3.75 GB kernel exists in exactly the two buffers handed in.

  The gradient is taken with respect to the hidden state only. Nothing else in
  this program is differentiated, so the head never appears in the tail's
  backward pass; `tail_gradients/7` carries the cotangent the rest of the way.
  """
  def head_terms(head, hidden, targets, opts) do
    kl_weight = Keyword.fetch!(opts, :kl_weight)

    Nx.Defn.value_and_grad(
      hidden,
      fn hidden -> head_components(targets, head_log_probs(head, hidden, opts)) end,
      fn terms -> Nx.add(terms.ce, Nx.multiply(terms.kl, kl_weight)) end
    )
  end

  defp head_log_probs(head, hidden, opts) do
    hidden
    |> final_norm(head.weight, Keyword.fetch!(opts, :epsilon))
    |> Nx.as_type(Nx.type(head.kernel))
    |> project(head.kernel_t, head.kernel)
    |> Nx.as_type({:f, 32})
    |> softcap(Keyword.fetch!(opts, :softcapping))
    |> log_probs()
  end

  # `Bumblebee.Layers.rms_norm/2` with `upcast: :all` and no shift, which is
  # what the head's `output_norm` is; the trainer runs it itself because the
  # tail model stops one node earlier.
  defp final_norm(hidden, weight, epsilon) do
    variance = hidden |> Nx.pow(2) |> Nx.mean(axes: [-1], keep_axes: true)

    hidden
    |> Nx.multiply(Nx.rsqrt(Nx.add(variance, epsilon)))
    |> Nx.multiply(weight)
  end

  defp softcap(logits, nil), do: logits

  defp softcap(logits, cap) do
    logits |> Nx.divide(cap) |> Nx.tanh() |> Nx.multiply(cap)
  end

  defnp project(hidden, kernel_t, kernel) do
    custom_grad(Nx.dot(hidden, [-1], kernel_t, [0]), [hidden], fn cotangent ->
      [Nx.dot(cotangent, [-1], kernel, [0])]
    end)
  end

  @doc """
  The gradient of the whole objective with respect to the expert and router
  parameters, plus the gate terms of this batch.

  `cotangent` is `head_terms/4`'s gradient, so the chain rule makes
  `sum(hidden * cotangent)` stand in for the head's half of the loss: its
  gradient in the expert is the one the full loss has, without the head being
  part of this program. The gate terms are recomputed here because they are
  differentiated too.
  """
  def tail_gradients(predict, frozen, expert, inputs, targets, cotangent, weights) do
    Nx.Defn.value_and_grad(
      expert,
      fn expert ->
        {hidden, gates} = tail_forward(predict, frozen, expert, inputs, targets)

        hidden
        |> Nx.multiply(cotangent)
        |> Nx.sum()
        |> Nx.add(Nx.multiply(gates.gate_open, weights.gate_open))
        |> Nx.add(Nx.multiply(gates.gate_closed, weights.gate_closed))
      end
    )
  end

  ## Optimizer

  @doc """
  Adam behind global-norm clipping, with the expert and the router on
  different learning rates, and `Finetune.guard_non_finite/1` so one
  non-finite step cannot poison the moments.

  Adam normalises by the gradient's own second moment, so scaling gradients
  per group would not change the step size; the rates are applied after it.
  """
  def optimizer(opts) do
    expert = Keyword.fetch!(opts, :expert_learning_rate)
    router = Keyword.fetch!(opts, :router_learning_rate)
    max_grad_norm = Keyword.get(opts, :max_grad_norm, 1.0)

    Polaris.Updates.clip_by_global_norm(max_norm: max_grad_norm)
    |> Polaris.Updates.scale_by_adam()
    |> Polaris.Updates.stateless(&scale_by_group(&1, &2, expert, router))
    |> Finetune.guard_non_finite()
  end

  defp scale_by_group(updates, _params, expert, router) do
    Map.new(updates, fn {node_name, parameters} ->
      rate = if router_node?(node_name), do: router, else: expert
      {node_name, Map.new(parameters, fn {name, u} -> {name, Nx.multiply(u, -rate)} end)}
    end)
  end

  @doc "True for a router's dense node, which trains at its own learning rate."
  def router_node?(name), do: String.ends_with?(name, ".router")

  ## Training

  @doc """
  Trains the experts over a cached prefix output.

  Required options: `:cache`, `:tail_artifact`, `:output`. The rest fall back
  to `defaults/0`; see `Gemma4MicTranscribe.SystemOneCLI` for the flags.
  """
  def train(opts) do
    log = Keyword.get(opts, :log, &default_log/1)
    cache = load_cache!(Keyword.fetch!(opts, :cache), Keyword.take(opts, [:limit, :kinds]))
    layers = Keyword.get(opts, :layers, @defaults.layers)
    output = Path.expand(Keyword.fetch!(opts, :output))
    checkpoint_every = Keyword.get(opts, :checkpoint_every, @defaults.checkpoint_every)
    log_every = Keyword.get(opts, :log_every, @defaults.log_every)
    max_steps = Keyword.get(opts, :max_steps)
    batch_size = Keyword.get(opts, :batch_size, @defaults.batch_size)
    responses = Keyword.get(opts, :max_response_tokens, @defaults.max_response_tokens)

    manifest = tail_manifest!(Keyword.fetch!(opts, :tail_artifact))
    spec = training_spec(manifest.spec, layers, opts)
    model = model(spec, layers, opts)
    plan = plan(cache, spec, layers, batch_size, responses, opts)

    log.(Map.put(plan, :event, "train_plan"))

    if Keyword.get(opts, :dry_run, false) do
      %{plan: plan, model: model, trained: nil}
    else
      run(model, spec, cache, plan, output, %{
        opts: opts,
        layers: layers,
        log: log,
        log_every: log_every,
        checkpoint_every: checkpoint_every,
        max_steps: max_steps
      })
    end
  end

  # A plain loop over three jitted programs instead of `Axon.Loop`, which
  # threads the whole model state through the training step and hands back a
  # fresh copy of it every iteration: a second 3.75 GB vocabulary kernel per
  # step is what used to run this out of device memory. Here the frozen
  # tensors are only ever arguments, and the only outputs are the expert
  # parameters, the optimizer state and scalars.
  defp run(model, spec, cache, plan, output, context) do
    opts = context.opts
    {:ok, backend} = resolve(Keyword.get(opts, :backend, "exla:rocm"))
    {:ok, stage} = resolve(Keyword.get(opts, :stage_backend, "torchx:cpu"))
    compiler_opts = compiler_opts(backend)

    tail = Gemma4MicTranscribe.Gemma4.DecoderBlockArtifact.load_tail!(opts[:tail_artifact], stage)

    frozen =
      dequantize(tail.params,
        stage_backend: stage,
        backend: backend,
        head_type: head_type(opts)
      )

    {head, frozen} = split_head(frozen, compiler_opts)

    {expert, resumed_step} = initial_expert(spec, context.layers, output, opts)
    params = transfer(expert.params, backend)

    model_for_step = hidden_model(spec, context.layers)
    {_init_fun, predict} = Axon.build(model_for_step, mode: :train)

    frozen =
      Map.merge(
        frozen,
        transfer(dropout_state(model_for_step, Keyword.get(opts, :seed, @defaults.seed)), backend)
      )

    {optimizer_init, optimizer_update} = optimizer(with_defaults(opts))
    weights = weights(opts)

    forward = Nx.Defn.jit(&tail_forward(predict, &1, &2, &3, &4), compiler_opts)
    head_step = Nx.Defn.jit(&head_terms(&1, &2, &3, head_opts(spec, weights)), compiler_opts)

    update =
      Nx.Defn.jit(
        fn frozen, params, inputs, targets, cotangent, optimizer_state ->
          {_surrogate, gradients} =
            tail_gradients(predict, frozen, params, inputs, targets, cotangent, weights)

          {updates, optimizer_state} = optimizer_update.(gradients, optimizer_state, params)
          {Polaris.Updates.apply_updates(params, updates), optimizer_state}
        end,
        compiler_opts
      )

    optimizer_state = Nx.Defn.jit(optimizer_init, compiler_opts).(params)

    stream =
      cache
      |> batches(
        batch_size: plan.batch_size,
        epochs: Keyword.get(opts, :epochs, @defaults.epochs),
        seed: Keyword.get(opts, :seed, @defaults.seed),
        sequence_length: plan.sequence_length,
        max_response_tokens: plan.max_response_tokens,
        hidden_size: plan.hidden_size,
        top_k: plan.top_k,
        lead_tokens: Keyword.get(opts, :lead_tokens, @defaults.lead_tokens),
        lead_weight: Keyword.get(opts, :lead_weight, @defaults.lead_weight),
        gate_mode: Keyword.get(opts, :gate_mode, @defaults.gate_mode),
        stage_backend: stage
      )
      |> Stream.drop(resumed_step)
      |> limit(context.max_steps)
      |> Stream.map(fn batch -> transfer_batch(batch, backend) end)

    {params, _optimizer_state, last_step} =
      stream
      |> Stream.with_index(resumed_step + 1)
      |> Enum.reduce({params, optimizer_state, resumed_step}, fn {{inputs, targets}, step},
                                                                 {params, optimizer_state, _last} ->
        started = System.monotonic_time(:millisecond)

        {hidden, gates} = forward.(frozen, params, inputs, targets)
        {terms, cotangent} = head_step.(head, hidden, targets)

        {params, optimizer_state} =
          update.(frozen, params, inputs, targets, cotangent, optimizer_state)

        elapsed = System.monotonic_time(:millisecond) - started
        progress(context, step, Map.merge(terms, gates), optimizer_state, elapsed)
        checkpoint(context, step, expert, params, output)

        # The step's intermediates are device buffers owned by NIF resources,
        # and a resource is freed when the reference to it is collected, not
        # when the variable goes out of scope. This process allocates almost
        # nothing on its own heap, so it runs for hundreds of steps between
        # major collections while the hidden states and the cotangent — the
        # latter is `batch * responses * vocab_size` floats, 200 MB a step at
        # a batch of four — pile up in the device arena. Releasing them here
        # is what keeps the arena flat: without it the run dies of an
        # out-of-memory a few hundred steps in, whatever the arena size, and
        # the dump shows every chunk of the leaked size still in use.
        Nx.backend_deallocate(hidden)
        Nx.backend_deallocate(cotangent)
        :erlang.garbage_collect()

        {params, optimizer_state, step}
      end)

    trained = model_state(params)

    artifact =
      SystemOneArtifact.from_model_state(expert, trained,
        step: last_step,
        meta: %{cache: cache.path, trained_at: DateTime.utc_now() |> DateTime.to_iso8601()}
      )

    path = SystemOneArtifact.save!(artifact, output, overwrite: true, keep: [@checkpoints])
    context.log.(%{event: "train_done", path: path, step: artifact.step})

    %{plan: plan, model: model, trained: trained, artifact: artifact, path: path}
  end

  # The head is not part of the model the step differentiates, so its two
  # tensors are lifted out of the frozen map. The transpose is jitted because
  # an eager Nx op on the ROCm client has segfaulted this box, and it is the
  # only thing that makes the kernel exist in both the layouts the forward and
  # the backward GEMM want without XLA materialising one of them per program.
  defp split_head(frozen, compiler_opts) do
    {head, rest} = Map.split(frozen, ["output_norm", "language_modeling_head.output"])
    kernel = head["language_modeling_head.output"]["kernel"]

    transpose = Nx.Defn.jit(&Nx.transpose/1, compiler_opts)

    {%{
       weight: head["output_norm"]["weight"],
       kernel: kernel,
       kernel_t: transpose.(kernel)
     }, rest}
  end

  defp head_opts(spec, weights) do
    [
      epsilon: spec.layer_norm_epsilon,
      softcapping: spec.final_logit_softcapping,
      kl_weight: weights.kl
    ]
  end

  defp progress(context, step, metrics, optimizer_state, elapsed) do
    if rem(step, context.log_every) == 0 do
      metrics = Map.new(metrics, fn {name, value} -> {"#{name}", scalar(value)} end)

      context.log.(
        Map.merge(metrics, %{
          event: "train_step",
          step: step,
          step_ms: elapsed,
          skipped: scalar(optimizer_state.skipped)
        })
      )
    end
  end

  defp checkpoint(context, step, expert, params, output) do
    if rem(step, context.checkpoint_every) == 0 do
      artifact = SystemOneArtifact.from_model_state(expert, model_state(params), step: step)
      path = SystemOneArtifact.save!(artifact, checkpoint_path(output, step), overwrite: true)
      context.log.(%{event: "train_checkpoint", step: step, path: path})
    end
  end

  ## Checkpoints

  @doc "Where the checkpoint after `step` steps lives."
  def checkpoint_path(output, step) do
    Path.join([
      Path.expand(output),
      @checkpoints,
      "step-#{String.pad_leading("#{step}", 6, "0")}"
    ])
  end

  @doc """
  The latest checkpoint under `output` as `{step, path}`, or `nil`. The step
  is read from the directory name, so a half-written checkpoint that was
  renamed into place is the only thing that can be found.
  """
  def latest_checkpoint(output) do
    directory = Path.join(Path.expand(output), @checkpoints)

    case File.ls(directory) do
      {:ok, names} ->
        names
        |> Enum.flat_map(&checkpoint_step/1)
        |> Enum.sort()
        |> List.last()
        |> case do
          nil -> nil
          step -> {step, checkpoint_path(output, step)}
        end

      {:error, _reason} ->
        nil
    end
  end

  defp checkpoint_step("step-" <> step) do
    case Integer.parse(step) do
      {step, ""} -> [step]
      _other -> []
    end
  end

  defp checkpoint_step(_name), do: []

  defp initial_expert(spec, layers, output, opts) do
    resume = Keyword.get(opts, :resume, true)

    case resume and latest_checkpoint(output) do
      {step, path} -> {SystemOneArtifact.load!(path, type: {:f, 32}), step}
      _none -> {fresh_expert(spec, layers, Keyword.get(opts, :init_from)), 0}
    end
  end

  # `--init-from` starts from another artifact's weights at step 0: no
  # optimizer state is carried over, so Adam's moments are rebuilt from
  # scratch and the run is a new one that happens to begin somewhere other
  # than a zero `down_e`. The gate floor is the new run's, not the one the
  # source artifact was trained at.
  defp fresh_expert(spec, layers, nil), do: SystemOneArtifact.new(spec, layers)

  defp fresh_expert(spec, layers, path) do
    artifact = SystemOneArtifact.load!(path, type: {:f, 32})

    if artifact.layers != layers do
      raise ArgumentError,
            "--init-from holds layers #{inspect(artifact.layers)}, this run trains #{inspect(layers)}"
    end

    if artifact.expert_size != SystemOne.expert_size(spec) do
      raise ArgumentError,
            "--init-from holds an expert of #{artifact.expert_size}, this run trains #{SystemOne.expert_size(spec)}"
    end

    %{artifact | gate_floor: SystemOne.gate_floor(spec), step: 0}
  end

  ## Plan and estimates

  @doc """
  What the run will do and what it will cost, without touching a device: the
  fixed shapes, the parameter counts and the bytes they need.
  """
  def plan(cache, spec, layers, batch_size, responses, opts \\ []) do
    counts = batch_count(cache, batch_size)
    epochs = Keyword.get(opts, :epochs, @defaults.epochs)
    steps = min_steps(counts.batches * epochs, Keyword.get(opts, :max_steps))
    sequence_length = cache.sequence_length
    top_k = Keyword.get(opts, :top_k, 1)

    %{
      rows: length(cache.rows),
      batch_size: batch_size,
      batches_per_epoch: counts.batches,
      dropped_rows: counts.dropped,
      epochs: epochs,
      steps: steps,
      layers: layers,
      sequence_length: sequence_length,
      max_response_tokens: responses,
      hidden_size: spec.hidden_size,
      top_k: top_k,
      parameters: parameter_counts(spec, layers),
      memory: memory_estimate(spec, layers, batch_size, sequence_length, responses, opts)
    }
  end

  defp min_steps(steps, nil), do: steps
  defp min_steps(steps, max_steps), do: min(steps, max_steps)

  @doc "Trained and frozen parameter counts of the tail."
  def parameter_counts(spec, layers) do
    hidden = spec.hidden_size
    expert_size = SystemOne.expert_size(spec)
    heads = spec.num_attention_heads * spec.attention_head_size
    kv = spec.num_key_value_heads * spec.attention_head_size

    attention = hidden * heads + 2 * hidden * kv + heads * hidden
    ffn = 3 * hidden * spec.intermediate_size
    expert = 3 * hidden * expert_size + hidden + 1

    %{
      expert: length(layers) * expert,
      frozen_layers: length(layers) * (attention + ffn),
      head: spec.vocab_size * hidden
    }
  end

  @doc """
  Bytes the training step needs, as a breakdown. The expert's Adam moments are
  exact; the activation figure is the batch tensors a decoder layer keeps for
  its backward pass and is the rough one.

  A frozen weight is not one buffer. It enters a program once and XLA then
  keeps the layouts the forward and the backward GEMM each want, and with the
  autotuner off nothing folds those transposes away: the allocator dump of a
  backward pass held about five live buffers per decoder matrix. The step
  runs the tail in two programs, the plain forward and the one that
  differentiates it, and each keeps its own set, which is where the eight
  below comes from: the 60-step shakedown peaked at 35.7 GB over the idle
  baseline against the 34.5 GB this predicts for it.

  The vocabulary kernel is the exception, because the step no longer
  differentiates through it: `head_terms/4` takes it in the two layouts its
  two GEMMs want and needs no other, so the head is exactly two buffers
  whatever the tail does.
  """
  @frozen_copies 8
  @head_copies 2
  @logit_copies 4

  def memory_estimate(spec, layers, batch_size, sequence_length, responses, opts \\ []) do
    counts = parameter_counts(spec, layers)
    tokens = batch_size * sequence_length
    head_bytes = bytes(head_type(opts) || {:f, 32})

    frozen = counts.frozen_layers * 4 * @frozen_copies
    head = counts.head * head_bytes * @head_copies

    # parameters, gradients, Adam's two moments and the updates, all f32
    expert = counts.expert * 4 * 5

    activations =
      length(layers) * tokens * (8 * spec.hidden_size + 3 * spec.intermediate_size) * 4

    # the head program's own tensors: the logits, the softcap, the log
    # probabilities and the cotangent, all over the gathered positions only
    logits = batch_size * responses * spec.vocab_size * 4 * @logit_copies
    cache = tokens * spec.hidden_size * 4

    %{
      frozen_bytes: frozen,
      head_bytes: head,
      expert_bytes: expert,
      activation_bytes: activations,
      logit_bytes: logits,
      batch_bytes: cache,
      total_bytes: frozen + head + expert + activations + logits + cache
    }
  end

  defp bytes({_kind, bits}), do: div(bits, 8)

  ## Helpers

  defp weights(opts) do
    %{
      kl: Keyword.get(opts, :kl_weight, @defaults.kl_weight),
      gate_open: Keyword.get(opts, :gate_open_weight, @defaults.gate_open_weight),
      gate_closed: Keyword.get(opts, :gate_closed_weight, @defaults.gate_closed_weight)
    }
  end

  defp with_defaults(opts) do
    Keyword.merge(
      [
        expert_learning_rate: @defaults.expert_learning_rate,
        router_learning_rate: @defaults.router_learning_rate,
        max_grad_norm: @defaults.max_grad_norm
      ],
      Keyword.take(opts, [:expert_learning_rate, :router_learning_rate, :max_grad_norm])
    )
  end

  defp model_state(data) do
    %Axon.ModelState{
      data: data,
      parameters: parameter_tree(data),
      state: %{},
      frozen_parameters: %{}
    }
    |> Axon.ModelState.freeze(fn [node_name | _rest] ->
      not SystemOne.parameter_node?(node_name)
    end)
  end

  defp parameter_tree(%Nx.Tensor{}), do: nil

  defp parameter_tree(map) when is_map(map),
    do: Map.new(map, fn {k, v} -> {k, parameter_tree(v)} end)

  defp transfer(params, backend) when is_map(params) do
    Map.new(params, fn
      {name, %Nx.Tensor{} = tensor} -> {name, to(tensor, backend)}
      {name, map} -> {name, transfer(map, backend)}
    end)
  end

  defp transfer_batch({inputs, targets}, backend) do
    {transfer(inputs, backend), transfer(targets, backend)}
  end

  defp limit(stream, nil), do: stream
  defp limit(stream, max_steps), do: Stream.take(stream, max_steps)

  defp resolve(backend), do: Gemma4MicTranscribe.Gemma4Unified.Runtime.resolve_backend(backend)

  defp compiler_opts({EXLA.Backend, backend_opts}),
    do: [compiler: EXLA] ++ Keyword.take(backend_opts, [:client, :device_id])

  defp compiler_opts(EXLA.Backend), do: [compiler: EXLA]
  defp compiler_opts(_backend), do: []

  defp tail_manifest!(path) do
    manifest =
      path
      |> Path.expand()
      |> Path.join("manifest.etf")
      |> File.read!()
      |> :erlang.binary_to_term()

    if manifest.kind != :decoder_tail do
      raise ArgumentError, "#{path} is not a decoder tail artifact"
    end

    manifest
  end

  defp scalar(%Nx.Tensor{} = tensor),
    do: tensor |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_number()

  defp scalar(number), do: number

  defp default_log(entry), do: IO.puts(Jason.encode!(entry))
end
