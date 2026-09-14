defmodule Gemma4MicTranscribe.LanguageId.Finetune do
  @moduledoc """
  End-to-end fine-tuning of a truncated audio tower for language
  identification, one depth at a time.

  The frozen-tower detectors keep every block's pretrained weights and only
  fit a logistic head on pooled features. That leaves shallow towers well
  short of the deeper ones, because the early blocks were never asked to
  separate languages. Here the truncated tower and the head are trained
  together with `Axon.Loop` and a Polaris optimizer so a shallower tower can
  be pushed towards the accuracy of a deeper frozen one.

  Mel inputs are decoded once per split and cached on disk (`Inputs`), the
  head starts from a logistic head fitted on the frozen tower's features with
  its standardization folded into the dense layer, and the tower starts from
  the pretrained checkpoint. The result is exported as a normal `Artifact`.
  """

  alias Gemma4MicTranscribe.Gemma4E4B.MelFeatures
  alias Gemma4MicTranscribe.Gemma4E4B.Spec
  alias Gemma4MicTranscribe.LanguageId.Artifact
  alias Gemma4MicTranscribe.LanguageId.Corpus
  alias Gemma4MicTranscribe.LanguageId.Encoder
  alias Gemma4MicTranscribe.LanguageId.Head
  alias Gemma4MicTranscribe.LanguageId.Runtime

  @sample_rate 16_000
  @inputs_file "inputs.safetensors"
  @manifest_file "manifest.json"

  defmodule Inputs do
    @moduledoc "Cached mel features, frame masks and labels for one split."
    defstruct [:features, :masks, :labels, :languages, :seconds, :keys]

    def count(%__MODULE__{labels: labels}), do: Nx.axis_size(labels, 0)
  end

  @doc """
  Decodes `clips` into fixed-window mel features and frame masks. Labels index
  into `languages` (sorted), so train and test splits share one label space.
  """
  def prepare_inputs(spec, clips, seconds, opts \\ []) do
    languages = Keyword.get(opts, :languages) || clips |> Enum.map(& &1.language) |> Enum.uniq() |> Enum.sort()
    concurrency = Keyword.get(opts, :concurrency, max(div(System.schedulers_online(), 2), 1))
    index = languages |> Enum.with_index() |> Map.new()
    window = window(spec, seconds)

    prepared =
      clips
      |> Task.async_stream(
        fn clip ->
          samples = Corpus.decode_clip!(clip, seconds)
          Runtime.prepare(window, samples)
        end,
        max_concurrency: concurrency,
        ordered: true,
        timeout: :infinity
      )
      |> Enum.map(fn {:ok, prepared} -> prepared end)

    %Inputs{
      features: prepared |> Enum.map(&elem(&1, 0)) |> Nx.stack(),
      masks: prepared |> Enum.map(&elem(&1, 1)) |> Nx.stack(),
      labels: clips |> Enum.map(&Map.fetch!(index, &1.language)) |> Nx.tensor(type: :s64),
      languages: languages,
      seconds: seconds,
      keys: Enum.map(clips, & &1.key)
    }
  end

  @doc "A runtime-shaped struct that can prepare clips without loaded parameters."
  def window(spec, seconds) do
    e4b_spec = Encoder.to_spec(spec)
    frames = MelFeatures.frame_count(seconds * @sample_rate, e4b_spec)

    %Runtime{
      spec: spec,
      seconds: seconds,
      frames: frames,
      tokens: Spec.audio_subsampled_length(e4b_spec, frames)
    }
  end

  def save_inputs!(%Inputs{} = inputs, path) do
    path = Path.expand(path)
    File.mkdir_p!(path)

    tensors = %{
      "input_features" => Nx.backend_copy(inputs.features, Nx.BinaryBackend),
      "frame_mask" => Nx.backend_copy(inputs.masks, Nx.BinaryBackend),
      "labels" => Nx.backend_copy(inputs.labels, Nx.BinaryBackend)
    }

    Safetensors.write!(Path.join(path, @inputs_file), tensors)

    manifest = %{
      version: 1,
      kind: "language_id_inputs",
      languages: inputs.languages,
      seconds: inputs.seconds,
      keys: inputs.keys
    }

    File.write!(Path.join(path, @manifest_file), Jason.encode!(manifest, pretty: true))
    path
  end

  def inputs_cached?(path), do: File.exists?(Path.join(Path.expand(path), @manifest_file))

  def load_inputs!(path) do
    path = Path.expand(path)
    manifest = path |> Path.join(@manifest_file) |> File.read!() |> Jason.decode!()
    tensors = path |> Path.join(@inputs_file) |> Safetensors.read!()

    %Inputs{
      features: tensors["input_features"],
      masks: tensors["frame_mask"],
      labels: tensors["labels"],
      languages: manifest["languages"],
      seconds: manifest["seconds"],
      keys: manifest["keys"]
    }
  end

  @doc """
  The trainable model: the truncated tower, its pooled output at `depth`,
  and a dense head with one logit per class. The head layer is named
  `"head"` so its parameters can be seeded and read back.
  """
  def model(spec, classes) do
    spec = Bumblebee.configure(spec, capture_all_depths: false)
    key = Encoder.depth_key(Encoder.depth(spec))

    spec
    |> Encoder.model()
    |> Axon.nx(fn outputs -> outputs[key] end, name: "pooled")
    |> Axon.dense(classes, name: "head")
  end

  @doc """
  Dense-head parameters equivalent to a logistic `Head`, with its input
  standardization folded into the kernel and bias.
  """
  def head_parameters(%Head{} = head) do
    scale = Nx.divide(1.0, head.std)
    kernel = Nx.multiply(head.kernel, Nx.new_axis(scale, 1))
    bias = Nx.subtract(head.bias, Nx.dot(Nx.multiply(head.mean, scale), head.kernel))
    %{"kernel" => Nx.as_type(kernel, :f32), "bias" => Nx.as_type(bias, :f32)}
  end

  @doc "A logistic `Head` (identity standardization) from trained dense parameters."
  def head_from_parameters(%{"kernel" => kernel, "bias" => bias}, languages) do
    {d, _classes} = Nx.shape(kernel)

    %Head{
      mean: Nx.broadcast(Nx.tensor(0.0, type: :f32), {d}),
      std: Nx.broadcast(Nx.tensor(1.0, type: :f32), {d}),
      kernel: Nx.as_type(kernel, :f32),
      bias: Nx.as_type(bias, :f32),
      languages: languages
    }
  end

  @doc """
  Fine-tunes a tower of `depth` blocks on cached `train` inputs.

  Options:

    * `:runtime` - a loaded f32 `Runtime` at (at least) `depth`
    * `:head` - a logistic `Head` for this depth used to seed the dense layer
    * `:test` - cached inputs evaluated after every epoch
    * `:epochs` (3), `:batch_size` (32), `:learning_rate` (2.0e-5)
    * `:freeze` - a list of layer-name prefixes whose parameters stay fixed
    * `:freeze_bounds` - keep the clipped linears' learned `input_min` /
      `input_max` / `output_min` / `output_max` calibration bounds fixed
      (default `true`); they are quantization ranges, not weights
    * `:max_grad_norm` - clip the global gradient norm before Adam
      (default `1.0`, `nil` disables)
    * `:seed` - shuffle seed (42)
    * `:compiler_opts` - defn options for the loop, default EXLA on `:rocm`
    * `:log` - function called with a progress map after each epoch
      (`epoch`, `loss`, `accuracy`, and `skipped`, the running count of
      steps dropped for non-finite gradients)

  Returns `%{model_state: state, model: model, history: [...]}`.
  """
  def train(%Inputs{} = train, depth, opts \\ []) do
    runtime = Keyword.fetch!(opts, :runtime)
    head = Keyword.fetch!(opts, :head)
    test = Keyword.get(opts, :test)
    epochs = Keyword.get(opts, :epochs, 3)
    batch_size = Keyword.get(opts, :batch_size, 32)
    learning_rate = Keyword.get(opts, :learning_rate, 2.0e-5)
    freeze = Keyword.get(opts, :freeze, [])
    freeze_bounds = Keyword.get(opts, :freeze_bounds, true)
    max_grad_norm = Keyword.get(opts, :max_grad_norm, 1.0)
    seed = Keyword.get(opts, :seed, 42)
    log = Keyword.get(opts, :log, fn _ -> :ok end)
    compiler_opts = Keyword.get(opts, :compiler_opts, [compiler: EXLA, client: :rocm])

    classes = length(train.languages)
    spec = Bumblebee.configure(runtime.spec, depth: depth, capture_all_depths: false)
    model = model(spec, classes)

    tower_data =
      runtime.params.data
      |> Enum.filter(fn {name, _} -> tower_layer?(name, depth) end)
      |> Map.new()

    data = Map.put(tower_data, "head", head_parameters(head))

    state =
      %Axon.ModelState{
        data: data,
        parameters: parameters_of(data),
        state: runtime.params.state |> Enum.filter(fn {name, _} -> tower_layer?(name, depth) end) |> Map.new(),
        frozen_parameters: %{}
      }
      |> Axon.ModelState.freeze(fn [layer | rest] ->
        Enum.any?(freeze, &String.starts_with?(layer, &1)) or
          (freeze_bounds and rest in [["input_min"], ["input_max"], ["output_min"], ["output_max"]])
      end)

    optimizer = optimizer(learning_rate, max_grad_norm)
    {_init, predict} = Axon.build(model, compiler_opts)

    evaluate = fn state ->
      if test, do: evaluate(predict, state, test, batch_size), else: nil
    end

    history_before = evaluate.(state)
    log.(%{epoch: 0, accuracy: history_before && history_before.accuracy, loss: nil, skipped: 0})

    loop =
      model
      |> Axon.Loop.trainer(&loss/2, optimizer, log: 0)
      |> Axon.Loop.handle_event(:epoch_completed, fn loop_state ->
        loss = loop_state.metrics["loss"] |> Nx.to_number()
        evaluation = evaluate.(loop_state.step_state.model_state)

        log.(%{
          epoch: Nx.to_number(loop_state.epoch) + 1,
          loss: loss,
          accuracy: evaluation && evaluation.accuracy,
          skipped: Nx.to_number(loop_state.step_state.optimizer_state.skipped)
        })

        {:continue, loop_state}
      end)

    trained =
      Axon.Loop.run(loop, batches(train, batch_size, seed, epochs), state, [epochs: epochs] ++ compiler_opts)

    %{model: model, model_state: trained, evaluation: evaluate.(trained)}
  end

  # Adam behind optional global-norm gradient clipping, with a guard that
  # skips the step when any gradient is non-finite.
  defp optimizer(learning_rate, nil) do
    guard_non_finite(Polaris.Optimizers.adam(learning_rate: learning_rate))
  end

  defp optimizer(learning_rate, max_grad_norm) do
    Polaris.Updates.clip_by_global_norm(max_norm: max_grad_norm)
    |> Polaris.Updates.scale_by_adam()
    |> Polaris.Updates.scale(-learning_rate)
    |> guard_non_finite()
  end

  @doc """
  Wraps a Polaris optimizer so a step whose gradients contain NaN or
  infinity applies no update and leaves the optimizer state untouched.

  On the ROCm client one gradient GEMM (the mel-level 3x3 convolution's
  kernel) occasionally returns NaN for a batch that computes cleanly when
  replayed, so a single flaky step must not poison Adam's moments for the
  rest of the run. The wrapped state carries a `:skipped` step counter.
  """
  def guard_non_finite({init_fn, update_fn}) do
    init = fn params ->
      %{inner: init_fn.(params), skipped: Nx.tensor(0, type: :s64)}
    end

    update = fn grads, %{inner: inner, skipped: skipped}, params ->
      finite = all_finite(grads)
      {updates, new_inner} = update_fn.(grads, inner, params)

      updates =
        Nx.Defn.Composite.traverse(updates, fn u ->
          Nx.select(Nx.broadcast(finite, Nx.shape(u)), u, Nx.tensor(0, type: Nx.type(u)))
        end)

      old = Nx.Defn.Composite.reduce(inner, [], fn t, acc -> [t | acc] end) |> Enum.reverse()

      {new_inner, []} =
        Nx.Defn.Composite.traverse(new_inner, old, fn new, [previous | rest] ->
          {Nx.select(Nx.broadcast(finite, Nx.shape(new)), new, previous), rest}
        end)

      {updates, %{inner: new_inner, skipped: Nx.add(skipped, Nx.logical_not(finite))}}
    end

    {init, update}
  end

  defp all_finite(tree) do
    Nx.Defn.Composite.reduce(tree, Nx.tensor(1, type: :u8), fn t, acc ->
      bad = Nx.logical_or(Nx.is_nan(t), Nx.is_infinity(t))
      Nx.logical_and(acc, Nx.logical_not(Nx.any(bad)))
    end)
  end

  # Cross-entropy from logits against integer labels.
  def loss(y_true, logits) do
    log_probs = Nx.subtract(logits, Nx.logsumexp(logits, axes: [-1], keep_axes: true))
    index = y_true |> Nx.as_type(:s64) |> Nx.new_axis(-1)
    log_probs |> Nx.take_along_axis(index, axis: -1) |> Nx.mean() |> Nx.negate()
  end

  # One epoch of full batches in a seeded shuffle; the loop stops each epoch
  # when the stream ends, and the stream reshuffles for the next one.
  defp batches(%Inputs{} = inputs, batch_size, seed, epochs) do
    count = Inputs.count(inputs)
    full = div(count, batch_size)

    Stream.flat_map(0..(epochs - 1), fn epoch ->
      # Deterministic shuffle without touching the process-wide RNG.
      order =
        0..(count - 1)
        |> Enum.map(fn i -> {:erlang.phash2({seed, epoch, i}), i} end)
        |> Enum.sort()
        |> Enum.map(&elem(&1, 1))

      order
      |> Enum.take(full * batch_size)
      |> Enum.chunk_every(batch_size)
      |> Stream.map(fn indices ->
        index = Nx.tensor(indices, type: :s64)

        {%{
           "input_features" => Nx.take(inputs.features, index),
           "frame_mask" => Nx.take(inputs.masks, index)
         }, Nx.take(inputs.labels, index)}
      end)
    end)
  end

  @doc "Accuracy of `predict` with `state` over cached inputs, in fixed batches."
  def evaluate(predict, state, %Inputs{} = inputs, batch_size) do
    count = Inputs.count(inputs)

    predictions =
      0..(count - 1)
      |> Enum.chunk_every(batch_size)
      |> Enum.map(fn indices ->
        n = length(indices)
        indices = indices ++ List.duplicate(0, batch_size - n)
        index = Nx.tensor(indices, type: :s64)

        inputs_map = %{
          "input_features" => Nx.take(inputs.features, index),
          "frame_mask" => Nx.take(inputs.masks, index)
        }

        predict.(state, inputs_map)
        |> Nx.argmax(axis: -1)
        |> Nx.backend_transfer(Nx.BinaryBackend)
        |> Nx.slice_along_axis(0, n, axis: 0)
      end)
      |> Nx.concatenate()

    labels = Nx.backend_copy(inputs.labels, Nx.BinaryBackend)
    correct = Nx.equal(predictions, labels)

    per_language =
      inputs.languages
      |> Enum.with_index()
      |> Map.new(fn {language, i} ->
        mask = Nx.equal(labels, i)
        n = mask |> Nx.sum() |> Nx.to_number()
        hits = correct |> Nx.logical_and(mask) |> Nx.sum() |> Nx.to_number()
        {language, if(n > 0, do: hits / n, else: nil)}
      end)

    accuracies = per_language |> Map.values() |> Enum.reject(&is_nil/1)

    %{
      accuracy: correct |> Nx.mean() |> Nx.to_number(),
      macro_accuracy: Enum.sum(accuracies) / max(length(accuracies), 1),
      per_language: per_language,
      samples: count
    }
  end

  @doc """
  Packages a fine-tuned model state as a bf16 `Artifact` at `depth`.
  """
  def to_artifact(%Runtime{} = runtime, %Axon.ModelState{} = state, depth, languages, seconds, opts \\ []) do
    {head_data, tower_data} = Map.pop!(state.data, "head")
    type = Keyword.get(opts, :type, {:bf, 16})

    tower =
      %Axon.ModelState{
        state
        | data: cast_tree(tower_data, type),
          parameters: Map.delete(state.parameters, "head"),
          frozen_parameters: %{}
      }

    spec = Bumblebee.configure(runtime.spec, depth: depth, capture_all_depths: false)

    exported =
      Runtime.load(
        spec: spec,
        params: tower,
        backend: "torchx:cpu",
        depth: depth,
        pooling: spec.pooling,
        seconds: seconds
      )

    head = head_from_parameters(cast_tree(head_data, {:f, 32}), languages)
    Artifact.build(exported, depth, head, opts)
  end

  defp cast_tree(%Nx.Tensor{} = tensor, type) do
    tensor |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.as_type(type)
  end

  defp cast_tree(%{} = map, type), do: Map.new(map, fn {k, v} -> {k, cast_tree(v, type)} end)

  defp parameters_of(%Nx.Tensor{}), do: nil
  defp parameters_of(%{} = map), do: Map.new(map, fn {k, v} -> {k, parameters_of(v)} end)

  defp tower_layer?("audio_encoder.subsample" <> _rest, _depth), do: true

  defp tower_layer?("audio_encoder.blocks." <> rest, depth) do
    case Integer.parse(rest) do
      {index, _} -> index < depth
      :error -> false
    end
  end

  defp tower_layer?(_name, _depth), do: false
end
