defmodule Gemma4MicTranscribe.LanguageId.Runtime do
  @moduledoc """
  Loads a truncated Gemma 4 audio tower and turns audio clips into pooled
  feature vectors.

  Parameters are staged on Torchx CPU and only then transferred to the
  selected backend, following the same rule as the transcription runtime:
  eager per-tensor checkpoint casts on the ROCm XLA client are not safe on
  gfx1151, fused executables are.
  """

  alias Gemma4MicTranscribe.Gemma4E4B.MelFeatures
  alias Gemma4MicTranscribe.Gemma4E4B.Spec
  alias Gemma4MicTranscribe.LanguageId.Encoder
  alias Gemma4MicTranscribe.RocmPreflight

  @sample_rate 16_000

  defstruct [:spec, :model, :params, :predict, :backend, :seconds, :frames, :tokens]

  @type t :: %__MODULE__{}

  @doc """
  Loads the encoder from a Hugging Face repository or a local checkpoint.

  Options:

    * `:repo` - `{:hf, id}` or `{:local, dir}`; default `google/gemma-4-E2B-it`
    * `:backend` - `"exla:rocm"`, `"exla:cuda"`, `"exla:host"`, `"torchx:cpu"`
    * `:depth` - conformer blocks to run; default all
    * `:capture_all_depths` - return one pooled vector per depth
    * `:pooling` - `:mean` (default) or `:mean_std`
    * `:seconds` - fixed clip length every input is padded or cut to
    * `:type` - parameter type, default bf16
    * `:params` - already-loaded `Axon.ModelState` (skips the checkpoint)
  """
  def load(opts \\ []) do
    repo = Keyword.get(opts, :repo, {:hf, "google/gemma-4-E2B-it"})
    backend = backend!(Keyword.get(opts, :backend, "torchx:cpu"))
    seconds = Keyword.get(opts, :seconds, 4)
    type = Keyword.get(opts, :type, {:bf, 16})

    spec =
      case Keyword.get(opts, :spec) do
        nil ->
          {:ok, spec} = Bumblebee.load_spec(repo, module: Encoder, architecture: :audio_encoder)
          spec

        spec ->
          spec
      end

    spec =
      Bumblebee.configure(spec,
        depth: Keyword.get(opts, :depth, spec.depth),
        capture_all_depths: Keyword.get(opts, :capture_all_depths, spec.capture_all_depths),
        pooling: Keyword.get(opts, :pooling, spec.pooling)
      )

    {model, params} =
      case Keyword.get(opts, :params) do
        nil ->
          {:ok, model_info} =
            Bumblebee.load_model(repo,
              spec: spec,
              backend: {Torchx.Backend, device: :cpu},
              type: type,
              log_params_diff: Keyword.get(opts, :log_params_diff, false)
            )

          {model_info.model, model_info.params}

        %Axon.ModelState{} = params ->
          {Encoder.model(spec), params}
      end

    params = transfer_params(params, backend)
    {_init, predict} = Axon.build(model, build_opts(backend))

    e4b_spec = Encoder.to_spec(spec)
    frames = MelFeatures.frame_count(seconds * @sample_rate, e4b_spec)

    %__MODULE__{
      spec: spec,
      model: model,
      params: params,
      predict: predict,
      backend: backend,
      seconds: seconds,
      frames: frames,
      tokens: Spec.audio_subsampled_length(e4b_spec, frames)
    }
  end

  @doc "Parameter count and byte size of the loaded tower."
  def size(%__MODULE__{params: params}) do
    tensors = params.data |> Enum.flat_map(fn {_layer, layer} -> Map.values(layer) end)

    %{
      parameters: tensors |> Enum.map(&Nx.size/1) |> Enum.sum(),
      bytes: tensors |> Enum.map(&Nx.byte_size/1) |> Enum.sum()
    }
  end

  @doc """
  Parameter counts of the subsampling stack and of each conformer block, so
  the size of a truncated tower can be computed without rebuilding it.
  """
  def layer_sizes(%__MODULE__{params: params}) do
    counts =
      Enum.reduce(params.data, %{}, fn {name, layer}, acc ->
        size = layer |> Map.values() |> Enum.map(&Nx.size/1) |> Enum.sum()

        key =
          case name do
            "audio_encoder.subsample" <> _rest -> :subsample
            "audio_encoder.blocks." <> rest -> {:block, rest |> Integer.parse() |> elem(0)}
            _other -> :other
          end

        Map.update(acc, key, size, &(&1 + size))
      end)

    blocks =
      counts
      |> Enum.filter(&match?({{:block, _}, _}, &1))
      |> Enum.sort()
      |> Enum.map(fn {_key, size} -> size end)

    %{subsample: Map.get(counts, :subsample, 0), blocks: blocks}
  end

  @doc "Parameter count of the tower truncated to `depth` blocks."
  def parameters_at_depth(%{subsample: subsample, blocks: blocks}, depth) do
    subsample + (blocks |> Enum.take(depth) |> Enum.sum())
  end

  @doc """
  Pads or cuts mono 16 kHz samples to the runtime's fixed length and returns
  the `{frames, mel}` features together with the `{tokens}` frame mask that
  marks encoder frames backed by real audio.
  """
  def prepare(%__MODULE__{} = runtime, samples) when is_list(samples) do
    e4b_spec = Encoder.to_spec(runtime.spec)
    total = runtime.seconds * @sample_rate
    real = Enum.take(samples, total)
    real_count = length(real)
    padded = real ++ List.duplicate(0.0, total - real_count)

    features = MelFeatures.extract(padded, e4b_spec)
    real_tokens = min(MelFeatures.audio_token_count(real_count, e4b_spec), runtime.tokens)

    mask =
      Nx.less(Nx.iota({runtime.tokens}, backend: Nx.BinaryBackend), real_tokens)
      |> Nx.as_type(:s64)

    {Nx.backend_transfer(features, Nx.BinaryBackend), mask}
  end

  @doc """
  Runs one fixed-size batch of prepared clips. Returns a map from depth key
  to a `{batch, hidden}` f32 tensor on the binary backend.
  """
  def encode(%__MODULE__{} = runtime, prepared) when is_list(prepared) and prepared != [] do
    features = prepared |> Enum.map(&elem(&1, 0)) |> Nx.stack()
    masks = prepared |> Enum.map(&elem(&1, 1)) |> Nx.stack()

    inputs = %{
      "input_features" => Nx.backend_transfer(features, runtime.backend),
      "frame_mask" => Nx.backend_transfer(masks, runtime.backend)
    }

    runtime.predict.(runtime.params, inputs)
    |> Map.new(fn {key, value} -> {key, Nx.backend_transfer(value, Nx.BinaryBackend)} end)
  end

  @doc """
  Encodes prepared clips in fixed-size batches, padding the last batch with
  silence so every batch compiles to the same executable.
  """
  def encode_all(%__MODULE__{} = runtime, prepared, batch_size) when batch_size > 0 do
    prepared
    |> Enum.chunk_every(batch_size)
    |> Enum.map(fn chunk ->
      count = length(chunk)
      chunk = chunk ++ List.duplicate(silent(runtime), batch_size - count)

      runtime
      |> encode(chunk)
      |> Map.new(fn {key, value} -> {key, Nx.slice_along_axis(value, 0, count, axis: 0)} end)
    end)
    |> Enum.reduce(fn batch, acc ->
      Map.merge(acc, batch, fn _key, left, right -> Nx.concatenate([left, right]) end)
    end)
  end

  defp silent(%__MODULE__{} = runtime) do
    e4b_spec = Encoder.to_spec(runtime.spec)
    features = Nx.broadcast(Nx.tensor(0.0, backend: Nx.BinaryBackend), {runtime.frames, e4b_spec.audio_mel_bins})
    {features, Nx.broadcast(Nx.tensor(0, type: :s64, backend: Nx.BinaryBackend), {runtime.tokens})}
  end

  @doc false
  def backend!("torchx:cpu"), do: {Torchx.Backend, device: :cpu}
  def backend!("torchx:cuda"), do: {Torchx.Backend, device: :cuda}
  def backend!("exla:host"), do: exla!(:host)
  def backend!("exla:cuda"), do: exla!(:cuda)
  def backend!("exla:rocm"), do: exla!(:rocm)
  def backend!({_module, _opts} = backend), do: backend

  def backend!(other) do
    raise ArgumentError,
          "unsupported backend #{inspect(other)}; expected torchx:cpu, torchx:cuda, exla:host, exla:cuda, or exla:rocm"
  end

  defp exla!(client) do
    if client == :rocm do
      {:ok, _changed?} = RocmPreflight.apply_runtime_workarounds()
    end

    {:ok, _started} = Application.ensure_all_started(:exla)
    {EXLA.Backend, client: client}
  end

  defp build_opts({EXLA.Backend, backend_opts}),
    do: [compiler: EXLA] ++ Keyword.take(backend_opts, [:client])

  defp build_opts(_backend), do: []

  defp transfer_params(%Axon.ModelState{} = params, backend) do
    %{params | data: deep_transfer(params.data, backend), state: deep_transfer(params.state, backend)}
  end

  defp deep_transfer(%Nx.Tensor{} = tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp deep_transfer(%{} = map, backend) when not is_struct(map),
    do: Map.new(map, fn {key, value} -> {key, deep_transfer(value, backend)} end)

  defp deep_transfer(other, _backend), do: other
end
