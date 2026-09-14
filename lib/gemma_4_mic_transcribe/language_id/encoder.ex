defmodule Gemma4MicTranscribe.LanguageId.Encoder do
  @moduledoc """
  Encoder-only Gemma 4 audio model for spoken language identification.

  Reuses the E2B/E4B conformer audio tower up to a configurable depth and
  mean-pools the hidden state over the real (unpadded) encoder frames. The
  decoder, embeddings, and the tower's own output projection are not built, so
  loading from a full checkpoint reads only the tensors the truncated tower
  needs.

  With `capture_all_depths: true` the model returns one pooled vector per
  depth (`"depth_0"` after the subsampling stack, `"depth_n"` after conformer
  block `n - 1`), so a single forward pass yields training features for every
  candidate truncation.
  """

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  import Nx.Defn

  alias Bumblebee.Layers
  alias Gemma4MicTranscribe.Gemma4E4B.AudioEncoder
  alias Gemma4MicTranscribe.Gemma4E4B.Spec

  @extra_fields [depth: nil, capture_all_depths: false, pooling: :mean]

  defstruct Keyword.merge(Map.to_list(Map.from_struct(%Spec{})), [architecture: :audio_encoder] ++ @extra_fields)

  @doc false
  def to_spec(%__MODULE__{} = model) do
    struct(Spec, model |> Map.from_struct() |> Map.drop(Keyword.keys(@extra_fields)))
  end

  @doc "Number of conformer blocks this model runs."
  def depth(%__MODULE__{depth: nil, audio_num_blocks: blocks}), do: blocks
  def depth(%__MODULE__{depth: depth}), do: depth

  @doc "Output key for the pooled vector after `depth` conformer blocks."
  def depth_key(depth) when is_integer(depth) and depth >= 0, do: "depth_#{depth}"

  @impl true
  def architectures, do: [:audio_encoder]

  @impl true
  def config(%__MODULE__{} = model, opts) do
    {extra, spec_opts} = Keyword.split(opts, Keyword.keys(@extra_fields))
    spec = model |> to_spec() |> Spec.config(spec_opts)

    kept = model |> Map.from_struct() |> Map.take(Keyword.keys(@extra_fields))
    model = struct(__MODULE__, Map.merge(Map.from_struct(spec), kept))
    model = struct!(model, extra)

    if model.depth != nil and (model.depth < 0 or model.depth > model.audio_num_blocks) do
      raise ArgumentError,
            "depth must be between 0 and #{model.audio_num_blocks}, got: #{inspect(model.depth)}"
    end

    if model.pooling not in [:mean, :mean_std] do
      raise ArgumentError, "pooling must be :mean or :mean_std, got: #{inspect(model.pooling)}"
    end

    %{model | architecture: :audio_encoder}
  end

  @impl true
  def input_template(%__MODULE__{} = model) do
    spec = to_spec(model)

    %{
      "input_features" => Nx.template({1, 4, spec.audio_mel_bins}, {:f, 32}),
      "frame_mask" => Nx.template({1, 1}, {:s, 64})
    }
  end

  @impl true
  def model(%__MODULE__{} = model_spec) do
    spec = to_spec(model_spec)
    depth = depth(model_spec)
    inputs = inputs(spec)
    frame_mask = inputs["frame_mask"]

    subsampled =
      AudioEncoder.subsample(inputs["input_features"], spec, name: "audio_encoder.subsample")

    {_hidden_state, pooled} =
      Enum.reduce(0..(depth - 1)//1, {subsampled, [{0, subsampled}]}, fn index,
                                                                         {hidden_state, acc} ->
        hidden_state =
          AudioEncoder.conformer_block(hidden_state, spec, name: "audio_encoder.blocks.#{index}")

        {hidden_state, [{index + 1, hidden_state} | acc]}
      end)

    pooled =
      pooled
      |> Enum.reverse()
      |> Enum.filter(fn {index, _state} -> model_spec.capture_all_depths or index == depth end)
      |> Map.new(fn {index, state} ->
        {depth_key(index),
         masked_pool(state, frame_mask, model_spec.pooling, name: "pooling.#{depth_key(index)}")}
      end)

    Layers.output(pooled)
  end

  defp inputs(spec) do
    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_features", shape: {nil, nil, spec.audio_mel_bins}),
      Axon.input("frame_mask", shape: {nil, nil})
    ])
  end

  # Statistics over encoder frames whose mask is set, computed in f32 so
  # bf16 towers do not lose precision in the sums. Frames masked off are
  # typically silence padding; the encoder is causal with a bounded left
  # context, so they never influence the frames that are kept. `:mean`
  # yields `{batch, hidden}`; `:mean_std` appends the per-feature standard
  # deviation for `{batch, 2 * hidden}`.
  defp masked_pool(hidden_state, frame_mask, pooling, opts) do
    Axon.layer(&masked_pool_impl/3, [hidden_state, frame_mask],
      name: opts[:name],
      op_name: :language_id_masked_pool,
      pooling: pooling
    )
  end

  defnp masked_pool_impl(hidden_state, frame_mask, opts \\ []) do
    opts = keyword!(opts, [:mode, pooling: :mean])
    hidden_state = Nx.as_type(hidden_state, :f32)
    mask = Nx.as_type(frame_mask, :f32)
    counts = mask |> Nx.sum(axes: [1], keep_axes: true) |> Nx.max(1.0)
    weights = mask / counts
    mean = Nx.dot(weights, [1], [0], hidden_state, [1], [0])

    case opts[:pooling] do
      :mean ->
        mean

      :mean_std ->
        centered = (hidden_state - Nx.new_axis(mean, 1)) * Nx.new_axis(mask, 2)
        variance = Nx.dot(weights, [1], [0], centered * centered, [1], [0])
        Nx.concatenate([mean, Nx.sqrt(variance + 1.0e-6)], axis: 1)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(model, data) do
      spec = Gemma4MicTranscribe.LanguageId.Encoder.to_spec(model)
      loaded = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)

      %{
        struct(Gemma4MicTranscribe.LanguageId.Encoder, Map.from_struct(loaded))
        | architecture: :audio_encoder,
          depth: model.depth,
          capture_all_depths: model.capture_all_depths,
          pooling: model.pooling
      }
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    # The tower's checkpoint layout is the one E4B's model already describes;
    # keep only the encoder entries so unrelated decoder tensors are never
    # read.
    def params_mapping(model) do
      e4b =
        struct(
          Gemma4MicTranscribe.Gemma4E4B.Model,
          Map.from_struct(Gemma4MicTranscribe.LanguageId.Encoder.to_spec(model))
        )

      e4b
      |> Bumblebee.HuggingFace.Transformers.Model.params_mapping()
      |> Enum.filter(fn {key, _source} -> String.starts_with?(key, "audio_encoder.") end)
      |> Map.new()
    end
  end
end
