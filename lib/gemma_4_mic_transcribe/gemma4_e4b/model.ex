defmodule Gemma4MicTranscribe.Gemma4E4B.Model do
  @moduledoc false

  # Assembles the Gemma 4 E4B audio path into one Bumblebee model:
  #
  #   mel features -> conformer encoder -> audio embeddings
  #   token ids    -> embedder          -> text embeddings
  #                                     -> merged, audio placeholders replaced
  #                                     -> decoder (per-layer inputs, KV sharing)
  #                                     -> language modeling head

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.Text.Generation

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers
  alias Gemma4MicTranscribe.Gemma4E4B.AudioEncoder
  alias Gemma4MicTranscribe.Gemma4E4B.Decoder
  alias Gemma4MicTranscribe.Gemma4E4B.Spec

  defstruct spec: %Spec{}

  @impl true
  def architectures, do: [:for_conditional_generation]

  @impl true
  def config(%__MODULE__{spec: spec} = model, opts) do
    %{model | spec: Spec.config(spec, opts)}
  end

  @impl true
  def input_template(%__MODULE__{spec: spec}) do
    %{
      "input_ids" => Nx.template({1, 1}, :s64),
      "attention_mask" => Nx.template({1, 1}, :s64),
      "position_ids" => Nx.template({1, 1}, :s64),
      "input_features" => Nx.template({1, 4, spec.audio_mel_bins}, {:f, 32})
    }
  end

  @impl true
  def init_cache(%__MODULE__{spec: spec}, batch_size, max_length, _inputs) do
    # Shared blocks read another block's cache, so only computing blocks get
    # an entry; the rest carry a placeholder to keep indices aligned.
    blocks =
      spec.layer_types
      |> Enum.with_index()
      |> Enum.map(fn {layer_type, index} ->
        head_size =
          if layer_type == :full_attention,
            do: spec.global_attention_head_size,
            else: spec.attention_head_size

        length = if Spec.kv_shared_layer?(spec, index), do: 1, else: max_length
        heads = if Spec.kv_shared_layer?(spec, index), do: 1, else: spec.num_attention_heads
        size = if Spec.kv_shared_layer?(spec, index), do: 1, else: head_size

        %{
          self_attention: attention_cache(batch_size, length, heads, size),
          cross_attention: attention_cache(batch_size, 1, 1, 1)
        }
      end)
      |> List.to_tuple()

    %{
      blocks: blocks,
      offset: Nx.tensor(0),
      attention_mask: Nx.broadcast(0, {batch_size, max_length})
    }
  end

  @impl true
  def traverse_cache(_model, cache, fun) do
    Layers.Decoder.traverse_cache(cache, fun)
  end

  @impl true
  def model(%__MODULE__{spec: spec}) do
    inputs = inputs(spec)

    input_ids = inputs["input_ids"]

    # Audio placeholders carry no text meaning, so they are embedded as pad
    # and then overwritten with the encoder output.
    text_ids =
      Axon.nx(input_ids, fn ids ->
        Nx.select(Nx.equal(ids, spec.audio_token_id), spec.pad_token_id, ids)
      end)

    audio_mask = Axon.nx(input_ids, &Nx.equal(&1, spec.audio_token_id))

    embeddings =
      text_ids
      |> Axon.embedding(spec.vocab_size, spec.hidden_size,
        name: join("embedder", "token_embedding")
      )
      |> Axon.nx(fn state ->
        Nx.multiply(state, Nx.sqrt(Nx.tensor(spec.hidden_size, type: Nx.type(state))))
      end)

    # One embedding per block per token, looked up from a single wide table
    # and reshaped into {batch, sequence, blocks, per_layer_size}.
    per_layer_inputs =
      text_ids
      |> Axon.embedding(
        spec.vocab_size_per_layer_input,
        spec.num_blocks * spec.hidden_size_per_layer_input,
        name: join("embedder", "per_layer_embedding")
      )
      |> Axon.nx(fn state ->
        {batch, sequence, _} = Nx.shape(state)
        Nx.reshape(state, {batch, sequence, spec.num_blocks, spec.hidden_size_per_layer_input})
      end)

    audio_embeddings = AudioEncoder.encode(inputs["input_features"], spec)

    hidden_state = replace_audio_embeddings(embeddings, audio_embeddings, audio_mask)

    outputs =
      Decoder.decode(
        hidden_state,
        per_layer_inputs,
        inputs["position_ids"],
        inputs["attention_mask"],
        inputs["cache"],
        spec
      )

    logits =
      outputs.hidden_state
      |> rms_norm(spec.layer_norm_epsilon, name: "output_norm")
      |> Layers.dense_transposed(spec.vocab_size, name: join("language_modeling_head", "output"))
      |> then(fn logits ->
        if spec.final_logit_softcapping do
          cap = spec.final_logit_softcapping
          Axon.nx(logits, fn l -> Nx.multiply(Nx.tanh(Nx.divide(l, cap)), cap) end)
        else
          logits
        end
      end)

    Layers.output(%{logits: logits, cache: outputs.cache})
  end

  defp inputs(spec) do
    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", shape: {nil, nil}),
      Axon.input("attention_mask", shape: {nil, nil}),
      Axon.input("position_ids", shape: {nil, nil}),
      Axon.input("input_features", shape: {nil, nil, spec.audio_mel_bins}),
      Axon.input("cache", optional: true)
    ])
  end

  # Audio placeholder positions take encoder frames in order.
  defp replace_audio_embeddings(embeddings, audio_embeddings, audio_mask) do
    Axon.layer(
      fn embeddings, audio_embeddings, audio_mask, _opts ->
        hidden_size = Nx.axis_size(embeddings, 2)
        frames = Nx.axis_size(audio_embeddings, 1)

        indices =
          audio_mask
          |> Nx.as_type({:s, 64})
          |> Nx.cumulative_sum(axis: 1)
          |> Nx.subtract(1)
          |> Nx.max(0)
          |> Nx.min(frames - 1)
          |> Nx.new_axis(-1)
          |> Nx.broadcast({Nx.axis_size(embeddings, 0), Nx.axis_size(embeddings, 1), hidden_size})

        gathered = Nx.take_along_axis(audio_embeddings, indices, axis: 1)

        mask =
          audio_mask
          |> Nx.new_axis(-1)
          |> Nx.broadcast({Nx.axis_size(embeddings, 0), Nx.axis_size(embeddings, 1), hidden_size})

        Nx.select(mask, gathered, embeddings)
      end,
      [embeddings, audio_embeddings, audio_mask],
      name: "audio_embedding_replacement"
    )
  end

  defp attention_cache(batch_size, sequence_length, num_heads, head_size) do
    zeros = Nx.broadcast(0.0, {batch_size, sequence_length, num_heads, head_size})
    %{key: zeros, value: zeros}
  end

  defp rms_norm(input, epsilon, opts) do
    Layers.rms_norm(input,
      name: opts[:name],
      shift: 0.0,
      epsilon: epsilon,
      initializer: Axon.Initializers.ones(),
      upcast: :all
    )
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(%{spec: spec} = model, data) do
      %{model | spec: Bumblebee.HuggingFace.Transformers.Config.load(spec, data)}
    end
  end
end
