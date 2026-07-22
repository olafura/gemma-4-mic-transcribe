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

  # The runtime reads fields like pad_token_id straight off the loaded spec,
  # so the model struct carries the configuration flat rather than nested.
  defstruct Map.to_list(Map.from_struct(%Spec{}))

  defp to_spec(%__MODULE__{} = model), do: struct(Spec, Map.from_struct(model))

  @impl true
  def architectures, do: [:for_conditional_generation]

  @impl true
  def config(%__MODULE__{} = model, opts) do
    spec = model |> to_spec() |> Spec.config(opts)
    struct(__MODULE__, Map.from_struct(spec))
  end

  @impl true
  def input_template(%__MODULE__{} = model) do
    spec = to_spec(model)

    %{
      "input_ids" => Nx.template({1, 1}, :s64),
      "attention_mask" => Nx.template({1, 1}, :s64),
      "position_ids" => Nx.template({1, 1}, :s64),
      "input_features" => Nx.template({1, 4, spec.audio_mel_bins}, {:f, 32})
    }
  end

  @impl true
  def init_cache(%__MODULE__{} = model, batch_size, max_length, _inputs) do
    spec = to_spec(model)
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
  def model(%__MODULE__{} = model_spec) do
    spec = to_spec(model_spec)
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

    # Token-identity half of the per-layer inputs: one embedding per block per
    # token, looked up from a single wide table, scaled like the main
    # embedding but by its own width.
    per_layer_embeddings =
      text_ids
      |> Axon.embedding(
        spec.vocab_size_per_layer_input,
        spec.num_blocks * spec.hidden_size_per_layer_input,
        name: join("embedder", "per_layer_embedding")
      )
      |> Axon.nx(fn state ->
        {batch, sequence, _} = Nx.shape(state)

        state
        |> Nx.reshape({batch, sequence, spec.num_blocks, spec.hidden_size_per_layer_input})
        |> Nx.multiply(
          Nx.sqrt(Nx.tensor(spec.hidden_size_per_layer_input, type: Nx.type(state)))
        )
      end)

    audio_embeddings = AudioEncoder.encode(inputs["input_features"], spec)

    hidden_state = replace_audio_embeddings(embeddings, audio_embeddings, audio_mask)

    # Context half: the merged embeddings, audio included, projected into the
    # per-layer width. This is the only path besides attention through which
    # audio reaches each block's per-layer gate.
    per_layer_projection =
      hidden_state
      |> Axon.dense(spec.num_blocks * spec.hidden_size_per_layer_input,
        use_bias: false,
        name: join("embedder", "per_layer_projection")
      )
      |> Axon.nx(fn state ->
        {batch, sequence, _} = Nx.shape(state)

        state
        |> Nx.multiply(Nx.rsqrt(Nx.tensor(spec.hidden_size, type: Nx.type(state))))
        |> Nx.reshape({batch, sequence, spec.num_blocks, spec.hidden_size_per_layer_input})
      end)
      |> rms_norm(spec.layer_norm_epsilon, name: join("embedder", "per_layer_projection_norm"))

    per_layer_inputs =
      Axon.add(per_layer_projection, per_layer_embeddings)
      |> Axon.nx(fn state ->
        Nx.multiply(state, Nx.rsqrt(Nx.tensor(2, type: Nx.type(state))))
      end)

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

  # Audio placeholder positions take the LAST n encoder frames, where n is
  # the number of placeholders. When the encoder input carries mel lookback
  # (incremental prefill re-encodes the previous chunk for exact context),
  # the lookback frames' outputs are boundary-contaminated and must be
  # dropped; when frames equal placeholders this reduces to taking them in
  # order.
  defp replace_audio_embeddings(embeddings, audio_embeddings, audio_mask) do
    Axon.layer(
      fn embeddings, audio_embeddings, audio_mask, _opts ->
        hidden_size = Nx.axis_size(embeddings, 2)
        frames = Nx.axis_size(audio_embeddings, 1)

        placeholders = audio_mask |> Nx.as_type({:s, 64}) |> Nx.sum()

        indices =
          audio_mask
          |> Nx.as_type({:s, 64})
          |> Nx.cumulative_sum(axis: 1)
          |> Nx.subtract(1)
          |> Nx.add(Nx.subtract(frames, placeholders))
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
    def load(model, data) do
      spec = struct(Gemma4MicTranscribe.Gemma4E4B.Spec, Map.from_struct(model))
      loaded = Bumblebee.HuggingFace.Transformers.Config.load(spec, data)
      struct(Gemma4MicTranscribe.Gemma4E4B.Model, Map.from_struct(loaded))
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      audio = "model.audio_tower"
      text = "model.language_model"

      %{
        # embeddings
        "embedder.token_embedding" => "#{text}.embed_tokens",
        "embedder.per_layer_embedding" => "#{text}.embed_tokens_per_layer",
        "embedder.per_layer_projection" => "#{text}.per_layer_model_projection",
        "embedder.per_layer_projection_norm" => "#{text}.per_layer_projection_norm",
        "output_norm" => "#{text}.norm",
        "language_modeling_head.output" =>
          if(spec.tie_word_embeddings, do: "#{text}.embed_tokens", else: "lm_head"),

        # audio subsampling and output
        "audio_encoder.subsample.layer0.conv" =>
          conv2d("#{audio}.subsample_conv_projection.layer0.conv"),
        "audio_encoder.subsample.layer0.norm" => "#{audio}.subsample_conv_projection.layer0.norm",
        "audio_encoder.subsample.layer1.conv" =>
          conv2d("#{audio}.subsample_conv_projection.layer1.conv"),
        "audio_encoder.subsample.layer1.norm" => "#{audio}.subsample_conv_projection.layer1.norm",
        "audio_encoder.subsample.input_proj_linear" =>
          "#{audio}.subsample_conv_projection.input_proj_linear",
        "audio_encoder.output_proj" => "#{audio}.output_proj",
        "embed_audio.embedding_projection" => "model.embed_audio.embedding_projection",

        # audio blocks
        "audio_encoder.blocks.{n}.ffn_start.pre_norm" =>
          "#{audio}.layers.{n}.feed_forward1.pre_layer_norm",
        "audio_encoder.blocks.{n}.ffn_start.post_norm" =>
          "#{audio}.layers.{n}.feed_forward1.post_layer_norm",
        "audio_encoder.blocks.{n}.ffn_start.intermediate" =>
          clipped("#{audio}.layers.{n}.feed_forward1.ffw_layer_1"),
        "audio_encoder.blocks.{n}.ffn_start.output" =>
          clipped("#{audio}.layers.{n}.feed_forward1.ffw_layer_2"),
        "audio_encoder.blocks.{n}.ffn_end.pre_norm" =>
          "#{audio}.layers.{n}.feed_forward2.pre_layer_norm",
        "audio_encoder.blocks.{n}.ffn_end.post_norm" =>
          "#{audio}.layers.{n}.feed_forward2.post_layer_norm",
        "audio_encoder.blocks.{n}.ffn_end.intermediate" =>
          clipped("#{audio}.layers.{n}.feed_forward2.ffw_layer_1"),
        "audio_encoder.blocks.{n}.ffn_end.output" =>
          clipped("#{audio}.layers.{n}.feed_forward2.ffw_layer_2"),
        "audio_encoder.blocks.{n}.attention.pre_norm" => "#{audio}.layers.{n}.norm_pre_attn",
        "audio_encoder.blocks.{n}.attention.post_norm" => "#{audio}.layers.{n}.norm_post_attn",
        "audio_encoder.blocks.{n}.attention.query" =>
          clipped("#{audio}.layers.{n}.self_attn.q_proj"),
        "audio_encoder.blocks.{n}.attention.key" =>
          clipped("#{audio}.layers.{n}.self_attn.k_proj"),
        "audio_encoder.blocks.{n}.attention.value" =>
          clipped("#{audio}.layers.{n}.self_attn.v_proj"),
        "audio_encoder.blocks.{n}.attention.chunked" => %{
          "per_dim_scale" => {[{"#{audio}.layers.{n}.self_attn", "per_dim_scale"}], &identity/1},
          # relative_k_proj projects the sinusoidal position table, so its
          # weight lives on the attention layer rather than on a dense
          "relative_key" =>
            {[{"#{audio}.layers.{n}.self_attn.relative_k_proj", "weight"}], &transpose/1}
        },
        "audio_encoder.blocks.{n}.attention.output" =>
          clipped("#{audio}.layers.{n}.self_attn.post"),
        "audio_encoder.blocks.{n}.conv.pre_layer_norm" =>
          "#{audio}.layers.{n}.lconv1d.pre_layer_norm",
        "audio_encoder.blocks.{n}.conv.linear_start" =>
          clipped("#{audio}.layers.{n}.lconv1d.linear_start"),
        "audio_encoder.blocks.{n}.conv.depthwise_conv1d" =>
          conv1d("#{audio}.layers.{n}.lconv1d.depthwise_conv1d"),
        "audio_encoder.blocks.{n}.conv.conv_norm" => "#{audio}.layers.{n}.lconv1d.conv_norm",
        "audio_encoder.blocks.{n}.conv.linear_end" =>
          clipped("#{audio}.layers.{n}.lconv1d.linear_end"),
        "audio_encoder.blocks.{n}.output_norm" => "#{audio}.layers.{n}.norm_out",

        # decoder blocks
        "decoder.blocks.{n}.self_attention_norm" => "#{text}.layers.{n}.input_layernorm",
        "decoder.blocks.{n}.post_attention_norm" => "#{text}.layers.{n}.post_attention_layernorm",
        "decoder.blocks.{n}.pre_ffn_norm" => "#{text}.layers.{n}.pre_feedforward_layernorm",
        "decoder.blocks.{n}.post_ffn_norm" => "#{text}.layers.{n}.post_feedforward_layernorm",
        "decoder.blocks.{n}.self_attention.query" => "#{text}.layers.{n}.self_attn.q_proj",
        "decoder.blocks.{n}.self_attention.key" => "#{text}.layers.{n}.self_attn.k_proj",
        "decoder.blocks.{n}.self_attention.value" => "#{text}.layers.{n}.self_attn.v_proj",
        "decoder.blocks.{n}.self_attention.output" => "#{text}.layers.{n}.self_attn.o_proj",
        "decoder.blocks.{n}.self_attention.query_norm" => "#{text}.layers.{n}.self_attn.q_norm",
        "decoder.blocks.{n}.self_attention.key_norm" => "#{text}.layers.{n}.self_attn.k_norm",
        "decoder.blocks.{n}.ffn.gate" => "#{text}.layers.{n}.mlp.gate_proj",
        "decoder.blocks.{n}.ffn.intermediate" => "#{text}.layers.{n}.mlp.up_proj",
        "decoder.blocks.{n}.ffn.output" => "#{text}.layers.{n}.mlp.down_proj",
        "decoder.blocks.{n}.layer_scalar" => "#{text}.layers.{n}",
        "decoder.blocks.{n}.per_layer.input_gate" => "#{text}.layers.{n}.per_layer_input_gate",
        "decoder.blocks.{n}.per_layer.projection" => "#{text}.layers.{n}.per_layer_projection",
        "decoder.blocks.{n}.per_layer.post_norm" => "#{text}.layers.{n}.post_per_layer_input_norm"
      }
    end

    # Clipped linears keep their weight under a nested "linear" and their
    # bounds beside it.
    defp clipped(source) do
      %{
        "kernel" => {[{"#{source}.linear", "weight"}], &transpose/1},
        "input_min" => {[{source, "input_min"}], &identity/1},
        "input_max" => {[{source, "input_max"}], &identity/1},
        "output_min" => {[{source, "output_min"}], &identity/1},
        "output_max" => {[{source, "output_max"}], &identity/1}
      }
    end

    # PyTorch stores conv kernels as {out, in, spatial...}; Axon wants
    # {spatial..., in, out}.
    defp conv2d(source) do
      %{"kernel" => {[{source, "weight"}], &transpose_conv2d/1}}
    end

    defp conv1d(source) do
      %{"kernel" => {[{source, "weight"}], &transpose_conv1d/1}}
    end

    defp transpose_conv2d([tensor]), do: Nx.transpose(tensor, axes: [2, 3, 1, 0])
    defp transpose_conv1d([tensor]), do: Nx.transpose(tensor, axes: [2, 1, 0])

    defp transpose([tensor]), do: Nx.transpose(tensor)
    defp identity([tensor]), do: tensor
  end
end
