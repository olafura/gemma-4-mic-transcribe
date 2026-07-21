defmodule Gemma4MicTranscribe.Gemma4E4B.AudioEncoder do
  @moduledoc false

  # Conformer audio encoder for Gemma 4 E4B.
  #
  # Where the 12B Unified model projects raw PCM frames straight into the
  # decoder, E4B runs mel features through a subsampling convolution stack and
  # 12 conformer blocks first. Each block is
  #
  #   half-step feed forward -> chunked local attention -> convolution ->
  #   half-step feed forward -> norm
  #
  # with residual_weight scaling the feed forward branches, which is why a
  # block is often called a "macaron" layer.

  import Nx.Defn

  alias Bumblebee.Layers
  alias Gemma4MicTranscribe.Gemma4E4B.Spec

  @doc """
  Builds the encoder over `{batch, frames, mel_bins}` features.

  Returns `{batch, subsampled_frames, hidden_size}` embeddings ready to be
  spliced into the decoder in place of audio placeholder tokens.
  """
  def encode(features, %Spec{} = spec, opts \\ []) do
    name = Keyword.get(opts, :name, "audio_encoder")

    features
    |> subsample(spec, name: join(name, "subsample"))
    |> then(fn hidden_state ->
      Enum.reduce(0..(spec.audio_num_blocks - 1), hidden_state, fn index, acc ->
        conformer_block(acc, spec, name: join(name, "blocks.#{index}"))
      end)
    end)
    |> rms_norm(spec.audio_rms_norm_epsilon, name: join(name, "output_norm"))
    # the tower ends at output_proj_dims with a bias, then embed_audio maps
    # that into the decoder width
    |> Axon.dense(spec.audio_output_proj_dims, use_bias: true, name: join(name, "output_proj"))
    |> Axon.dense(spec.hidden_size,
      use_bias: false,
      name: "embed_audio.embedding_projection"
    )
  end

  # Two strided convolutions over {batch, frames, mel} treated as a single
  # channel image, each halving the time axis, then a projection into the
  # encoder width.
  defp subsample(features, %Spec{} = spec, opts) do
    name = opts[:name]
    [first_channels, second_channels] = spec.audio_subsampling_conv_channels

    features
    |> Axon.nx(&Nx.new_axis(&1, -1))
    |> Axon.conv(first_channels,
      kernel_size: {3, 3},
      strides: [2, 2],
      padding: [{1, 1}, {1, 1}],
      use_bias: false,
      name: join(name, "layer0.conv")
    )
    |> rms_norm(spec.audio_rms_norm_epsilon, name: join(name, "layer0.norm"))
    |> Axon.activation(spec.audio_activation)
    |> Axon.conv(second_channels,
      kernel_size: {3, 3},
      strides: [2, 2],
      padding: [{1, 1}, {1, 1}],
      use_bias: false,
      name: join(name, "layer1.conv")
    )
    |> rms_norm(spec.audio_rms_norm_epsilon, name: join(name, "layer1.norm"))
    |> Axon.activation(spec.audio_activation)
    |> Axon.nx(fn state ->
      {batch, frames, freq, channels} = Nx.shape(state)
      Nx.reshape(state, {batch, frames, freq * channels})
    end)
    |> Axon.dense(spec.audio_hidden_size,
      use_bias: false,
      name: join(name, "input_proj_linear")
    )
  end

  defp conformer_block(hidden_state, %Spec{} = spec, opts) do
    name = opts[:name]

    hidden_state
    |> half_feed_forward(spec, name: join(name, "ffn_start"))
    |> attention(spec, name: join(name, "attention"))
    |> convolution(spec, name: join(name, "conv"))
    |> half_feed_forward(spec, name: join(name, "ffn_end"))
    |> rms_norm(spec.audio_rms_norm_epsilon, name: join(name, "output_norm"))
  end

  # residual_weight (0.5) makes each feed forward a half step, so the two
  # around the attention sum to one full-width feed forward.
  defp half_feed_forward(hidden_state, %Spec{} = spec, opts) do
    name = opts[:name]
    weight = spec.audio_residual_weight

    residual = hidden_state

    hidden_state
    |> rms_norm(spec.audio_rms_norm_epsilon, name: join(name, "pre_norm"))
    |> dense_maybe_clipped(4 * spec.audio_hidden_size, spec, name: join(name, "intermediate"))
    |> Axon.activation(spec.audio_activation)
    |> dense_maybe_clipped(spec.audio_hidden_size, spec, name: join(name, "output"))
    |> rms_norm(spec.audio_rms_norm_epsilon, name: join(name, "post_norm"))
    |> Axon.nx(&Nx.multiply(&1, weight))
    |> then(&Axon.add(residual, &1))
  end

  defp attention(hidden_state, %Spec{} = spec, opts) do
    name = opts[:name]
    head_size = div(spec.audio_hidden_size, spec.audio_num_attention_heads)
    residual = hidden_state

    normed = rms_norm(hidden_state, spec.audio_rms_norm_epsilon, name: join(name, "pre_norm"))

    query =
      normed
      |> dense_maybe_clipped(spec.audio_hidden_size, spec, name: join(name, "query"))
      |> Layers.split_heads(spec.audio_num_attention_heads)

    key =
      normed
      |> dense_maybe_clipped(spec.audio_hidden_size, spec, name: join(name, "key"))
      |> Layers.split_heads(spec.audio_num_attention_heads)

    value =
      normed
      |> dense_maybe_clipped(spec.audio_hidden_size, spec, name: join(name, "value"))
      |> Layers.split_heads(spec.audio_num_attention_heads)

    # The checkpoint carries relative_k_proj and per_dim_scale beside the
    # usual projections, so attention adds a relative position bias and scales
    # each query dimension rather than the whole head uniformly.
    # relative_k_proj is a plain weight in the checkpoint, with none of the
    # clip bounds the other audio linears carry
    relative_key =
      normed
      |> Axon.dense(spec.audio_hidden_size, use_bias: false, name: join(name, "relative_key"))
      |> Layers.split_heads(spec.audio_num_attention_heads)

    Axon.layer(
      &chunked_attention/6,
      [
        query,
        key,
        value,
        relative_key,
        Axon.param("per_dim_scale", {head_size})
      ],
      name: join(name, "chunked"),
      op_name: :gemma4_audio_attention,
      chunk_size: spec.audio_attention_chunk_size,
      context_left: spec.audio_attention_context_left,
      context_right: spec.audio_attention_context_right,
      logit_cap: spec.audio_attention_logit_cap,
      invalid_value: spec.audio_attention_invalid_logits_value,
      head_size: head_size
    )
    |> Layers.flatten_trailing()
    |> dense_maybe_clipped(spec.audio_hidden_size, spec, name: join(name, "output"))
    |> rms_norm(spec.audio_rms_norm_epsilon, name: join(name, "post_norm"))
    |> then(&Axon.add(residual, &1))
  end

  # Depthwise convolution over time, gated by a pointwise projection.
  defp convolution(hidden_state, %Spec{} = spec, opts) do
    name = opts[:name]
    residual = hidden_state
    hidden_size = spec.audio_hidden_size

    normed =
      rms_norm(hidden_state, spec.audio_rms_norm_epsilon, name: join(name, "pre_layer_norm"))

    # linear_start is twice the width: half carries signal, half gates it
    normed
    |> dense_maybe_clipped(2 * hidden_size, spec, name: join(name, "linear_start"))
    |> Axon.nx(fn state ->
      half = div(Nx.axis_size(state, 2), 2)
      signal = Nx.slice_along_axis(state, 0, half, axis: 2)
      gate = Nx.slice_along_axis(state, half, half, axis: 2)
      Nx.multiply(signal, Nx.sigmoid(gate))
    end)
    # causal padding: the encoder never looks right of the current frame
    |> Axon.conv(hidden_size,
      kernel_size: spec.audio_conv_kernel_size,
      padding: [{spec.audio_conv_kernel_size - 1, 0}],
      feature_group_size: hidden_size,
      use_bias: false,
      name: join(name, "depthwise_conv1d")
    )
    |> rms_norm(spec.audio_rms_norm_epsilon, name: join(name, "conv_norm"))
    |> Axon.activation(spec.audio_activation)
    |> dense_maybe_clipped(hidden_size, spec, name: join(name, "linear_end"))
    |> then(&Axon.add(residual, &1))
  end

  # The checkpoint stores input_min/input_max and output_min/output_max beside
  # each clipped linear, so the bounds are learned per layer rather than the
  # fixed logit cap.
  defp dense_maybe_clipped(input, units, %Spec{audio_use_clipped_linears: true}, opts) do
    name = opts[:name]

    Axon.layer(
      &clipped_dense/7,
      [
        input,
        Axon.param("kernel", fn shape -> {elem(shape, tuple_size(shape) - 1), units} end),
        # Default to a wide range so an untrained layer passes signal through;
        # loaded checkpoints replace these with their learned bounds.
        Axon.param("input_min", {},
          initializer: fn _shape, type, _key ->
            Nx.tensor(-1.0e4, type: type)
          end
        ),
        Axon.param("input_max", {},
          initializer: fn _shape, type, _key ->
            Nx.tensor(1.0e4, type: type)
          end
        ),
        Axon.param("output_min", {},
          initializer: fn _shape, type, _key ->
            Nx.tensor(-1.0e4, type: type)
          end
        ),
        Axon.param("output_max", {},
          initializer: fn _shape, type, _key ->
            Nx.tensor(1.0e4, type: type)
          end
        )
      ],
      name: name,
      op_name: :gemma4_audio_clipped_dense
    )
  end

  defp dense_maybe_clipped(input, units, _spec, opts) do
    Axon.dense(input, units, use_bias: false, name: opts[:name])
  end

  defnp clipped_dense(input, kernel, input_min, input_max, output_min, output_max, _opts \\ []) do
    # bounds are stored as {1} tensors; clip wants scalars
    input
    |> Nx.clip(input_min, input_max)
    |> Nx.dot([Nx.rank(input) - 1], [], kernel, [0], [])
    |> Nx.clip(output_min, output_max)
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

  defp join(nil, suffix), do: suffix
  defp join(prefix, suffix), do: "#{prefix}.#{suffix}"

  @doc """
  Shifts a relative score matrix so entry `{i, j}` scores the key `j - i`
  frames from the query, which is what a relative position term means.
  """
  defn relative_shift(scores) do
    {batch, heads, queries, keys} = Nx.shape(scores)

    scores
    |> Nx.pad(0.0, [{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {1, 0, 0}])
    |> Nx.reshape({batch, heads, keys + 1, queries})
    |> Nx.slice_along_axis(1, keys, axis: 2)
    |> Nx.reshape({batch, heads, queries, keys})
  end

  @doc """
  Mask for chunked local attention.

  A query attends to its own chunk, `context_left` frames before the chunk
  starts, and `context_right` frames after it ends. Everything else is
  invalid. Returns a `{queries, keys}` boolean tensor.
  """
  defn chunk_mask(opts \\ []) do
    opts =
      keyword!(opts, [:length, :chunk_size, :context_left, :context_right])

    length = opts[:length]
    chunk_size = opts[:chunk_size]

    queries = Nx.iota({length, 1})
    keys = Nx.iota({1, length})

    chunk_start = Nx.quotient(queries, chunk_size) * chunk_size
    chunk_end = chunk_start + chunk_size - 1

    keys >= chunk_start - opts[:context_left] and keys <= chunk_end + opts[:context_right]
  end

  defnp chunked_attention(query, key, value, relative_key, per_dim_scale, opts \\ []) do
    opts =
      keyword!(opts, [
        :chunk_size,
        :context_left,
        :context_right,
        :logit_cap,
        :invalid_value,
        :head_size,
        mode: :inference
      ])

    # Nx.dot requires batch axes to be successive from 0, so move heads next to
    # batch: {batch, seq, heads, head_size} -> {batch, heads, seq, head_size}
    # per_dim_scale weights each query dimension before the dot product, so
    # softplus keeps the learned scale positive as the reference does.
    scale = Nx.log(1.0 + Nx.exp(per_dim_scale)) / Nx.sqrt(opts[:head_size])
    query = query * scale

    query = Nx.transpose(query, axes: [0, 2, 1, 3])
    key = Nx.transpose(key, axes: [0, 2, 1, 3])
    value = Nx.transpose(value, axes: [0, 2, 1, 3])
    relative_key = Nx.transpose(relative_key, axes: [0, 2, 1, 3])

    length = Nx.axis_size(query, 2)

    # content term plus a relative position term, the latter shifted so that
    # position j scores against the key that sits j - i frames away
    content = Nx.dot(query, [3], [0, 1], key, [3], [0, 1])
    relative = relative_shift(Nx.dot(query, [3], [0, 1], relative_key, [3], [0, 1]))

    weights = content + relative

    # tanh soft cap keeps logits inside the trained range
    cap = opts[:logit_cap]
    weights = Nx.tanh(weights / cap) * cap

    mask =
      chunk_mask(
        length: length,
        chunk_size: opts[:chunk_size],
        context_left: opts[:context_left],
        context_right: opts[:context_right]
      )
      |> Nx.new_axis(0)
      |> Nx.new_axis(0)
      # select takes its output shape from the predicate, so the mask has to
      # cover every batch and head rather than relying on broadcast
      |> Nx.broadcast(Nx.shape(weights))

    # opts values are compile time constants, so the fill broadcasts as a
    # scalar without building a tensor inside defn
    weights =
      Nx.select(mask, weights, opts[:invalid_value])
      |> Axon.Activations.softmax(axis: -1)

    # {batch, heads, queries, head_size} -> {batch, queries, heads, head_size}
    Nx.dot(weights, [3], [0, 1], value, [2], [0, 1])
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end
end
