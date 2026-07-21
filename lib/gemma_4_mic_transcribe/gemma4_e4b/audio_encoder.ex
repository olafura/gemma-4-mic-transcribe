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
    # the checkpoint has no norm between the last block and output_proj
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

    # The subsampling stack normalises with a mean-centred LayerNorm over the
    # channel axis and activates with ReLU, unlike the conformer blocks above
    # it, which use RMS norm and the configured activation.
    features
    |> Axon.nx(&Nx.new_axis(&1, -1))
    |> Axon.conv(first_channels,
      kernel_size: {3, 3},
      strides: [2, 2],
      padding: [{1, 1}, {1, 1}],
      use_bias: false,
      name: join(name, "layer0.conv")
    )
    |> layer_norm(spec.audio_rms_norm_epsilon, name: join(name, "layer0.norm"))
    |> Axon.activation(:relu)
    |> Axon.conv(second_channels,
      kernel_size: {3, 3},
      strides: [2, 2],
      padding: [{1, 1}, {1, 1}],
      use_bias: false,
      name: join(name, "layer1.conv")
    )
    |> layer_norm(spec.audio_rms_norm_epsilon, name: join(name, "layer1.norm"))
    |> Axon.activation(:relu)
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

    # relative_k_proj projects a fixed sinusoidal table of relative positions,
    # not the hidden state, so it is a parameter of the attention layer rather
    # than a projection of its input. It is a plain weight in the checkpoint,
    # with none of the clip bounds the other audio linears carry.
    Axon.layer(
      &chunked_attention/6,
      [
        query,
        key,
        value,
        Axon.param("per_dim_scale", {head_size}),
        Axon.param("relative_key", {spec.audio_hidden_size, spec.audio_hidden_size})
      ],
      name: join(name, "chunked"),
      op_name: :gemma4_audio_attention,
      chunk_size: spec.audio_attention_chunk_size,
      context_left: spec.audio_attention_context_left,
      context_right: spec.audio_attention_context_right,
      logit_cap: spec.audio_attention_logit_cap,
      invalid_value: spec.audio_attention_invalid_logits_value,
      head_size: head_size,
      hidden_size: spec.audio_hidden_size,
      num_heads: spec.audio_num_attention_heads,
      # defn cannot call :math, so the scalar constants are folded here: the
      # reference runs the softmax in base 2, folding 1/log(2) into the query
      # scale and log(1 + e)/log(2) into the key scale
      query_scale: :math.pow(head_size, -0.5) / :math.log(2),
      key_scale: :math.log(1 + :math.exp(1)) / :math.log(2),
      timescale_increment:
        :math.log(10_000.0) / max(div(spec.audio_hidden_size, 2) - 1, 1)
    )
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

  # Mean-centred norm over the channel axis, scale only. The subsampling stack
  # uses this where the conformer blocks use RMS norm.
  defp layer_norm(input, epsilon, opts) do
    Axon.layer(
      &layer_norm_impl/3,
      [
        input,
        Axon.param("weight", fn shape -> {elem(shape, tuple_size(shape) - 1)} end,
          initializer: Axon.Initializers.ones()
        )
      ],
      name: opts[:name],
      epsilon: epsilon,
      op_name: :gemma4_audio_layer_norm
    )
  end

  defnp layer_norm_impl(input, weight, opts \\ []) do
    opts = keyword!(opts, [:epsilon, mode: :inference])

    input = Nx.as_type(input, :f32)
    centered = input - Nx.mean(input, axes: [-1], keep_axes: true)
    variance = Nx.mean(centered ** 2, axes: [-1], keep_axes: true)

    centered * Nx.rsqrt(variance + opts[:epsilon]) * weight
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
  Shifts a blocked relative score matrix so that entry `{j, o}` scores the key
  sitting at context offset `o` from query `j`.

  Takes `{batch, heads, blocks, chunk_size, positions}` and returns
  `{batch, heads, blocks, chunk_size, context_size}`.
  """
  defn relative_shift(scores, opts \\ []) do
    opts = keyword!(opts, [:context_size])
    context = opts[:context_size]

    {batch, heads, blocks, chunk, positions} = Nx.shape(scores)

    scores
    |> Nx.pad(0.0, [{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, context + 1 - positions, 0}])
    |> Nx.reshape({batch, heads, blocks, chunk * (context + 1)})
    |> Nx.slice_along_axis(0, chunk * context, axis: 3)
    |> Nx.reshape({batch, heads, blocks, chunk, context})
  end

  @doc """
  Mask for chunked local attention in blocked form.

  Each query carries its own sliding window rather than whole-chunk
  visibility. The reference mask allows `0 <= dist < max_past` looking back
  and `0 < -dist < max_future` looking forward (`dist` is query minus key),
  so `max_past` counts the query itself: a query sees `max_past - 1` earlier
  frames, never `max_past`. Keys must also fall inside the real sequence.
  Returns a `{blocks, chunk_size, context_size}` boolean tensor.
  """
  defn chunk_mask(opts \\ []) do
    opts =
      keyword!(opts, [:length, :blocks, :chunk_size, :max_past, :max_future])

    context = opts[:chunk_size] + opts[:max_past] + opts[:max_future]

    block = Nx.iota({opts[:blocks], 1, 1})
    query = Nx.iota({1, opts[:chunk_size], 1})
    offset = Nx.iota({1, 1, context})

    # the key a context offset refers to, in sequence coordinates
    key = block * opts[:chunk_size] + offset - opts[:max_past]

    dist = query + opts[:max_past] - offset

    in_window =
      (dist >= 0 and dist < opts[:max_past]) or (dist < 0 and -dist < opts[:max_future])

    in_window and key >= 0 and key < opts[:length]
  end

  defnp chunked_attention(query, key, value, per_dim_scale, relative_key, opts \\ []) do
    opts =
      keyword!(opts, [
        :chunk_size,
        :context_left,
        :context_right,
        :logit_cap,
        :invalid_value,
        :head_size,
        :hidden_size,
        :num_heads,
        :query_scale,
        :key_scale,
        :timescale_increment,
        mode: :inference
      ])

    chunk = opts[:chunk_size]
    head_size = opts[:head_size]
    heads = opts[:num_heads]

    # context_left counts the query frame itself, so the past horizon is one
    # frame shorter than the configured context
    max_past = opts[:context_left] - 1
    max_future = opts[:context_right]
    context = chunk + max_past + max_future

    {batch, length, _, _} = Nx.shape(query)
    blocks = div(length + chunk - 1, chunk)
    padded = blocks * chunk

    # per_dim_scale weights each query dimension, through softplus to keep it
    # positive
    query = query * opts[:query_scale] * Nx.log1p(Nx.exp(per_dim_scale))
    key = key * opts[:key_scale]

    # queries split into non-overlapping blocks; keys and values gather an
    # overlapping context window per block
    query =
      query
      |> Nx.pad(0.0, [{0, 0, 0}, {0, padded - length, 0}, {0, 0, 0}, {0, 0, 0}])
      |> Nx.reshape({batch, blocks, chunk, heads, head_size})
      |> Nx.transpose(axes: [0, 3, 1, 2, 4])

    key =
      key
      |> context_windows(blocks: blocks, chunk: chunk, past: max_past, future: max_future)
      |> Nx.transpose(axes: [0, 3, 1, 4, 2])

    value =
      value
      |> context_windows(blocks: blocks, chunk: chunk, past: max_past, future: max_future)
      |> Nx.transpose(axes: [0, 3, 1, 2, 4])

    content = Nx.dot(query, [4], [0, 1, 2], key, [3], [0, 1, 2])

    # The relative term projects a sinusoidal position table, one row per
    # distinct relative offset, then shifts it into per-query alignment.
    positions = div(context, 2) + 1

    relative =
      position_table(
        context: context,
        hidden_size: opts[:hidden_size],
        increment: opts[:timescale_increment]
      )
      |> Nx.dot(relative_key)
      |> Nx.reshape({positions, heads, head_size})
      |> Nx.transpose(axes: [1, 2, 0])

    relative =
      query
      |> Nx.reshape({batch, heads, blocks * chunk, head_size})
      # Nx.dot needs batch axes to start at 0, so heads leads going in and the
      # result is transposed back afterwards
      |> Nx.transpose(axes: [1, 0, 2, 3])
      |> Nx.dot([3], [0], relative, [1], [0])
      |> Nx.transpose(axes: [1, 0, 2, 3])
      |> Nx.reshape({batch, heads, blocks, chunk, positions})
      |> relative_shift(context_size: context)

    # tanh soft cap keeps logits inside the trained range
    cap = opts[:logit_cap]
    weights = Nx.tanh((content + relative) / cap) * cap

    mask =
      chunk_mask(
        length: length,
        blocks: blocks,
        chunk_size: chunk,
        max_past: max_past,
        max_future: max_future
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

    weights
    |> Nx.dot([4], [0, 1, 2], value, [3], [0, 1, 2])
    |> Nx.transpose(axes: [0, 2, 3, 1, 4])
    |> Nx.reshape({batch, padded, heads * head_size})
    |> Nx.slice_along_axis(0, length, axis: 1)
  end

  # Gathers, for each block, the window of frames its queries may attend to:
  # max_past frames before the block starts through max_future frames after it
  # ends.
  defnp context_windows(state, opts \\ []) do
    opts = keyword!(opts, [:blocks, :chunk, :past, :future])
    chunk = opts[:chunk]
    context = chunk + opts[:past] + opts[:future]

    indices = Nx.iota({opts[:blocks], 1}) * chunk + Nx.iota({1, context})

    state
    |> Nx.pad(0.0, [{0, 0, 0}, {opts[:past], opts[:future] + chunk - 1, 0}, {0, 0, 0}, {0, 0, 0}])
    |> Nx.take(indices, axis: 1)
  end

  # Sinusoidal table over the distinct relative offsets a context window spans,
  # laid out as concatenated [sin, cos] halves.
  defnp position_table(opts \\ []) do
    opts = keyword!(opts, [:context, :hidden_size, :increment])

    timescales = div(opts[:hidden_size], 2)

    inverse = Nx.exp(Nx.iota({1, timescales}, type: :f32) * -opts[:increment])

    # offsets run from the furthest past frame down to the current one
    positions = div(opts[:context], 2) - Nx.iota({div(opts[:context], 2) + 1, 1}, type: :f32)

    scaled = positions * inverse

    Nx.concatenate([Nx.sin(scaled), Nx.cos(scaled)], axis: -1)
  end
end
