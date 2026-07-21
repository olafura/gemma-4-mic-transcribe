defmodule Gemma4MicTranscribe.Gemma4E4B.Decoder do
  @moduledoc false

  # Gemma 4 E4B decoder.
  #
  # Differs from the 12B Unified decoder in three ways that change the graph
  # rather than just its sizes:
  #
  #   * every block reads a per-layer input embedding alongside the hidden
  #     state, looked up from its own embedding table
  #   * the last num_kv_shared_layers blocks reuse the key/value state of the
  #     last block that computed its own, so those blocks project queries only
  #   * 8 query heads share 2 key/value heads

  import Nx.Defn

  alias Bumblebee.Layers
  alias Gemma4MicTranscribe.Gemma4E4B.Spec

  @doc """
  Runs the decoder stack over `hidden_state`.

  `per_layer_inputs` is `{batch, sequence, num_blocks, hidden_size_per_layer_input}`,
  one embedding per block per token.
  """
  def decode(
        hidden_state,
        per_layer_inputs,
        position_ids,
        attention_mask,
        cache,
        spec,
        opts \\ []
      ) do
    name = Keyword.get(opts, :name, "decoder")

    {attention_mask, cache} = Layers.Decoder.cached_attention_mask(attention_mask, cache)
    offset = Layers.Decoder.get_cache_offset(cache)

    # Sliding and full attention blocks use different head sizes, so a shared
    # block has to reuse the last computing block of its own type; taking the
    # last computing block regardless would hand a 512 wide key to a layer
    # whose rotary is built for 256.
    state = %{
      hidden_state: hidden_state,
      cache: cache,
      shared: %{}
    }

    outputs =
      spec.layer_types
      |> Enum.with_index()
      |> Enum.reduce(state, fn {layer_type, index}, state ->
        block(state, per_layer_inputs, position_ids, attention_mask, offset, layer_type, index,
          spec: spec,
          name: join(name, "blocks.#{index}")
        )
      end)

    cache = Layers.Decoder.update_cache_offset(outputs.cache, outputs.hidden_state)

    %{hidden_state: outputs.hidden_state, cache: cache}
  end

  defp block(
         state,
         per_layer_inputs,
         position_ids,
         attention_mask,
         offset,
         layer_type,
         index,
         opts
       ) do
    spec = opts[:spec]
    name = opts[:name]
    shared? = Spec.kv_shared_layer?(spec, index)

    block_cache = Layers.Decoder.get_block_cache(state.cache, index)
    shortcut = state.hidden_state

    normed =
      rms_norm(state.hidden_state, spec.layer_norm_epsilon,
        name: join(name, "self_attention_norm")
      )

    {attention_output, block_cache, key, value} =
      self_attention(
        normed,
        position_ids,
        attention_mask,
        block_cache,
        offset,
        layer_type,
        shared?,
        state.shared[layer_type],
        spec,
        name: join(name, "self_attention")
      )

    hidden_state =
      attention_output
      |> rms_norm(spec.layer_norm_epsilon, name: join(name, "post_attention_norm"))
      |> then(&Axon.add(shortcut, &1))

    shortcut = hidden_state

    hidden_state =
      hidden_state
      |> rms_norm(spec.layer_norm_epsilon, name: join(name, "pre_ffn_norm"))
      |> gated_ffn(spec, name: join(name, "ffn"))
      |> rms_norm(spec.layer_norm_epsilon, name: join(name, "post_ffn_norm"))
      |> then(&Axon.add(shortcut, &1))

    # Each block mixes in its own slice of the per-layer embeddings.
    hidden_state =
      merge_per_layer_input(hidden_state, per_layer_inputs, index, spec,
        name: join(name, "per_layer")
      )
      # every block carries a learned scalar on its output
      |> layer_scalar(name: join(name, "layer_scalar"))

    %{
      hidden_state: hidden_state,
      cache: Layers.Decoder.put_block_cache(state.cache, index, block_cache),
      shared:
        if(shared?,
          do: state.shared,
          else: Map.put(state.shared, layer_type, {key, value})
        )
    }
  end

  defp self_attention(
         hidden_state,
         position_ids,
         attention_mask,
         block_cache,
         offset,
         layer_type,
         shared?,
         shared_kv,
         spec,
         opts
       ) do
    name = opts[:name]

    {self_attention_cache, cross_attention_cache} =
      Layers.Decoder.get_attention_caches(block_cache)

    full? = layer_type == :full_attention
    head_size = if full?, do: spec.global_attention_head_size, else: spec.attention_head_size
    num_kv_heads = spec.num_key_value_heads

    query =
      hidden_state
      |> Axon.dense(spec.num_attention_heads * head_size,
        use_bias: false,
        name: join(name, "query")
      )
      |> Layers.split_heads(spec.num_attention_heads)
      |> rms_norm(spec.layer_norm_epsilon, name: join(name, "query_norm"))

    {key, value} =
      if shared? and shared_kv != nil do
        # The checkpoint keeps k_proj and v_proj for these blocks, so sharing
        # is of the cached state rather than of the weights: the block reuses
        # what the last computing block of the same type put in the cache.
        shared_kv
      else
        key =
          hidden_state
          |> Axon.dense(num_kv_heads * head_size, use_bias: false, name: join(name, "key"))
          |> Layers.split_heads(num_kv_heads)
          |> rms_norm(spec.layer_norm_epsilon, name: join(name, "key_norm"))

        value =
          hidden_state
          |> Axon.dense(num_kv_heads * head_size, use_bias: false, name: join(name, "value"))
          |> Layers.split_heads(num_kv_heads)

        {key, value}
      end

    {rotary_query, rotary_key} =
      rotary_embedding(query, key, position_ids, attention_mask, layer_type, head_size, spec)

    groups = div(spec.num_attention_heads, num_kv_heads)
    repeated_key = repeat_kv(rotary_key, groups)
    repeated_value = repeat_kv(value, groups)

    {cached_key, cached_value, self_attention_cache} =
      Layers.Decoder.cached_attention_key_values(
        repeated_key,
        repeated_value,
        self_attention_cache,
        offset
      )

    window_size =
      case layer_type do
        :sliding_attention -> {spec.attention_window_size, 0}
        :full_attention -> nil
      end

    {attention_output, _weights} =
      Layers.attention(
        rotary_query,
        cached_key,
        cached_value,
        attention_mask,
        Layers.none(),
        Layers.none(),
        offset,
        causal: true,
        window_size: window_size,
        scale: 1.0
      )

    attention_output =
      attention_output
      |> Layers.flatten_trailing()
      |> Axon.dense(spec.hidden_size, use_bias: false, name: join(name, "output"))

    block_cache =
      Layers.Decoder.put_attention_caches(
        block_cache,
        self_attention_cache,
        cross_attention_cache
      )

    # Pass the pre-rotary key and the value on, so a shared block reuses the
    # same projections rather than positions already baked in.
    {attention_output, block_cache, key, value}
  end

  # The checkpoint carries per_layer_input_gate (hidden -> per_layer_size),
  # per_layer_projection (per_layer_size -> hidden) and
  # post_per_layer_input_norm, so the block gates its own state against the
  # per-layer embedding rather than simply adding a projection of it.
  defp merge_per_layer_input(hidden_state, per_layer_inputs, index, spec, opts) do
    name = opts[:name]

    slice = Axon.nx(per_layer_inputs, fn inputs -> inputs[[.., .., index, ..]] end)

    gate =
      hidden_state
      |> Axon.dense(spec.hidden_size_per_layer_input,
        use_bias: false,
        name: join(name, "input_gate")
      )
      |> Layers.activation(spec.activation)

    gated =
      gate
      |> Axon.multiply(slice)
      |> Axon.dense(spec.hidden_size, use_bias: false, name: join(name, "projection"))
      |> rms_norm(spec.layer_norm_epsilon, name: join(name, "post_norm"))

    Axon.add(hidden_state, gated)
  end

  defp layer_scalar(input, opts) do
    Axon.layer(
      fn input, scalar, _opts -> Nx.multiply(input, scalar) end,
      [input, Axon.param("layer_scalar", {1}, initializer: Axon.Initializers.ones())],
      name: opts[:name],
      op_name: :gemma4_layer_scalar
    )
  end

  defp gated_ffn(hidden_state, spec, opts) do
    name = opts[:name]

    intermediate =
      if spec.use_double_wide_mlp, do: 2 * spec.intermediate_size, else: spec.intermediate_size

    gate =
      hidden_state
      |> Axon.dense(intermediate, use_bias: false, name: join(name, "gate"))
      |> Layers.activation(spec.activation)

    up = Axon.dense(hidden_state, intermediate, use_bias: false, name: join(name, "intermediate"))

    gate
    |> Axon.multiply(up)
    |> Axon.dense(spec.hidden_size, use_bias: false, name: join(name, "output"))
  end

  defp rotary_embedding(
         query,
         key,
         position_ids,
         attention_mask,
         :sliding_attention,
         head_size,
         spec
       ) do
    Layers.rotary_embedding(query, key, position_ids, attention_mask, head_size,
      base: spec.rotary_embedding_base_local,
      max_positions: spec.max_positions
    )
  end

  defp rotary_embedding(
         query,
         key,
         position_ids,
         _attention_mask,
         :full_attention,
         head_size,
         spec
       ) do
    Axon.layer(
      &proportional_rotary_embedding/4,
      [query, key, position_ids],
      head_size: head_size,
      base: spec.rotary_embedding_base,
      rotated_angles: trunc(spec.full_attention_rotary_percentage * head_size / 2)
    )
    |> Layers.unwrap_tuple(2)
  end

  defp repeat_kv(state, 1), do: state

  defp repeat_kv(state, groups) do
    Axon.nx(state, fn state ->
      {batch, sequence, heads, head_size} = Nx.shape(state)

      state
      |> Nx.new_axis(3)
      |> Nx.broadcast({batch, sequence, heads, groups, head_size})
      |> Nx.reshape({batch, sequence, heads * groups, head_size})
    end)
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

  defnp proportional_rotary_embedding(query, key, position_ids, opts \\ []) do
    opts = keyword!(opts, [:head_size, :base, :rotated_angles, mode: :inference])

    head_size = opts[:head_size]
    half_size = div(head_size, 2)

    angle_positions = Nx.iota({half_size})

    inv_freq =
      Nx.select(
        angle_positions < opts[:rotated_angles],
        Nx.pow(opts[:base], -(angle_positions * 2 / head_size)),
        0.0
      )

    freqs = Nx.multiply(Nx.new_axis(position_ids, -1), inv_freq)
    embeddings = Nx.concatenate([freqs, freqs], axis: -1)
    cos = embeddings |> Nx.cos() |> Nx.new_axis(2)
    sin = embeddings |> Nx.sin() |> Nx.new_axis(2)

    {apply_rotary(query, cos, sin), apply_rotary(key, cos, sin)}
  end

  defnp apply_rotary(x, cos, sin) do
    half = div(Nx.axis_size(x, -1), 2)
    left = x[[.., .., .., 0..(half - 1)//1]]
    right = x[[.., .., .., half..-1//1]]
    rotated = Nx.concatenate([-right, left], axis: -1)

    x * cos + rotated * sin
  end
end
