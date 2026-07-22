defmodule Gemma4MicTranscribe.Gemma4Unified.Model do
  @moduledoc false

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.Text.Generation

  import Nx.Defn
  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers
  alias Gemma4MicTranscribe.Gemma4Unified.CompressedTensors
  alias Gemma4MicTranscribe.Gemma4Unified.Q4Gemv

  defstruct architecture: :for_conditional_generation,
            vocab_size: 262_144,
            max_positions: 262_144,
            hidden_size: 3840,
            intermediate_size: 15_360,
            enable_moe_block: false,
            moe_intermediate_size: nil,
            num_experts: 0,
            top_k_experts: 0,
            num_blocks: 48,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            num_global_key_value_heads: 1,
            attention_head_size: 256,
            global_attention_head_size: 512,
            activation: :gelu_approx_tanh,
            attention_k_eq_v: true,
            rotary_embedding_base: 1_000_000,
            rotary_embedding_base_local: 10_000,
            full_attention_rotary_percentage: 0.25,
            use_attention_bias: false,
            layer_norm_epsilon: 1.0e-6,
            initializer_scale: 0.02,
            attention_window_size: 1024,
            layer_types: nil,
            tie_word_embeddings: true,
            final_logit_softcapping: 30.0,
            pad_token_id: 0,
            bos_token_id: 2,
            eos_token_id: [1, 106],
            boa_token_id: 256_000,
            audio_token_id: 258_881,
            eoa_token_id: 258_883,
            audio_embed_dim: 640,
            audio_rms_norm_epsilon: 1.0e-6,
            quantization_config: nil,
            logits_last_only: false,
            cache_type: {:f, 32},
            # false dequantizes int4 weights to bf16 at load: 4x the resident
            # memory, but prefill uses rocBLAS instead of the hand int4 GEMM.
            packed_linear: true,
            # true also loads dequantized bf16 kernels, so prefill can use rocBLAS
            # matrix cores while decode still reads packed int4
            hybrid_linear: false

  @impl true
  def architectures, do: [:for_conditional_generation]

  @impl true
  def config(spec, opts) do
    struct!(spec, opts)
    |> normalize_layer_types()
  end

  @impl true
  def input_template(spec) do
    %{
      "input_ids" => Nx.template({1, 1}, :s64),
      "attention_mask" => Nx.template({1, 1}, :s64),
      "position_ids" => Nx.template({1, 1}, :s64),
      "input_features" => Nx.template({1, 1, spec.audio_embed_dim}, {:f, 32}),
      "input_features_mask" => Nx.template({1, 1}, :s64)
    }
  end

  @impl true
  def init_cache(spec, batch_size, max_length, _inputs) do
    blocks =
      spec
      |> layer_types()
      |> Enum.map(fn layer_type ->
        head_size =
          if layer_type == :full_attention,
            do: spec.global_attention_head_size,
            else: spec.attention_head_size

        %{
          self_attention:
            attention_cache(
              batch_size,
              max_length,
              spec.num_attention_heads,
              head_size,
              spec.cache_type
            ),
          cross_attention: attention_cache(batch_size, 1, 1, 1, spec.cache_type)
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
  def traverse_cache(_spec, cache, fun) do
    Layers.Decoder.traverse_cache(cache, fun)
  end

  @impl true
  def model(%__MODULE__{architecture: :for_conditional_generation, enable_moe_block: true}) do
    raise ArgumentError,
          "Gemma 4 MoE inference is not implemented yet; use Gemma4MicTranscribe.Gemma4.Experts.list/2 to inspect its experts"
  end

  def model(%__MODULE__{architecture: :for_conditional_generation} = spec) do
    inputs = inputs(spec)

    outputs = core(inputs, spec)

    hidden_state =
      if spec.logits_last_only do
        Axon.nx(outputs.hidden_state, fn hidden_state ->
          last = Nx.axis_size(hidden_state, 1) - 1
          Nx.slice_along_axis(hidden_state, last, 1, axis: 1)
        end)
      else
        outputs.hidden_state
      end

    logits = language_modeling_head(hidden_state, spec, name: "language_modeling_head")

    logits =
      if spec.final_logit_softcapping do
        Axon.nx(logits, fn logits ->
          logits
          |> Nx.divide(spec.final_logit_softcapping)
          |> Nx.tanh()
          |> Nx.multiply(spec.final_logit_softcapping)
        end)
      else
        logits
      end

    Layers.output(%{logits: logits, cache: outputs.cache})
  end

  @doc false
  def decoder_block_model(%__MODULE__{} = spec, layer_index)
      when is_integer(layer_index) and layer_index >= 0 and layer_index < spec.num_blocks do
    decoder_block_chain_model(spec, [layer_index])
  end

  def decoder_block_model(%__MODULE__{} = spec, layer_index) do
    raise ArgumentError,
          "expected a Gemma 4 layer index in 0..#{spec.num_blocks - 1}, got: #{inspect(layer_index)}"
  end

  @doc false
  def decoder_block_chain_model(%__MODULE__{} = spec, [first | _rest] = layer_indices) do
    last = List.last(layer_indices)

    unless Enum.all?(layer_indices, &(&1 in 0..(spec.num_blocks - 1))) and
             last >= first and layer_indices == Enum.to_list(first..last//1) do
      raise ArgumentError, "decoder block chains must contain contiguous ascending layer indices"
    end

    hidden_state =
      Axon.input("hidden_state", shape: {nil, nil, spec.hidden_size})

    position_ids = Axon.input("position_ids", shape: {nil, nil})
    attention_mask = Axon.input("attention_mask", shape: {nil, nil})

    apply_decoder_blocks(hidden_state, position_ids, attention_mask, spec, layer_indices)
  end

  @doc false
  def decoder_prefix_model(%__MODULE__{} = spec, last_layer)
      when is_integer(last_layer) and last_layer >= 0 and last_layer < spec.num_blocks do
    inputs = inputs(spec)
    input_ids = inputs["input_ids"]

    audio_mask = Axon.nx(input_ids, &Nx.equal(&1, spec.audio_token_id))

    llm_input_ids =
      Axon.nx(input_ids, fn input_ids ->
        Nx.select(Nx.equal(input_ids, spec.audio_token_id), spec.pad_token_id, input_ids)
      end)

    hidden_state =
      llm_input_ids
      |> embedder(spec, name: "embedder")
      |> replace_audio_embeddings(
        audio_embedder(inputs["input_features"], spec, name: "audio_embedder"),
        audio_mask
      )

    apply_decoder_blocks(
      hidden_state,
      inputs["position_ids"],
      inputs["attention_mask"],
      spec,
      Enum.to_list(0..last_layer)
    )
  end

  def decoder_prefix_model(%__MODULE__{} = spec, last_layer) do
    raise ArgumentError,
          "expected a Gemma 4 prefix endpoint in 0..#{spec.num_blocks - 1}, got: #{inspect(last_layer)}"
  end

  @doc false
  def cached_decoder_prefix_model(%__MODULE__{} = spec, last_layer)
      when is_integer(last_layer) and last_layer >= 0 and last_layer < spec.num_blocks do
    inputs = inputs(spec)
    hidden_state = input_hidden_state(inputs, spec)

    {attention_mask, cache} =
      Layers.Decoder.cached_attention_mask(inputs["attention_mask"], inputs["cache"])

    offset = Layers.Decoder.get_cache_offset(cache)

    outputs =
      apply_cached_decoder_blocks(
        hidden_state,
        inputs["position_ids"],
        attention_mask,
        cache,
        offset,
        spec,
        Enum.to_list(0..last_layer)
      )

    Axon.container(%{
      hidden_state: outputs.hidden_state,
      attention_mask: attention_mask,
      cache: outputs.cache
    })
  end

  @doc false
  def cached_decoder_tail_model(%__MODULE__{} = spec, layer_indices) do
    hidden_state = Axon.input("hidden_state", shape: {nil, nil, spec.hidden_size})
    position_ids = Axon.input("position_ids", shape: {nil, nil})
    attention_mask = Axon.input("attention_mask", shape: {nil, nil})
    cache = Axon.input("cache", optional: true)
    offset = Layers.Decoder.get_cache_offset(cache)

    outputs =
      apply_cached_decoder_blocks(
        hidden_state,
        position_ids,
        attention_mask,
        cache,
        offset,
        spec,
        layer_indices
      )

    cache = Layers.Decoder.update_cache_offset(outputs.cache, outputs.hidden_state)
    logits = output_logits(outputs.hidden_state, spec)
    Axon.container(%{logits: logits, cache: cache})
  end

  @doc false
  def cached_decoder_bypass_model(%__MODULE__{} = spec, bypass_layers) do
    bypass_layers = MapSet.new(bypass_layers)
    inputs = inputs(spec)
    hidden_state = input_hidden_state(inputs, spec)

    {attention_mask, cache} =
      Layers.Decoder.cached_attention_mask(inputs["attention_mask"], inputs["cache"])

    offset = Layers.Decoder.get_cache_offset(cache)

    outputs =
      apply_cached_decoder_blocks(
        hidden_state,
        inputs["position_ids"],
        attention_mask,
        cache,
        offset,
        spec,
        Enum.to_list(0..(spec.num_blocks - 1)),
        bypass_layers
      )

    cache = Layers.Decoder.update_cache_offset(outputs.cache, outputs.hidden_state)
    logits = output_logits(outputs.hidden_state, spec)
    Axon.container(%{logits: logits, cache: cache})
  end

  defp apply_decoder_blocks(hidden_state, position_ids, attention_mask, spec, layer_indices) do
    Enum.reduce(layer_indices, hidden_state, fn layer_index, hidden_state ->
      block_cache =
        Axon.container(%{
          self_attention: Layers.none(),
          cross_attention: Layers.none()
        })

      {hidden_state, _block_cache} =
        decoder_block(
          hidden_state,
          position_ids,
          attention_mask,
          block_cache,
          Layers.none(),
          Enum.fetch!(layer_types(spec), layer_index),
          spec,
          name: "decoder.blocks.#{layer_index}"
        )

      hidden_state
    end)
  end

  defp apply_cached_decoder_blocks(
         hidden_state,
         position_ids,
         attention_mask,
         cache,
         offset,
         spec,
         layer_indices,
         bypass_layers \\ MapSet.new()
       ) do
    Enum.reduce(layer_indices, %{hidden_state: hidden_state, cache: cache}, fn layer_index,
                                                                               state ->
      if MapSet.member?(bypass_layers, layer_index) do
        state
      else
        block_cache = Layers.Decoder.get_block_cache(state.cache, layer_index)

        {hidden_state, block_cache} =
          decoder_block(
            state.hidden_state,
            position_ids,
            attention_mask,
            block_cache,
            offset,
            Enum.fetch!(layer_types(spec), layer_index),
            spec,
            name: "decoder.blocks.#{layer_index}"
          )

        %{
          hidden_state: hidden_state,
          cache: Layers.Decoder.put_block_cache(state.cache, layer_index, block_cache)
        }
      end
    end)
  end

  defp input_hidden_state(inputs, spec) do
    input_ids = inputs["input_ids"]
    audio_mask = Axon.nx(input_ids, &Nx.equal(&1, spec.audio_token_id))

    llm_input_ids =
      Axon.nx(input_ids, fn input_ids ->
        Nx.select(Nx.equal(input_ids, spec.audio_token_id), spec.pad_token_id, input_ids)
      end)

    llm_input_ids
    |> embedder(spec, name: "embedder")
    |> replace_audio_embeddings(
      audio_embedder(inputs["input_features"], spec, name: "audio_embedder"),
      audio_mask
    )
  end

  defp output_logits(hidden_state, spec) do
    logits =
      hidden_state
      |> Axon.nx(
        fn hidden_state ->
          hidden_state
          |> Nx.slice_along_axis(Nx.axis_size(hidden_state, 1) - 1, 1, axis: 1)
          |> Nx.squeeze(axes: [1])
        end,
        name: "decoder_tail.last_hidden_state"
      )
      |> rms_norm(spec.hidden_size,
        name: "output_norm",
        epsilon: spec.layer_norm_epsilon
      )
      |> language_modeling_head(spec, name: "language_modeling_head")

    if spec.final_logit_softcapping do
      Axon.nx(logits, fn logits ->
        logits
        |> Nx.divide(spec.final_logit_softcapping)
        |> Nx.tanh()
        |> Nx.multiply(spec.final_logit_softcapping)
      end)
    else
      logits
    end
  end

  @doc false
  def decoder_tail_model(%__MODULE__{} = spec, layer_indices) do
    spec
    |> decoder_block_chain_model(layer_indices)
    |> output_logits(spec)
  end

  defp inputs(spec) do
    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", shape: {nil, nil}),
      Axon.input("attention_mask", shape: {nil, nil}),
      Axon.input("position_ids", shape: {nil, nil}),
      Axon.input("input_features", shape: {nil, nil, spec.audio_embed_dim}),
      Axon.input("input_features_mask", shape: {nil, nil}),
      Axon.input("cache", optional: true)
    ])
  end

  defp core(inputs, spec) do
    input_ids = inputs["input_ids"]

    audio_mask =
      Axon.nx(input_ids, fn input_ids ->
        Nx.equal(input_ids, spec.audio_token_id)
      end)

    llm_input_ids =
      Axon.nx(input_ids, fn input_ids ->
        Nx.select(Nx.equal(input_ids, spec.audio_token_id), spec.pad_token_id, input_ids)
      end)

    embeddings = embedder(llm_input_ids, spec, name: "embedder")
    audio_embeddings = audio_embedder(inputs["input_features"], spec, name: "audio_embedder")
    hidden_state = replace_audio_embeddings(embeddings, audio_embeddings, audio_mask)

    decoder_outputs =
      decoder(
        hidden_state,
        inputs["position_ids"],
        inputs["attention_mask"],
        inputs["cache"],
        spec,
        name: "decoder"
      )

    hidden_state =
      rms_norm(decoder_outputs.hidden_state, spec.hidden_size,
        name: "output_norm",
        epsilon: spec.layer_norm_epsilon
      )

    %{hidden_state: hidden_state, cache: decoder_outputs.cache}
  end

  defp embedder(input_ids, spec, opts) do
    name = opts[:name]

    Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "token_embedding")
    )
    |> Axon.nx(fn embeddings ->
      scale =
        spec.hidden_size
        |> Nx.tensor(type: Nx.type(embeddings))
        |> Nx.sqrt()

      Nx.multiply(embeddings, scale)
    end)
  end

  defp audio_embedder(input_features, spec, opts) do
    name = opts[:name]

    input_features
    |> rms_norm_no_scale(epsilon: spec.audio_rms_norm_epsilon)
    |> Axon.dense(spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "projection"),
      use_bias: false
    )
  end

  defp replace_audio_embeddings(embeddings, audio_embeddings, audio_mask) do
    Axon.layer(
      fn embeddings, audio_embeddings, audio_mask, _opts ->
        hidden_size = Nx.axis_size(embeddings, 2)
        audio_count = Nx.axis_size(audio_embeddings, 1)

        audio_indices =
          audio_mask
          |> Nx.as_type({:s, 64})
          |> Nx.cumulative_sum(axis: 1)
          |> Nx.subtract(1)
          |> Nx.max(0)
          |> Nx.min(audio_count - 1)
          |> Nx.new_axis(-1)
          |> Nx.broadcast({Nx.axis_size(embeddings, 0), Nx.axis_size(embeddings, 1), hidden_size})

        gathered = Nx.take_along_axis(audio_embeddings, audio_indices, axis: 1)

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

  defp decoder(hidden_state, position_ids, attention_mask, cache, spec, opts) do
    name = opts[:name]
    {attention_mask, cache} = Layers.Decoder.cached_attention_mask(attention_mask, cache)
    offset = Layers.Decoder.get_cache_offset(cache)

    outputs =
      spec
      |> layer_types()
      |> Enum.with_index()
      |> Enum.reduce(%{hidden_state: hidden_state, cache: cache}, fn {layer_type, idx}, state ->
        block_cache = Layers.Decoder.get_block_cache(state.cache, idx)

        {hidden_state, block_cache} =
          decoder_block(
            state.hidden_state,
            position_ids,
            attention_mask,
            block_cache,
            offset,
            layer_type,
            spec,
            name: join(name, "blocks.#{idx}")
          )

        cache = Layers.Decoder.put_block_cache(state.cache, idx, block_cache)

        %{hidden_state: hidden_state, cache: cache}
      end)

    cache = Layers.Decoder.update_cache_offset(outputs.cache, outputs.hidden_state)

    %{outputs | cache: cache}
  end

  defp decoder_block(
         hidden_state,
         position_ids,
         attention_mask,
         block_cache,
         offset,
         layer_type,
         spec,
         opts
       ) do
    name = opts[:name]

    shortcut = hidden_state

    {hidden_state, block_cache} =
      hidden_state
      |> rms_norm(spec.hidden_size,
        name: join(name, "self_attention_norm"),
        epsilon: spec.layer_norm_epsilon
      )
      |> self_attention(position_ids, attention_mask, block_cache, offset, layer_type, spec,
        name: join(name, "self_attention")
      )

    hidden_state =
      hidden_state
      |> rms_norm(spec.hidden_size,
        name: join(name, "post_attention_norm"),
        epsilon: spec.layer_norm_epsilon
      )

    hidden_state = Axon.add(shortcut, hidden_state)
    shortcut = hidden_state

    hidden_state =
      hidden_state
      |> rms_norm(spec.hidden_size,
        name: join(name, "pre_ffn_norm"),
        epsilon: spec.layer_norm_epsilon
      )
      |> gated_ffn(spec.intermediate_size, spec.hidden_size, spec,
        name: join(name, "ffn"),
        activation: spec.activation,
        kernel_initializer: kernel_initializer(spec)
      )
      |> rms_norm(spec.hidden_size,
        name: join(name, "post_ffn_norm"),
        epsilon: spec.layer_norm_epsilon
      )

    hidden_state =
      shortcut
      |> Axon.add(hidden_state)
      |> layer_scalar(name: join(name, "layer_scalar"))

    {hidden_state, block_cache}
  end

  defp self_attention(
         hidden_state,
         position_ids,
         attention_mask,
         block_cache,
         offset,
         layer_type,
         spec,
         opts
       ) do
    {self_attention_cache, cross_attention_cache} =
      Layers.Decoder.get_attention_caches(block_cache)

    name = opts[:name]
    full_attention? = layer_type == :full_attention

    head_size =
      if full_attention?, do: spec.global_attention_head_size, else: spec.attention_head_size

    num_key_value_heads =
      if full_attention? and spec.num_global_key_value_heads do
        spec.num_global_key_value_heads
      else
        spec.num_key_value_heads
      end

    query =
      hidden_state
      |> linear(spec.num_attention_heads * head_size, spec,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "query"),
        use_bias: spec.use_attention_bias
      )
      |> Layers.split_heads(spec.num_attention_heads)
      |> rms_norm(head_size,
        name: join(name, "query_norm"),
        epsilon: spec.layer_norm_epsilon
      )

    key_projection =
      hidden_state
      |> linear(num_key_value_heads * head_size, spec,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "key"),
        use_bias: spec.use_attention_bias
      )
      |> Layers.split_heads(num_key_value_heads)

    key =
      key_projection
      |> rms_norm(head_size,
        name: join(name, "key_norm"),
        epsilon: spec.layer_norm_epsilon
      )

    value_projection =
      if full_attention? and spec.attention_k_eq_v do
        key_projection
      else
        hidden_state
        |> linear(num_key_value_heads * head_size, spec,
          kernel_initializer: kernel_initializer(spec),
          name: join(name, "value"),
          use_bias: spec.use_attention_bias
        )
        |> Layers.split_heads(num_key_value_heads)
      end

    value = rms_norm_no_scale(value_projection, epsilon: spec.layer_norm_epsilon)

    {query, key} =
      rotary_embedding(query, key, position_ids, attention_mask, layer_type, head_size, spec)

    num_key_value_groups = div(spec.num_attention_heads, num_key_value_heads)
    key = repeat_kv(key, num_key_value_groups)
    value = repeat_kv(value, num_key_value_groups)

    {key, value, self_attention_cache} =
      Layers.Decoder.cached_attention_key_values(key, value, self_attention_cache, offset)

    window_size =
      case layer_type do
        :sliding_attention -> {spec.attention_window_size, 0}
        :full_attention -> nil
      end

    {attention_output, _attention_weights} =
      Layers.attention(
        query,
        key,
        value,
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
      |> linear(spec.hidden_size, spec,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "output"),
        use_bias: spec.use_attention_bias
      )

    block_cache =
      Layers.Decoder.put_attention_caches(
        block_cache,
        self_attention_cache,
        cross_attention_cache
      )

    {attention_output, block_cache}
  end

  defp attention_cache(batch_size, sequence_length, num_heads, head_size, cache_type) do
    shape = {batch_size, sequence_length, num_heads, head_size}
    zeros = Nx.broadcast(Nx.tensor(0.0, type: cache_type), shape)
    %{key: zeros, value: zeros}
  end

  # The attention mask must be passed through: it sizes the sinusoidal position
  # table that position_ids index into. With Layers.none() the table is sized by
  # the current call's token count, so a cached decode step (one token at
  # position p) builds a one-row table and Nx.take clamps p to row 0, applying
  # position-0 rotary to every generated token.
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

  defp rms_norm(input, _size, opts) do
    Layers.rms_norm(input,
      name: opts[:name],
      shift: 0.0,
      epsilon: opts[:epsilon],
      initializer: Axon.Initializers.ones(),
      upcast: :all
    )
  end

  defp rms_norm_no_scale(input, opts) do
    Axon.nx(input, fn input ->
      epsilon = opts[:epsilon]

      norm =
        input
        |> Nx.as_type(:f32)
        |> Nx.pow(2)
        |> Nx.mean(axes: [-1], keep_axes: true)
        |> Nx.add(epsilon)
        |> Nx.pow(-0.5)

      input
      |> Nx.as_type(:f32)
      |> Nx.multiply(norm)
      |> Nx.as_type(Nx.type(input))
    end)
  end

  defp layer_scalar(input, opts) do
    Axon.layer(
      fn input, scalar, _opts -> Nx.multiply(input, scalar) end,
      [input, Axon.param("layer_scalar", {1}, initializer: Axon.Initializers.ones())],
      name: opts[:name],
      op_name: :gemma4_layer_scalar
    )
  end

  defp gated_ffn(hidden_state, intermediate_size, output_size, spec, opts) do
    name = opts[:name]

    gate =
      hidden_state
      |> linear(intermediate_size, spec,
        name: join(name, "gate"),
        kernel_initializer: opts[:kernel_initializer],
        use_bias: false
      )
      |> Layers.activation(opts[:activation])

    intermediate =
      linear(hidden_state, intermediate_size, spec,
        name: join(name, "intermediate"),
        kernel_initializer: opts[:kernel_initializer],
        use_bias: false
      )

    Axon.multiply(gate, intermediate)
    |> linear(output_size, spec,
      name: join(name, "output"),
      kernel_initializer: opts[:kernel_initializer],
      use_bias: false
    )
  end

  defp language_modeling_head(hidden_state, spec, opts) do
    name = opts[:name]

    Layers.dense_transposed(hidden_state, spec.vocab_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "output")
    )
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defp quantized?(%__MODULE__{
         quantization_config: %{"quant_method" => "compressed-tensors"},
         packed_linear: true
       }),
       do: true

  defp quantized?(_spec), do: false

  # Linear layers whose weights come from a compressed-tensors checkpoint keep
  # their int4 weights packed instead of being dequantized at load, so decode
  # can read 4-bit traffic through the GEMV kernel.
  defp linear(input, units, spec, opts) do
    if quantized?(spec) and not opts[:use_bias] do
      quantized_dense(input, units, name: opts[:name], hybrid: spec.hybrid_linear)
    else
      Axon.dense(input, units,
        kernel_initializer: opts[:kernel_initializer],
        name: opts[:name],
        use_bias: opts[:use_bias]
      )
    end
  end

  defp quantized_dense(input, units, opts) do
    group_size = CompressedTensors.quant_group_size()

    packed =
      Axon.param(
        "packed",
        fn shape -> {div(elem(shape, tuple_size(shape) - 1), 8), units} end,
        type: {:s, 32},
        initializer: :zeros
      )

    scales =
      Axon.param(
        "scales",
        fn shape -> {div(elem(shape, tuple_size(shape) - 1), group_size), units} end,
        type: {:bf, 16},
        initializer: :zeros
      )

    if opts[:hybrid] do
      kernel =
        Axon.param(
          "kernel",
          fn shape -> {elem(shape, tuple_size(shape) - 1), units} end,
          type: {:bf, 16},
          initializer: :zeros
        )

      Axon.layer(&hybrid_dense_impl/5, [input, packed, scales, kernel],
        name: opts[:name],
        op_name: :gemma4_q4_dense,
        group_size: group_size
      )
    else
      Axon.layer(&quantized_dense_impl/4, [input, packed, scales],
        name: opts[:name],
        op_name: :gemma4_q4_dense,
        group_size: group_size
      )
    end
  end

  # Decode is a GEMV over one token and is memory bound, so it reads packed
  # int4. Prefill is a GEMM over the whole sequence and is compute bound, so it
  # uses the dequantized kernel and rocBLAS matrix cores, which beat the hand
  # int4 GEMM by roughly 4x.
  defp hybrid_dense_impl(input, packed, scales, kernel, opts) do
    shape = Nx.shape(input)

    if single_token?(shape) do
      quantized_dense_impl(input, packed, scales, opts)
    else
      input
      |> Nx.dot(Nx.as_type(kernel, Nx.type(input)))
    end
  end

  # Decode runs one token at a time, which is exactly the GEMV the kernel
  # implements. Prefill multiplies a whole sequence, so it dequantizes and
  # uses a regular GEMM.
  defp quantized_dense_impl(input, packed, scales, opts) do
    group_size = opts[:group_size]
    shape = Nx.shape(input)
    hidden_size = elem(shape, tuple_size(shape) - 1)
    {_words, units} = Nx.shape(packed)
    input_type = Nx.type(input)

    if single_token?(shape) do
      # Types must match the kernel's FFI signature exactly, otherwise the
      # custom call is skipped and this silently falls back to dequantization.
      input
      |> Nx.reshape({hidden_size})
      |> Nx.as_type({:bf, 16})
      |> Q4Gemv.dot(Nx.as_type(packed, {:s, 32}), Nx.as_type(scales, {:bf, 16}),
        group_size: group_size
      )
      |> Nx.reshape(put_elem(shape, tuple_size(shape) - 1, units))
      |> Nx.as_type(input_type)
    else
      seq = div(Nx.size(input), hidden_size)

      input
      |> Nx.reshape({seq, hidden_size})
      |> Nx.as_type({:bf, 16})
      |> Q4Gemv.matmul(Nx.as_type(packed, {:s, 32}), Nx.as_type(scales, {:bf, 16}),
        group_size: group_size
      )
      |> Nx.reshape(put_elem(shape, tuple_size(shape) - 1, units))
      |> Nx.as_type(input_type)
    end
  end

  defp single_token?({1, 1, _hidden}), do: true
  defp single_token?({1, _hidden}), do: true
  defp single_token?(_shape), do: false

  defp normalize_layer_types(%__MODULE__{layer_types: nil} = spec) do
    layer_types =
      Enum.map(0..(spec.num_blocks - 1), fn idx ->
        if rem(idx + 1, 6) == 0, do: :full_attention, else: :sliding_attention
      end)

    %{spec | layer_types: force_last_full_attention(layer_types)}
  end

  defp normalize_layer_types(%__MODULE__{layer_types: layer_types} = spec) do
    layer_types =
      Enum.map(layer_types, fn
        "full_attention" -> :full_attention
        "sliding_attention" -> :sliding_attention
        :full_attention -> :full_attention
        :sliding_attention -> :sliding_attention
      end)

    %{spec | layer_types: force_last_full_attention(layer_types)}
  end

  defp force_last_full_attention([]), do: []

  defp force_last_full_attention(layer_types) do
    List.replace_at(layer_types, -1, :full_attention)
  end

  defp layer_types(spec), do: spec.layer_types || normalize_layer_types(spec).layer_types

  defnp proportional_rotary_embedding(query, key, position_ids, opts \\ []) do
    opts = keyword!(opts, [:head_size, :base, :rotated_angles, mode: :inference])

    head_size = opts[:head_size]
    half_size = div(head_size, 2)
    rotated_angles = opts[:rotated_angles]

    angle_positions = Nx.iota({half_size})

    inv_freq =
      Nx.select(
        angle_positions < rotated_angles,
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

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      text_config = Map.fetch!(data, "text_config")
      audio_config = Map.get(data, "audio_config", %{})

      @for.config(spec,
        vocab_size: get_number(text_config, "vocab_size", spec.vocab_size),
        max_positions: get_number(text_config, "max_position_embeddings", spec.max_positions),
        hidden_size: get_number(text_config, "hidden_size", spec.hidden_size),
        intermediate_size: get_number(text_config, "intermediate_size", spec.intermediate_size),
        enable_moe_block: Map.get(text_config, "enable_moe_block", spec.enable_moe_block),
        moe_intermediate_size:
          Map.get(text_config, "moe_intermediate_size", spec.moe_intermediate_size),
        num_experts: Map.get(text_config, "num_experts", spec.num_experts),
        top_k_experts: Map.get(text_config, "top_k_experts", spec.top_k_experts),
        num_blocks: get_number(text_config, "num_hidden_layers", spec.num_blocks),
        num_attention_heads:
          get_number(text_config, "num_attention_heads", spec.num_attention_heads),
        num_key_value_heads:
          get_number(text_config, "num_key_value_heads", spec.num_key_value_heads),
        num_global_key_value_heads:
          Map.get(text_config, "num_global_key_value_heads", spec.num_global_key_value_heads),
        attention_head_size: get_number(text_config, "head_dim", spec.attention_head_size),
        global_attention_head_size:
          get_number(text_config, "global_head_dim", spec.global_attention_head_size),
        activation: activation(Map.get(text_config, "hidden_activation")),
        attention_k_eq_v: Map.get(text_config, "attention_k_eq_v", spec.attention_k_eq_v),
        use_attention_bias: Map.get(text_config, "attention_bias", spec.use_attention_bias),
        rotary_embedding_base:
          get_in(text_config, ["rope_parameters", "full_attention", "rope_theta"]) ||
            spec.rotary_embedding_base,
        rotary_embedding_base_local:
          get_in(text_config, ["rope_parameters", "sliding_attention", "rope_theta"]) ||
            spec.rotary_embedding_base_local,
        full_attention_rotary_percentage:
          get_in(text_config, ["rope_parameters", "full_attention", "partial_rotary_factor"]) ||
            spec.full_attention_rotary_percentage,
        layer_norm_epsilon: get_number(text_config, "rms_norm_eps", spec.layer_norm_epsilon),
        initializer_scale: get_number(data, "initializer_range", spec.initializer_scale),
        attention_window_size:
          get_number(text_config, "sliding_window", spec.attention_window_size),
        layer_types: Map.get(text_config, "layer_types", spec.layer_types),
        tie_word_embeddings:
          Map.get(text_config, "tie_word_embeddings", spec.tie_word_embeddings),
        final_logit_softcapping:
          Map.get(text_config, "final_logit_softcapping", spec.final_logit_softcapping),
        pad_token_id: Map.get(text_config, "pad_token_id", spec.pad_token_id),
        bos_token_id: Map.get(text_config, "bos_token_id", spec.bos_token_id),
        eos_token_id: Map.get(data, "eos_token_id", spec.eos_token_id),
        boa_token_id: Map.get(data, "boa_token_id", spec.boa_token_id),
        audio_token_id: Map.get(data, "audio_token_id", spec.audio_token_id),
        eoa_token_id: Map.get(data, "eoa_token_index", spec.eoa_token_id),
        audio_embed_dim: get_number(audio_config, "audio_embed_dim", spec.audio_embed_dim),
        audio_rms_norm_epsilon:
          get_number(audio_config, "rms_norm_eps", spec.audio_rms_norm_epsilon),
        quantization_config: Map.get(data, "quantization_config", spec.quantization_config)
      )
    end

    defp get_number(data, key, default), do: Map.get(data, key, default)

    defp activation("gelu_pytorch_tanh"), do: :gelu_approx_tanh
    defp activation("gelu_new"), do: :gelu_approx_tanh
    defp activation("quick_gelu"), do: :gelu_approx_sigmoid
    defp activation(value) when is_binary(value), do: String.to_existing_atom(value)
    defp activation(_), do: :gelu_approx_tanh
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      linear_source = linear_source_fun(spec)

      %{
        "embedder.token_embedding" => "model.language_model.embed_tokens",
        "audio_embedder.projection" => "model.embed_audio.embedding_projection",
        "decoder.blocks.{n}.self_attention.query" =>
          linear_source.("model.language_model.layers.{n}.self_attn.q_proj"),
        "decoder.blocks.{n}.self_attention.key" =>
          linear_source.("model.language_model.layers.{n}.self_attn.k_proj"),
        "decoder.blocks.{n}.self_attention.value" =>
          linear_source.("model.language_model.layers.{n}.self_attn.v_proj"),
        "decoder.blocks.{n}.self_attention.output" =>
          linear_source.("model.language_model.layers.{n}.self_attn.o_proj"),
        "decoder.blocks.{n}.self_attention.query_norm" =>
          "model.language_model.layers.{n}.self_attn.q_norm",
        "decoder.blocks.{n}.self_attention.key_norm" =>
          "model.language_model.layers.{n}.self_attn.k_norm",
        "decoder.blocks.{n}.self_attention_norm" =>
          "model.language_model.layers.{n}.input_layernorm",
        "decoder.blocks.{n}.post_attention_norm" =>
          "model.language_model.layers.{n}.post_attention_layernorm",
        "decoder.blocks.{n}.pre_ffn_norm" =>
          "model.language_model.layers.{n}.pre_feedforward_layernorm",
        "decoder.blocks.{n}.post_ffn_norm" =>
          "model.language_model.layers.{n}.post_feedforward_layernorm",
        "decoder.blocks.{n}.layer_scalar" => "model.language_model.layers.{n}",
        "decoder.blocks.{n}.ffn.gate" =>
          linear_source.("model.language_model.layers.{n}.mlp.gate_proj"),
        "decoder.blocks.{n}.ffn.intermediate" =>
          linear_source.("model.language_model.layers.{n}.mlp.up_proj"),
        "decoder.blocks.{n}.ffn.output" =>
          linear_source.("model.language_model.layers.{n}.mlp.down_proj"),
        "output_norm" => "model.language_model.norm",
        "language_modeling_head.output" =>
          if(spec.tie_word_embeddings, do: "model.language_model.embed_tokens", else: "lm_head")
      }
    end

    # Weights stay packed: the layer reads int4 directly on decode instead of
    # loading a dequantized bf16 matrix four times the size.
    defp linear_source_fun(%{
           quantization_config: %{"quant_method" => "compressed-tensors"},
           packed_linear: true,
           hybrid_linear: true
         }) do
      fn layer_name ->
        %{
          "kernel" =>
            {[{layer_name, "weight_packed"}, {layer_name, "weight_scale"}],
             &Gemma4MicTranscribe.Gemma4Unified.CompressedTensors.linear_kernel_bf16/1},
          "packed" =>
            {[{layer_name, "weight_packed"}],
             &Gemma4MicTranscribe.Gemma4Unified.CompressedTensors.repack_kernel/1},
          "scales" =>
            {[{layer_name, "weight_scale"}],
             &Gemma4MicTranscribe.Gemma4Unified.CompressedTensors.repack_scales/1}
        }
      end
    end

    defp linear_source_fun(%{
           quantization_config: %{"quant_method" => "compressed-tensors"},
           packed_linear: true
         }) do
      fn layer_name ->
        %{
          "packed" =>
            {[{layer_name, "weight_packed"}],
             &Gemma4MicTranscribe.Gemma4Unified.CompressedTensors.repack_kernel/1},
          "scales" =>
            {[{layer_name, "weight_scale"}],
             &Gemma4MicTranscribe.Gemma4Unified.CompressedTensors.repack_scales/1}
        }
      end
    end

    defp linear_source_fun(%{quantization_config: %{"quant_method" => "compressed-tensors"}}) do
      fn layer_name ->
        %{
          "kernel" =>
            {[{layer_name, "weight_packed"}, {layer_name, "weight_scale"}],
             &Gemma4MicTranscribe.Gemma4Unified.CompressedTensors.linear_kernel/1}
        }
      end
    end

    defp linear_source_fun(_spec) do
      fn layer_name -> layer_name end
    end
  end
end
