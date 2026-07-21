defmodule Gemma4MicTranscribe.Gemma4E4B.Spec do
  @moduledoc false

  # Gemma 4 E4B configuration.
  #
  # E4B is not a smaller 12B Unified: it keeps a conformer audio encoder where
  # the 12B projects raw PCM frames directly, and its decoder adds per-layer
  # input embeddings, KV sharing across a suffix of layers, and a double-wide
  # MLP. This module holds the configuration for that architecture so the
  # layers can be built against a checked spec rather than guessed constants.

  @behaviour Bumblebee.Configurable

  defstruct architecture: :for_conditional_generation,

            # decoder
            vocab_size: 262_144,
            vocab_size_per_layer_input: 262_144,
            max_positions: 262_144,
            hidden_size: 2560,
            hidden_size_per_layer_input: 256,
            intermediate_size: 10_240,
            num_blocks: 42,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            num_global_key_value_heads: nil,
            attention_head_size: 256,
            global_attention_head_size: 512,
            num_kv_shared_layers: 18,
            use_double_wide_mlp: false,
            use_bidirectional_attention: false,
            attention_k_eq_v: false,
            activation: :gelu_approx_tanh,
            rotary_embedding_base: 1_000_000,
            rotary_embedding_base_local: 10_000,
            full_attention_rotary_percentage: 0.25,
            attention_window_size: 512,
            layer_norm_epsilon: 1.0e-6,
            final_logit_softcapping: 30.0,
            tie_word_embeddings: true,
            layer_types: nil,

            # audio encoder (model_type "gemma4_audio")
            audio_hidden_size: 1024,
            audio_num_blocks: 12,
            audio_num_attention_heads: 8,
            audio_conv_kernel_size: 5,
            audio_subsampling_conv_channels: [128, 32],
            audio_output_proj_dims: 1536,
            audio_attention_chunk_size: 12,
            audio_attention_context_left: 13,
            audio_attention_context_right: 0,
            audio_attention_logit_cap: 50.0,
            audio_attention_invalid_logits_value: -1.0e9,
            audio_residual_weight: 0.5,
            audio_activation: :silu,
            audio_use_clipped_linears: true,
            audio_rms_norm_epsilon: 1.0e-6,
            # E4B consumes mel spectrogram frames, not the raw 640 sample PCM
            # frames the 12B Unified model takes.
            audio_mel_bins: 128,
            audio_frame_length_ms: 32.0,
            audio_frame_step_ms: 10.0,

            # tokens
            pad_token_id: 0,
            bos_token_id: 2,
            eos_token_id: [1, 106],
            boa_token_id: 256_000,
            audio_token_id: 258_881,
            eoa_token_id: 258_883,
            quantization_config: nil

  @impl true
  def config(spec, opts) do
    spec |> struct!(opts) |> normalize_layer_types()
  end

  @doc """
  Layers whose key/value projections are shared rather than computed.

  The last `num_kv_shared_layers` blocks reuse the key/value state of the last
  block that computed its own, so the cache only holds entries for the
  computing layers.
  """
  def kv_shared_layer?(%__MODULE__{num_blocks: blocks, num_kv_shared_layers: shared}, index)
      when is_integer(shared) and shared > 0 do
    index >= blocks - shared
  end

  def kv_shared_layer?(_spec, _index), do: false

  @doc """
  Index of the block whose key/value state a shared block reuses: the last
  block before the shared suffix begins.
  """
  def kv_source_layer(%__MODULE__{num_blocks: blocks, num_kv_shared_layers: shared}) do
    max(blocks - shared - 1, 0)
  end

  @doc """
  Output length of the audio subsampling stack for a given input frame count.

  Each subsampling convolution halves the time axis, so the encoder sees one
  frame per `2^length(channels)` input frames.
  """
  def audio_subsampled_length(%__MODULE__{audio_subsampling_conv_channels: channels}, frames) do
    Enum.reduce(channels, frames, fn _channel, acc -> div(acc + 1, 2) end)
  end

  defp normalize_layer_types(%__MODULE__{layer_types: nil} = spec) do
    layer_types =
      Enum.map(0..(spec.num_blocks - 1), fn index ->
        if rem(index + 1, 6) == 0, do: :full_attention, else: :sliding_attention
      end)

    %{spec | layer_types: layer_types}
  end

  defp normalize_layer_types(%__MODULE__{layer_types: layer_types} = spec) do
    %{
      spec
      | layer_types:
          Enum.map(layer_types, fn
            type when type in [:full_attention, :sliding_attention] -> type
            "full_attention" -> :full_attention
            "sliding_attention" -> :sliding_attention
          end)
    }
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      text = Map.fetch!(data, "text_config")
      audio = Map.get(data, "audio_config", %{})

      @for.config(spec,
        vocab_size: Map.get(text, "vocab_size", spec.vocab_size),
        vocab_size_per_layer_input:
          Map.get(text, "vocab_size_per_layer_input", spec.vocab_size_per_layer_input),
        max_positions: Map.get(text, "max_position_embeddings", spec.max_positions),
        hidden_size: Map.get(text, "hidden_size", spec.hidden_size),
        hidden_size_per_layer_input:
          Map.get(text, "hidden_size_per_layer_input", spec.hidden_size_per_layer_input),
        intermediate_size: Map.get(text, "intermediate_size", spec.intermediate_size),
        num_blocks: Map.get(text, "num_hidden_layers", spec.num_blocks),
        num_attention_heads: Map.get(text, "num_attention_heads", spec.num_attention_heads),
        num_key_value_heads: Map.get(text, "num_key_value_heads", spec.num_key_value_heads),
        num_global_key_value_heads:
          Map.get(text, "num_global_key_value_heads", spec.num_global_key_value_heads),
        attention_head_size: Map.get(text, "head_dim", spec.attention_head_size),
        global_attention_head_size:
          Map.get(text, "global_head_dim", spec.global_attention_head_size),
        num_kv_shared_layers: Map.get(text, "num_kv_shared_layers", spec.num_kv_shared_layers),
        use_double_wide_mlp: Map.get(text, "use_double_wide_mlp", spec.use_double_wide_mlp),
        use_bidirectional_attention:
          Map.get(text, "use_bidirectional_attention", spec.use_bidirectional_attention),
        attention_k_eq_v: Map.get(text, "attention_k_eq_v", spec.attention_k_eq_v),
        activation: activation(Map.get(text, "hidden_activation")),
        rotary_embedding_base:
          get_in(text, ["rope_parameters", "full_attention", "rope_theta"]) ||
            spec.rotary_embedding_base,
        rotary_embedding_base_local:
          get_in(text, ["rope_parameters", "sliding_attention", "rope_theta"]) ||
            spec.rotary_embedding_base_local,
        full_attention_rotary_percentage:
          get_in(text, ["rope_parameters", "full_attention", "partial_rotary_factor"]) ||
            spec.full_attention_rotary_percentage,
        attention_window_size: Map.get(text, "sliding_window", spec.attention_window_size),
        layer_norm_epsilon: Map.get(text, "rms_norm_eps", spec.layer_norm_epsilon),
        final_logit_softcapping:
          Map.get(text, "final_logit_softcapping", spec.final_logit_softcapping),
        tie_word_embeddings: Map.get(text, "tie_word_embeddings", spec.tie_word_embeddings),
        layer_types: Map.get(text, "layer_types", spec.layer_types),
        pad_token_id: Map.get(text, "pad_token_id", spec.pad_token_id),
        bos_token_id: Map.get(text, "bos_token_id", spec.bos_token_id),
        eos_token_id: Map.get(data, "eos_token_id", spec.eos_token_id),
        boa_token_id: Map.get(data, "boa_token_id", spec.boa_token_id),
        audio_token_id: Map.get(data, "audio_token_id", spec.audio_token_id),
        eoa_token_id: Map.get(data, "eoa_token_index", spec.eoa_token_id),
        audio_hidden_size: Map.get(audio, "hidden_size", spec.audio_hidden_size),
        audio_num_blocks: Map.get(audio, "num_hidden_layers", spec.audio_num_blocks),
        audio_num_attention_heads:
          Map.get(audio, "num_attention_heads", spec.audio_num_attention_heads),
        audio_conv_kernel_size: Map.get(audio, "conv_kernel_size", spec.audio_conv_kernel_size),
        audio_subsampling_conv_channels:
          Map.get(audio, "subsampling_conv_channels", spec.audio_subsampling_conv_channels),
        audio_output_proj_dims: Map.get(audio, "output_proj_dims", spec.audio_output_proj_dims),
        audio_attention_chunk_size:
          Map.get(audio, "attention_chunk_size", spec.audio_attention_chunk_size),
        audio_attention_context_left:
          Map.get(audio, "attention_context_left", spec.audio_attention_context_left),
        audio_attention_context_right:
          Map.get(audio, "attention_context_right", spec.audio_attention_context_right),
        audio_attention_logit_cap:
          Map.get(audio, "attention_logit_cap", spec.audio_attention_logit_cap),
        audio_attention_invalid_logits_value:
          Map.get(
            audio,
            "attention_invalid_logits_value",
            spec.audio_attention_invalid_logits_value
          ),
        audio_residual_weight: Map.get(audio, "residual_weight", spec.audio_residual_weight),
        audio_activation: activation(Map.get(audio, "hidden_act")),
        audio_use_clipped_linears:
          Map.get(audio, "use_clipped_linears", spec.audio_use_clipped_linears),
        audio_rms_norm_epsilon: Map.get(audio, "rms_norm_eps", spec.audio_rms_norm_epsilon),
        audio_mel_bins: Map.get(audio, "num_mel_bins", spec.audio_mel_bins),
        quantization_config: Map.get(data, "quantization_config", spec.quantization_config)
      )
    end

    defp activation("gelu_pytorch_tanh"), do: :gelu_approx_tanh
    defp activation("gelu_new"), do: :gelu_approx_tanh
    defp activation("silu"), do: :silu
    defp activation(value) when is_binary(value), do: String.to_existing_atom(value)
    defp activation(_), do: :gelu_approx_tanh
  end
end
