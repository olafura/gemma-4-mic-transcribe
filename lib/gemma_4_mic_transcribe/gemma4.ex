defmodule Gemma4MicTranscribe.Gemma4 do
  @moduledoc "Convenience functions for inspecting Gemma 4 model components."

  alias Gemma4MicTranscribe.Gemma4.Experts
  alias Gemma4MicTranscribe.Gemma4.DecoderBlocks
  alias Gemma4MicTranscribe.Gemma4.DecoderPipeline
  alias Gemma4MicTranscribe.Gemma4.FFNs

  @doc "Lists the routed and shared experts in a Gemma 4 MoE model."
  defdelegate list_experts(model_or_config, opts \\ []), to: Experts, as: :list

  @doc "Lists the dense gated FFN in every Gemma 4 decoder layer."
  defdelegate list_ffns(model_or_config, opts \\ []), to: FFNs, as: :list

  @doc "Extracts one unified Gemma 4 decoder block for standalone execution."
  defdelegate extract_decoder_block(runtime, layer_index), to: DecoderBlocks, as: :extract

  @doc "Extracts contiguous unified Gemma 4 decoder blocks as one standalone graph."
  defdelegate extract_decoder_chain(runtime, layer_indices),
    to: DecoderBlocks,
    as: :extract_chain

  @doc "Extracts a final decoder chain with its vocabulary output head."
  defdelegate extract_decoder_tail(runtime, layer_indices),
    to: DecoderBlocks,
    as: :extract_tail

  @doc "Splits a unified model into a raw-input prefix and replaceable decoder tail."
  defdelegate extract_decoder_pipeline(runtime, tail_layers),
    to: DecoderPipeline,
    as: :extract
end
