defmodule Gemma4MicTranscribe.Gemma4Unified.DecodeState do
  @moduledoc false

  defstruct [
    :prompt_length,
    :max_sequence_length,
    :input_ids,
    :attention_mask,
    :position_ids
  ]

  def new(prompt_input_ids, max_response_tokens, pad_token_id) when max_response_tokens >= 0 do
    prompt_length = length(prompt_input_ids)
    max_sequence_length = prompt_length + max_response_tokens

    position_ids =
      if max_sequence_length == 0, do: [], else: Enum.to_list(0..(max_sequence_length - 1))

    %__MODULE__{
      prompt_length: prompt_length,
      max_sequence_length: max_sequence_length,
      input_ids: prompt_input_ids ++ List.duplicate(pad_token_id, max_response_tokens),
      attention_mask: List.duplicate(1, prompt_length) ++ List.duplicate(0, max_response_tokens),
      position_ids: position_ids
    }
  end

  def context_length(%__MODULE__{prompt_length: prompt_length}, generated_count) do
    prompt_length + generated_count
  end

  def append(%__MODULE__{} = state, generated_count, token_id) do
    write_index = context_length(state, generated_count)

    if write_index >= state.max_sequence_length do
      raise ArgumentError,
            "cannot append token beyond fixed decode length #{state.max_sequence_length}"
    end

    %{
      state
      | input_ids: List.replace_at(state.input_ids, write_index, token_id),
        attention_mask: List.replace_at(state.attention_mask, write_index, 1)
    }
  end
end
