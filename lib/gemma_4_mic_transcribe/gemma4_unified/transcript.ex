defmodule Gemma4MicTranscribe.Gemma4Unified.Transcript do
  @moduledoc false

  def decode(tokenizer, token_ids) do
    token_ids =
      token_ids
      |> strip_tagged_span(token_id(tokenizer, "<|channel>"), token_id(tokenizer, "<channel|>"))
      |> strip_tagged_span(token_id(tokenizer, "<|tool>"), token_id(tokenizer, "<tool|>"))
      |> strip_tagged_span(
        token_id(tokenizer, "<|tool_call>"),
        token_id(tokenizer, "<tool_call|>")
      )
      |> strip_tagged_span(
        token_id(tokenizer, "<|tool_response>"),
        token_id(tokenizer, "<tool_response|>")
      )
      |> Enum.reject(&standalone_control_token?(tokenizer, &1))

    tokenizer
    |> Bumblebee.Tokenizer.decode(token_ids)
    |> String.trim()
  end

  def strip_tagged_span(token_ids, nil, _end_token_id), do: token_ids

  def strip_tagged_span(token_ids, _start_token_id, nil), do: token_ids

  def strip_tagged_span(token_ids, start_token_id, end_token_id) do
    {kept, _skipping?} =
      Enum.reduce(token_ids, {[], false}, fn token_id, {kept, skipping?} ->
        cond do
          token_id == start_token_id ->
            {kept, true}

          skipping? and token_id == end_token_id ->
            {kept, false}

          skipping? ->
            {kept, true}

          token_id == end_token_id ->
            {kept, false}

          true ->
            {[token_id | kept], false}
        end
      end)

    Enum.reverse(kept)
  end

  defp standalone_control_token?(tokenizer, token_id) do
    token_id in [
      token_id(tokenizer, "<|think|>")
    ]
  end

  defp token_id(tokenizer, token) do
    Bumblebee.Tokenizer.token_to_id(tokenizer, token)
  end
end
