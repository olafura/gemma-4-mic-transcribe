defmodule Gemma4MicTranscribe.Gemma4Unified.Prompt do
  @moduledoc false

  @audio_placeholder "<audio_soft_token>"

  def audio_placeholder, do: @audio_placeholder

  def build(system_message, prompt, audio_token_count)
      when is_integer(audio_token_count) and audio_token_count >= 0 do
    system_message = normalize_text(system_message)
    prompt = normalize_text(prompt)
    audio_tokens = String.duplicate(@audio_placeholder, audio_token_count)

    user_content =
      [system_message, "<boa>" <> audio_tokens <> "<eoa>", prompt]
      |> Enum.reject(&(&1 == ""))
      |> Enum.join("\n\n")

    """
    <bos><start_of_turn>user
    #{user_content}<end_of_turn>
    <start_of_turn>model
    """
    |> String.trim()
  end

  defp normalize_text(nil), do: ""
  defp normalize_text(text) when is_binary(text), do: String.trim(text)
end
