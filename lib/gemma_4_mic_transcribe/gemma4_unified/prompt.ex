defmodule Gemma4MicTranscribe.Gemma4Unified.Prompt do
  @moduledoc false

  @audio_begin "<|audio>"
  @audio_token "<|audio|>"
  @audio_end "<audio|>"

  def audio_begin, do: @audio_begin
  def audio_token, do: @audio_token
  def audio_end, do: @audio_end
  def audio_placeholder, do: @audio_token

  def build(system_message, prompt, audio_token_count)
      when is_integer(audio_token_count) and audio_token_count >= 0 do
    system_message = normalize_text(system_message)
    prompt = normalize_text(prompt)
    audio_block = @audio_begin <> String.duplicate(@audio_token, audio_token_count) <> @audio_end

    system_turn =
      if system_message == "" do
        ""
      else
        "<|turn>system\n#{system_message}<turn|>\n"
      end

    "<bos>" <>
      system_turn <>
      "<|turn>user\n" <>
      audio_block <>
      prompt <>
      "<turn|>\n" <>
      "<|turn>model\n"
  end

  defp normalize_text(nil), do: ""
  defp normalize_text(text) when is_binary(text), do: String.trim(text)
end
