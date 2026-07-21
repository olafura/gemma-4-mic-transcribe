defmodule Gemma4MicTranscribe.Gemma4Unified.Prompt do
  @moduledoc false

  @audio_begin "<|audio>"
  @audio_token "<|audio|>"
  @audio_end "<audio|>"
  @empty_thought_channel "<|channel>thought\n<channel|>"

  def audio_begin, do: @audio_begin
  def audio_token, do: @audio_token
  def audio_end, do: @audio_end
  def audio_placeholder, do: @audio_token

  def build(system_message, prompt, audio_token_count, opts \\ [])
      when is_integer(audio_token_count) and audio_token_count >= 0 do
    prefix(system_message, prompt) <>
      String.duplicate(@audio_token, audio_token_count) <> suffix(opts)
  end

  @doc """
  Everything up to and including the audio begin marker.

  Incremental prefill splits the prompt here: this part plus the audio soft
  tokens grows append-only as audio arrives, so its KV cache can be reused
  across partial transcripts instead of being recomputed each time.
  """
  def prefix(system_message, prompt) do
    system_message = normalize_text(system_message)
    prompt = normalize_text(prompt)

    system_turn =
      if system_message == "" do
        ""
      else
        "<|turn>system\n#{system_message}<turn|>\n"
      end

    "<bos>" <>
      system_turn <>
      "<|turn>user\n" <>
      prompt <>
      "\n\n" <>
      @audio_begin
  end

  @doc """
  Everything after the audio soft tokens, which closes the user turn and opens
  the model turn. Prefilled fresh on top of the cached audio prefix.

  The 12B Unified model speaks the channel protocol, so its generation starts
  after an empty thought channel. E4B was not trained on channels - handed
  one, it writes deliberation into it instead of transcribing - so
  `thought_channel: false` ends the prompt at the model turn, matching the
  reference.
  """
  def suffix(opts \\ []) do
    channel =
      if Keyword.get(opts, :thought_channel, true), do: @empty_thought_channel, else: ""

    @audio_end <>
      "<turn|>\n" <>
      "<|turn>model\n" <>
      channel
  end

  defp normalize_text(nil), do: ""
  defp normalize_text(text) when is_binary(text), do: String.trim(text)
end
