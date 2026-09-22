defmodule Gemma4MicTranscribe.Gemma4Unified.Prompt do
  @moduledoc false

  @audio_begin "<|audio>"
  @audio_token "<|audio|>"
  @audio_end "<audio|>"
  @turn_end "<turn|>"
  @empty_thought_channel "<|channel>thought\n<channel|>"

  def audio_begin, do: @audio_begin
  def audio_token, do: @audio_token
  def audio_end, do: @audio_end
  def audio_placeholder, do: @audio_token
  def turn_end, do: @turn_end

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
    "<bos>" <>
      system_turn(system_message) <>
      "<|turn>user\n" <>
      normalize_text(prompt) <>
      "\n\n" <>
      @audio_begin
  end

  @doc """
  A text-only prompt, up to and including the model turn header.

  No audio markers are emitted at all: this is byte for byte what the official
  Gemma 4 chat template renders for a single user turn with
  `add_generation_prompt=True` and `enable_thinking=False`, so a text prompt
  reaches the model in the form it was trained on.

      <bos><|turn>user\\n{prompt}<turn|>\\n<|turn>model\\n<|channel>thought\\n<channel|>

  `thought_channel: false` stops at the model turn, which is the template's
  `enable_thinking=True` generation prompt: the model then opens its own
  thought channel instead of finding it already closed.
  """
  def text(system_message, prompt, opts \\ []) do
    "<bos>" <>
      system_turn(system_message) <>
      "<|turn>user\n" <>
      normalize_text(prompt) <>
      @turn_end <>
      "\n" <>
      "<|turn>model\n" <>
      thought_channel(opts)
  end

  @doc """
  A teacher-forced model turn: the response text and the end-of-turn token the
  model would have to emit to close it, appended after `text/3`.
  """
  def forced_response(response), do: normalize_text(response) <> @turn_end

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
    @audio_end <>
      @turn_end <>
      "\n" <>
      "<|turn>model\n" <>
      thought_channel(opts)
  end

  defp thought_channel(opts) do
    if Keyword.get(opts, :thought_channel, true), do: @empty_thought_channel, else: ""
  end

  defp system_turn(system_message) do
    case normalize_text(system_message) do
      "" -> ""
      text -> "<|turn>system\n" <> text <> @turn_end <> "\n"
    end
  end

  defp normalize_text(nil), do: ""
  defp normalize_text(text) when is_binary(text), do: String.trim(text)
end
