defmodule Gemma4MicTranscribe.Gemma4Unified.Prompt do
  @moduledoc false

  @audio_begin "<|audio>"
  @audio_token "<|audio|>"
  @audio_end "<audio|>"
  @turn_end "<turn|>"
  @empty_thought_channel "<|channel>thought\n<channel|>"
  @open_thought_channel "<|channel>thought\n"
  @think "<|think|>"

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

  `thought_channel: false` stops at the model turn, so the model writes its
  own thought channel. On its own that does not turn thinking on: without the
  think token the 12B opens the channel and closes it straight away.

  `think: true` is the 12B template's `enable_thinking=True`: a system turn
  that starts with `<|think|>` (opened even without a system message) and a
  prompt that stops at the model turn, where the model then reasons inside its
  thought channel before answering. The E4B template puts a newline after
  `<|think|>`; the 12B's does not, and this follows the 12B.

  The packed 12B does not open that channel itself under greedy decoding: it
  spells `<thought` out as text and never closes it. So `think: true` also
  opens the channel (`thought_channel: :open`), and the model's first
  generated token is already thought, to be closed with `<channel|>` before
  the reply. Under greedy decoding the 12B reaches the answer inside the
  thought and then keeps re-checking it ("Wait, let me double-check...")
  without closing; thinking needs sampling or a budget.

      <bos><|turn>system\\n<|think|>{system}<turn|>\\n<|turn>user\\n{prompt}<turn|>\\n<|turn>model\\n<|channel>thought\\n
  """
  def text(system_message, prompt, opts \\ []) do
    think = Keyword.get(opts, :think, false)
    opts = if think, do: Keyword.put(opts, :thought_channel, :open), else: opts

    "<bos>" <>
      system_turn(system_message, think) <>
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
    case Keyword.get(opts, :thought_channel, true) do
      :open -> @open_thought_channel
      true -> @empty_thought_channel
      false -> ""
    end
  end

  defp system_turn(system_message, think \\ false)

  defp system_turn(system_message, true) do
    "<|turn>system\n" <> @think <> normalize_text(system_message) <> @turn_end <> "\n"
  end

  defp system_turn(system_message, false) do
    case normalize_text(system_message) do
      "" -> ""
      text -> "<|turn>system\n" <> text <> @turn_end <> "\n"
    end
  end

  defp normalize_text(nil), do: ""
  defp normalize_text(text) when is_binary(text), do: String.trim(text)
end
