defmodule Gemma4MicTranscribe.Gemma4Unified.Input do
  @moduledoc false

  alias Gemma4MicTranscribe.Gemma4Unified.AudioFeatureExtractor
  alias Gemma4MicTranscribe.Gemma4Unified.Prompt

  def build(samples, opts \\ []) do
    features = AudioFeatureExtractor.extract(samples, opts)
    system_message = Keyword.get(opts, :system_message)
    user_prompt = Keyword.fetch!(opts, :prompt)
    prompt_opts = Keyword.take(opts, [:thought_channel])

    %{
      # kept so a runtime with a different audio front end (E4B mel features)
      # can rebuild the features it needs
      samples: Enum.to_list(samples),
      system_message: system_message,
      user_prompt: user_prompt,
      prompt: Prompt.build(system_message, user_prompt, features.token_count, prompt_opts),
      audio: features
    }
  end

  @doc """
  Builds a text-only input: no audio markers in the prompt and no audio tokens.

  Options are `:system_message`, `:thought_channel`, `:think` (see
  `Gemma4MicTranscribe.Gemma4Unified.Prompt.text/3`) and `:response`. With
  `:response` the model turn is teacher-forced, so `:prompt` runs past the
  model turn header into the response and its end-of-turn token, and
  `:prompt_without_response` holds everything before it; `response_range/3`
  turns that split into the token index range of the response.
  """
  def build_text(prompt, opts \\ []) do
    system_message = Keyword.get(opts, :system_message)
    think = Keyword.get(opts, :think, false)
    thought_channel = if think, do: :open, else: Keyword.get(opts, :thought_channel, true)
    response = Keyword.get(opts, :response)

    head = Prompt.text(system_message, prompt, thought_channel: thought_channel, think: think)

    %{
      samples: [],
      system_message: system_message,
      user_prompt: prompt,
      response: response,
      prompt: head <> forced_response(response),
      prompt_without_response: head,
      thought_channel: thought_channel,
      audio: silent_audio()
    }
  end

  @doc """
  Token index range of the teacher-forced response, given the tokenized
  `:prompt` and `:prompt_without_response` of a `build_text/2` input.

  Both are tokenized separately rather than searched for, because only the
  split point is known here; the response itself carries no marker.
  """
  def response_range(%{response: nil}, _prompt_ids, _head_ids), do: nil

  def response_range(%{}, prompt_ids, head_ids) do
    start = length(head_ids)
    %{start: start, length: length(prompt_ids) - start}
  end

  defp forced_response(nil), do: ""
  defp forced_response(response), do: Prompt.forced_response(response)

  # The graph still takes audio, so a text-only input hands it one silent
  # frame masked out of attention. A zero-sized audio axis would be a zero
  # dimension for XLA, which the ROCm client does not compile.
  defp silent_audio do
    %{AudioFeatureExtractor.extract([], audio_token_count: 1) | token_count: 0}
  end
end
