defmodule Gemma4MicTranscribe.Gemma4Unified.Input do
  @moduledoc false

  alias Gemma4MicTranscribe.Gemma4Unified.AudioFeatureExtractor
  alias Gemma4MicTranscribe.Gemma4Unified.Prompt

  def build(samples, opts \\ []) do
    features = AudioFeatureExtractor.extract(samples, opts)
    system_message = Keyword.get(opts, :system_message)
    user_prompt = Keyword.fetch!(opts, :prompt)

    %{
      # kept so a runtime with a different audio front end (E4B mel features)
      # can rebuild the features it needs
      samples: Enum.to_list(samples),
      system_message: system_message,
      user_prompt: user_prompt,
      prompt: Prompt.build(system_message, user_prompt, features.token_count),
      audio: features
    }
  end
end
