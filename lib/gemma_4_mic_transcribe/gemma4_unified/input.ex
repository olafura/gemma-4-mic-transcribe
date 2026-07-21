defmodule Gemma4MicTranscribe.Gemma4Unified.Input do
  @moduledoc false

  alias Gemma4MicTranscribe.Gemma4Unified.AudioFeatureExtractor
  alias Gemma4MicTranscribe.Gemma4Unified.Prompt

  def build(samples, opts \\ []) do
    features = AudioFeatureExtractor.extract(samples, opts)

    %{
      # kept so a runtime with a different audio front end (E4B mel features)
      # can rebuild the features it needs
      samples: Enum.to_list(samples),
      prompt:
        Prompt.build(
          Keyword.get(opts, :system_message),
          Keyword.fetch!(opts, :prompt),
          features.token_count
        ),
      audio: features
    }
  end
end
