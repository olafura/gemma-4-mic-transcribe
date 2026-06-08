defmodule Gemma4MicTranscribe.Gemma4UnifiedTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4Unified.AudioFeatureExtractor
  alias Gemma4MicTranscribe.Gemma4Unified.Input
  alias Gemma4MicTranscribe.Gemma4Unified.Prompt

  test "audio feature extractor chunks raw 16 kHz audio into 640-sample soft tokens" do
    features = AudioFeatureExtractor.extract(List.duplicate(0.25, 641))

    assert features.token_count == 2
    assert Nx.shape(features.input_features) == {2, 640}
    assert Nx.to_flat_list(features.attention_mask) == [1, 1]
    assert features.input_features[1][1] |> Nx.to_number() == 0.0
  end

  test "audio feature extractor truncates at max token count" do
    features = AudioFeatureExtractor.extract(List.duplicate(0.0, 2_000), max_tokens: 2)

    assert features.token_count == 2
    assert Nx.shape(features.input_features) == {2, 640}
  end

  test "prompt inserts one audio placeholder per audio token" do
    prompt = Prompt.build("System", "Transcribe.", 3)

    assert prompt =~ "<boa>"
    assert prompt =~ "<eoa>"
    assert prompt =~ "System"
    assert prompt =~ "Transcribe."
    assert prompt |> String.split(Prompt.audio_placeholder()) |> length() == 4
  end

  test "input builder combines prompt and audio features" do
    input = Input.build(List.duplicate(0.0, 640), prompt: "Transcribe.")

    assert input.audio.token_count == 1
    assert input.prompt =~ Prompt.audio_placeholder()
  end
end
