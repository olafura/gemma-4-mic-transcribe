defmodule Gemma4MicTranscribe.Gemma4E4BMelFeaturesTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4E4B.MelFeatures
  alias Gemma4MicTranscribe.Gemma4E4B.Spec

  @sample_rate 16_000

  defp spec, do: %Spec{audio_mel_bins: 32, audio_frame_length_ms: 32.0, audio_frame_step_ms: 10.0}

  defp tone(hz, seconds) do
    count = round(@sample_rate * seconds)

    for index <- 0..(count - 1) do
      :math.sin(2 * :math.pi() * hz * index / @sample_rate)
    end
  end

  test "frame count follows length and hop" do
    s = spec()

    # 32ms frames hopping 10ms: 512 samples long, 160 apart
    assert MelFeatures.frame_count(512, s, sample_rate: @sample_rate) == 1
    # shorter than one frame is padded up to one frame, not dropped
    assert MelFeatures.frame_count(511, s, sample_rate: @sample_rate) == 1
    assert MelFeatures.frame_count(512 + 160, s, sample_rate: @sample_rate) == 2
    assert MelFeatures.frame_count(16_000, s, sample_rate: @sample_rate) == 97
  end

  test "audio token count is the frame count after subsampling" do
    s = spec()

    # one second gives 97 mel frames, subsampled twice by two -> 25 tokens,
    # which matches Gemma 4's documented 25 audio tokens per second
    assert MelFeatures.audio_token_count(16_000, s, sample_rate: @sample_rate) == 25
  end

  test "extracts one row per frame with the configured mel bins" do
    s = spec()
    features = MelFeatures.extract(tone(440, 0.1), s, sample_rate: @sample_rate)

    frames = MelFeatures.frame_count(1600, s, sample_rate: @sample_rate)
    assert Nx.shape(features) == {frames, 32}
    assert features |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
  end

  test "a pure tone peaks in a higher mel bin as its pitch rises" do
    s = spec()

    peak_bin = fn hz ->
      MelFeatures.extract(tone(hz, 0.2), s, sample_rate: @sample_rate)
      |> Nx.mean(axes: [0])
      |> Nx.argmax()
      |> Nx.to_number()
    end

    low = peak_bin.(300)
    mid = peak_bin.(1000)
    high = peak_bin.(3000)

    assert low < mid
    assert mid < high
  end

  test "silence is quieter than a tone in every mel bin" do
    s = spec()

    silent =
      MelFeatures.extract(List.duplicate(0.0, 3200), s, sample_rate: @sample_rate)
      |> Nx.mean()
      |> Nx.to_number()

    loud =
      MelFeatures.extract(tone(1000, 0.2), s, sample_rate: @sample_rate)
      |> Nx.mean()
      |> Nx.to_number()

    assert silent < loud
  end

  test "audio shorter than one frame is padded to a single frame" do
    s = spec()
    features = MelFeatures.extract(List.duplicate(0.1, 100), s, sample_rate: @sample_rate)

    assert Nx.shape(features) == {1, 32}
    assert MelFeatures.audio_token_count(100, s, sample_rate: @sample_rate) == 1
  end

  test "filterbank rows are non-negative and cover the spectrum" do
    bank = MelFeatures.mel_filterbank(16, 512, @sample_rate)

    assert Nx.shape(bank) == {257, 16}
    assert bank |> Nx.less(0.0) |> Nx.any() |> Nx.to_number() == 0
    # every filter has some weight somewhere
    assert bank |> Nx.sum(axes: [0]) |> Nx.greater(0.0) |> Nx.all() |> Nx.to_number() == 1
  end
end
