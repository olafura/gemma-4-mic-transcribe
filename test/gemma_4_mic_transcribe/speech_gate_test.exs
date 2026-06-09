defmodule Gemma4MicTranscribe.SpeechGateTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.SpeechGate

  test "rejects windows shorter than the minimum speech duration" do
    analysis = SpeechGate.analyze(List.duplicate(0.2, 160), sample_rate: 16_000)

    refute analysis.speech?
    assert analysis.reason == :too_short
    assert analysis.min_samples == 4_000
  end

  test "rejects silence" do
    analysis = SpeechGate.analyze(List.duplicate(0.0, 16_000), sample_rate: 16_000)

    refute analysis.speech?
    assert analysis.reason == :low_rms
  end

  test "accepts a sustained voiced signal" do
    samples = sine_samples(440, 0.05, 0.3, 16_000)
    analysis = SpeechGate.analyze(samples, sample_rate: 16_000)

    assert analysis.speech?
    assert analysis.reason == :speech
  end

  test "rejects high zero-crossing noise" do
    samples =
      1..16_000
      |> Enum.map(fn index -> if rem(index, 2) == 0, do: 0.05, else: -0.05 end)

    analysis = SpeechGate.analyze(samples, sample_rate: 16_000)

    refute analysis.speech?
    assert analysis.reason == :high_zero_crossing_rate
  end

  defp sine_samples(frequency, amplitude, seconds, sample_rate) do
    sample_count = round(seconds * sample_rate)

    for index <- 0..(sample_count - 1) do
      amplitude * :math.sin(2.0 * :math.pi() * frequency * index / sample_rate)
    end
  end
end
