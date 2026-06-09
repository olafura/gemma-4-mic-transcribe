defmodule Gemma4MicTranscribe.SpeechGate do
  @moduledoc false

  def analyze(samples, opts \\ []) do
    samples = Enum.to_list(samples)
    sample_count = length(samples)
    sample_rate = Keyword.fetch!(opts, :sample_rate)
    threshold = Keyword.get(opts, :threshold, 0.01)
    min_speech_seconds = Keyword.get(opts, :min_speech_seconds, 0.25)
    min_active_ratio = Keyword.get(opts, :min_active_ratio, 0.2)
    max_zero_crossing_rate = Keyword.get(opts, :max_zero_crossing_rate, 0.35)
    frame_seconds = Keyword.get(opts, :frame_seconds, 0.02)

    rms = rms(samples)
    peak = peak(samples)
    zero_crossing_rate = zero_crossing_rate(samples)
    active_ratio = active_ratio(samples, sample_rate, frame_seconds, threshold)
    min_samples = max(1, round(sample_rate * min_speech_seconds))

    reason =
      cond do
        sample_count < min_samples -> :too_short
        rms < threshold -> :low_rms
        active_ratio < min_active_ratio -> :low_active_ratio
        zero_crossing_rate > max_zero_crossing_rate -> :high_zero_crossing_rate
        true -> :speech
      end

    %{
      speech?: reason == :speech,
      reason: reason,
      sample_count: sample_count,
      min_samples: min_samples,
      rms: rms,
      peak: peak,
      active_ratio: active_ratio,
      zero_crossing_rate: zero_crossing_rate,
      threshold: threshold
    }
  end

  def speech?(samples, opts \\ []), do: analyze(samples, opts).speech?

  defp rms([]), do: 0.0

  defp rms(samples) do
    mean_square =
      samples
      |> Enum.reduce(0.0, fn sample, acc -> acc + sample * sample end)
      |> Kernel./(length(samples))

    :math.sqrt(mean_square)
  end

  defp peak(samples) do
    samples
    |> Enum.map(&abs/1)
    |> Enum.max(fn -> 0.0 end)
  end

  defp active_ratio([], _sample_rate, _frame_seconds, _threshold), do: 0.0

  defp active_ratio(samples, sample_rate, frame_seconds, threshold) do
    frame_samples = max(1, round(sample_rate * frame_seconds))
    frames = Enum.chunk_every(samples, frame_samples)
    active_frames = Enum.count(frames, &(rms(&1) >= threshold))

    active_frames / length(frames)
  end

  defp zero_crossing_rate(samples) when length(samples) < 2, do: 0.0

  defp zero_crossing_rate(samples) do
    {crossings, _last_sign} =
      samples
      |> Enum.map(&sign/1)
      |> Enum.reduce({0, nil}, fn
        sign, {crossings, nil} -> {crossings, sign}
        sign, {crossings, sign} -> {crossings, sign}
        sign, {crossings, _last_sign} -> {crossings + 1, sign}
      end)

    crossings / (length(samples) - 1)
  end

  defp sign(sample) when sample < 0.0, do: -1
  defp sign(_sample), do: 1
end
