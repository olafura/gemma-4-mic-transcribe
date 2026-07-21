defmodule Gemma4MicTranscribe.Gemma4E4B.MelFeatures do
  @moduledoc false

  # Log-mel spectrogram frames for the E4B audio encoder.
  #
  # The 12B Unified model takes raw PCM frames straight into a linear
  # projection, so this pipeline does not exist there. E4B expects mel frames,
  # which the conformer subsampling stack then reduces by 4 along time.

  import Nx.Defn

  alias Gemma4MicTranscribe.Gemma4E4B.Spec

  @doc """
  Extracts `{frames, mel_bins}` log-mel features from mono samples.

  Frames are `audio_frame_length_ms` long and hop by `audio_frame_step_ms`,
  matching the encoder's expected rate.
  """
  def extract(samples, %Spec{} = spec, opts \\ []) do
    sample_rate = Keyword.get(opts, :sample_rate, 16_000)

    frame_length = round(sample_rate * spec.audio_frame_length_ms / 1000)
    frame_step = round(sample_rate * spec.audio_frame_step_ms / 1000)
    fft_length = next_power_of_two(frame_length)

    # Nx has no zero-size dimensions, and a partial frame still carries audio,
    # so short input is padded up to one whole frame rather than dropped.
    samples = samples |> Enum.to_list() |> pad_to_frame(frame_length)
    samples = Nx.tensor(samples, type: :f32)

    frames = frame_signal(samples, frame_length, frame_step)
    filterbank = mel_filterbank(spec.audio_mel_bins, fft_length, sample_rate)
    window = hann_window(frame_length)

    frames
    |> power_spectrum(window, fft_length: fft_length)
    |> Nx.dot(filterbank)
    |> log_compress()
  end

  defp pad_to_frame(samples, frame_length) do
    missing = frame_length - length(samples)
    if missing > 0, do: samples ++ List.duplicate(0.0, missing), else: samples
  end

  @doc """
  Number of mel frames a sample count produces.
  """
  def frame_count(sample_count, %Spec{} = spec, opts \\ []) do
    sample_rate = Keyword.get(opts, :sample_rate, 16_000)
    frame_length = round(sample_rate * spec.audio_frame_length_ms / 1000)
    frame_step = round(sample_rate * spec.audio_frame_step_ms / 1000)

    # short input is padded to one frame rather than dropped
    div(max(sample_count, frame_length) - frame_length, frame_step) + 1
  end

  @doc """
  Audio placeholder tokens needed for a sample count.

  The prompt must carry one placeholder per encoder frame, which is the mel
  frame count after the subsampling stack, not a count derived from samples.
  """
  def audio_token_count(sample_count, %Spec{} = spec, opts \\ []) do
    sample_count
    |> frame_count(spec, opts)
    |> then(&Spec.audio_subsampled_length(spec, &1))
  end

  # Slices the signal into overlapping frames without materialising an index
  # tensor per frame.
  defp frame_signal(samples, frame_length, frame_step) do
    total = Nx.axis_size(samples, 0)

    count = div(total - frame_length, frame_step) + 1

    offsets = Nx.iota({count, 1}) |> Nx.multiply(frame_step)
    indices = Nx.add(offsets, Nx.iota({1, frame_length}))

    Nx.take(samples, indices)
  end

  defnp power_spectrum(frames, window, opts \\ []) do
    opts = keyword!(opts, [:fft_length, mode: :inference])
    fft_length = opts[:fft_length]
    frame_length = Nx.axis_size(window, 0)

    windowed = frames * window

    padded =
      Nx.pad(windowed, 0.0, [{0, 0, 0}, {0, fft_length - frame_length, 0}])

    padded
    |> Nx.fft(length: fft_length)
    |> then(fn spectrum ->
      # keep the non-redundant half, magnitude squared
      half = div(fft_length, 2) + 1
      spectrum = spectrum[[.., 0..(half - 1)//1]]
      Nx.real(spectrum) ** 2 + Nx.imag(spectrum) ** 2
    end)
  end

  # Built outside defn: the length is a shape, not a runtime value.
  defp hann_window(length) do
    positions = Nx.iota({length}, type: :f32)

    Nx.subtract(
      0.5,
      Nx.multiply(0.5, Nx.cos(Nx.multiply(2 * :math.pi() / (length - 1), positions)))
    )
  end

  defnp log_compress(mel) do
    Nx.log(mel + 1.0e-6)
  end

  @doc """
  Triangular mel filterbank of shape `{fft_bins, mel_bins}`.
  """
  def mel_filterbank(mel_bins, fft_length, sample_rate, opts \\ []) do
    low_hz = Keyword.get(opts, :low_hz, 0.0)
    high_hz = Keyword.get(opts, :high_hz, sample_rate / 2)

    fft_bins = div(fft_length, 2) + 1

    low_mel = hz_to_mel(low_hz)
    high_mel = hz_to_mel(high_hz)

    # mel_bins + 2 edges give each filter a left, centre and right point
    edges =
      for index <- 0..(mel_bins + 1) do
        mel = low_mel + (high_mel - low_mel) * index / (mel_bins + 1)
        mel_to_hz(mel) * fft_length / sample_rate
      end

    bin_positions = Enum.map(0..(fft_bins - 1), &(&1 * 1.0))

    weights =
      for bin <- bin_positions do
        for filter <- 0..(mel_bins - 1) do
          left = Enum.at(edges, filter)
          centre = Enum.at(edges, filter + 1)
          right = Enum.at(edges, filter + 2)

          cond do
            bin <= left or bin >= right -> 0.0
            bin <= centre and centre > left -> (bin - left) / (centre - left)
            right > centre -> (right - bin) / (right - centre)
            true -> 0.0
          end
        end
      end

    Nx.tensor(weights, type: :f32)
  end

  defp hz_to_mel(hz), do: 2595.0 * :math.log10(1.0 + hz / 700.0)
  defp mel_to_hz(mel), do: 700.0 * (:math.pow(10.0, mel / 2595.0) - 1.0)

  defp next_power_of_two(value) do
    Stream.iterate(1, &(&1 * 2)) |> Enum.find(&(&1 >= value))
  end
end
