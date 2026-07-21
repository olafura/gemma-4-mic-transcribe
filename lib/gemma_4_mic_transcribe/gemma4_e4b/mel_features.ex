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

  Matches the reference `Gemma4AudioFeatureExtractor`: semicausal padding of
  `frame_length / 2` zeros so the first frame is centred at t=0, a periodic
  Hann window, magnitude (not power) spectrum, and `log(mel + mel_floor)`.
  """
  def extract(samples, %Spec{} = spec, opts \\ []) do
    # Pin the eager tensor work to libtorch. The launcher does not load Mix
    # config, so on the EXLA path the ambient default backend is
    # Nx.BinaryBackend, whose pure-Elixir fft and dot turn this call from
    # milliseconds into seconds.
    Nx.with_default_backend(Torchx.Backend, fn -> do_extract(samples, spec, opts) end)
  end

  defp do_extract(samples, %Spec{} = spec, opts) do
    sample_rate = Keyword.get(opts, :sample_rate, 16_000)

    frame_length = round(sample_rate * spec.audio_frame_length_ms / 1000)
    frame_step = round(sample_rate * spec.audio_frame_step_ms / 1000)
    fft_length = spec.audio_fft_length
    pad_left = div(frame_length, 2)

    # Semicausal padding centres frame 0 at t=0. The right pad only tops up
    # degenerate short input; the unfold below never reads past a whole frame.
    samples = Enum.to_list(samples)
    total = pad_left + length(samples)
    right = max(frame_length + 1 - total, 0)

    samples =
      List.duplicate(0.0, pad_left) ++ samples ++ List.duplicate(0.0, right)

    t0 = System.monotonic_time(:millisecond)
    samples = Nx.tensor(samples, type: :f32)
    t1 = System.monotonic_time(:millisecond)

    frames = frame_signal(samples, frame_length, frame_step)
    t2 = System.monotonic_time(:millisecond)

    filterbank =
      cached_filterbank(spec.audio_mel_bins, fft_length, sample_rate, spec.audio_max_frequency)
      # cached storage is backend-neutral binary; the dot needs it local
      |> Nx.backend_copy(Nx.default_backend())

    window = hann_window(frame_length)
    t3 = System.monotonic_time(:millisecond)

    spectrum = magnitude_spectrum(frames, window, fft_length: fft_length)
    t4 = System.monotonic_time(:millisecond)

    result =
      spectrum
      |> Nx.dot(filterbank)
      |> log_compress(floor: spec.audio_mel_floor)

    t5 = System.monotonic_time(:millisecond)

    if t5 - t0 > 500 do
      require Logger

      Logger.debug(
        "mel: slow extract tensor=#{t1 - t0}ms frame=#{t2 - t1}ms window=#{t3 - t2}ms " <>
          "fft=#{t4 - t3}ms dot=#{t5 - t4}ms backend=#{inspect(Nx.default_backend())} " <>
          "defn=#{inspect(Nx.Defn.default_options())}"
      )
    end

    result
  end

  @doc """
  Number of mel frames a sample count produces.

  The reference unfolds `frame_length + 1` wide windows over the semicausally
  padded signal, so a frame exists for every full window, and short input is
  padded up to one window rather than dropped.
  """
  def frame_count(sample_count, %Spec{} = spec, opts \\ []) do
    sample_rate = Keyword.get(opts, :sample_rate, 16_000)
    frame_length = round(sample_rate * spec.audio_frame_length_ms / 1000)
    frame_step = round(sample_rate * spec.audio_frame_step_ms / 1000)

    total = max(sample_count + div(frame_length, 2), frame_length + 1)
    div(total - (frame_length + 1), frame_step) + 1
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
  # tensor per frame. The count follows the reference's frame_length + 1 wide
  # unfold (the extra sample feeds preemphasis, which E4B has disabled), so a
  # window short of that final sample is dropped even though the frame itself
  # would fit.
  defp frame_signal(samples, frame_length, frame_step) do
    total = Nx.axis_size(samples, 0)

    count = div(total - (frame_length + 1), frame_step) + 1

    offsets = Nx.iota({count, 1}) |> Nx.multiply(frame_step)
    indices = Nx.add(offsets, Nx.iota({1, frame_length}))

    Nx.take(samples, indices)
  end

  # The reference takes the magnitude, not the power: log-mel is built on
  # |X|, so squaring here would double every value in the log domain.
  defnp magnitude_spectrum(frames, window, opts \\ []) do
    opts = keyword!(opts, [:fft_length, mode: :inference])
    fft_length = opts[:fft_length]
    frame_length = Nx.axis_size(window, 0)

    windowed = frames * window

    padded =
      Nx.pad(windowed, 0.0, [{0, 0, 0}, {0, fft_length - frame_length, 0}])

    padded
    |> Nx.fft(length: fft_length)
    |> then(fn spectrum ->
      # keep the non-redundant half
      half = div(fft_length, 2) + 1
      spectrum = spectrum[[.., 0..(half - 1)//1]]
      Nx.sqrt(Nx.real(spectrum) ** 2 + Nx.imag(spectrum) ** 2)
    end)
  end

  # Built outside defn: the length is a shape, not a runtime value.
  # Periodic Hann (divide by N, not N - 1), matching the reference's
  # signal.hann_window default.
  defp hann_window(length) do
    positions = Nx.iota({length}, type: :f32)

    Nx.subtract(
      0.5,
      Nx.multiply(0.5, Nx.cos(Nx.multiply(2 * :math.pi() / length, positions)))
    )
  end

  # mel_floor is added before the log, so quiet bins approach log(floor)
  # smoothly rather than clipping at it.
  defnp log_compress(mel, opts \\ []) do
    opts = keyword!(opts, [:floor, mode: :inference])
    Nx.log(mel + opts[:floor])
  end

  # The filterbank is a pure function of four scalars but costs a 257 x 128
  # Elixir comprehension to build, so one copy is kept per configuration.
  defp cached_filterbank(mel_bins, fft_length, sample_rate, high_hz) do
    key = {__MODULE__, :filterbank, mel_bins, fft_length, sample_rate, high_hz}

    case :persistent_term.get(key, nil) do
      nil ->
        bank =
          mel_filterbank(mel_bins, fft_length, sample_rate, high_hz: high_hz)
          |> Nx.backend_copy(Nx.BinaryBackend)

        :persistent_term.put(key, bank)
        bank

      bank ->
        bank
    end
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
end
