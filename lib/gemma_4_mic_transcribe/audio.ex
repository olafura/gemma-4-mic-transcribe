defmodule Gemma4MicTranscribe.Audio do
  @moduledoc false

  defmodule Window do
    @moduledoc false
    defstruct [:samples, :start_frame, :end_frame, :sample_rate]
  end

  def windows_from_samples(samples, sample_rate, window_seconds, stride_seconds) do
    validate_positive!(sample_rate, "sample_rate")
    validate_positive!(window_seconds, "window_seconds")
    validate_positive!(stride_seconds, "stride_seconds")

    samples = Enum.to_list(samples)
    total_frames = length(samples)
    window_frames = max(1, trunc(sample_rate * window_seconds))
    stride_frames = max(1, trunc(sample_rate * stride_seconds))

    do_windows(samples, total_frames, sample_rate, window_frames, stride_frames, 0, [])
  end

  def stream_wav_windows(path, sample_rate, window_seconds, stride_seconds) do
    path
    |> stream_f32le_audio(sample_rate)
    |> Stream.flat_map(&binary_to_f32_samples/1)
    |> Enum.to_list()
    |> windows_from_samples(sample_rate, window_seconds, stride_seconds)
  end

  def stream_f32le_audio(path, sample_rate) do
    ensure_boombox_started!()

    Boombox.run(
      input: path,
      output:
        {:stream,
         video: false,
         audio: :binary,
         audio_rate: sample_rate,
         audio_channels: 1,
         audio_format: :f32le}
    )
    |> Stream.filter(&match?(%Boombox.Packet{kind: :audio}, &1))
    |> Stream.map(& &1.payload)
  end

  def binary_to_f32_samples(binary) when is_binary(binary) do
    for <<sample::little-float-32 <- binary>>, do: sample
  end

  def frames_to_timestamp(frames, sample_rate) when sample_rate > 0 do
    seconds = frames / sample_rate
    minutes = trunc(seconds / 60)
    remaining = seconds - minutes * 60

    seconds_text =
      :io_lib.format("~.1f", [remaining])
      |> IO.iodata_to_binary()
      |> String.pad_leading(4, "0")

    String.pad_leading(Integer.to_string(minutes), 2, "0") <> ":" <> seconds_text
  end

  defp do_windows(_samples, 0, _sample_rate, _window_frames, _stride_frames, _start_frame, []),
    do: []

  defp do_windows(
         samples,
         total_frames,
         sample_rate,
         window_frames,
         stride_frames,
         start_frame,
         acc
       )
       when start_frame < total_frames do
    end_frame = min(total_frames, start_frame + window_frames)

    window = %Window{
      samples: Enum.slice(samples, start_frame, end_frame - start_frame),
      start_frame: start_frame,
      end_frame: end_frame,
      sample_rate: sample_rate
    }

    if end_frame == total_frames do
      Enum.reverse([window | acc])
    else
      do_windows(
        samples,
        total_frames,
        sample_rate,
        window_frames,
        stride_frames,
        start_frame + stride_frames,
        [
          window | acc
        ]
      )
    end
  end

  defp do_windows(
         _samples,
         _total_frames,
         _sample_rate,
         _window_frames,
         _stride_frames,
         _start_frame,
         acc
       ) do
    Enum.reverse(acc)
  end

  defp validate_positive!(value, _name) when is_number(value) and value > 0, do: :ok
  defp validate_positive!(_value, name), do: raise(ArgumentError, "#{name} must be positive")

  defp ensure_boombox_started! do
    case Application.ensure_all_started(:boombox) do
      {:ok, _apps} -> :ok
      {:error, reason} -> raise "failed to start Boombox: #{inspect(reason)}"
    end
  end
end
