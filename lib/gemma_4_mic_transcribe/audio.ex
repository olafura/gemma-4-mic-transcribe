defmodule Gemma4MicTranscribe.Audio do
  @moduledoc false

  defmodule Window do
    @moduledoc false
    defstruct [:samples, :start_frame, :end_frame, :sample_rate]
  end

  def windows_from_samples(samples, sample_rate, window_seconds, stride_seconds) do
    samples
    |> stream_windows_from_samples(sample_rate, window_seconds, stride_seconds)
    |> Enum.to_list()
  end

  def stream_windows_from_samples(samples, sample_rate, window_seconds, stride_seconds) do
    validate_positive!(sample_rate, "sample_rate")
    validate_positive!(window_seconds, "window_seconds")
    validate_positive!(stride_seconds, "stride_seconds")

    samples = Enum.to_list(samples)
    total_frames = length(samples)
    window_frames = max(1, trunc(sample_rate * window_seconds))
    stride_frames = max(1, trunc(sample_rate * stride_seconds))

    if total_frames == 0 do
      []
    else
      Stream.unfold(0, fn
        start_frame when start_frame < total_frames ->
          end_frame = min(total_frames, start_frame + window_frames)

          window = %Window{
            samples: Enum.slice(samples, start_frame, end_frame - start_frame),
            start_frame: start_frame,
            end_frame: end_frame,
            sample_rate: sample_rate
          }

          next_start_frame =
            if end_frame == total_frames do
              total_frames
            else
              start_frame + stride_frames
            end

          {window, next_start_frame}

        _start_frame ->
          nil
      end)
    end
  end

  def stream_wav_windows(path, sample_rate, window_seconds, stride_seconds) do
    path
    |> read_wav_samples!(sample_rate)
    |> stream_windows_from_samples(sample_rate, window_seconds, stride_seconds)
  end

  def stream_sample_chunks(samples, sample_rate, chunk_ms) do
    validate_positive!(sample_rate, "sample_rate")
    validate_positive!(chunk_ms, "chunk_ms")

    chunk_frames = max(1, round(sample_rate * chunk_ms / 1000))

    samples = Enum.to_list(samples)

    Stream.unfold({samples, 0}, fn
      {[], _start_frame} ->
        nil

      {remaining, start_frame} ->
        {chunk, rest} = Enum.split(remaining, chunk_frames)
        timestamp_ms = start_frame * 1000 / sample_rate
        {{chunk, timestamp_ms}, {rest, start_frame + length(chunk)}}
    end)
  end

  def read_wav_samples!(path, target_sample_rate) do
    path
    |> File.read!()
    |> decode_wav!(target_sample_rate)
  end

  def decode_wav!(
        <<"RIFF", _riff_size::little-unsigned-integer-size(32), "WAVE", chunks::binary>>,
        target_sample_rate
      ) do
    validate_positive!(target_sample_rate, "target_sample_rate")

    %{fmt: fmt, data: data} = parse_wav_chunks(chunks, %{fmt: nil, data: nil})
    samples = data_to_mono_samples(data, fmt)

    resample(samples, fmt.sample_rate, target_sample_rate)
  end

  def decode_wav!(_binary, _target_sample_rate),
    do: raise(ArgumentError, "expected a RIFF/WAVE file")

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

  defp validate_positive!(value, _name) when is_number(value) and value > 0, do: :ok
  defp validate_positive!(_value, name), do: raise(ArgumentError, "#{name} must be positive")

  defp parse_wav_chunks(<<>>, %{fmt: nil}), do: raise(ArgumentError, "WAV fmt chunk not found")
  defp parse_wav_chunks(<<>>, %{data: nil}), do: raise(ArgumentError, "WAV data chunk not found")
  defp parse_wav_chunks(<<>>, chunks), do: chunks

  defp parse_wav_chunks(<<_trailing_padding>>, chunks), do: parse_wav_chunks(<<>>, chunks)

  defp parse_wav_chunks(
         <<chunk_id::binary-size(4), chunk_size::little-unsigned-integer-size(32), rest::binary>>,
         chunks
       )
       when byte_size(rest) >= chunk_size do
    <<payload::binary-size(^chunk_size), tail::binary>> = rest

    chunks =
      case chunk_id do
        "fmt " -> %{chunks | fmt: parse_fmt_chunk(payload)}
        "data" -> %{chunks | data: payload}
        _unknown -> chunks
      end

    parse_wav_chunks(drop_chunk_padding(tail, chunk_size), chunks)
  end

  defp parse_wav_chunks(_binary, _chunks), do: raise(ArgumentError, "truncated WAV chunk")

  defp drop_chunk_padding(<<_padding, tail::binary>>, chunk_size) when rem(chunk_size, 2) == 1,
    do: tail

  defp drop_chunk_padding(tail, _chunk_size), do: tail

  defp parse_fmt_chunk(
         <<audio_format::little-unsigned-integer-size(16),
           channels::little-unsigned-integer-size(16),
           sample_rate::little-unsigned-integer-size(32),
           _byte_rate::little-unsigned-integer-size(32),
           block_align::little-unsigned-integer-size(16),
           bits_per_sample::little-unsigned-integer-size(16), _rest::binary>>
       )
       when channels > 0 and sample_rate > 0 and block_align > 0 do
    %{
      audio_format: audio_format,
      channels: channels,
      sample_rate: sample_rate,
      block_align: block_align,
      bits_per_sample: bits_per_sample
    }
  end

  defp parse_fmt_chunk(_payload), do: raise(ArgumentError, "invalid WAV fmt chunk")

  defp data_to_mono_samples(data, %{
         audio_format: 1,
         bits_per_sample: 16,
         channels: channels,
         block_align: block_align
       }) do
    for <<frame::binary-size(^block_align) <- data>> do
      frame
      |> pcm16_frame_to_samples()
      |> average_channels(channels)
    end
  end

  defp data_to_mono_samples(data, %{
         audio_format: 3,
         bits_per_sample: 32,
         channels: channels,
         block_align: block_align
       }) do
    for <<frame::binary-size(^block_align) <- data>> do
      frame
      |> binary_to_f32_samples()
      |> average_channels(channels)
    end
  end

  defp data_to_mono_samples(_data, %{audio_format: format, bits_per_sample: bits}) do
    raise ArgumentError, "unsupported WAV format #{format} with #{bits} bits per sample"
  end

  defp pcm16_frame_to_samples(frame) do
    for <<sample::little-signed-integer-size(16) <- frame>>, do: sample / 32768.0
  end

  defp average_channels(samples, channels) do
    Enum.sum(Enum.take(samples, channels)) / channels
  end

  defp resample(samples, sample_rate, sample_rate), do: samples
  defp resample([], _source_sample_rate, _target_sample_rate), do: []

  defp resample(samples, source_sample_rate, target_sample_rate) do
    sample_count = length(samples)
    output_count = max(1, round(sample_count * target_sample_rate / source_sample_rate))
    source = List.to_tuple(samples)
    last_index = sample_count - 1

    for output_index <- 0..(output_count - 1) do
      source_position = output_index * source_sample_rate / target_sample_rate
      left_index = min(trunc(source_position), last_index)
      right_index = min(left_index + 1, last_index)
      fraction = source_position - left_index
      left = elem(source, left_index)
      right = elem(source, right_index)

      left + (right - left) * fraction
    end
  end

  defp ensure_boombox_started! do
    case Application.ensure_all_started(:boombox) do
      {:ok, _apps} -> :ok
      {:error, reason} -> raise "failed to start Boombox: #{inspect(reason)}"
    end
  end
end
