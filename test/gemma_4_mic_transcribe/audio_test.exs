defmodule Gemma4MicTranscribe.AudioTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Audio

  test "windows_from_samples matches the old rolling wav split semantics" do
    windows = Audio.windows_from_samples(Enum.to_list(0..7), 4, 1.0, 0.5)

    assert Enum.map(windows, &{&1.start_frame, &1.end_frame}) == [{0, 4}, {2, 6}, {4, 8}]

    assert Enum.map(windows, & &1.samples) == [
             Enum.to_list(0..3),
             Enum.to_list(2..5),
             Enum.to_list(4..7)
           ]
  end

  test "stream_windows_from_samples can take an early prefix" do
    windows =
      0..10_000
      |> Audio.stream_windows_from_samples(4, 1.0, 0.5)
      |> Enum.take(2)

    assert Enum.map(windows, &{&1.start_frame, &1.end_frame}) == [{0, 4}, {2, 6}]
  end

  test "stream_sample_chunks emits timestamped chunks" do
    chunks = Audio.stream_sample_chunks(Enum.to_list(0..4), 10, 200.0) |> Enum.to_list()

    assert chunks == [
             {[0, 1], 0.0},
             {[2, 3], 200.0},
             {[4], 400.0}
           ]
  end

  test "frames_to_timestamp formats minute and fractional seconds" do
    assert Audio.frames_to_timestamp(100, 10) == "00:10.0"
    assert Audio.frames_to_timestamp(650, 10) == "01:05.0"
  end

  test "binary_to_f32_samples parses little-endian PCM float payloads" do
    binary = <<0.5::little-float-32, -1.0::little-float-32>>

    assert Audio.binary_to_f32_samples(binary) == [0.5, -1.0]
  end

  test "decode_wav! skips metadata chunks before PCM data" do
    wav =
      wav_binary([
        {"fmt ",
         <<1::little-unsigned-integer-size(16), 2::little-unsigned-integer-size(16),
           48_000::little-unsigned-integer-size(32), 192_000::little-unsigned-integer-size(32),
           4::little-unsigned-integer-size(16), 16::little-unsigned-integer-size(16)>>},
        {"LIST", "INFOx"},
        {"data",
         <<32767::little-signed-integer-size(16), 32767::little-signed-integer-size(16),
           -32768::little-signed-integer-size(16), -32768::little-signed-integer-size(16)>>}
      ])

    samples = Audio.decode_wav!(wav, 48_000)

    assert_in_delta Enum.at(samples, 0), 32767 / 32768.0, 0.000001
    assert Enum.at(samples, 1) == -1.0
  end

  test "decode_wav! resamples PCM data to the target sample rate" do
    wav =
      wav_binary([
        {"fmt ",
         <<1::little-unsigned-integer-size(16), 1::little-unsigned-integer-size(16),
           48_000::little-unsigned-integer-size(32), 96_000::little-unsigned-integer-size(32),
           2::little-unsigned-integer-size(16), 16::little-unsigned-integer-size(16)>>},
        {"data",
         <<0::little-signed-integer-size(16), 4096::little-signed-integer-size(16),
           8192::little-signed-integer-size(16)>>}
      ])

    assert Audio.decode_wav!(wav, 16_000) == [0.0]
  end

  defp wav_binary(chunks) do
    chunks =
      chunks
      |> Enum.map(fn {id, payload} ->
        padding = if rem(byte_size(payload), 2) == 1, do: <<0>>, else: <<>>

        <<id::binary-size(4), byte_size(payload)::little-unsigned-integer-size(32),
          payload::binary, padding::binary>>
      end)
      |> IO.iodata_to_binary()

    <<"RIFF", byte_size("WAVE" <> chunks)::little-unsigned-integer-size(32), "WAVE",
      chunks::binary>>
  end
end
