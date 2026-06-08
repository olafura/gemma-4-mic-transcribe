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

  test "frames_to_timestamp formats minute and fractional seconds" do
    assert Audio.frames_to_timestamp(100, 10) == "00:10.0"
    assert Audio.frames_to_timestamp(650, 10) == "01:05.0"
  end

  test "binary_to_f32_samples parses little-endian PCM float payloads" do
    binary = <<0.5::little-float-32, -1.0::little-float-32>>

    assert Audio.binary_to_f32_samples(binary) == [0.5, -1.0]
  end
end
