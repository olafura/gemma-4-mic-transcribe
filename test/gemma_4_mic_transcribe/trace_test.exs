defmodule Gemma4MicTranscribe.TraceTest do
  # :dbg tracing is node-global, so this cannot run alongside other tests.
  use ExUnit.Case, async: false

  alias Gemma4MicTranscribe.SpeechGate
  alias Gemma4MicTranscribe.Trace

  test "logs call durations for traced modules" do
    {:ok, device} = StringIO.open("")

    :ok = Trace.enable(modules: [SpeechGate], device: device, min_us: 0)

    try do
      analysis = SpeechGate.analyze(List.duplicate(0.5, 8_000), sample_rate: 16_000)
      assert analysis.speech?

      output = wait_for_output(device, "SpeechGate.analyze/2")
      assert output =~ "trace: "
      assert output =~ "Gemma4MicTranscribe.SpeechGate.analyze/2"
      assert output =~ "ms"
    after
      Trace.disable()
    end
  end

  defp wait_for_output(device, marker, attempts \\ 100) do
    {_input, output} = StringIO.contents(device)

    cond do
      output =~ marker ->
        output

      attempts == 0 ->
        output

      true ->
        Process.sleep(10)
        wait_for_output(device, marker, attempts - 1)
    end
  end
end
