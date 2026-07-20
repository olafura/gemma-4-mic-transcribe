defmodule Gemma4MicTranscribe.WebRTC.TestHarness do
  @moduledoc false

  alias Gemma4MicTranscribe.Audio
  alias Gemma4MicTranscribe.StreamingSession

  def available? do
    Code.ensure_loaded?(ExWebRTC.PeerConnection)
  end

  def push_audio_payload(session, payload, timestamp_ms, opts \\ [])

  def push_audio_payload(session, payload, timestamp_ms, opts) when is_binary(payload) do
    format = Keyword.get(opts, :format, :f32le)

    samples =
      case format do
        :f32le -> Audio.binary_to_f32_samples(payload)
      end

    StreamingSession.push_audio(session, samples, timestamp_ms)
  end

  def push_audio_payload(session, samples, timestamp_ms, _opts) when is_list(samples) do
    StreamingSession.push_audio(session, samples, timestamp_ms)
  end
end
