defmodule ExLibSRT.Client do
  @moduledoc false

  @type t :: pid()
  @type srt_client_started :: :srt_client_started
  @type srt_client_disconnected :: :srt_client_disconnected
  @type srt_client_error :: {:srt_client_error, reason :: String.t()}

  @spec start(String.t(), non_neg_integer(), String.t()) ::
          {:ok, t()} | {:error, String.t(), integer()}
  @spec start(String.t(), non_neg_integer(), String.t(), String.t()) ::
          {:ok, t()} | {:error, String.t(), integer()}
  def start(_address, _port, _stream_id, _password \\ "") do
    if stub_success?(), do: {:ok, self()}, else: {:error, unsupported(), 0}
  end

  @spec start_link(String.t(), non_neg_integer(), String.t(), String.t(), integer()) ::
          {:ok, t()} | {:error, String.t(), integer()}
  def start_link(_address, _port, _stream_id, _password \\ "", _latency_ms \\ -1) do
    if stub_success?(), do: {:ok, self()}, else: {:error, unsupported(), 0}
  end

  @spec stop(t()) :: :ok
  def stop(_client), do: :ok

  @spec send_data(binary(), t()) :: :ok | {:error, :payload_too_large | String.t()}
  def send_data(payload, _client) when byte_size(payload) > 1316, do: {:error, :payload_too_large}

  def send_data(_payload, _client) do
    if stub_success?(), do: :ok, else: {:error, unsupported()}
  end

  @spec read_socket_stats(t()) :: {:ok, ExLibSRT.SocketStats.t()} | {:error, String.t()}
  def read_socket_stats(_client) do
    if stub_success?(), do: {:ok, %ExLibSRT.SocketStats{}}, else: {:error, unsupported()}
  end

  defp unsupported do
    "SRT support is stubbed in gemma_4_mic_transcribe; this CLI only uses Boombox file audio"
  end

  defp stub_success? do
    Application.get_env(:ex_libsrt, :stub_success, false) == true
  end
end
