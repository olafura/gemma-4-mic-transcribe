defmodule ExLibSRT.Server do
  @moduledoc false

  @type t :: pid()
  @type connection_id :: non_neg_integer()
  @type srt_server_conn :: {:srt_server_conn, connection_id(), stream_id :: String.t()}
  @type srt_server_conn_closed :: {:srt_server_conn_closed, connection_id()}
  @type srt_server_error :: {:srt_server_error, connection_id(), error :: String.t()}
  @type srt_data :: {:srt_data, connection_id(), data :: binary()}
  @type srt_server_connect_request ::
          {:srt_server_connect_request, address :: String.t(), stream_id :: String.t()}

  @spec start(String.t(), non_neg_integer()) :: {:ok, t()} | {:error, String.t(), integer()}
  @spec start(String.t(), non_neg_integer(), String.t()) ::
          {:ok, t()} | {:error, String.t(), integer()}
  def start(_address, _port, _password \\ "") do
    if stub_success?(), do: {:ok, self()}, else: {:error, unsupported(), 0}
  end

  @spec start_link(String.t(), non_neg_integer(), String.t(), integer()) ::
          {:ok, t()} | {:error, String.t(), integer()}
  def start_link(_address, _port, _password \\ "", _latency_ms \\ -1) do
    if stub_success?(), do: {:ok, self()}, else: {:error, unsupported(), 0}
  end

  @spec stop(t()) :: :ok | {:error, String.t()}
  def stop(_server), do: :ok

  @spec accept_awaiting_connect_request(t()) :: :ok | {:error, String.t()}
  def accept_awaiting_connect_request(_server) do
    if stub_success?(), do: :ok, else: {:error, unsupported()}
  end

  @spec accept_awaiting_connect_request_with_handler(ExLibSRT.Connection.Handler.t(), t()) ::
          {:ok, ExLibSRT.Connection.t()} | {:error, any()}
  def accept_awaiting_connect_request_with_handler(handler, _server) do
    if stub_success?(), do: ExLibSRT.Connection.start(handler), else: {:error, unsupported()}
  end

  @spec reject_awaiting_connect_request(t()) :: :ok | {:error, String.t()}
  def reject_awaiting_connect_request(_server) do
    if stub_success?(), do: :ok, else: {:error, unsupported()}
  end

  @spec close_server_connection(connection_id(), t()) :: :ok | {:error, String.t()}
  def close_server_connection(_connection_id, _server) do
    if stub_success?(), do: :ok, else: {:error, unsupported()}
  end

  @spec read_socket_stats(connection_id(), t()) ::
          {:ok, ExLibSRT.SocketStats.t()} | {:error, String.t()}
  def read_socket_stats(_connection_id, _server) do
    if stub_success?(), do: {:ok, %ExLibSRT.SocketStats{}}, else: {:error, unsupported()}
  end

  defp unsupported do
    "SRT support is stubbed in gemma_4_mic_transcribe; this CLI only uses Boombox file audio"
  end

  defp stub_success? do
    Application.get_env(:ex_libsrt, :stub_success, false) == true
  end
end
