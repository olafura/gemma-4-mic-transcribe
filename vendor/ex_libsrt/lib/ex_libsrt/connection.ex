defmodule ExLibSRT.Connection do
  @moduledoc false

  @type t :: GenServer.server()

  defmodule Handler do
    @moduledoc false

    @type t :: module() | struct()
    @type connection_id :: non_neg_integer()
    @type stream_id :: String.t()
    @type state :: any()

    @callback init(t()) :: state()
    @callback handle_connected(connection_id(), stream_id(), state()) :: {:ok, state()} | :stop
    @callback handle_disconnected(state()) :: :ok
    @callback handle_data(binary(), state()) :: {:ok, state()} | :stop
  end

  def start(_handler), do: {:error, "SRT support is stubbed"}
  def stop(_handler), do: :ok
end
