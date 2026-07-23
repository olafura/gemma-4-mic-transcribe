defmodule Gemma4MicTranscribe.Gemma4.ExtractedGeneratorServer do
  @moduledoc """
  Long-running owner for an independently extracted Gemma 4 model.

  Model parameters, compiled XLA functions, sparse decoder shells, and the
  routed-expert LRU remain owned by this process across calls. Each
  `generate/2` call still creates and releases its own K/V caches.
  """

  use GenServer

  alias Gemma4MicTranscribe.Gemma4.ExtractedGenerator

  defstruct [
    :generator,
    :model,
    requests: 0,
    total_processing_us: 0
  ]

  @doc "Starts a persistent extracted-model owner."
  def start_link(opts) do
    {server_opts, load_opts} = Keyword.split(opts, [:name, :timeout])
    GenServer.start_link(__MODULE__, load_opts, Keyword.take(server_opts, [:name]))
  end

  @doc "Generates with fresh per-request token and K/V state."
  def generate(server, opts) do
    GenServer.call(server, {:generate, opts}, :infinity)
  end

  @doc "Returns cumulative request and retained-resource measurements."
  def stats(server) do
    GenServer.call(server, :stats)
  end

  @impl true
  def init(opts) do
    generator = Keyword.get(opts, :generator, ExtractedGenerator)
    model = Keyword.get_lazy(opts, :model, fn -> generator.load!(opts) end)

    {:ok, %__MODULE__{generator: generator, model: model}}
  end

  @impl true
  def handle_call({:generate, opts}, _from, state) do
    {result, model} = state.generator.generate!(state.model, opts)
    processing_us = Map.get(result, :elapsed_us, 0)

    state = %{
      state
      | model: model,
        requests: state.requests + 1,
        total_processing_us: state.total_processing_us + processing_us
    }

    {:reply, result, state}
  end

  def handle_call(:stats, _from, state) do
    mean_processing_us =
      if state.requests == 0 do
        0.0
      else
        state.total_processing_us / state.requests
      end

    stats =
      state.generator.stats(state.model)
      |> Map.merge(%{
        requests: state.requests,
        total_processing_us: state.total_processing_us,
        mean_processing_us: mean_processing_us
      })

    {:reply, stats, state}
  end

  @impl true
  def terminate(_reason, state) do
    state.generator.release(state.model)
  end
end
