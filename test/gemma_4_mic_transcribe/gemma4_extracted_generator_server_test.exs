defmodule Gemma4MicTranscribe.Gemma4.ExtractedGeneratorServerTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4.ExtractedGeneratorServer

  defmodule FakeGenerator do
    def load!(opts) do
      %{owner: Keyword.fetch!(opts, :owner), generations: 0, cache_entries: 0}
    end

    def generate!(model, opts) do
      send(model.owner, {:generated, model.generations, opts})

      updated = %{
        model
        | generations: model.generations + 1,
          cache_entries: model.cache_entries + Keyword.get(opts, :new_experts, 0)
      }

      {%{generated_text: "request-#{updated.generations}", elapsed_us: 100}, updated}
    end

    def stats(model) do
      %{generations: model.generations, expert_cache: %{entries: model.cache_entries}}
    end

    def release(model) do
      send(model.owner, {:released, model.generations, model.cache_entries})
      :ok
    end
  end

  test "retains model state across requests and releases it on shutdown" do
    {:ok, server} =
      ExtractedGeneratorServer.start_link(generator: FakeGenerator, owner: self())

    assert %{generated_text: "request-1"} =
             ExtractedGeneratorServer.generate(server, new_experts: 3)

    assert_received {:generated, 0, [new_experts: 3]}

    assert %{generated_text: "request-2"} =
             ExtractedGeneratorServer.generate(server, new_experts: 2)

    assert_received {:generated, 1, [new_experts: 2]}

    assert %{
             requests: 2,
             total_processing_us: 200,
             mean_processing_us: 100.0,
             generations: 2,
             expert_cache: %{entries: 5}
           } = ExtractedGeneratorServer.stats(server)

    :ok = GenServer.stop(server)
    assert_received {:released, 2, 5}
  end

  test "accepts an already loaded model" do
    model = %{owner: self(), generations: 4, cache_entries: 9}

    {:ok, server} =
      ExtractedGeneratorServer.start_link(generator: FakeGenerator, model: model)

    assert %{generated_text: "request-5"} = ExtractedGeneratorServer.generate(server, [])
    :ok = GenServer.stop(server)
    assert_received {:released, 5, 9}
  end
end
