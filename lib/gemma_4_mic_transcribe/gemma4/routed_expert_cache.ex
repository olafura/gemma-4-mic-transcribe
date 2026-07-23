defmodule Gemma4MicTranscribe.Gemma4.RoutedExpertCache do
  @moduledoc """
  A bounded LRU of exact routed-expert matrices on an Nx backend.

  Entries are keyed by decoder layer and expert index. `checkout!/5` returns a
  compact route-ordered bank suitable for one-token MoE execution.
  """

  alias Gemma4MicTranscribe.Gemma4.MoeLayerArtifact

  defstruct entries: %{},
            bytes: 0,
            max_bytes: 0,
            clock: 0,
            hits: 0,
            misses: 0,
            evictions: 0

  @doc "Creates an empty expert cache with a byte limit."
  def new(max_bytes) when is_integer(max_bytes) and max_bytes >= 0 do
    %__MODULE__{max_bytes: max_bytes}
  end

  @doc "Returns route-ordered expert matrices and the updated cache."
  def checkout!(
        %__MODULE__{} = cache,
        artifact,
        manifest,
        expert_indices,
        backend
      )
      when is_list(expert_indices) and expert_indices != [] do
    layer = manifest.layer_index
    clock = cache.clock + 1
    keys = Enum.map(expert_indices, &{layer, &1})
    missing_keys = Enum.reject(keys, &Map.has_key?(cache.entries, &1))
    missing_indices = Enum.map(missing_keys, &elem(&1, 1))
    entry_bytes = expert_bytes(manifest)

    cache =
      %{cache | clock: clock}
      |> Map.update!(:hits, &(&1 + length(keys) - length(missing_keys)))
      |> Map.update!(:misses, &(&1 + length(missing_keys)))
      |> touch(keys, clock)

    cache =
      if missing_indices == [] do
        cache
      else
        {_manifest, loaded} =
          MoeLayerArtifact.load_routed_experts!(
            artifact,
            missing_indices,
            backend,
            verify_checksum: false
          )

        cache =
          missing_keys
          |> Enum.with_index()
          |> Enum.reduce(cache, fn {key, position}, cache ->
            entry = %{
              experts_gate_up:
                loaded.experts_gate_up
                |> Nx.slice_along_axis(position, 1, axis: 0)
                |> Nx.backend_copy(backend),
              experts_down:
                loaded.experts_down
                |> Nx.slice_along_axis(position, 1, axis: 0)
                |> Nx.backend_copy(backend),
              bytes: entry_bytes,
              last_used: clock
            }

            %{
              cache
              | entries: Map.put(cache.entries, key, entry),
                bytes: cache.bytes + entry_bytes
            }
          end)

        Nx.backend_deallocate(loaded)
        evict(cache, MapSet.new(keys))
      end

    params = compact_params!(cache, keys)
    {params, cache}
  end

  @doc "Returns cumulative cache measurements."
  def stats(%__MODULE__{} = cache) do
    requests = cache.hits + cache.misses

    %{
      entries: map_size(cache.entries),
      bytes: cache.bytes,
      max_bytes: cache.max_bytes,
      hits: cache.hits,
      misses: cache.misses,
      evictions: cache.evictions,
      hit_rate: if(requests == 0, do: 0.0, else: cache.hits / requests)
    }
  end

  @doc "Explicitly releases every cached backend tensor."
  def release(%__MODULE__{} = cache) do
    Enum.each(cache.entries, fn {_key, entry} ->
      Nx.backend_deallocate(entry.experts_gate_up)
      Nx.backend_deallocate(entry.experts_down)
    end)

    :ok
  end

  defp touch(cache, keys, clock) do
    entries =
      Enum.reduce(keys, cache.entries, fn key, entries ->
        case entries do
          %{^key => entry} -> Map.put(entries, key, %{entry | last_used: clock})
          %{} -> entries
        end
      end)

    %{cache | entries: entries}
  end

  defp evict(cache, protected) do
    if cache.bytes <= cache.max_bytes do
      cache
    else
      candidate =
        cache.entries
        |> Enum.reject(fn {key, _entry} -> MapSet.member?(protected, key) end)
        |> Enum.min_by(fn {_key, entry} -> entry.last_used end, fn -> nil end)

      case candidate do
        nil ->
          cache

        {key, entry} ->
          Nx.backend_deallocate(entry.experts_gate_up)
          Nx.backend_deallocate(entry.experts_down)

          evict(
            %{
              cache
              | entries: Map.delete(cache.entries, key),
                bytes: cache.bytes - entry.bytes,
                evictions: cache.evictions + 1
            },
            protected
          )
      end
    end
  end

  defp compact_params!(cache, keys) do
    entries = Enum.map(keys, &Map.fetch!(cache.entries, &1))

    %{
      experts_gate_up:
        Nx.concatenate(
          Enum.map(entries, &Nx.backend_copy(&1.experts_gate_up)),
          axis: 0
        ),
      experts_down:
        Nx.concatenate(
          Enum.map(entries, &Nx.backend_copy(&1.experts_down)),
          axis: 0
        )
    }
  end

  defp expert_bytes(manifest) do
    Enum.reduce(["experts_gate_up", "experts_down"], 0, fn name, bytes ->
      metadata = Map.fetch!(manifest.tensors, name)
      bytes + div(metadata.byte_size, manifest.num_experts)
    end)
  end
end
