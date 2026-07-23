defmodule Gemma4MicTranscribe.Gemma4.RoutedExpertCache do
  @moduledoc """
  A bounded LRU of exact routed-expert matrices on an Nx backend.

  Entries are keyed by decoder layer and expert index. Their tensors are stored
  in one contiguous GPU table per layer, so cached execution can index the
  resident table directly without copying and concatenating eight matrices on
  every token.
  """

  alias Gemma4MicTranscribe.Gemma4.MoeLayerArtifact

  defstruct entries: %{},
            tables: %{},
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
    {table_params, slots, cache} =
      checkout_table!(cache, artifact, manifest, expert_indices, backend)

    indices =
      slots
      |> Nx.tensor(type: :s64)
      |> transfer(backend)

    params = %{
      experts_gate_up:
        table_params.experts_gate_up
        |> Nx.take(indices, axis: 0)
        |> Nx.backend_copy(backend),
      experts_down:
        table_params.experts_down
        |> Nx.take(indices, axis: 0)
        |> Nx.backend_copy(backend)
    }

    Nx.backend_deallocate(indices)
    {params, cache}
  end

  @doc """
  Returns a resident per-layer expert table, requested table slots, and the
  updated cache.

  The returned table remains owned by the cache and must not be deallocated by
  the caller.
  """
  def checkout_table!(
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
    missing_keys = keys |> Enum.reject(&Map.has_key?(cache.entries, &1)) |> Enum.uniq()
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

        cache = append_to_table(cache, layer, missing_keys, loaded, entry_bytes, clock, backend)
        evict(cache, MapSet.new(keys))
      end

    table = Map.fetch!(cache.tables, layer)
    slots = Enum.map(expert_indices, &Map.fetch!(table.slots, &1))
    {table.params, slots, cache}
  end

  @doc "Returns cumulative cache measurements."
  def stats(%__MODULE__{} = cache) do
    requests = cache.hits + cache.misses

    %{
      entries: map_size(cache.entries),
      tables: map_size(cache.tables),
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
    Enum.each(cache.tables, fn {_layer, table} ->
      Nx.backend_deallocate(table.params)
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
          evict(
            remove_entry(cache, key, entry),
            protected
          )
      end
    end
  end

  defp append_to_table(cache, layer, missing_keys, loaded, entry_bytes, clock, backend) do
    {expert_ids, params} =
      case Map.get(cache.tables, layer) do
        nil ->
          {Enum.map(missing_keys, &elem(&1, 1)), copy_params(loaded, backend)}

        table ->
          params = %{
            experts_gate_up:
              Nx.concatenate(
                [
                  Nx.backend_copy(table.params.experts_gate_up, backend),
                  Nx.backend_copy(loaded.experts_gate_up, backend)
                ],
                axis: 0
              ),
            experts_down:
              Nx.concatenate(
                [
                  Nx.backend_copy(table.params.experts_down, backend),
                  Nx.backend_copy(loaded.experts_down, backend)
                ],
                axis: 0
              )
          }

          Nx.backend_deallocate(table.params)
          {table.expert_ids ++ Enum.map(missing_keys, &elem(&1, 1)), params}
      end

    Nx.backend_deallocate(loaded)

    slots =
      expert_ids
      |> Enum.with_index()
      |> Map.new()

    entries =
      Enum.reduce(missing_keys, cache.entries, fn key, entries ->
        Map.put(entries, key, %{bytes: entry_bytes, last_used: clock})
      end)

    table = %{expert_ids: expert_ids, slots: slots, params: params, backend: backend}

    %{
      cache
      | entries: entries,
        tables: Map.put(cache.tables, layer, table),
        bytes: cache.bytes + length(missing_keys) * entry_bytes
    }
  end

  defp remove_entry(cache, {layer, expert_index} = key, entry) do
    table = Map.fetch!(cache.tables, layer)
    remaining_ids = List.delete(table.expert_ids, expert_index)

    tables =
      case remaining_ids do
        [] ->
          Nx.backend_deallocate(table.params)
          Map.delete(cache.tables, layer)

        remaining_ids ->
          positions = Enum.map(remaining_ids, &Map.fetch!(table.slots, &1))

          indices =
            positions
            |> Nx.tensor(type: :s64)
            |> transfer(table.backend)

          params = %{
            experts_gate_up:
              table.params.experts_gate_up
              |> Nx.take(indices, axis: 0)
              |> Nx.backend_copy(table.backend),
            experts_down:
              table.params.experts_down
              |> Nx.take(indices, axis: 0)
              |> Nx.backend_copy(table.backend)
          }

          Nx.backend_deallocate(indices)
          Nx.backend_deallocate(table.params)

          slots =
            remaining_ids
            |> Enum.with_index()
            |> Map.new()

          Map.put(cache.tables, layer, %{
            expert_ids: remaining_ids,
            slots: slots,
            params: params,
            backend: table.backend
          })
      end

    %{
      cache
      | entries: Map.delete(cache.entries, key),
        tables: tables,
        bytes: cache.bytes - entry.bytes,
        evictions: cache.evictions + 1
    }
  end

  defp copy_params(params, backend) do
    %{
      experts_gate_up: Nx.backend_copy(params.experts_gate_up, backend),
      experts_down: Nx.backend_copy(params.experts_down, backend)
    }
  end

  defp transfer(tensor, Nx.BinaryBackend), do: Nx.backend_copy(tensor, Nx.BinaryBackend)
  defp transfer(tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp expert_bytes(manifest) do
    Enum.reduce(["experts_gate_up", "experts_down"], 0, fn name, bytes ->
      metadata = Map.fetch!(manifest.tensors, name)
      bytes + div(metadata.byte_size, manifest.num_experts)
    end)
  end
end
