defmodule Gemma4MicTranscribe.Gemma4Unified.TokenSelection do
  @moduledoc false

  import Nx.Defn

  def suppression_mask(token_ids, vocab_size, backend) do
    token_id_set = MapSet.new(token_ids)

    0..(vocab_size - 1)
    |> Enum.map(fn token_id -> if MapSet.member?(token_id_set, token_id), do: 1, else: 0 end)
    |> Nx.tensor(type: :u8)
    |> Nx.backend_copy(backend)
  end

  def next_token_id(logits, suppression_mask) do
    with_tensor_backend(suppression_mask, fn ->
      logits
      |> next_token_id_tensor(suppression_mask)
      |> Nx.backend_copy(Nx.BinaryBackend)
      |> Nx.to_number()
    end)
  end

  @doc """
  Picks the best token from the last sequence position that is not in
  `banned_ids`, falling back to the overall best candidate when every
  candidate is banned.

  The ban set is tiny and changes every step, so instead of rebuilding a
  device-side mask (which would defeat executable reuse) the top candidates
  are pulled to the host and filtered there.
  """
  def next_allowed_token_id_from_sequence(logits, suppression_mask, banned_ids, top_k \\ 8)

  def next_allowed_token_id_from_sequence(logits, suppression_mask, [], _top_k) do
    next_token_id_from_sequence(logits, suppression_mask)
  end

  def next_allowed_token_id_from_sequence(logits, suppression_mask, banned_ids, top_k) do
    top_k = min(top_k, Nx.axis_size(logits, -1))
    candidates = top_tokens_from_sequence(logits, suppression_mask, top_k)
    banned = MapSet.new(banned_ids)

    case Enum.find(candidates, fn {token_id, _score} -> token_id not in banned end) do
      {token_id, _score} -> token_id
      nil -> candidates |> List.first() |> elem(0)
    end
  end

  def next_token_id_from_sequence(logits, suppression_mask) do
    with_tensor_backend(suppression_mask, fn ->
      logits
      |> next_token_id_tensor(suppression_mask)
      |> Nx.backend_copy(Nx.BinaryBackend)
      |> Nx.to_flat_list()
      |> List.last()
    end)
  end

  def next_token_with_margin_from_sequence(logits, suppression_mask, banned_ids \\ []) do
    count =
      if banned_ids == [],
        do: min(2, Nx.axis_size(logits, -1)),
        else: min(8, Nx.axis_size(logits, -1))

    banned = MapSet.new(banned_ids)

    allowed =
      logits
      |> top_tokens_from_sequence(suppression_mask, count)
      |> Enum.reject(fn {token_id, _score} -> MapSet.member?(banned, token_id) end)

    case allowed do
      [{token_id, score}, {_runner_up, runner_up_score} | _rest] ->
        {token_id, score - runner_up_score}

      [{token_id, _score}] ->
        {token_id, 0.0}

      [] ->
        {next_token_id_from_sequence(logits, suppression_mask), 0.0}
    end
  end

  def top_tokens(logits, suppression_mask, count) when is_integer(count) and count > 0 do
    with_tensor_backend(suppression_mask, fn ->
      {values, indices} = top_tokens_tensor(logits, suppression_mask, k: count)

      indices
      |> Nx.backend_copy(Nx.BinaryBackend)
      |> Nx.to_flat_list()
      |> Enum.zip(values |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list())
    end)
  end

  def top_tokens_from_sequence(logits, suppression_mask, count)
      when is_integer(count) and count > 0 do
    with_tensor_backend(suppression_mask, fn ->
      {values, indices} = top_tokens_tensor(logits, suppression_mask, k: count)

      indices =
        indices |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list() |> Enum.take(-count)

      values =
        values |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list() |> Enum.take(-count)

      Enum.zip(indices, values)
    end)
  end

  defn next_token_id_tensor(logits, suppression_mask) do
    replacement = Nx.Constants.min_finite(Nx.type(logits))
    suppression_mask = Nx.broadcast(suppression_mask, Nx.shape(logits))
    logits = Nx.select(suppression_mask == 1, replacement, logits)
    Nx.argmax(logits, axis: -1)
  end

  defn top_tokens_tensor(logits, suppression_mask, opts \\ []) do
    opts = keyword!(opts, [:k])
    replacement = Nx.Constants.min_finite(Nx.type(logits))
    suppression_mask = Nx.broadcast(suppression_mask, Nx.shape(logits))
    logits = Nx.select(suppression_mask == 1, replacement, logits)

    Nx.top_k(logits, k: opts[:k])
  end

  defp with_tensor_backend(tensor, fun) do
    Nx.with_default_backend(tensor_backend(tensor), fun)
  end

  defp tensor_backend(%Nx.Tensor{data: %EXLA.Backend{buffer: buffer}}) do
    {EXLA.Backend, client: buffer.client_name, device_id: buffer.device_id}
  end

  defp tensor_backend(%Nx.Tensor{data: data}), do: data.__struct__
end
