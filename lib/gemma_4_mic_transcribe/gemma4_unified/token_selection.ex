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

  def next_token_id_from_sequence(logits, suppression_mask) do
    with_tensor_backend(suppression_mask, fn ->
      logits
      |> next_token_id_tensor(suppression_mask)
      |> Nx.backend_copy(Nx.BinaryBackend)
      |> Nx.to_flat_list()
      |> List.last()
    end)
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
