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
    logits
    |> next_token_id_tensor(suppression_mask)
    |> Nx.to_number()
  end

  def top_tokens(logits, suppression_mask, count) when is_integer(count) and count > 0 do
    {values, indices} = top_tokens_tensor(logits, suppression_mask, k: count)

    indices
    |> Nx.to_flat_list()
    |> Enum.zip(Nx.to_flat_list(values))
  end

  defn next_token_id_tensor(logits, suppression_mask) do
    replacement = Nx.Constants.min_finite(Nx.type(logits))
    logits = Nx.select(suppression_mask == 1, replacement, logits)
    Nx.argmax(logits, axis: -1)
  end

  defn top_tokens_tensor(logits, suppression_mask, opts \\ []) do
    opts = keyword!(opts, [:k])
    replacement = Nx.Constants.min_finite(Nx.type(logits))
    logits = Nx.select(suppression_mask == 1, replacement, logits)

    Nx.top_k(logits, k: opts[:k])
  end
end
