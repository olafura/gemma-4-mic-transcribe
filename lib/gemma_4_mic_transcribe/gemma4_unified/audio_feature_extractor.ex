defmodule Gemma4MicTranscribe.Gemma4Unified.AudioFeatureExtractor do
  @moduledoc false

  @samples_per_token 640
  @default_max_tokens 750

  def samples_per_token, do: @samples_per_token
  def default_max_tokens, do: @default_max_tokens

  def extract(samples, opts \\ []) do
    samples_per_token = Keyword.get(opts, :samples_per_token, @samples_per_token)
    max_tokens = Keyword.get(opts, :max_tokens, @default_max_tokens)

    samples = samples |> Enum.to_list() |> truncate_samples(samples_per_token, max_tokens)
    token_count = ceil_div(length(samples), samples_per_token)
    padded = samples ++ List.duplicate(0.0, token_count * samples_per_token - length(samples))

    frames =
      padded
      |> Enum.chunk_every(samples_per_token)
      |> Nx.tensor(type: {:f, 32})

    attention_mask =
      List.duplicate(1, token_count)
      |> Nx.tensor(type: {:s, 64})

    %{
      input_features: frames,
      attention_mask: attention_mask,
      token_count: token_count,
      samples_per_token: samples_per_token
    }
  end

  defp truncate_samples(samples, _samples_per_token, nil), do: samples

  defp truncate_samples(samples, samples_per_token, max_tokens)
       when is_integer(max_tokens) and max_tokens > 0 do
    Enum.take(samples, samples_per_token * max_tokens)
  end

  defp ceil_div(0, _denominator), do: 0
  defp ceil_div(numerator, denominator), do: div(numerator + denominator - 1, denominator)
end
