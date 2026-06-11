defmodule Gemma4MicTranscribe.Gemma4Unified.CompressedTensors do
  @moduledoc false

  @int4_values_per_i32 8
  @quant_group_size 32
  @nibble_shifts for index <- 0..(@int4_values_per_i32 - 1), do: index * 4

  def linear_kernel([packed, scales]) do
    {out_features, packed_cols} = Nx.shape(packed)
    {_out_features, scale_cols} = Nx.shape(scales)

    unpacked =
      @nibble_shifts
      |> Enum.map(&signed_int4_at(packed, &1))
      |> Nx.stack(axis: -1)
      |> Nx.reshape({out_features, packed_cols * @int4_values_per_i32})

    expanded_scales =
      scales
      |> Nx.new_axis(-1)
      |> Nx.broadcast({out_features, scale_cols, @quant_group_size})
      |> Nx.reshape({out_features, scale_cols * @quant_group_size})

    unpacked
    |> Nx.as_type(Nx.type(scales))
    |> Nx.multiply(expanded_scales)
    |> Nx.transpose()
  end

  defp signed_int4_at(packed, shift) do
    nibble =
      packed
      |> Nx.right_shift(shift)
      |> Nx.bitwise_and(0xF)

    Nx.select(Nx.greater_equal(nibble, 8), Nx.subtract(nibble, 16), nibble)
  end
end
