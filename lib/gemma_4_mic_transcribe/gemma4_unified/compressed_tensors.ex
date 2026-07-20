defmodule Gemma4MicTranscribe.Gemma4Unified.CompressedTensors do
  @moduledoc false

  @int4_values_per_i32 8
  @quant_group_size 32
  @nibble_shifts for index <- 0..(@int4_values_per_i32 - 1), do: index * 4

  @doc """
  Repacks checkpoint weights into the layout the int4 GEMV kernel reads.

  Checkpoints store `{out_features, in_features / 8}` with 8 nibbles per word
  along `in`. The kernel indexes `packed[word * n + col]`, so it needs
  `{in_features / 8, out_features}` with the nibbles along `in` instead, which
  requires unpacking, transposing, and repacking rather than a plain transpose.

  Nibbles stay biased by 8, exactly as stored, so the kernel's `- 8` still
  applies.
  """
  def repack_kernel([packed]) do
    {out_features, packed_cols} = Nx.shape(packed)
    in_features = packed_cols * @int4_values_per_i32

    nibbles =
      @nibble_shifts
      |> Enum.map(fn shift -> packed |> Nx.right_shift(shift) |> Nx.bitwise_and(0xF) end)
      |> Nx.stack(axis: -1)
      |> Nx.reshape({out_features, in_features})
      |> Nx.transpose()
      |> Nx.reshape({div(in_features, @int4_values_per_i32), @int4_values_per_i32, out_features})

    # Accumulate in u32: the top nibble shifted by 28 does not fit in s32.
    multipliers =
      @nibble_shifts
      |> Enum.map(&:erlang.bsl(1, &1))
      |> Nx.tensor(type: :u32)
      |> Nx.reshape({1, @int4_values_per_i32, 1})

    nibbles
    |> Nx.as_type(:u32)
    |> Nx.multiply(multipliers)
    |> Nx.sum(axes: [1])
    |> Nx.bitcast(:s32)
  end

  @doc """
  Transposes checkpoint scales `{out_features, in_features / group_size}` into
  the `{in_features / group_size, out_features}` layout the kernel indexes.
  """
  def repack_scales([scales]) do
    scales |> Nx.transpose() |> Nx.as_type({:bf, 16})
  end

  def quant_group_size, do: @quant_group_size

  def linear_kernel([packed, scales]) do
    {out_features, packed_cols} = Nx.shape(packed)
    {_out_features, scale_cols} = Nx.shape(scales)

    unpacked =
      @nibble_shifts
      |> Enum.map(&biased_int4_at(packed, &1))
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

  defp biased_int4_at(packed, shift) do
    packed
    |> Nx.right_shift(shift)
    |> Nx.bitwise_and(0xF)
    |> Nx.subtract(8)
  end
end
