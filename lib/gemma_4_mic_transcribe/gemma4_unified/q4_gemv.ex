defmodule Gemma4MicTranscribe.Gemma4Unified.Q4Gemv do
  @moduledoc false

  # Weight-only int4 matrix-vector product.
  #
  # XLA's generic codegen dequantizes packed int4 weights to bf16 before the
  # GEMM, so decode reads the dequantized size instead of the packed size. On
  # ROCm this block lowers to the hand-written HIP kernel registered in
  # libexla.so, which reads packed nibbles directly. Everywhere else the
  # default implementation below runs, which is also the correctness oracle
  # the kernel is tested against.

  import Nx.Defn

  @nibbles_per_word 8

  defstruct [:group_size]

  @doc """
  Computes `x . dequantize(packed, scales)`.

    * `x` - `{k}` bf16 activations
    * `packed` - `{k / 8, n}` s32, 8 biased nibbles per word along k
    * `scales` - `{k / group_size, n}` bf16

  Returns `{n}` f32.
  """
  deftransform dot(x, packed, scales, opts \\ []) do
    group_size = Keyword.get(opts, :group_size, 32)
    {words, n} = Nx.shape(packed)
    k = words * @nibbles_per_word

    Nx.block(
      %__MODULE__{group_size: group_size},
      [x, packed, scales],
      Nx.template({n}, :f32),
      fn %__MODULE__{}, x, packed, scales ->
        dequantized_dot(x, packed, scales, group_size: group_size, k: k, n: n)
      end
    )
  end

  defnp dequantized_dot(x, packed, scales, opts \\ []) do
    opts = keyword!(opts, [:group_size, :k, :n])
    group_size = opts[:group_size]
    k = opts[:k]
    n = opts[:n]

    # {words, 1, n} >> {1, 8, 1} unpacks all nibbles without a comprehension,
    # which defn does not allow.
    shifts =
      Nx.iota({1, @nibbles_per_word, 1}, axis: 1, type: :s32)
      |> Nx.multiply(4)

    weights =
      packed
      |> Nx.new_axis(1)
      |> Nx.right_shift(shifts)
      |> Nx.bitwise_and(0xF)
      |> Nx.subtract(8)
      |> Nx.reshape({k, n})

    expanded_scales =
      scales
      |> Nx.new_axis(1)
      |> Nx.broadcast({div(k, group_size), group_size, n})
      |> Nx.reshape({k, n})

    weights
    |> Nx.as_type(:f32)
    |> Nx.multiply(Nx.as_type(expanded_scales, :f32))
    |> then(&Nx.dot(Nx.as_type(x, :f32), &1))
  end
end

defimpl EXLA.CustomCall, for: Gemma4MicTranscribe.Gemma4Unified.Q4Gemv do
  def call(%{group_size: group_size}, _out, [x, packed, scales], %{platform: :rocm}) do
    with {:bf, 16} <- Nx.type(x),
         {:s, 32} <- Nx.type(packed),
         {:bf, 16} <- Nx.type(scales) do
      {:ok,
       %EXLA.CustomCall.Spec{
         call_target_name: "exla_q4_gemv",
         attributes: [{"group_size", "#{group_size} : i64"}]
       }}
    else
      _other -> :skip
    end
  end

  def call(_block, _out, _args, _client), do: :skip
end
