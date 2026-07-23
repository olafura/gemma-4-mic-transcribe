alias Gemma4MicTranscribe.Gemma4Unified.Q4DualGemv
alias Gemma4MicTranscribe.Gemma4Unified.Q4Gemv

client =
  case System.get_env("EXLA_CLIENT", "cuda") do
    "cuda" -> :cuda
    "rocm" -> :rocm
    value -> raise "EXLA_CLIENT must be cuda or rocm, got: #{inspect(value)}"
  end

{:ok, _applications} = Application.ensure_all_started(:exla)

group_size = 32
units = 7

x =
  Nx.iota({group_size}, type: :f32)
  |> Nx.subtract(15)
  |> Nx.divide(16)
  |> Nx.as_type(:bf16)

matrix_x =
  Nx.iota({3, group_size}, type: :f32)
  |> Nx.subtract(47)
  |> Nx.divide(32)
  |> Nx.as_type(:bf16)

packed =
  Nx.iota({div(group_size, 8), units}, type: :s32)
  |> Nx.multiply(0x1020_304)
  |> Nx.add(0x7654_3210)

scales =
  Nx.iota({1, units}, type: :f32)
  |> Nx.add(1)
  |> Nx.divide(32)
  |> Nx.as_type(:bf16)

second_scales = Nx.multiply(scales, Nx.tensor(0.5, type: :bf16))

dot_reference =
  Nx.dot(Nx.as_type(x, :f32), Q4Gemv.dequantize(packed, scales, group_size))

matrix_reference =
  Nx.dot(
    Nx.as_type(matrix_x, :f32),
    Q4Gemv.dequantize(packed, scales, group_size)
  )

dual_reference =
  Nx.concatenate([
    dot_reference,
    Nx.dot(
      Nx.as_type(x, :f32),
      Q4Gemv.dequantize(packed, second_scales, group_size)
    )
  ])

jit_opts = [compiler: EXLA, client: client]

dot =
  Nx.Defn.jit(
    fn x, packed, scales ->
      Q4Gemv.dot(x, packed, scales, group_size: group_size)
    end,
    jit_opts
  )

matrix =
  Nx.Defn.jit(
    fn x, packed, scales ->
      Q4Gemv.matmul(x, packed, scales, group_size: group_size)
    end,
    jit_opts
  )

dual =
  Nx.Defn.jit(
    fn x, packed, scales, second_scales ->
      Q4DualGemv.dot(x, packed, scales, packed, second_scales, group_size: group_size)
    end,
    jit_opts
  )

results = [
  gemv: {dot.(x, packed, scales), dot_reference},
  gemm: {matrix.(matrix_x, packed, scales), matrix_reference},
  dual_gemv: {dual.(x, packed, scales, second_scales), dual_reference}
]

Enum.each(results, fn {name, {actual, expected}} ->
  max_abs_diff =
    actual
    |> Nx.backend_transfer(Nx.BinaryBackend)
    |> Nx.subtract(Nx.backend_transfer(expected, Nx.BinaryBackend))
    |> Nx.abs()
    |> Nx.reduce_max()
    |> Nx.to_number()

  IO.puts("#{name} max_abs_diff=#{max_abs_diff}")

  if max_abs_diff > 0.02 do
    raise "#{name} exceeded tolerance: #{max_abs_diff}"
  end
end)
