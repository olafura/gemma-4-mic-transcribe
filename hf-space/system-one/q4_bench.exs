# Times the packed-int4 decode kernel (exla_q4_gemv) against a bf16 dot of
# the same shape, for every projection shape of the 12B, on the job's GPU.
# Run from job.sh with BENCH=/data/jobs/q4_bench.exs.

Application.ensure_all_started(:exla)

defmodule Q4Bench do
  import Nx.Defn

  alias Gemma4MicTranscribe.Gemma4Unified.Q4Gemv

  defn q4_loop(x, packed, scales, opts \\ []) do
    opts = keyword!(opts, [:iters])
    acc = Q4Gemv.dot(x, packed, scales)

    {_, acc, _, _, _} =
      while {i = 1, acc, x, packed, scales}, i < opts[:iters] do
        # Scaling x per step keeps XLA from hoisting the call out of the loop.
        xi = Nx.as_type(Nx.as_type(x, :f32) * (1 + i * 1.0e-3), :bf16)
        {i + 1, acc + Q4Gemv.dot(xi, packed, scales), x, packed, scales}
      end

    acc
  end

  defn bf16_loop(x, w, opts \\ []) do
    opts = keyword!(opts, [:iters])
    acc = Nx.dot(x, w) |> Nx.as_type(:f32)

    {_, acc, _, _} =
      while {i = 1, acc, x, w}, i < opts[:iters] do
        xi = Nx.as_type(Nx.as_type(x, :f32) * (1 + i * 1.0e-3), :bf16)
        {i + 1, acc + Nx.as_type(Nx.dot(xi, w), :f32), x, w}
      end

    acc
  end

  defn matmul_loop(x, packed, scales, opts \\ []) do
    opts = keyword!(opts, [:iters])
    acc = Q4Gemv.matmul(x, packed, scales)

    {_, acc, _, _, _} =
      while {i = 1, acc, x, packed, scales}, i < opts[:iters] do
        xi = Nx.as_type(Nx.as_type(x, :f32) * (1 + i * 1.0e-3), :bf16)
        {i + 1, acc + Q4Gemv.matmul(xi, packed, scales), x, packed, scales}
      end

    acc
  end

  def matmul_error(x, packed, scales) do
    relative_error(Q4Gemv.matmul(x, packed, scales), x, packed, scales)
  end

  # Relative max error of the kernel against dequantize-then-dot. Plain
  # functions rather than defn, since `Q4Gemv.dequantize/3` is not defn; jit
  # traces them all the same.
  def error(x, packed, scales),
    do: relative_error(Q4Gemv.dot(x, packed, scales), x, packed, scales)

  defp relative_error(kernel, x, packed, scales) do
    reference = Nx.dot(Nx.as_type(x, :f32), Q4Gemv.dequantize(packed, scales, 32))

    Nx.subtract(kernel, reference)
    |> Nx.abs()
    |> Nx.reduce_max()
    |> Nx.divide(Nx.add(Nx.reduce_max(Nx.abs(reference)), 1.0e-6))
  end
end

iters = String.to_integer(System.get_env("BENCH_ITERS", "200"))
# BENCH_SEQ > 0 times prefill (Q4Gemv.matmul over {seq, k}) instead of decode;
# GEMMA_Q4_CUDA_PREFILL=packed routes it to the packed kernel.
seq = String.to_integer(System.get_env("BENCH_SEQ", "0"))
client = String.to_atom(System.get_env("BENCH_CLIENT", "cuda"))

inputs = fn words, n ->
  k = words * 8

  Nx.Defn.jit(
    fn seed ->
      packed =
        Nx.iota({words, n}, type: :u32)
        |> Nx.add(seed)
        |> Nx.multiply(2_654_435_761)
        |> Nx.bitcast(:s32)

      scales = Nx.broadcast(Nx.tensor(0.01, type: :bf16), {div(k, 32), n})

      x =
        if seq > 0,
          do: Nx.iota({seq, k}, type: :f32) |> Nx.remainder(13) |> Nx.divide(13),
          else: Nx.iota({k}, type: :f32) |> Nx.divide(k)

      x = Nx.as_type(x, :bf16)
      w = Nx.iota({k, n}, type: :f32) |> Nx.remainder(7) |> Nx.multiply(0.01) |> Nx.as_type(:bf16)
      {x, packed, scales, w}
    end,
    compiler: EXLA,
    client: client
  ).(Nx.tensor(1, type: :u32))
end

time = fn fun, args ->
  apply(fun, args) |> Nx.sum() |> Nx.to_number()

  {us, _} =
    :timer.tc(fn ->
      for _ <- 1..3, do: apply(fun, args) |> Nx.sum() |> Nx.to_number()
    end)

  us / 3 / iters
end

# {words, n} and how many times one decode token runs each shape (48 layers:
# 40 sliding with q/k/v/o, 8 global, gate and up and down everywhere).
shapes = [
  {{480, 15360}, 96},
  {{1920, 3840}, 48},
  {{480, 2048}, 80},
  {{480, 4096}, 40},
  {{512, 3840}, 40},
  {{480, 512}, 8},
  {{480, 8192}, 8},
  {{1024, 3840}, 8}
]

totals =
  Enum.reduce(shapes, {0.0, 0.0}, fn {{words, n}, count}, {q4_total, bf16_total} ->
    {x, packed, scales, w} = inputs.(words, n)

    {q4_fun, error_fun} =
      if seq > 0,
        do: {&Q4Bench.matmul_loop(&1, &2, &3, iters: iters), &Q4Bench.matmul_error/3},
        else: {&Q4Bench.q4_loop(&1, &2, &3, iters: iters), &Q4Bench.error/3}

    q4 = Nx.Defn.jit(q4_fun, compiler: EXLA, client: client)
    bf = Nx.Defn.jit(&Q4Bench.bf16_loop(&1, &2, iters: iters), compiler: EXLA, client: client)

    err =
      Nx.Defn.jit(error_fun, compiler: EXLA, client: client).(x, packed, scales)
      |> Nx.to_number()

    q4_us = time.(q4, [x, packed, scales])
    bf16_us = time.(bf, [x, w])
    q4_bytes = words * n * 4 + div(words * 8, 32) * n * 2
    bf16_bytes = words * 8 * n * 2

    rate = fn bytes, us ->
      if seq > 0,
        do: "#{Float.round(2 * seq * words * 8 * n / us / 1.0e6, 1)} TFLOPS",
        else: "#{Float.round(bytes / us / 1000, 0)} GB/s"
    end

    IO.puts(
      "shape #{words}x#{n} x#{count}: err #{Float.round(err, 5)} q4 #{Float.round(q4_us, 1)} us " <>
        "(#{rate.(q4_bytes, q4_us)})  bf16 #{Float.round(bf16_us, 1)} us " <>
        "(#{rate.(bf16_bytes, bf16_us)})"
    )

    {q4_total + count * q4_us, bf16_total + count * bf16_us}
  end)

{q4_total, bf16_total} = totals

IO.puts(
  "per #{if seq > 0, do: "prefill of #{seq} tokens", else: "decode token"}, projections only: q4 #{Float.round(q4_total / 1000, 1)} ms, " <>
    "bf16 #{Float.round(bf16_total / 1000, 1)} ms"
)
