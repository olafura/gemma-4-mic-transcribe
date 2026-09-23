# Times the 12B's tied LM head for one decode token with the table stored as
# f32 (as the packed artifacts do) against bf16 upcast inside the graph,
# which gives the same f32 dot when the f32 values are exact bf16. Run with
# BENCH_SCRIPT=head_bench.exs from bench.sh.

Application.ensure_all_started(:exla)

client = String.to_atom(System.get_env("BENCH_CLIENT", "cuda"))
calls = String.to_integer(System.get_env("BENCH_ITERS", "50"))
{vocab, hidden} = {262_144, 3840}

jit = &Nx.Defn.jit(&1, compiler: EXLA, client: client)

table =
  jit.(fn seed ->
    Nx.iota({vocab, hidden}, type: :f32)
    |> Nx.add(seed)
    |> Nx.remainder(251)
    |> Nx.divide(1000)
    |> Nx.as_type(:bf16)
  end).(Nx.tensor(0.0))

f32_table = jit.(&Nx.as_type(&1, :f32)).(table)
x = jit.(fn -> Nx.iota({1, 1, hidden}, type: :f32) |> Nx.divide(hidden) end).()

head = jit.(fn x, table -> Nx.dot(x, [2], Nx.as_type(table, :f32), [1]) end)

time = fn table ->
  head.(x, table) |> Nx.sum() |> Nx.to_number()

  {us, _} =
    :timer.tc(fn ->
      Enum.reduce(1..calls, nil, fn _, _ -> head.(x, table) end) |> Nx.sum() |> Nx.to_number()
    end)

  us / calls
end

same = jit.(&Nx.all(Nx.equal(&1, &2))).(head.(x, f32_table), head.(x, table)) |> Nx.to_number()

IO.puts(
  "lm head per decode token: f32 table #{Float.round(time.(f32_table) / 1000, 2)} ms, " <>
    "bf16 table #{Float.round(time.(table) / 1000, 2)} ms, bitwise equal #{same == 1}"
)
