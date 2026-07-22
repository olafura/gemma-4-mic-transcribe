defmodule Gemma4MicTranscribe.Gemma4Unified.Q4DualGemv do
  @moduledoc false

  import Nx.Defn

  alias Gemma4MicTranscribe.Gemma4Unified.Q4Gemv

  defstruct [:group_size]

  @doc "Computes two packed int4 projections of the same activation vector."
  deftransform dot(x, packed_a, scales_a, packed_b, scales_b, opts \\ []) do
    group_size = Keyword.get(opts, :group_size, 32)
    {_words, units} = Nx.shape(packed_a)

    Nx.block(
      %__MODULE__{group_size: group_size},
      [x, packed_a, scales_a, packed_b, scales_b],
      Nx.template({2 * units}, :f32),
      fn %__MODULE__{}, x, packed_a, scales_a, packed_b, scales_b ->
        first =
          Nx.dot(
            Nx.as_type(x, :f32),
            Q4Gemv.dequantize(packed_a, scales_a, group_size)
          )

        second =
          Nx.dot(
            Nx.as_type(x, :f32),
            Q4Gemv.dequantize(packed_b, scales_b, group_size)
          )

        Nx.concatenate([first, second])
      end
    )
  end
end

defimpl EXLA.CustomCall, for: Gemma4MicTranscribe.Gemma4Unified.Q4DualGemv do
  require Logger

  def call(
        %{group_size: group_size},
        _out,
        [x, packed_a, scales_a, packed_b, scales_b],
        %{platform: :rocm}
      ) do
    with 1 <- Nx.rank(x),
         {:bf, 16} <- Nx.type(x),
         {:s, 32} <- Nx.type(packed_a),
         {:bf, 16} <- Nx.type(scales_a),
         {:s, 32} <- Nx.type(packed_b),
         {:bf, 16} <- Nx.type(scales_b) do
      Logger.info(fn ->
        "q4_dual_gemv: fused projections selected for #{inspect(Nx.shape(packed_a))}"
      end)

      {:ok,
       %EXLA.CustomCall.Spec{
         call_target_name: "exla_q4_dual_gemv",
         attributes: [{"group_size", "#{group_size} : i64"}]
       }}
    else
      _other -> :skip
    end
  end

  def call(_block, _out, _args, _context), do: :skip
end
