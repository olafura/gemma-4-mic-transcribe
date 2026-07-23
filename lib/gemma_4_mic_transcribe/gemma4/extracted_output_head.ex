defmodule Gemma4MicTranscribe.Gemma4.ExtractedOutputHead do
  @moduledoc """
  Runs Gemma 4's extracted final norm and tied language-model output head.
  """

  import Nx.Defn

  alias Gemma4MicTranscribe.Gemma4.OutputHeadArtifact

  defstruct [:manifest, :params, :predict_fun, :backend]

  @doc "Loads and compiles an extracted output head."
  def load!(artifact, backend \\ Nx.BinaryBackend) do
    {manifest, params} = OutputHeadArtifact.load!(artifact, backend)
    eps = manifest.rms_norm_eps

    predict_fun =
      Nx.Defn.jit(
        fn hidden, params -> raw_logits(hidden, params, eps: eps) end,
        build_opts(backend)
      )

    %__MODULE__{
      manifest: manifest,
      params: params,
      predict_fun: predict_fun,
      backend: backend
    }
  end

  @doc "Returns logits and top-k predictions for the final hidden-state row."
  def run(%__MODULE__{} = head, hidden, opts \\ []) do
    hidden = Nx.to_tensor(hidden)
    expected = head.manifest.hidden_size

    unless Nx.rank(hidden) == 2 and elem(Nx.shape(hidden), 1) == expected do
      raise ArgumentError,
            "expected output-head input shape {tokens, #{expected}}, got #{inspect(Nx.shape(hidden))}"
    end

    top_k = Keyword.get(opts, :top_k, 10)

    unless is_integer(top_k) and top_k > 0 and top_k <= head.manifest.vocab_size do
      raise ArgumentError, "top_k must be in 1..#{head.manifest.vocab_size}"
    end

    raw_logits =
      hidden
      |> Nx.as_type(:bf16)
      |> transfer(head.backend)
      |> head.predict_fun.(head.params)

    logits = softcap(raw_logits, head.manifest.final_logit_softcapping)
    {raw_top_k_values, top_k_indices} = Nx.top_k(raw_logits, k: top_k)
    top_k_values = softcap(raw_top_k_values, head.manifest.final_logit_softcapping)

    %{
      logits: logits,
      raw_logits: raw_logits,
      top_k_values: top_k_values,
      raw_top_k_values: raw_top_k_values,
      top_k_indices: top_k_indices
    }
  end

  @doc false
  defn raw_logits(hidden, params, opts \\ []) do
    eps = opts[:eps]
    last = Nx.axis_size(hidden, 0) - 1
    hidden = Nx.slice_along_axis(hidden, last, 1, axis: 0)
    hidden_f32 = Nx.as_type(hidden, :f32)

    normalized =
      hidden_f32
      |> Nx.pow(2)
      |> Nx.mean(axes: [1], keep_axes: true)
      |> Nx.add(eps)
      |> Nx.pow(-0.5)
      |> Nx.multiply(hidden_f32)
      |> Nx.multiply(Nx.as_type(params.norm, :f32))
      |> Nx.as_type(:bf16)

    normalized
    |> Nx.dot(Nx.transpose(params.embedding))
    |> Nx.as_type(:f32)
    |> Nx.squeeze(axes: [0])
  end

  defp softcap(logits, value), do: Nx.multiply(Nx.tanh(Nx.divide(logits, value)), value)

  defp transfer(tensor, nil), do: tensor
  defp transfer(tensor, Nx.BinaryBackend), do: Nx.backend_copy(tensor, Nx.BinaryBackend)
  defp transfer(tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp build_opts(EXLA.Backend), do: [compiler: EXLA]

  defp build_opts({EXLA.Backend, backend_opts}),
    do: [compiler: EXLA] ++ Keyword.take(backend_opts, [:client, :device_id])

  defp build_opts(_backend), do: []
end
