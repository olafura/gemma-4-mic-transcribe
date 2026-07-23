defmodule Gemma4MicTranscribe.Gemma4.ExtractedExpert do
  @moduledoc """
  Runs one range-extracted Gemma 4 routed expert independently.

  The input is the pre-FFN-normalized hidden state with shape `[tokens, hidden]`.
  Router weighting, the shared expert, surrounding norms, and residual addition
  belong to the MoE block and are intentionally outside this module.
  """

  import Nx.Defn

  alias Gemma4MicTranscribe.Gemma4.ExpertArtifact

  defstruct [:manifest, :params, :predict_fun, :backend]

  @doc "Loads an expert artifact and builds its backend-specific predictor."
  def load!(path, backend \\ Nx.BinaryBackend) do
    {manifest, params} = ExpertArtifact.load!(path, backend)

    %__MODULE__{
      manifest: manifest,
      params: params,
      predict_fun: Nx.Defn.jit(&forward/2, build_opts(backend)),
      backend: backend
    }
  end

  @doc "Runs the expert over `[tokens, hidden]` states."
  def run(%__MODULE__{} = expert, input) do
    input = Nx.to_tensor(input)

    unless Nx.rank(input) == 2 and elem(Nx.shape(input), 1) == expert.manifest.input_size do
      raise ArgumentError,
            "expected expert input shape {tokens, #{expert.manifest.input_size}}, got #{inspect(Nx.shape(input))}"
    end

    input
    |> Nx.as_type(expert.manifest.parameter_type)
    |> transfer(expert.backend)
    |> expert.predict_fun.(expert.params)
  end

  @doc "Compiles the predictor for a fixed token count."
  def warmup(%__MODULE__{} = expert, token_count \\ 1) do
    expert
    |> run(Nx.broadcast(0.0, {token_count, expert.manifest.input_size}))
    |> Nx.backend_copy(Nx.BinaryBackend)

    :ok
  end

  @doc false
  defn forward(input, params) do
    gate = Nx.dot(input, Nx.transpose(params.gate))
    up = Nx.dot(input, Nx.transpose(params.up))
    hidden = Bumblebee.Layers.gelu_approx_tanh(gate) * up
    Nx.dot(hidden, Nx.transpose(params.down))
  end

  defp transfer(tensor, nil), do: tensor
  defp transfer(tensor, Nx.BinaryBackend), do: Nx.backend_copy(tensor, Nx.BinaryBackend)
  defp transfer(tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp build_opts(EXLA.Backend), do: [compiler: EXLA]

  defp build_opts({EXLA.Backend, backend_opts}),
    do: [compiler: EXLA] ++ Keyword.take(backend_opts, [:client, :device_id])

  defp build_opts(_backend), do: []
end
