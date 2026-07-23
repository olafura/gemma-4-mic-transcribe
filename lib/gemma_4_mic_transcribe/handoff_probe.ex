defmodule Gemma4MicTranscribe.HandoffProbe do
  @moduledoc """
  Runs Cactus Compute's Gemma 4 E2B correctness probe independently.

  The probe consumes one decoder-layer-28 row for every accepted generated
  token and returns `confidence = 1 - p_wrong`. Its parameters are kept in a
  small standalone artifact; no Cactus base-model tensors are loaded.
  """

  import Nx.Defn

  alias Gemma4MicTranscribe.HandoffProbe.Artifact

  defstruct [:params, :predict_fun, :backend, :manifest]

  @max_tokens 1024

  @doc "Loads a standalone probe artifact and compiles its scorer for `backend`."
  def load!(path, backend) do
    {manifest, params} = Artifact.load!(path, backend)

    predict_fun = Nx.Defn.jit(&confidence/3, build_opts(backend))

    %__MODULE__{
      params: params,
      predict_fun: predict_fun,
      backend: backend,
      manifest: manifest
    }
  end

  @doc "Scores captured `[1, hidden]` rows and returns confidence in `[0, 1]`."
  def score(%__MODULE__{} = probe, rows) when is_list(rows) do
    rows = Enum.take(rows, @max_tokens)

    case rows do
      [] ->
        nil

      rows ->
        token_count = length(rows)

        hidden_states =
          rows
          |> Enum.map(&Nx.as_type(&1, :f32))
          |> Nx.concatenate(axis: 0)
          |> Nx.pad(0.0, [{0, @max_tokens - token_count, 0}, {0, 0, 0}])
          |> transfer(probe.backend)

        probe.predict_fun.(hidden_states, Nx.tensor(token_count), probe.params)
        |> Nx.backend_copy(Nx.BinaryBackend)
        |> Nx.to_number()
    end
  end

  @doc "Compiles the fixed-shape scorer outside request processing."
  def warmup(%__MODULE__{} = probe) do
    _confidence = score(probe, [Nx.broadcast(0.0, {1, probe.manifest.feature_size})])
    :ok
  end

  @doc false
  defn confidence(hidden_states, token_count, params) do
    x = Nx.as_type(hidden_states, :f32)
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    variance = Nx.mean(Nx.pow(x - mean, 2), axes: [-1], keep_axes: true)

    x =
      (x - mean) * Nx.rsqrt(variance + 1.0e-5) * params.norm_weight + params.norm_bias

    projected =
      x
      |> Nx.dot(Nx.transpose(params.proj_weight))
      |> Nx.add(params.proj_bias)
      |> Nx.max(0.0)

    scores = Nx.dot(projected, params.attn_query) / Nx.sqrt(Nx.tensor(32.0, type: :f32))
    valid_tokens = Nx.iota({@max_tokens}) < token_count
    scores = Nx.select(valid_tokens, scores, Nx.tensor(-1.0e30, type: :f32))
    weights = Nx.exp(scores - Nx.reduce_max(scores))
    weights = weights / Nx.sum(weights)
    pooled = Nx.sum(projected * Nx.new_axis(weights, -1), axes: [0])

    hidden =
      pooled
      |> Nx.dot(Nx.transpose(params.head_0_weight))
      |> Nx.add(params.head_0_bias)
      |> Nx.max(0.0)
      |> Nx.dot(Nx.transpose(params.head_2_weight))
      |> Nx.add(params.head_2_bias)
      |> Nx.max(0.0)

    logit = Nx.dot(hidden, Nx.transpose(params.head_4_weight)) + params.head_4_bias
    p_wrong = Nx.sigmoid(logit)
    Nx.squeeze(1.0 - p_wrong)
  end

  defp transfer(tensor, nil), do: tensor
  defp transfer(tensor, Nx.BinaryBackend), do: Nx.backend_copy(tensor, Nx.BinaryBackend)
  defp transfer(tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp build_opts(EXLA.Backend), do: [compiler: EXLA]

  defp build_opts({EXLA.Backend, backend_opts}),
    do: [compiler: EXLA] ++ Keyword.take(backend_opts, [:client, :device_id])

  defp build_opts(_backend), do: []
end
