defmodule Gemma4MicTranscribe.Gemma4.ExtractedDecoderLayer do
  @moduledoc """
  Runs one complete extracted Gemma 4 decoder layer.

  The attention artifact and MoE artifact remain separately replaceable. Layer
  0 accepts raw token-embedding rows and applies the model's embedding scale;
  later layers accept the preceding layer's output directly.
  """

  import Nx.Defn

  alias Gemma4MicTranscribe.Gemma4.ExpertCaller
  alias Gemma4MicTranscribe.Gemma4.ExpertCallerArtifact
  alias Gemma4MicTranscribe.Gemma4.ExtractedMoeLayer
  alias Gemma4MicTranscribe.Gemma4.MoeLayerArtifact

  defstruct [
    :manifest,
    :attention_params,
    :moe_manifest,
    :moe_params,
    :predict_fun,
    :backend
  ]

  @doc "Loads matching attention and MoE artifacts onto an Nx backend."
  def load!(caller_artifact, moe_artifact, backend \\ Nx.BinaryBackend) do
    {manifest, attention_params} = ExpertCallerArtifact.load!(caller_artifact, backend)
    {moe_manifest, moe_params} = MoeLayerArtifact.load!(moe_artifact, backend)

    unless manifest.layer_index == moe_manifest.layer_index do
      raise ArgumentError, "attention and MoE artifacts must use the same decoder layer"
    end

    top_k = moe_manifest.top_k_experts
    eps = manifest.rms_norm_eps
    router_scalar = manifest.hidden_size ** -0.5

    attention_opts = [
      top_k: top_k,
      eps: eps,
      router_scalar: router_scalar,
      embedding_scalar: if(manifest.layer_index == 0, do: manifest.hidden_size ** 0.5, else: 1.0),
      heads: manifest.num_attention_heads,
      kv_heads: manifest.num_key_value_heads,
      head_dim: manifest.head_dim,
      rope_theta: manifest.rope_theta,
      sliding_window: manifest.sliding_window
    ]

    predict_fun =
      Nx.Defn.jit(
        fn input, attention_params, moe_params ->
          forward(input, attention_params, moe_params, attention_opts)
        end,
        build_opts(backend)
      )

    %__MODULE__{
      manifest: manifest,
      attention_params: attention_params,
      moe_manifest: moe_manifest,
      moe_params: moe_params,
      predict_fun: predict_fun,
      backend: backend
    }
  end

  @doc "Runs `[tokens, hidden]` states through attention and the complete MoE shell."
  def run(%__MODULE__{} = layer, input) do
    input = Nx.to_tensor(input)
    expected = layer.manifest.hidden_size

    unless Nx.rank(input) == 2 and elem(Nx.shape(input), 1) == expected do
      raise ArgumentError,
            "expected decoder input shape {tokens, #{expected}}, got #{inspect(Nx.shape(input))}"
    end

    input =
      input
      |> Nx.as_type(:bf16)
      |> transfer(layer.backend)

    layer.predict_fun.(input, layer.attention_params, layer.moe_params)
  end

  @doc false
  defn forward(input, attention_params, moe_params, opts \\ []) do
    attention = ExpertCaller.forward(input, attention_params, moe_params, opts)

    feed_forward =
      ExtractedMoeLayer.forward(attention.residual_after_attention, moe_params,
        top_k: opts[:top_k],
        eps: opts[:eps],
        router_scalar: opts[:router_scalar]
      )

    %{
      output: feed_forward.output,
      residual_after_attention: attention.residual_after_attention,
      shared_output: feed_forward.shared_output,
      routed_output: feed_forward.routed_output,
      router_probabilities: feed_forward.router_probabilities,
      top_k_indices: feed_forward.top_k_indices,
      top_k_weights: feed_forward.top_k_weights
    }
  end

  defp transfer(tensor, nil), do: tensor
  defp transfer(tensor, Nx.BinaryBackend), do: Nx.backend_copy(tensor, Nx.BinaryBackend)
  defp transfer(tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp build_opts(EXLA.Backend), do: [compiler: EXLA]

  defp build_opts({EXLA.Backend, backend_opts}),
    do: [compiler: EXLA] ++ Keyword.take(backend_opts, [:client, :device_id])

  defp build_opts(_backend), do: []
end
