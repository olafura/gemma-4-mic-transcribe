defmodule Gemma4MicTranscribe.Gemma4.ExtractedSparseDecoderLayer do
  @moduledoc """
  Runs one-token cached decoder steps without loading complete expert banks.

  Attention, shared-FFN, router, and normalization parameters load first. After
  routing, only the selected expert slices are read and transferred.
  """

  alias Gemma4MicTranscribe.Gemma4.ExpertCaller
  alias Gemma4MicTranscribe.Gemma4.ExpertCallerArtifact
  alias Gemma4MicTranscribe.Gemma4.ExtractedMoeLayer
  alias Gemma4MicTranscribe.Gemma4.MoeLayerArtifact
  alias Gemma4MicTranscribe.Gemma4.RoutedExpertCache

  defstruct [
    :manifest,
    :attention_params,
    :moe_manifest,
    :moe_params,
    :prepare_fun,
    :finish_fun,
    :backend,
    :moe_artifact
  ]

  @doc "Loads attention and the non-routed-expert MoE parameters."
  def load!(caller_artifact, moe_artifact, backend \\ Nx.BinaryBackend, opts \\ []) do
    {manifest, attention_params} = ExpertCallerArtifact.load!(caller_artifact, backend, opts)
    {moe_manifest, moe_params} = MoeLayerArtifact.load_sparse_base!(moe_artifact, backend, opts)

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
      partial_rotary_factor: Map.get(manifest, :partial_rotary_factor, 1.0),
      rotary_angles:
        trunc(Map.get(manifest, :partial_rotary_factor, 1.0) * div(manifest.head_dim, 2)),
      alternative_attention: Map.get(manifest, :alternative_attention, false),
      sliding_window: manifest.sliding_window
    ]

    prepare_fun =
      Nx.Defn.jit(
        fn input, attention_params, moe_params, key_cache, value_cache, offset ->
          attention =
            ExpertCaller.forward_cached(
              input,
              attention_params,
              moe_params,
              key_cache,
              value_cache,
              offset,
              attention_opts
            )

          prepared =
            ExtractedMoeLayer.prepare_sparse(
              attention.residual_after_attention,
              moe_params,
              attention.top_k_indices,
              attention.top_k_weights,
              eps: eps
            )

          Map.merge(prepared, %{
            key_cache: attention.key_cache,
            value_cache: attention.value_cache
          })
        end,
        build_opts(backend)
      )

    finish_fun =
      Nx.Defn.jit(
        fn prepared, moe_params, expert_params ->
          ExtractedMoeLayer.finish_sparse(prepared, moe_params, expert_params, eps: eps).output
        end,
        build_opts(backend)
      )

    %__MODULE__{
      manifest: manifest,
      attention_params: attention_params,
      moe_manifest: moe_manifest,
      moe_params: moe_params,
      prepare_fun: prepare_fun,
      finish_fun: finish_fun,
      backend: backend,
      moe_artifact: Path.expand(moe_artifact)
    }
  end

  @doc "Allocates a fixed-shape attention cache for the sparse layer."
  def init_cache(%__MODULE__{} = layer, max_length)
      when is_integer(max_length) and max_length > 0 do
    shape = {
      max_length,
      layer.manifest.num_key_value_heads,
      layer.manifest.head_dim
    }

    Nx.broadcast(Nx.tensor(0.0, type: :bf16), shape)
    |> transfer(layer.backend)
    |> then(&%{key: &1, value: Nx.backend_copy(&1, layer.backend)})
  end

  @doc "Runs one cached token using only its selected routed experts."
  def run_cached(%__MODULE__{} = layer, input, %{key: key, value: value}, offset, opts \\ [])
      when is_integer(offset) and offset >= 0 do
    input = prepare_input!(layer, input)
    token_count = Nx.axis_size(input, 0)
    cache_length = Nx.axis_size(key, 0)

    unless token_count == 1 do
      raise ArgumentError, "sparse decoder execution requires exactly one input token"
    end

    unless Nx.shape(key) == Nx.shape(value) and offset + token_count <= cache_length do
      raise ArgumentError,
            "decoder cache cannot place #{token_count} tokens at offset #{offset} " <>
              "in cache length #{cache_length}"
    end

    offset_tensor = Nx.tensor(offset, type: :s64) |> transfer(layer.backend)

    prepared =
      layer.prepare_fun.(
        input,
        layer.attention_params,
        layer.moe_params,
        key,
        value,
        offset_tensor
      )

    expert_indices =
      prepared.top_k_indices
      |> Nx.backend_copy(Nx.BinaryBackend)
      |> Nx.to_flat_list()

    {expert_params, expert_cache} =
      case Keyword.get(opts, :expert_cache) do
        nil ->
          {_manifest, expert_params} =
            MoeLayerArtifact.load_routed_experts!(
              layer.moe_artifact,
              expert_indices,
              layer.backend,
              verify_checksum: false
            )

          {expert_params, nil}

        %RoutedExpertCache{} = cache ->
          RoutedExpertCache.checkout!(
            cache,
            layer.moe_artifact,
            layer.moe_manifest,
            expert_indices,
            layer.backend
          )
      end

    finish_input =
      Map.take(prepared, [
        :residual,
        :shared_output,
        :routed_input,
        :top_k_indices,
        :top_k_weights
      ])

    output = layer.finish_fun.(finish_input, layer.moe_params, expert_params)
    Nx.backend_deallocate(expert_params)

    Nx.backend_deallocate(Map.drop(prepared, [:key_cache, :value_cache]))

    %{
      output: output,
      key_cache: prepared.key_cache,
      value_cache: prepared.value_cache,
      selected_experts: expert_indices,
      expert_cache: expert_cache
    }
  end

  defp prepare_input!(layer, input) do
    input = Nx.to_tensor(input)
    expected = layer.manifest.hidden_size

    unless Nx.rank(input) == 2 and elem(Nx.shape(input), 1) == expected do
      raise ArgumentError,
            "expected decoder input shape {tokens, #{expected}}, got #{inspect(Nx.shape(input))}"
    end

    input
    |> Nx.as_type(:bf16)
    |> transfer(layer.backend)
  end

  defp transfer(tensor, nil), do: tensor
  defp transfer(tensor, Nx.BinaryBackend), do: Nx.backend_copy(tensor, Nx.BinaryBackend)
  defp transfer(tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp build_opts(EXLA.Backend), do: [compiler: EXLA]

  defp build_opts({EXLA.Backend, backend_opts}),
    do: [compiler: EXLA] ++ Keyword.take(backend_opts, [:client, :device_id])

  defp build_opts(_backend), do: []
end
