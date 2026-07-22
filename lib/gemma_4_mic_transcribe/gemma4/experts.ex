defmodule Gemma4MicTranscribe.Gemma4.Experts do
  @moduledoc """
  Describes the experts in a Gemma 4 Mixture-of-Experts text model.

  `list/2` only needs a model spec or Hugging Face configuration; it does not
  load the checkpoint weights. Each descriptor includes the exact checkpoint
  tensors (and, for routed experts, the slice on their expert axis) needed to
  extract or replace that expert later.

  Dense Gemma models return an empty list. In the 26B-A4B model every decoder
  layer has one shared FFN and a bank of routed experts, so the shared FFN is
  included by default.
  """

  defmodule Descriptor do
    @moduledoc "A stable description of one shared or routed Gemma 4 expert."

    @enforce_keys [
      :id,
      :kind,
      :layer_index,
      :expert_index,
      :input_size,
      :intermediate_size,
      :output_size,
      :activation,
      :parameter_count,
      :weights
    ]
    defstruct @enforce_keys ++ [:router]

    @type t :: %__MODULE__{
            id: String.t(),
            kind: :shared | :routed,
            layer_index: non_neg_integer(),
            expert_index: :shared | non_neg_integer(),
            input_size: pos_integer(),
            intermediate_size: pos_integer(),
            output_size: pos_integer(),
            activation: atom() | String.t(),
            parameter_count: pos_integer(),
            weights: map(),
            router: map() | nil
          }
  end

  @doc """
  Lists every expert described by a Gemma 4 model spec or configuration.

  Accepted inputs include a Bumblebee spec, a raw Hugging Face `config.json`
  map, a `%{spec: spec}` model-info map, or a runtime containing `model_info`.

  ## Options

    * `:include_shared` - includes the always-on shared FFN (default: `true`)

  """
  @spec list(struct() | map(), keyword()) :: [Descriptor.t()]
  def list(model_or_config, opts \\ [])

  def list(%{model_info: model_info}, opts), do: list(model_info, opts)
  def list(%{spec: spec}, opts), do: list(spec, opts)

  def list(model_or_config, opts) when is_map(model_or_config) do
    config = text_config(model_or_config)

    if field(config, :enable_moe_block, false) do
      build_descriptors(config, Keyword.get(opts, :include_shared, true))
    else
      []
    end
  end

  defp build_descriptors(config, include_shared?) do
    num_layers = positive_field!(config, :num_blocks, :num_hidden_layers)
    num_experts = positive_field!(config, :num_experts)
    hidden_size = positive_field!(config, :hidden_size)
    shared_size = positive_field!(config, :intermediate_size)
    expert_size = positive_field!(config, :moe_intermediate_size)
    top_k = positive_field!(config, :top_k_experts)
    activation = field(config, :activation, field(config, :hidden_activation, :gelu_approx_tanh))

    if top_k > num_experts do
      raise ArgumentError,
            "Gemma 4 top_k_experts (#{top_k}) cannot exceed num_experts (#{num_experts})"
    end

    0..(num_layers - 1)
    |> Enum.flat_map(fn layer_index ->
      shared =
        if include_shared? do
          [shared_descriptor(layer_index, hidden_size, shared_size, activation)]
        else
          []
        end

      routed =
        Enum.map(0..(num_experts - 1), fn expert_index ->
          routed_descriptor(
            layer_index,
            expert_index,
            num_experts,
            top_k,
            hidden_size,
            expert_size,
            activation
          )
        end)

      shared ++ routed
    end)
  end

  defp shared_descriptor(layer, hidden_size, intermediate_size, activation) do
    prefix = "model.language_model.layers.#{layer}.mlp"

    %Descriptor{
      id: "language_model.layer.#{layer}.shared",
      kind: :shared,
      layer_index: layer,
      expert_index: :shared,
      input_size: hidden_size,
      intermediate_size: intermediate_size,
      output_size: hidden_size,
      activation: activation,
      parameter_count: 3 * hidden_size * intermediate_size,
      weights: %{
        gate: tensor_ref("#{prefix}.gate_proj.weight", {intermediate_size, hidden_size}),
        up: tensor_ref("#{prefix}.up_proj.weight", {intermediate_size, hidden_size}),
        down: tensor_ref("#{prefix}.down_proj.weight", {hidden_size, intermediate_size})
      },
      router: nil
    }
  end

  defp routed_descriptor(
         layer,
         expert,
         num_experts,
         top_k,
         hidden_size,
         intermediate_size,
         activation
       ) do
    layer_prefix = "model.language_model.layers.#{layer}"
    experts_prefix = "#{layer_prefix}.experts"

    %Descriptor{
      id: "language_model.layer.#{layer}.expert.#{expert}",
      kind: :routed,
      layer_index: layer,
      expert_index: expert,
      input_size: hidden_size,
      intermediate_size: intermediate_size,
      output_size: hidden_size,
      activation: activation,
      parameter_count: 3 * hidden_size * intermediate_size,
      weights: %{
        gate_up:
          sliced_tensor_ref(
            "#{experts_prefix}.gate_up_proj",
            {num_experts, 2 * intermediate_size, hidden_size},
            expert,
            {2 * intermediate_size, hidden_size},
            %{
              gate: %{axis: 0, start: 0, length: intermediate_size},
              up: %{axis: 0, start: intermediate_size, length: intermediate_size}
            }
          ),
        down:
          sliced_tensor_ref(
            "#{experts_prefix}.down_proj",
            {num_experts, hidden_size, intermediate_size},
            expert,
            {hidden_size, intermediate_size}
          )
      },
      router: %{
        top_k: top_k,
        projection: "#{layer_prefix}.router.proj.weight",
        scale: "#{layer_prefix}.router.scale",
        per_expert_scale: "#{layer_prefix}.router.per_expert_scale"
      }
    }
  end

  defp tensor_ref(tensor, shape), do: %{tensor: tensor, checkpoint_shape: shape}

  defp sliced_tensor_ref(tensor, checkpoint_shape, index, slice_shape, splits \\ nil) do
    %{
      tensor: tensor,
      checkpoint_shape: checkpoint_shape,
      slice: %{axis: 0, index: index, shape: slice_shape}
    }
    |> maybe_put_splits(splits)
  end

  defp maybe_put_splits(reference, nil), do: reference
  defp maybe_put_splits(reference, splits), do: Map.put(reference, :splits, splits)

  defp text_config(config) do
    Map.get(config, :text_config) || Map.get(config, "text_config") || config
  end

  defp positive_field!(config, primary, fallback \\ nil) do
    value = field(config, primary, if(fallback, do: field(config, fallback, nil), else: nil))

    if is_integer(value) and value > 0 do
      value
    else
      names = if fallback, do: "#{primary}/#{fallback}", else: to_string(primary)

      raise ArgumentError,
            "expected positive Gemma 4 configuration field #{names}, got: #{inspect(value)}"
    end
  end

  defp field(config, key, default) do
    Map.get(config, key, Map.get(config, Atom.to_string(key), default))
  end
end
