defmodule Gemma4MicTranscribe.Gemma4.FFNs do
  @moduledoc """
  Describes the dense gated FFNs in Gemma 4 decoder layers.

  Every dense Gemma 4 decoder layer applies the same three-projection operation:

      down(activation(gate(x)) * up(x))

  There is no router: every token passes through that layer's FFN. In an MoE
  Gemma model this dense FFN is the always-on shared expert, so its descriptor
  has `kind: :shared`; otherwise it has `kind: :dense`.

  The FFN itself receives the result of the layer's pre-feedforward RMS norm.
  Its output is subsequently post-normalized and added to the residual stream;
  those norms and the residual addition are not part of the extracted FFN.
  """

  defmodule Descriptor do
    @moduledoc "A stable description of one Gemma 4 decoder FFN."

    @enforce_keys [
      :id,
      :kind,
      :layer_index,
      :input_size,
      :intermediate_size,
      :output_size,
      :activation,
      :parameter_count,
      :operation,
      :weights,
      :context
    ]
    defstruct @enforce_keys

    @type t :: %__MODULE__{
            id: String.t(),
            kind: :dense | :shared,
            layer_index: non_neg_integer(),
            input_size: pos_integer(),
            intermediate_size: pos_integer(),
            output_size: pos_integer(),
            activation: atom() | String.t(),
            parameter_count: pos_integer(),
            operation: String.t(),
            weights: map(),
            context: map()
          }
  end

  @operation "down(activation(gate(x)) * up(x))"

  @doc """
  Lists the dense FFN in every decoder layer described by a Gemma 4 model.

  Accepted inputs include a Bumblebee spec, a raw Hugging Face `config.json`
  map, a `%{spec: spec}` model-info map, or a runtime containing `model_info`.

  ## Options

    * `:layers` - restricts the result to the given layer indices

  """
  @spec list(struct() | map(), keyword()) :: [Descriptor.t()]
  def list(model_or_config, opts \\ [])

  def list(%{model_info: model_info}, opts), do: list(model_info, opts)
  def list(%{spec: spec}, opts), do: list(spec, opts)

  def list(model_or_config, opts) when is_map(model_or_config) do
    config = text_config(model_or_config)
    num_layers = positive_field!(config, :num_blocks, :num_hidden_layers)
    hidden_size = positive_field!(config, :hidden_size)
    configured_intermediate_size = positive_field!(config, :intermediate_size)

    intermediate_size =
      if field(config, :use_double_wide_mlp, false) do
        2 * configured_intermediate_size
      else
        configured_intermediate_size
      end

    activation = field(config, :activation, field(config, :hidden_activation, :gelu_approx_tanh))
    kind = if field(config, :enable_moe_block, false), do: :shared, else: :dense

    requested_layers = Keyword.get(opts, :layers, 0..(num_layers - 1))

    Enum.map(requested_layers, fn layer_index ->
      validate_layer!(layer_index, num_layers)
      descriptor(layer_index, kind, hidden_size, intermediate_size, activation)
    end)
  end

  @doc "Returns the operation performed by every listed FFN."
  def operation, do: @operation

  defp descriptor(layer, kind, hidden_size, intermediate_size, activation) do
    checkpoint_prefix = "model.language_model.layers.#{layer}"
    axon_prefix = "decoder.blocks.#{layer}"

    %Descriptor{
      id: "language_model.layer.#{layer}.ffn",
      kind: kind,
      layer_index: layer,
      input_size: hidden_size,
      intermediate_size: intermediate_size,
      output_size: hidden_size,
      activation: activation,
      parameter_count: 3 * hidden_size * intermediate_size,
      operation: @operation,
      weights: %{
        gate:
          weight_ref(
            "#{checkpoint_prefix}.mlp.gate_proj.weight",
            {intermediate_size, hidden_size},
            "#{axon_prefix}.ffn.gate.kernel",
            {hidden_size, intermediate_size}
          ),
        up:
          weight_ref(
            "#{checkpoint_prefix}.mlp.up_proj.weight",
            {intermediate_size, hidden_size},
            "#{axon_prefix}.ffn.intermediate.kernel",
            {hidden_size, intermediate_size}
          ),
        down:
          weight_ref(
            "#{checkpoint_prefix}.mlp.down_proj.weight",
            {hidden_size, intermediate_size},
            "#{axon_prefix}.ffn.output.kernel",
            {intermediate_size, hidden_size}
          )
      },
      context: %{
        input: :after_pre_feedforward_rms_norm,
        output: :before_post_feedforward_rms_norm_and_residual,
        pre_norm: "#{checkpoint_prefix}.pre_feedforward_layernorm.weight",
        post_norm: "#{checkpoint_prefix}.post_feedforward_layernorm.weight",
        residual: true
      }
    }
  end

  defp weight_ref(checkpoint_tensor, checkpoint_shape, axon_parameter, axon_shape) do
    %{
      checkpoint_tensor: checkpoint_tensor,
      checkpoint_shape: checkpoint_shape,
      axon_parameter: axon_parameter,
      axon_shape: axon_shape
    }
  end

  defp validate_layer!(layer, num_layers)
       when is_integer(layer) and layer >= 0 and layer < num_layers,
       do: :ok

  defp validate_layer!(layer, num_layers) do
    raise ArgumentError,
          "expected a Gemma 4 layer index in 0..#{num_layers - 1}, got: #{inspect(layer)}"
  end

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
