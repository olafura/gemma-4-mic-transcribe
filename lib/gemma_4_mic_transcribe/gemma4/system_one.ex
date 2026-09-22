defmodule Gemma4MicTranscribe.Gemma4.SystemOne do
  @moduledoc """
  The routed "quick decision" expert added beside a decoder block's FFN.

  See `docs/system-one-expert-plan.md`, section 1. For the layers named by
  `spec.system_one_layers` the block computes

      ffn_out(x) = base_ffn(x) + g(x) * expert(x)
      expert(x)  = down_e(act(gate_e(x)) * up_e(x))     3840 -> 2048 -> 3840
      g(x)       = sigmoid(w . x + b)                   one scalar per token

  where `x` is the existing `pre_ffn_norm` output, so the router needs no norm
  of its own and the sum lands before `post_ffn_norm` and `layer_scalar`. The
  expert is a sibling subgraph of `gated_ffn`, so the packed `Q4Gemv` and
  `Q4DualGemv` custom calls keep their operand shapes.

  With `system_one_layers` empty `attach/4` adds no nodes at all, so the built
  graph is the one the repo had before this module existed. `down_e` starts at
  zeros and the router bias at -4, so an untrained expert contributes exactly
  nothing.
  """

  import Nx.Defn
  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @default_expert_size 2048
  @default_gate_floor 0.05

  # sigmoid(-4) is 0.018, under the default gate floor, so a freshly
  # initialised router is clamped shut before `down_e` has moved off zero.
  @closed_router_bias -4.0

  @doc "The layer indices that carry an expert, `[]` for a spec without the field."
  def layers(spec), do: Map.get(spec, :system_one_layers, []) || []

  @doc "The inference gate floor: a gate under it is clamped to exactly 0."
  def gate_floor(spec), do: Map.get(spec, :system_one_gate_floor, @default_gate_floor)

  @doc "The expert's bottleneck width."
  def expert_size(spec), do: Map.get(spec, :system_one_expert_size, @default_expert_size)

  @doc "The node name prefix the expert's parameters live under for `layer_index`."
  def node_prefix(layer_index), do: "decoder.blocks.#{layer_index}.system_one"

  @doc "The name of the gate node of `layer_index`, the node `gate_nodes/2` reads."
  def gate_name(layer_index), do: join(node_prefix(layer_index), "gate")

  @doc "The name of the router's dense node of `layer_index`, the node `router_logit_nodes/2` reads."
  def router_name(layer_index), do: join(node_prefix(layer_index), "router")

  @doc """
  Adds `g(x) * expert(x)` to a block's FFN output.

  `ffn_output` is what `gated_ffn/5` returned and `ffn_input` the
  `pre_ffn_norm` output both it and the expert read. Returns `ffn_output`
  unchanged, with no new graph nodes, unless `opts[:layer_index]` is one of
  `spec.system_one_layers`.
  """
  def attach(ffn_output, ffn_input, spec, opts) do
    layer_index = opts[:layer_index]

    if layer_index != nil and layer_index in layers(spec) do
      name = node_prefix(layer_index)
      gate = router(ffn_input, name: router_name(layer_index), gate_floor: gate_floor(spec))

      expert =
        expert(ffn_input, spec,
          name: join(name, "expert"),
          units: expert_size(spec)
        )

      expert
      |> Axon.multiply(gate, name: join(name, "gated"))
      |> then(&Axon.add(ffn_output, &1, name: join(name, "sum")))
    else
      ffn_output
    end
  end

  @doc """
  A model whose only parameters are the experts and routers of `layer_indices`.

  It is the expert subgraph on its own, with the block's FFN output stood in
  for by the same input, so `Axon.build/2` can initialise exactly the tensors
  an artifact holds without building a decoder layer.
  """
  def parameter_model(spec, layer_indices) do
    hidden_size = Map.fetch!(spec, :hidden_size)
    spec = Map.put(spec, :system_one_layers, layer_indices)
    input = Axon.input("hidden_state", shape: {nil, nil, hidden_size})

    layer_indices
    |> Map.new(fn index -> {"#{index}", attach(input, input, spec, layer_index: index)} end)
    |> Axon.container()
  end

  @doc """
  Freshly initialised expert parameters for `layer_indices`, on the binary
  backend: random `gate_e` and `up_e`, zero `down_e`, and a router at the
  closed bias.
  """
  def init_parameters(spec, layer_indices) do
    hidden_size = Map.fetch!(spec, :hidden_size)

    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      {init_fun, _predict_fun} = Axon.build(parameter_model(spec, layer_indices))

      state =
        init_fun.(
          %{"hidden_state" => Nx.template({1, 1, hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      Map.filter(state.data, fn {name, _parameters} -> parameter_node?(name) end)
    end)
  end

  @doc "True for the parameter node names an expert artifact owns."
  def parameter_node?("decoder.blocks." <> rest) do
    case String.split(rest, ".", parts: 2) do
      [index, "system_one." <> _name] -> match?({_index, ""}, Integer.parse(index))
      _other -> false
    end
  end

  def parameter_node?(_name), do: false

  @doc """
  The gated bottleneck FFN, `hidden_size -> units -> hidden_size`.

  Same shape as `Model.gated_ffn/5` and the same activation, but always a
  plain dense: the expert is trained, so its weights are never packed int4 and
  gradients flow through `Nx.dot`. `output` starts at zeros, which is what
  makes step 0 exactly Gemma 4.
  """
  def expert(hidden_state, spec, opts) do
    name = opts[:name]
    units = Keyword.get(opts, :units, expert_size(spec))
    hidden_size = Map.fetch!(spec, :hidden_size)
    initializer = Axon.Initializers.normal(scale: Map.get(spec, :initializer_scale, 0.02))

    gate =
      Axon.dense(hidden_state, units,
        name: join(name, "gate"),
        kernel_initializer: initializer,
        use_bias: false
      )

    intermediate =
      Axon.dense(hidden_state, units,
        name: join(name, "intermediate"),
        kernel_initializer: initializer,
        use_bias: false
      )

    gate
    |> Layers.activation(Map.get(spec, :activation, :gelu_approx_tanh),
      name: join(name, "activation")
    )
    |> Axon.multiply(intermediate, name: join(name, "product"))
    |> Axon.dense(hidden_size,
      name: join(name, "output"),
      kernel_initializer: Axon.Initializers.zeros(),
      use_bias: false
    )
  end

  @doc """
  One scalar gate per token: a single dense unit and a sigmoid, shaped
  `{batch, sequence, 1}` so it broadcasts over the expert output.

  At inference a gate under `:gate_floor` is clamped to exactly 0, so the
  expert term is exactly 0.0 and the block output is bit-identical to base
  Gemma; a floor of `1.0` therefore forces the router closed. In `:train` mode
  the floor is not applied, because `Nx.select` below it would leave the
  router with no gradient and nothing to reopen it with.
  """
  def router(hidden_state, opts) do
    name = opts[:name]
    gate_name = Keyword.get(opts, :gate_name, default_gate_name(name))
    floor = Keyword.get(opts, :gate_floor, @default_gate_floor)

    logit =
      Axon.dense(hidden_state, 1,
        name: name,
        kernel_initializer: Axon.Initializers.zeros(),
        bias_initializer: Axon.Initializers.full(@closed_router_bias),
        use_bias: true
      )

    Axon.layer(&gate_impl/2, [logit],
      name: gate_name,
      op_name: :system_one_gate,
      gate_floor: floor
    )
  end

  defp default_gate_name("" <> name) do
    case String.split(name, ".") do
      [_single] -> "gate"
      parts -> parts |> Enum.drop(-1) |> Enum.concat(["gate"]) |> Enum.join(".")
    end
  end

  @doc "The bias a router is initialised with, and what `closed?/1` compares against."
  def closed_router_bias, do: @closed_router_bias

  @doc """
  Clamps a gate tensor the way inference does: values under `floor` become
  exactly 0, the rest pass through.
  """
  def clamp_gate(gate, floor) when is_number(floor) do
    if floor <= 0.0 do
      gate
    else
      Nx.multiply(gate, Nx.as_type(Nx.greater_equal(gate, floor), Nx.type(gate)))
    end
  end

  @doc """
  Handles on the gate nodes of a built model, as `%{layer_index => %Axon{}}`.

  Axon has no way to name an intermediate output, so the gates are read back
  by rebuilding the model struct around the gate node's id. The trainer needs
  them as a second output (the gate losses are per token) and the evaluator
  reports their mean.
  """
  def gate_nodes(model, layer_indices), do: nodes_named(model, layer_indices, &gate_name/1)

  @doc """
  Handles on the routers' pre-sigmoid dense nodes, the same way as
  `gate_nodes/2`.

  The trainer supervises the gate with a binary cross-entropy in both
  directions, and read back off the sigmoid one of `log g` and `log (1 - g)`
  is always the log of a rounded-off zero: a router at -40 has `g == 0.0` in
  f32 exactly. `softplus` of the logit is the same two numbers without the
  cancellation, which is why the logit is an output of its own.
  """
  def router_logit_nodes(model, layer_indices),
    do: nodes_named(model, layer_indices, &router_name/1)

  defp nodes_named(%Axon{nodes: nodes} = model, layer_indices, name_of) do
    wanted = Map.new(layer_indices, &{name_of.(&1), &1})

    nodes
    |> Enum.flat_map(fn {id, node} ->
      case Map.fetch(wanted, node_name(node)) do
        {:ok, layer_index} -> [{layer_index, %{model | output: id}}]
        :error -> []
      end
    end)
    |> Map.new()
  end

  defp node_name(%Axon.Node{name: name, op_name: op_name}) when is_function(name, 2),
    do: name.(op_name, %{})

  defp node_name(%Axon.Node{name: name}), do: name

  defp gate_impl(logit, opts) do
    opts = Keyword.validate!(opts, [:gate_floor, mode: :inference])

    gate = stable_sigmoid(logit)

    case opts[:mode] do
      :train -> gate
      _inference -> clamp_gate(gate, opts[:gate_floor])
    end
  end

  # Nx differentiates `sigmoid` as `exp(-x) * s * s`, which is `inf * 0` for a
  # strongly negative pre-activation, and the router starts at -4 and is
  # pushed further out by the gate losses. Same forward pass, derivative
  # rewritten as `s * (1 - s)`; see `Gemma4E4B.AudioEncoder.stable_sigmoid/1`.
  defn stable_sigmoid(x) do
    s = Nx.sigmoid(x)
    custom_grad(s, [x], fn g -> [g * s * (1 - s)] end)
  end
end
