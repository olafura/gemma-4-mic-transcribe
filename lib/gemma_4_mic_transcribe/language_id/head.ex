defmodule Gemma4MicTranscribe.LanguageId.Head do
  @moduledoc """
  A multinomial logistic-regression head over pooled encoder features.

  Inputs are standardized with training-set statistics, then projected to
  one logit per language. The head is trained full-batch with Adam and L2
  weight decay; at a few thousand clips and a 1024-wide feature this takes
  seconds, so it can be retrained for every candidate encoder depth.
  """

  import Nx.Defn

  defstruct [:mean, :std, :kernel, :bias, :languages]

  @type t :: %__MODULE__{}

  @doc """
  Trains a head on `x` (`{n, d}`) with integer labels `y` (`{n}`).

  Options: `:steps` (default 400), `:learning_rate` (0.01), `:weight_decay`
  (1.0e-3), `:seed`, `:classes` (defaults to `max(y) + 1`).
  """
  def train(x, y, languages, opts \\ []) do
    steps = Keyword.get(opts, :steps, 400)
    learning_rate = Keyword.get(opts, :learning_rate, 0.01)
    weight_decay = Keyword.get(opts, :weight_decay, 1.0e-3)
    classes = Keyword.get(opts, :classes, length(languages))

    x = Nx.as_type(x, :f32)
    y = Nx.as_type(y, :s64)
    {_n, d} = Nx.shape(x)

    mean = Nx.mean(x, axes: [0])
    std = x |> Nx.standard_deviation(axes: [0]) |> Nx.max(1.0e-6)
    x = standardize(x, mean, std)

    state = %{
      kernel: Nx.broadcast(Nx.tensor(0.0, type: :f32), {d, classes}),
      bias: Nx.broadcast(Nx.tensor(0.0, type: :f32), {classes}),
      m_kernel: Nx.broadcast(Nx.tensor(0.0, type: :f32), {d, classes}),
      v_kernel: Nx.broadcast(Nx.tensor(0.0, type: :f32), {d, classes}),
      m_bias: Nx.broadcast(Nx.tensor(0.0, type: :f32), {classes}),
      v_bias: Nx.broadcast(Nx.tensor(0.0, type: :f32), {classes})
    }

    hyper = [learning_rate: learning_rate, weight_decay: weight_decay, classes: classes]

    state =
      Enum.reduce(1..steps, state, fn step, state ->
        adam_step(state, x, y, Nx.tensor(step, type: :f32), hyper)
      end)

    %__MODULE__{
      mean: mean,
      std: std,
      kernel: state.kernel,
      bias: state.bias,
      languages: languages
    }
  end

  defnp standardize(x, mean, std), do: (x - mean) / std

  defnp loss(kernel, bias, x, y, opts \\ []) do
    opts = keyword!(opts, [:classes, :weight_decay])
    logits = Nx.dot(x, kernel) + bias
    log_probs = logits - Nx.logsumexp(logits, axes: [1], keep_axes: true)
    onehot = Nx.equal(Nx.new_axis(y, 1), Nx.iota({1, opts[:classes]}))
    nll = -Nx.mean(Nx.sum(log_probs * onehot, axes: [1]))
    nll + opts[:weight_decay] * Nx.sum(kernel * kernel) / 2
  end

  defnp adam_step(state, x, y, step, opts \\ []) do
    opts = keyword!(opts, [:classes, :weight_decay, :learning_rate])

    {grad_kernel, grad_bias} =
      grad({state.kernel, state.bias}, fn {kernel, bias} ->
        loss(kernel, bias, x, y, classes: opts[:classes], weight_decay: opts[:weight_decay])
      end)

    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8

    m_kernel = beta1 * state.m_kernel + (1 - beta1) * grad_kernel
    v_kernel = beta2 * state.v_kernel + (1 - beta2) * grad_kernel * grad_kernel
    m_bias = beta1 * state.m_bias + (1 - beta1) * grad_bias
    v_bias = beta2 * state.v_bias + (1 - beta2) * grad_bias * grad_bias

    correction1 = 1 - Nx.pow(beta1, step)
    correction2 = 1 - Nx.pow(beta2, step)
    lr = opts[:learning_rate] * Nx.sqrt(correction2) / correction1

    %{
      state
      | kernel: state.kernel - lr * m_kernel / (Nx.sqrt(v_kernel) + eps),
        bias: state.bias - lr * m_bias / (Nx.sqrt(v_bias) + eps),
        m_kernel: m_kernel,
        v_kernel: v_kernel,
        m_bias: m_bias,
        v_bias: v_bias
    }
  end

  @doc "Class log-probabilities, `{n, classes}`."
  def log_probs(%__MODULE__{} = head, x) do
    log_probs_impl(Nx.as_type(x, :f32), head.mean, head.std, head.kernel, head.bias)
  end

  defnp log_probs_impl(x, mean, std, kernel, bias) do
    logits = Nx.dot(standardize(x, mean, std), kernel) + bias
    logits - Nx.logsumexp(logits, axes: [1], keep_axes: true)
  end

  @doc "Predicted class index per row."
  def predict(%__MODULE__{} = head, x) do
    head |> log_probs(x) |> Nx.argmax(axis: 1)
  end

  @doc """
  Accuracy, macro-averaged per-language accuracy, and per-language details on
  labeled data.
  """
  def evaluate(%__MODULE__{} = head, x, y) do
    predicted = head |> predict(x) |> Nx.to_flat_list()
    labels = Nx.to_flat_list(y)
    pairs = Enum.zip(labels, predicted)
    correct = Enum.count(pairs, fn {label, prediction} -> label == prediction end)

    per_language =
      pairs
      |> Enum.group_by(fn {label, _prediction} -> label end)
      |> Enum.map(fn {label, group} ->
        hits = Enum.count(group, fn {l, p} -> l == p end)

        confusions =
          group
          |> Enum.reject(fn {l, p} -> l == p end)
          |> Enum.frequencies_by(fn {_l, p} -> Enum.at(head.languages, p) end)

        {Enum.at(head.languages, label),
         %{samples: length(group), correct: hits, accuracy: hits / length(group), confusions: confusions}}
      end)
      |> Map.new()

    macro =
      if map_size(per_language) == 0,
        do: 0.0,
        else: per_language |> Map.values() |> Enum.map(& &1.accuracy) |> Enum.sum() |> Kernel./(map_size(per_language))

    %{
      samples: length(pairs),
      correct: correct,
      accuracy: if(pairs == [], do: 0.0, else: correct / length(pairs)),
      macro_accuracy: macro,
      per_language: per_language
    }
  end

  @doc "Parameter count of the head, including the standardization vectors."
  def parameter_count(%__MODULE__{} = head) do
    Nx.size(head.mean) + Nx.size(head.std) + Nx.size(head.kernel) + Nx.size(head.bias)
  end

  def to_tensors(%__MODULE__{} = head) do
    %{
      "head.mean" => head.mean,
      "head.std" => head.std,
      "head.kernel" => head.kernel,
      "head.bias" => head.bias
    }
  end

  def from_tensors(tensors, languages) do
    %__MODULE__{
      mean: Map.fetch!(tensors, "head.mean"),
      std: Map.fetch!(tensors, "head.std"),
      kernel: Map.fetch!(tensors, "head.kernel"),
      bias: Map.fetch!(tensors, "head.bias"),
      languages: languages
    }
  end
end
