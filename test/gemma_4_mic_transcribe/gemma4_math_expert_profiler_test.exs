defmodule Gemma4MicTranscribe.Gemma4.MathExpertProfilerTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4.MathExpertProfiler

  test "ranks experts by math-specific selection and probability lift" do
    math = %{
      router_probabilities:
        Nx.tensor([
          [0.05, 0.15, 0.80],
          [0.10, 0.20, 0.70],
          [0.05, 0.25, 0.70]
        ]),
      top_k_indices: Nx.tensor([[2], [2], [2]]),
      top_k_weights: Nx.tensor([[1.0], [1.0], [1.0]])
    }

    control = %{
      router_probabilities:
        Nx.tensor([
          [0.45, 0.45, 0.10],
          [0.55, 0.35, 0.10],
          [0.40, 0.50, 0.10]
        ]),
      top_k_indices: Nx.tensor([[0], [0], [1]]),
      top_k_weights: Nx.tensor([[1.0], [1.0], [1.0]])
    }

    [candidate | _] = MathExpertProfiler.rank(math, control, 3)

    assert candidate.expert == 2
    assert candidate.math_selections == 3
    assert candidate.control_selections == 0
    assert candidate.selection_lift == 1.0
    assert_in_delta candidate.probability_lift, 0.633333, 1.0e-5
  end
end
