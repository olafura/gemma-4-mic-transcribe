defmodule Gemma4MicTranscribe.Gemma4.SystemOne.Router do
  @moduledoc """
  A router in front of unmodified Gemma that picks, per request, whether to
  answer now, reason first or ask back (`docs/system-one-expert-plan.md`,
  "Route 1"). Nothing about the model changes; the router only chooses which
  prompt Gemma is given.

    1. Render the request in the direct form, which asks for a single
       `Answer: <x>` line, and run the prefix over it.
    2. **Ask back** when the ask-back probe, a logistic regression on the
       prefix output at the last prompt token, scores above its threshold:
       the state does not settle the question.
    3. Otherwise generate the direct answer and read its confidence, the
       lowest probability among the answer tokens. **Answer now** at or above
       the cutoff.
    4. Otherwise **reason**: the request again, asking for a worked reply that
       ends in an `Answer:` line.

  The probe was trained on direct-form prompts and only works on that
  wording, so `body/1` and `direct_prompt/1` must not change without
  retraining it (`scripts/system_one/export_ask_probe.py`).
  """

  @ask_instruction "The information above does not settle this. Reply with only the one short " <>
                     "question you would ask to settle it, and nothing else."

  @doc """
  The answer's shape as the instructions name it: the row's `"answer"`
  (`number`, `letter`, ...), else `option name` for a System One item and
  `answer` for a bare prompt.
  """
  def answer_form(%{"answer" => form}) when is_binary(form), do: form
  def answer_form(%{"state" => _state}), do: "option name"
  def answer_form(_row), do: "answer"

  @doc """
  The request itself, in the wording the probe was trained on
  (`scripts/system_one/build_router_probe_set.py`): a bare `"prompt"` as it
  is, or a System One item as its state, its question and its options with
  their descriptions. Unlike `SystemOne.Prompt`, which names the options
  only, the router has no scorecard to hold the descriptions, so Gemma is
  shown them.
  """
  def body(%{"state" => state} = row) do
    ["State: " <> state_json(state), row |> fetch!("question") |> to_string() |> String.trim()]
    |> Enum.concat(options_block(Map.get(row, "options")))
    |> Enum.join("\n\n")
  end

  def body(%{"prompt" => prompt}) when is_binary(prompt), do: String.trim(prompt)

  def body(_row), do: raise(ArgumentError, "a request needs a \"prompt\" or a \"state\"")

  @doc "The request with the one-line answer instruction; the probe reads this prompt."
  def direct_prompt(row) do
    body(row) <>
      "\n\nReply with only one line of the form 'Answer: <#{answer_form(row)}>' and nothing else."
  end

  @doc "The request with room to reason, ending in an answer line."
  def reason_prompt(row) do
    body(row) <>
      "\n\nEnd your reply with a line of the form 'Answer: <#{answer_form(row)}>'."
  end

  @doc "The request with an instruction to ask the one follow-up question that would settle it."
  def ask_prompt(row), do: body(row) <> "\n\n" <> @ask_instruction

  defp state_json(state) when is_binary(state), do: String.trim(state)
  defp state_json(state), do: Jason.encode!(state)

  defp options_block(nil), do: []
  defp options_block([]), do: []
  defp options_block(options) when map_size(options) == 0, do: []

  defp options_block(options) when is_list(options),
    do: ["Options:\n" <> Enum.map_join(options, "\n", &"- #{&1}")]

  defp options_block(options) when is_map(options) do
    lines =
      options
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.map_join("\n", fn {name, text} -> "- #{name}: #{String.trim(to_string(text))}" end)

    ["Options:\n" <> lines]
  end

  defp fetch!(row, key) do
    case Map.fetch(row, key) do
      {:ok, value} -> value
      :error -> raise ArgumentError, "a System One request is missing #{inspect(key)}"
    end
  end

  @doc """
  Loads a probe written by `scripts/system_one/export_ask_probe.py`: a folded
  weight vector, a bias, and the ask-back threshold it was exported with.
  """
  def load_probe!(dir) do
    meta = dir |> Path.join("probe.json") |> File.read!() |> Jason.decode!()

    tensors =
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        Safetensors.read!(Path.join(dir, "probe.safetensors"))
      end)

    weight = Nx.backend_copy(tensors["weight"], Nx.BinaryBackend)

    if Nx.shape(weight) != {meta["hidden_size"]} do
      raise ArgumentError,
            "probe weight has shape #{inspect(Nx.shape(weight))}, expected {#{meta["hidden_size"]}}"
    end

    %{
      weight: weight,
      bias: tensors["bias"] |> Nx.to_flat_list() |> hd(),
      threshold: meta["threshold"],
      meta: meta
    }
  end

  @doc "The probability that the state does not settle the request, for one hidden-state vector."
  def probe_score(%{weight: weight, bias: bias}, vector) do
    logit =
      vector
      |> Nx.backend_copy(Nx.BinaryBackend)
      |> Nx.as_type(:f32)
      |> Nx.dot(weight)
      |> Nx.to_number()

    1.0 / (1.0 + :math.exp(-(logit + bias)))
  end

  @doc """
  The direct answer's confidence: the lowest probability among the tokens
  after `Answer:` that print something, given each generated token's text
  and log-probability. A reply with no `Answer:` line has confidence 0.0, so
  it is always reasoned.
  """
  def answer_confidence(pieces, logprobs) when length(pieces) == length(logprobs) do
    case answer_start(pieces) do
      nil ->
        0.0

      start ->
        pieces
        |> Enum.zip(logprobs)
        |> Enum.drop(start)
        |> Enum.filter(fn {piece, logprob} -> String.trim(piece) != "" and is_number(logprob) end)
        |> case do
          [] -> 0.0
          kept -> kept |> Enum.map(&elem(&1, 1)) |> Enum.min() |> :math.exp()
        end
    end
  end

  defp answer_start(pieces) do
    pieces
    |> Enum.scan("", fn piece, text -> text <> piece end)
    |> Enum.find_index(&String.contains?(&1, "Answer:"))
    |> case do
      nil -> nil
      index -> index + 1
    end
  end

  @doc "Ask back when the probe scores strictly above the threshold."
  def ask?(ask_score, threshold), do: ask_score > threshold

  @doc "Answer now when the direct answer's confidence reaches the cutoff."
  def answer_now?(confidence, cutoff), do: confidence >= cutoff
end
