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

  @spoken_question "The question is spoken in the audio that follows."
  @spoken_request "The request is spoken in the audio that follows."

  @expert_system_message "Answer in one short line. Name exactly one of the options you are given. " <>
                           "If the state you are given does not determine the answer, do not guess: " <>
                           "ask one short question for the missing detail instead."

  @question_marks "?？؟⁇⁈⁉፧"
  @sentence_ends "." <> @question_marks <> "!！。۔\n"
  @closers " \t\"'”’)]}»」』*_"

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

  A spoken request carries `"audio"`, a WAV the caller puts in the audio
  slot after this text. The written question is then replaced by a line
  that points at the audio, as in `SystemOne.Prompt.render_audio/1`; a
  spoken request without a state is the audio alone, after any `"prompt"`
  given as context.
  """
  def body(%{"audio" => audio, "state" => state} = row) when is_binary(audio) do
    ["State: " <> state_json(state), @spoken_question]
    |> Enum.concat(options_block(Map.get(row, "options")))
    |> Enum.join("\n\n")
  end

  def body(%{"audio" => audio} = row) when is_binary(audio) do
    case Map.get(row, "prompt") do
      prompt when is_binary(prompt) -> String.trim(prompt) <> "\n\n" <> @spoken_request
      _none -> @spoken_request
    end
  end

  def body(%{"state" => state} = row) do
    ["State: " <> state_json(state), row |> fetch!("question") |> to_string() |> String.trim()]
    |> Enum.concat(options_block(Map.get(row, "options")))
    |> Enum.join("\n\n")
  end

  def body(%{"prompt" => prompt}) when is_binary(prompt), do: String.trim(prompt)

  def body(_row), do: raise(ArgumentError, "a request needs a \"prompt\" or a \"state\"")

  @doc "Whether the request is spoken: it carries an `\"audio\"` WAV path."
  def spoken?(%{"audio" => audio}) when is_binary(audio), do: true
  def spoken?(_row), do: false

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

  @doc """
  Whether the round-3 expert is consulted: a System One item whose probe
  score is above `low` but not above the ask threshold. The expert asks on
  more unclear items than the probe but also on many clear ones; below the
  band the probe is trusted to answer (`docs/system-one-expert-plan.md`,
  "The router and the expert together").
  """
  def expert_band?(%{"state" => _state}, ask_score, low, threshold),
    do: ask_score > low and ask_score <= threshold

  def expert_band?(_row, _ask_score, _low, _threshold), do: false

  @doc "The system turn the round-3 expert was evaluated with."
  def expert_system_message, do: @expert_system_message

  @doc """
  True when the reply's last sentence is a question: the deterministic rule
  of `scripts/system_one/scorecard.py` (`asks_question_rule`), which agrees
  with the full judge on every held-out expert reply.
  """
  def asks_question?(reply) when is_binary(reply) do
    sentence = last_sentence(reply)
    bare = trim_chars(sentence, "", @closers)

    cond do
      sentence == "" -> false
      String.contains?(sentence, "¿") -> true
      String.ends_with?(bare, String.graphemes(@question_marks)) -> true
      bare |> trim_chars("", "。.．") |> String.ends_with?(["か", "の", "カ"]) -> true
      true -> question_opening?(sentence)
    end
  end

  defp last_sentence(reply) do
    ~r/[^.?？؟⁇⁈⁉፧!！。۔\n]+[.?？؟⁇⁈⁉፧!！。۔\n]*/u
    |> Regex.scan(reply)
    |> Enum.map(fn [part] -> String.trim(part) end)
    |> Enum.filter(
      &(trim_chars(&1, @closers <> @sentence_ends, @closers <> @sentence_ends) != "")
    )
    |> List.last("")
  end

  defp question_opening?(sentence) do
    head = ~r/^[^\w¿]+/u |> Regex.replace(sentence, "") |> String.downcase()

    Regex.match?(~r/^(which|what|whats|who|whom|whose|when|where|why|how)\b/u, head) or
      Regex.match?(
        ~r/^(do|does|did|is|are|was|were|can|could|will|would|should|shall|may|might|have|has|had|am)\s+(i|you|we|they|he|she|it|there|this|that|these|those|the|your|his|her|their|any)\b/u,
        head
      ) or
      Regex.match?(
        ~r/^(could|can|would)\s+you\b|^(please\s+)?(tell|let)\s+(me|us)\b|^i\s+(need|would\s+need)\s+to\s+know\b|^(please\s+)?(clarify|specify|confirm)\b|^before\s+i\b.*\b(tell|know|confirm|clarify)\b/u,
        head
      )
  end

  # Python's str.strip(chars): drop any of the leading and trailing characters.
  defp trim_chars(text, leading, trailing) do
    text
    |> drop_while_in(String.graphemes(leading))
    |> String.reverse()
    |> drop_while_in(String.graphemes(trailing))
    |> String.reverse()
  end

  defp drop_while_in(text, []), do: text

  defp drop_while_in(text, chars) do
    case String.next_grapheme(text) do
      {first, rest} -> if first in chars, do: drop_while_in(rest, chars), else: text
      nil -> text
    end
  end

  @doc "Answer now when the direct answer's confidence reaches the cutoff."
  def answer_now?(confidence, cutoff), do: confidence >= cutoff
end
