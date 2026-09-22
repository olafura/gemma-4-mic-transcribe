defmodule Gemma4MicTranscribe.Gemma4.SystemOne.Prompt do
  @moduledoc """
  The one fixed template every System One item is rendered with.

  An item is a map with `"state"` (a JSON object, or a string already in the
  shape the author wants), `"question"`, and optionally `"options"` (names
  with descriptions, or a bare list of names).

  A replay item carries a bare `"prompt"` instead and no `"state"`: it is
  ordinary traffic the model must keep answering as it does today, so it is
  passed through untouched rather than wrapped in the System One template.

  The template deliberately says nothing about answering briefly or about
  asking a follow-up when the state does not decide the question. That
  behaviour is what the expert is trained to carry
  (`docs/system-one-expert-plan.md`), so putting it in the prompt would leave
  the expert and the prompt-only baseline measuring the same thing.
  """

  @doc "Renders one item into the user prompt text."
  def render(item) when is_map(item) do
    case {Map.get(item, "state"), Map.get(item, "prompt")} do
      {nil, prompt} when is_binary(prompt) -> String.trim(prompt)
      _ -> render_item(item)
    end
  end

  defp render_item(item) do
    [state_block(item), question(item)]
    |> Enum.concat(options_block(item))
    |> Enum.join("\n\n")
  end

  defp state_block(item) do
    "State: " <> state_json(fetch!(item, "state"))
  end

  # Compact, one line, so the state reads as data rather than prose. Maps of up
  # to 32 keys enumerate in key order, so the rendering is stable for a given
  # item.
  defp state_json(state) when is_binary(state), do: String.trim(state)
  defp state_json(state), do: Jason.encode!(state)

  defp question(item) do
    item |> fetch!("question") |> to_string() |> String.trim()
  end

  # Only the option names: their descriptions are the scorecard's business
  # (Laya is asked which option a reply commits to), not the model's.
  defp options_block(item) do
    case option_names(Map.get(item, "options")) do
      [] -> []
      names -> ["Options: " <> Enum.join(names, ", ")]
    end
  end

  defp option_names(nil), do: []
  defp option_names(options) when is_list(options), do: Enum.map(options, &to_string/1)
  defp option_names(options) when is_map(options), do: options |> Map.keys() |> Enum.sort()

  defp fetch!(item, key) do
    case Map.fetch(item, key) do
      {:ok, value} -> value
      :error -> raise ArgumentError, "System One item is missing #{inspect(key)}"
    end
  end
end
