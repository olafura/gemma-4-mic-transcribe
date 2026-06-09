defmodule Gemma4MicTranscribe.Gemma4Unified.Transcript do
  @moduledoc false

  @non_transcript_lines MapSet.new([
                          "analysis",
                          "assistant",
                          "commentary",
                          "final",
                          "model",
                          "thinking",
                          "thought",
                          "transcript",
                          "```"
                        ])

  def clean(text) when is_binary(text) do
    text
    |> String.split(~r/\R+/)
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == ""))
    |> Enum.reject(&non_transcript_line?/1)
    |> Enum.join("\n")
    |> String.trim()
  end

  defp non_transcript_line?(line) do
    line
    |> String.replace(~r/^[\s:：]+|[\s:：]+$/, "")
    |> String.downcase()
    |> then(&MapSet.member?(@non_transcript_lines, &1))
  end
end
