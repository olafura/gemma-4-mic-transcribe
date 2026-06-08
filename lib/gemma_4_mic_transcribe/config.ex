defmodule Gemma4MicTranscribe.Config do
  @moduledoc false

  @default_model_name "google/gemma-4-12B-it"
  @default_prompt "Transcribe the spoken audio exactly. Return only the transcript for this audio window."

  def default_model_name, do: @default_model_name
  def default_prompt, do: @default_prompt

  def max_response_tokens do
    positive_int_env("MAX_RESPONSE_TOKENS", 512)
  end

  def request_timeout_seconds do
    positive_float_env("REQUEST_TIMEOUT_SECONDS", 30.0)
  end

  def backend do
    env("GEMMA_BACKEND", "host")
  end

  def model_cache_dir do
    case System.get_env("MODEL_CACHE_DIR") do
      nil -> nil
      "" -> nil
      value -> value
    end
  end

  defp env(name, default) do
    case System.get_env(name) do
      nil -> default
      "" -> default
      value -> value
    end
  end

  defp positive_int_env(name, default) do
    case Integer.parse(System.get_env(name, "")) do
      {value, ""} when value > 0 -> value
      _ -> default
    end
  end

  defp positive_float_env(name, default) do
    case Float.parse(System.get_env(name, "")) do
      {value, ""} when value > 0 -> value
      _ -> default
    end
  end
end
