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
    env("GEMMA_BACKEND", "torchx")
  end

  def speech_gate? do
    boolean_env("SPEECH_GATE", true)
  end

  def min_speech_seconds do
    positive_float_env("MIN_SPEECH_SECONDS", 0.25)
  end

  def speech_threshold do
    positive_float_env("SPEECH_THRESHOLD", 0.01)
  end

  def speech_min_active_ratio do
    ratio_env("SPEECH_MIN_ACTIVE_RATIO", 0.2)
  end

  def speech_max_zero_crossing_rate do
    ratio_env("SPEECH_MAX_ZERO_CROSSING_RATE", 0.35)
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

  defp ratio_env(name, default) do
    case Float.parse(System.get_env(name, "")) do
      {value, ""} when value >= 0.0 and value <= 1.0 -> value
      _ -> default
    end
  end

  defp boolean_env(name, default) do
    case String.downcase(System.get_env(name, "")) do
      "" -> default
      value when value in ["1", "true", "yes", "on"] -> true
      value when value in ["0", "false", "no", "off"] -> false
      _other -> default
    end
  end
end
