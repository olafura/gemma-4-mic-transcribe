defmodule Gemma4MicTranscribe.Gemma4Unified.Runtime do
  @moduledoc false

  @unsupported """
  Gemma 4 Unified direct-audio inference is not supported by the current Elixir Bumblebee runtime in this app.

  The CLI already builds the Gemma 4 Unified audio input: raw 16 kHz waveform frames of 640 samples,
  <boa>/<eoa> prompt placeholders, and the text prompt. The remaining missing piece is a Bumblebee/Nx
  implementation of Gemma4UnifiedForConditionalGeneration, including the gemma4_unified_text backbone
  and multimodal RMSNorm -> Linear embedding injection.

  No Python, LiteRT, or Whisper fallback was used.
  """

  defstruct [:model_name, :backend, :max_response_tokens]

  def unsupported_message, do: String.trim(@unsupported)

  def load(opts \\ []) do
    runtime = %__MODULE__{
      model_name: Keyword.fetch!(opts, :model_name),
      backend: Keyword.get(opts, :backend, "host"),
      max_response_tokens: Keyword.get(opts, :max_response_tokens, 512)
    }

    case verify_bumblebee_available() do
      :ok -> {:error, {:unsupported_runtime, unsupported_message(), runtime}}
      {:error, reason} -> {:error, reason}
    end
  end

  def generate(_runtime, _input, _opts \\ []) do
    {:error, {:unsupported_runtime, unsupported_message()}}
  end

  defp verify_bumblebee_available do
    if Code.ensure_loaded?(Bumblebee) do
      :ok
    else
      {:error, {:missing_dependency, :bumblebee}}
    end
  end
end
