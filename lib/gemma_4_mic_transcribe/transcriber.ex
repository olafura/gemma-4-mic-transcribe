defmodule Gemma4MicTranscribe.Transcriber do
  @moduledoc false

  alias Gemma4MicTranscribe.Gemma4Unified.Input
  alias Gemma4MicTranscribe.Gemma4Unified.Runtime

  def transcribe_windows(windows, opts \\ []) do
    runtime_module = Keyword.get(opts, :runtime_module, Runtime)

    with {:ok, runtime} <- runtime_module.load(opts) do
      windows
      |> Enum.map(fn window ->
        input =
          Input.build(window.samples,
            prompt: Keyword.fetch!(opts, :prompt),
            system_message: Keyword.get(opts, :system_message)
          )

        case runtime_module.generate(runtime, input,
               timeout_seconds: Keyword.get(opts, :request_timeout_seconds)
             ) do
          {:ok, text} -> {:ok, window, String.trim(text)}
          {:error, reason} -> {:error, reason}
        end
      end)
      |> collect_results()
    end
  end

  defp collect_results(results) do
    case Enum.find(results, &match?({:error, _reason}, &1)) do
      nil -> {:ok, results}
      {:error, reason} -> {:error, reason}
    end
  end
end
