defmodule Gemma4MicTranscribe.ModelCatalog do
  @moduledoc false

  alias Gemma4MicTranscribe.Config

  @models [
    %{
      name: "gemma4-12b-unified",
      hf_repo: "google/gemma-4-12B-it",
      description: "Gemma 4 12B instruction-tuned Unified model with native audio input",
      runtime: "local Bumblebee/Axon Gemma4Unified audio runtime"
    },
    %{
      name: "gemma4-12b-qat-q4_0-gguf",
      hf_repo: "google/gemma-4-12B-it-qat-q4_0-gguf",
      description: "Gemma 4 12B QAT Q4_0 GGUF",
      runtime: "non-Bumblebee GGUF runtimes only"
    },
    %{
      name: "gemma4-12b-qat-w4a16-ct",
      hf_repo: "google/gemma-4-12B-it-qat-w4a16-ct",
      description: "Gemma 4 12B QAT w4a16 compressed tensors",
      runtime: "Transformers/vLLM compressed-tensors workflows"
    }
  ]

  def default_model_name, do: Config.default_model_name()
  def all, do: @models

  def resolve(name) do
    case Enum.find(@models, &(&1.name == name || &1.hf_repo == name)) do
      nil -> name
      model -> model.hf_repo
    end
  end

  def format do
    [
      "Gemma 4 direct-audio models:",
      "  default: #{Config.default_model_name()}",
      ""
      | Enum.flat_map(@models, fn model ->
          [
            "  #{model.name}: #{model.description}",
            "    repo: #{model.hf_repo}",
            "    runtime: #{model.runtime}"
          ]
        end)
    ]
    |> Enum.join("\n")
  end
end
