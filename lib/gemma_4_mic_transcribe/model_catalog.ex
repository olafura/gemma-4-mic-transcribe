defmodule Gemma4MicTranscribe.ModelCatalog do
  @moduledoc false

  alias Gemma4MicTranscribe.Config

  @models [
    %{
      name: "gemma4-12b-unified",
      hf_repo: "google/gemma-4-12B-it",
      description: "Gemma 4 12B instruction-tuned Unified model with native audio input",
      artifact_format: :transformers_safetensors,
      runtime_kind: :bumblebee_axon,
      runtime: "local Bumblebee/Axon Gemma4Unified audio runtime"
    },
    %{
      name: "gemma4-12b-qat-q4_0-gguf",
      hf_repo: "google/gemma-4-12B-it-qat-q4_0-gguf",
      description: "Gemma 4 12B QAT Q4_0 GGUF",
      artifact_format: :gguf,
      runtime_kind: :llama_cpp,
      runtime: "llama.cpp/Ollama/LM Studio GGUF runtime"
    },
    %{
      name: "gemma4-12b-qat-w4a16-ct",
      hf_repo: "google/gemma-4-12B-it-qat-w4a16-ct",
      description: "Gemma 4 12B QAT w4a16 compressed tensors",
      artifact_format: :compressed_tensors,
      runtime_kind: :bumblebee_axon,
      runtime: "local Bumblebee/Axon Gemma4Unified audio runtime with compressed-tensors unpacking"
    }
  ]

  def default_model_name, do: Config.default_model_name()
  def all, do: @models

  def get(name) do
    Enum.find(@models, &(&1.name == name || &1.hf_repo == name))
  end

  def resolve(name) do
    case get(name) do
      nil -> name
      model -> model.hf_repo
    end
  end

  def runtime_kind(name) do
    case get(name) do
      nil -> :bumblebee_axon
      model -> model.runtime_kind
    end
  end

  def artifact_format(name) do
    case get(name) do
      nil -> :transformers_safetensors
      model -> model.artifact_format
    end
  end

  def runtime_module(name) do
    case runtime_kind(name) do
      :bumblebee_axon ->
        {:ok, Gemma4MicTranscribe.Gemma4Unified.Runtime}

      :llama_cpp ->
        {:error,
         "#{name} is a GGUF artifact. This CLI does not include a llama.cpp runtime bridge yet; use llama.cpp/Ollama/LM Studio for this repo."}
    end
  end

  def format do
    [
      "Gemma 4 model variants:",
      "  default: #{Config.default_model_name()}",
      ""
      | Enum.flat_map(@models, fn model ->
          [
            "  #{model.name}: #{model.description}",
            "    repo: #{model.hf_repo}",
            "    artifact: #{model.artifact_format}",
            "    runtime: #{model.runtime}"
          ]
        end)
    ]
    |> Enum.join("\n")
  end
end
