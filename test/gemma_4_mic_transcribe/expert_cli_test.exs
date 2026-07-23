defmodule Gemma4MicTranscribe.ExpertCLITest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.ExpertCLI

  test "parses separate extraction and execution commands" do
    assert {:extract, extract} =
             ExpertCLI.parse([
               "extract",
               "--artifact",
               "expert",
               "--layer",
               "7",
               "--expert",
               "42"
             ])

    assert extract.layer == 7
    assert extract.expert == 42
    assert extract.repo == "google/gemma-4-26B-A4B-it"

    assert {:run, run} =
             ExpertCLI.parse([
               "run",
               "--artifact",
               "expert",
               "--backend",
               "exla:rocm",
               "--tokens",
               "4",
               "--runs",
               "2"
             ])

    assert run.backend == "exla:rocm"
    assert run.tokens == 4
    assert run.runs == 2
  end

  test "validates required paths and indices" do
    assert {:error, "--artifact PATH is required"} = ExpertCLI.parse(["extract"])

    assert {:error, "--expert must be non-negative"} =
             ExpertCLI.parse(["extract", "--artifact", "expert", "--expert", "-1"])
  end

  test "parses complete MoE layer extraction and execution commands" do
    assert {:extract_layer, extract} =
             ExpertCLI.parse([
               "extract-layer",
               "--artifact",
               "moe-layer",
               "--layer",
               "12"
             ])

    assert extract.layer == 12
    assert extract.repo == "google/gemma-4-26B-A4B-it"

    assert {:inspect_layer, "moe-layer"} =
             ExpertCLI.parse(["inspect-layer", "--artifact", "moe-layer"])

    assert {:run_layer, run} =
             ExpertCLI.parse([
               "run-layer",
               "--artifact",
               "moe-layer",
               "--backend",
               "exla:rocm",
               "--tokens",
               "3",
               "--runs",
               "5"
             ])

    assert run.backend == "exla:rocm"
    assert run.tokens == 3
    assert run.runs == 5

    assert {:extract_head, head} =
             ExpertCLI.parse(["extract-head", "--artifact", "output-head"])

    assert head.artifact == "output-head"
    assert head.repo == "google/gemma-4-26B-A4B-it"

    assert {:profile_math, profile} =
             ExpertCLI.parse([
               "profile-math",
               "--artifact",
               "moe-layer",
               "--backend",
               "exla:rocm",
               "--limit",
               "7"
             ])

    assert profile.backend == "exla:rocm"
    assert profile.limit == 7
  end

  test "parses caller extraction and a text-to-expert call" do
    assert {:extract_caller, extract} =
             ExpertCLI.parse(["extract-caller", "--artifact", "caller"])

    assert extract.artifact == "caller"
    assert extract.repo == "google/gemma-4-26B-A4B-it"
    assert extract.layer == 0

    assert {:call_expert, call} =
             ExpertCLI.parse([
               "call-expert",
               "--artifact",
               "moe-layer",
               "--caller-artifact",
               "caller",
               "--expert-artifact",
               "expert-112",
               "--text",
               "Prove the theorem."
             ])

    assert call.artifact == "moe-layer"
    assert call.caller_artifact == "caller"
    assert call.expert_artifact == "expert-112"
    assert call.text == "Prove the theorem."
    assert call.backend == "exla:rocm"

    assert {:call_layer, layer} =
             ExpertCLI.parse([
               "call-layer",
               "--artifact",
               "moe-layer",
               "--caller-artifact",
               "caller",
               "--expert-artifact",
               "expert-112",
               "--expert-scale",
               "0.5",
               "--text",
               "Prove the theorem."
             ])

    assert layer.artifact == "moe-layer"
    assert layer.expert_scale == 0.5
    assert layer.text == "Prove the theorem."

    assert {:call_chain, chain} =
             ExpertCLI.parse([
               "call-chain",
               "--artifact",
               "layer-0-moe",
               "--caller-artifact",
               "layer-0-caller",
               "--expert-artifact",
               "expert-112",
               "--next-artifact",
               "layer-1-moe",
               "--next-caller-artifact",
               "layer-1-caller",
               "--next-artifact",
               "layer-2-moe",
               "--next-caller-artifact",
               "layer-2-caller",
               "--text",
               "Prove the theorem."
             ])

    assert chain.layers == [
             %{caller_artifact: "layer-1-caller", moe_artifact: "layer-1-moe"},
             %{caller_artifact: "layer-2-caller", moe_artifact: "layer-2-moe"}
           ]

    assert chain.expert_scale == 1.0

    assert {:call_chain, prefix} =
             ExpertCLI.parse([
               "call-prefix",
               "--artifact-prefix",
               "artifacts/gemma4-26b",
               "--expert-artifact",
               "expert-112",
               "--last-layer",
               "3",
               "--head-artifact",
               "output-head",
               "--chat",
               "--expert-scale",
               "0.0",
               "--text",
               "Prove the theorem."
             ])

    assert prefix.artifact == "artifacts/gemma4-26b-layer0-moe"
    assert prefix.caller_artifact == "artifacts/gemma4-26b-layer0-caller"

    assert prefix.layers == [
             %{
               caller_artifact: "artifacts/gemma4-26b-layer1-caller",
               moe_artifact: "artifacts/gemma4-26b-layer1-moe"
             },
             %{
               caller_artifact: "artifacts/gemma4-26b-layer2-caller",
               moe_artifact: "artifacts/gemma4-26b-layer2-moe"
             },
             %{
               caller_artifact: "artifacts/gemma4-26b-layer3-caller",
               moe_artifact: "artifacts/gemma4-26b-layer3-moe"
             }
           ]

    assert prefix.expert_scale == 0.0
    assert prefix.head_artifact == "output-head"

    assert prefix.input_text ==
             "<|turn>user\nProve the theorem.<turn|>\n" <>
               "<|turn>model\n<|channel>thought\n<channel|>"

    assert {:generate_prefix, generation} =
             ExpertCLI.parse([
               "generate-prefix",
               "--artifact-prefix",
               "artifacts/gemma4-26b",
               "--expert-artifact",
               "expert-112",
               "--head-artifact",
               "output-head",
               "--max-new-tokens",
               "3",
               "--chat",
               "--expert-scale",
               "0.0",
               "--text",
               "Prove the theorem."
             ])

    assert generation.max_new_tokens == 3
    assert generation.expert_scale == 0.0
    assert generation.head_artifact == "output-head"
    assert generation.input_text == prefix.input_text
  end

  test "validates caller command paths and text" do
    assert {:error, "--caller-artifact PATH is required"} =
             ExpertCLI.parse([
               "call-expert",
               "--artifact",
               "moe-layer",
               "--expert-artifact",
               "expert-112",
               "--text",
               "x"
             ])

    assert {:error, "--next-artifact and --next-caller-artifact counts must match"} =
             ExpertCLI.parse([
               "call-chain",
               "--artifact",
               "layer-0-moe",
               "--caller-artifact",
               "layer-0-caller",
               "--expert-artifact",
               "expert-112",
               "--next-artifact",
               "layer-1-moe",
               "--next-artifact",
               "layer-2-moe",
               "--next-caller-artifact",
               "layer-1-caller",
               "--text",
               "x"
             ])

    assert {:error, "--last-layer must be an integer from 1 through 29"} =
             ExpertCLI.parse([
               "call-prefix",
               "--artifact-prefix",
               "artifacts/gemma4-26b",
               "--expert-artifact",
               "expert-112",
               "--last-layer",
               "30",
               "--text",
               "x"
             ])
  end
end
