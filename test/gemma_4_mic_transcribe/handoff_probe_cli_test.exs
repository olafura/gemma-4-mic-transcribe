defmodule Gemma4MicTranscribe.HandoffProbeCLITest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.HandoffProbeCLI

  test "parses extraction separately from runtime execution" do
    assert {:extract, opts} =
             HandoffProbeCLI.parse([
               "extract",
               "--artifact",
               "artifacts/probe",
               "--revision",
               "abc123"
             ])

    assert opts.artifact == "artifacts/probe"
    assert opts.revision == "abc123"
    assert opts.repo == "Cactus-Compute/gemma-4-e2b-it-hybrid"
  end

  test "requires an artifact path" do
    assert {:error, "--artifact PATH is required"} = HandoffProbeCLI.parse(["extract"])
    assert {:error, "--artifact PATH is required"} = HandoffProbeCLI.parse(["inspect"])
  end
end
