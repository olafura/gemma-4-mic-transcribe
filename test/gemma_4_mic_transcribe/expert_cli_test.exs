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
end
