defmodule Gemma4MicTranscribe.HandoffProbeTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.HandoffProbe
  alias Gemma4MicTranscribe.HandoffProbe.Artifact

  @tag :tmp_dir
  test "range-extracts a standalone artifact and scores captured rows", %{tmp_dir: tmp_dir} do
    source = source_tensors() |> Safetensors.dump() |> IO.iodata_to_binary()
    artifact_path = Path.join(tmp_dir, "probe")

    fetch_range = fn first, last -> binary_part(source, first, last - first + 1) end

    manifest =
      Artifact.extract!(artifact_path,
        url: "memory://cactus-probe",
        fetch_range: fetch_range
      )

    assert manifest.kind == :gemma4_e2b_handoff_probe
    assert manifest.probe_layer == 28
    assert manifest.feature_size == 1536
    assert manifest.parameter_count == 64_833
    assert manifest.downloaded_bytes < byte_size(source)

    probe = HandoffProbe.load!(artifact_path, Nx.BinaryBackend)
    confidence = HandoffProbe.score(probe, [Nx.broadcast(3.0, {1, 1536})])

    assert_in_delta confidence, 0.5, 1.0e-6
  end

  @tag :tmp_dir
  test "rejects changes to standalone probe parameters", %{tmp_dir: tmp_dir} do
    source = source_tensors() |> Safetensors.dump() |> IO.iodata_to_binary()
    artifact_path = Path.join(tmp_dir, "probe")
    fetch_range = fn first, last -> binary_part(source, first, last - first + 1) end

    Artifact.extract!(artifact_path, fetch_range: fetch_range)
    File.write!(Path.join(artifact_path, "parameters.safetensors"), "corrupt")

    assert_raise ArgumentError, ~r/checksum mismatch/, fn ->
      Artifact.load!(artifact_path, Nx.BinaryBackend)
    end
  end

  test "empty captures are not scoreable" do
    probe = %HandoffProbe{manifest: %{max_tokens: 1024}}
    assert HandoffProbe.score(probe, []) == nil
  end

  defp source_tensors do
    %{
      "handoff_probe.attn_query" => Nx.broadcast(0.0, {32}),
      "handoff_probe.head.0.bias" => Nx.broadcast(0.0, {128}),
      "handoff_probe.head.0.weight" => Nx.broadcast(0.0, {128, 32}),
      "handoff_probe.head.2.bias" => Nx.broadcast(0.0, {64}),
      "handoff_probe.head.2.weight" => Nx.broadcast(0.0, {64, 128}),
      "handoff_probe.head.4.bias" => Nx.broadcast(0.0, {1}),
      "handoff_probe.head.4.weight" => Nx.broadcast(0.0, {1, 64}),
      "handoff_probe.norm.bias" => Nx.broadcast(0.0, {1536}),
      "handoff_probe.norm.weight" => Nx.broadcast(1.0, {1536}),
      "handoff_probe.proj.bias" => Nx.broadcast(0.0, {32}),
      "handoff_probe.proj.weight" => Nx.broadcast(0.0, {32, 1536}),
      # Proves extraction does not download unrelated checkpoint data.
      "z.base_model.weight" => Nx.broadcast(0.0, {262_144})
    }
  end
end
