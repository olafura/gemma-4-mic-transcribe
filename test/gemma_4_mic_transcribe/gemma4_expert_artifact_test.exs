defmodule Gemma4MicTranscribe.Gemma4.ExpertArtifactTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4.ExpertArtifact
  alias Gemma4MicTranscribe.Gemma4.ExtractedExpert

  @config %{
    "text_config" => %{
      "enable_moe_block" => true,
      "num_hidden_layers" => 1,
      "num_experts" => 3,
      "top_k_experts" => 2,
      "hidden_size" => 8,
      "intermediate_size" => 12,
      "moe_intermediate_size" => 4,
      "hidden_activation" => "gelu_pytorch_tanh"
    }
  }

  @tag :tmp_dir
  test "range-extracts and independently runs one routed expert", %{tmp_dir: tmp_dir} do
    gate_up =
      Nx.iota({3, 8, 8}, type: :bf16)
      |> Nx.divide(100)

    down =
      Nx.iota({3, 8, 4}, type: :bf16)
      |> Nx.divide(100)

    source =
      %{
        "model.language_model.layers.0.experts.gate_up_proj" => gate_up,
        "model.language_model.layers.0.experts.down_proj" => down
      }
      |> Safetensors.dump()
      |> IO.iodata_to_binary()

    index = %{
      "metadata" => %{"total_size" => 51_611_872_412},
      "weight_map" => %{
        "model.language_model.layers.0.experts.gate_up_proj" => "model.safetensors",
        "model.language_model.layers.0.experts.down_proj" => "model.safetensors"
      }
    }

    fetch_json = fn url ->
      if String.ends_with?(url, "config.json"), do: @config, else: index
    end

    fetch_range = fn _url, first, last ->
      binary_part(source, first, last - first + 1)
    end

    artifact = Path.join(tmp_dir, "expert")

    manifest =
      ExpertArtifact.extract!(artifact,
        layer: 0,
        expert: 1,
        fetch_json: fetch_json,
        fetch_range: fetch_range
      )

    assert manifest.layer_index == 0
    assert manifest.expert_index == 1
    assert manifest.parameter_count == 96
    assert manifest.parameter_type == {:bf, 16}
    assert manifest.downloaded_bytes < byte_size(source)

    {_, params} = ExpertArtifact.load!(artifact)
    expected_gate_up = Nx.slice_along_axis(gate_up, 1, 1, axis: 0) |> Nx.squeeze(axes: [0])
    expected_down = Nx.slice_along_axis(down, 1, 1, axis: 0) |> Nx.squeeze(axes: [0])

    assert Nx.to_flat_list(params.gate) ==
             expected_gate_up |> Nx.slice_along_axis(0, 4, axis: 0) |> Nx.to_flat_list()

    assert Nx.to_flat_list(params.up) ==
             expected_gate_up |> Nx.slice_along_axis(4, 4, axis: 0) |> Nx.to_flat_list()

    assert Nx.to_flat_list(params.down) == Nx.to_flat_list(expected_down)

    expert = ExtractedExpert.load!(artifact)
    output = ExtractedExpert.run(expert, Nx.broadcast(0.01, {2, 8}))

    assert Nx.shape(output) == {2, 8}
    assert output |> Nx.as_type(:f32) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number() > 0
  end

  @tag :tmp_dir
  test "rejects an incomplete artifact manifest", %{tmp_dir: tmp_dir} do
    artifact = Path.join(tmp_dir, "expert")
    File.mkdir_p!(artifact)

    File.write!(
      Path.join(artifact, "manifest.etf"),
      :erlang.term_to_binary(%{version: 1, kind: :gemma4_routed_expert})
    )

    assert_raise KeyError, fn -> ExpertArtifact.load!(artifact) end
  end
end
