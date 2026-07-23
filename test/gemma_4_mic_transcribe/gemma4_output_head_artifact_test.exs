defmodule Gemma4MicTranscribe.Gemma4.OutputHeadArtifactTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4.ExtractedOutputHead
  alias Gemma4MicTranscribe.Gemma4.OutputHeadArtifact

  @hidden 4
  @vocab 8
  @eps 1.0e-6
  @softcap 30.0

  @tag :tmp_dir
  test "range-extracts and runs the tied output head", %{tmp_dir: tmp_dir} do
    embedding =
      Nx.tensor(
        [
          [0.5, -0.25, 0.125, 0.75],
          [-0.5, 0.75, 0.25, -0.125],
          [0.25, 0.5, -0.75, 0.375],
          [-0.125, -0.5, 0.625, 0.25],
          [0.75, 0.25, 0.5, -0.25],
          [-0.25, 0.125, 0.75, 0.5],
          [0.375, -0.75, -0.25, 0.125],
          [0.125, 0.375, -0.5, 0.75]
        ],
        type: :bf16
      )

    norm = Nx.tensor([0.75, 1.0, 1.25, 1.5], type: :bf16)

    tensors = %{
      "model.language_model.embed_tokens.weight" => embedding,
      "model.language_model.norm.weight" => norm
    }

    source = tensors |> Safetensors.dump() |> IO.iodata_to_binary()

    config = %{
      "text_config" => %{
        "hidden_size" => @hidden,
        "vocab_size" => @vocab,
        "rms_norm_eps" => @eps,
        "final_logit_softcapping" => @softcap
      }
    }

    index = %{
      "metadata" => %{"total_size" => byte_size(source)},
      "weight_map" => Map.new(tensors, fn {name, _tensor} -> {name, "model.safetensors"} end)
    }

    fetch_json = fn url ->
      if String.ends_with?(url, "config.json"), do: config, else: index
    end

    fetch_range = fn _url, first, last ->
      binary_part(source, first, last - first + 1)
    end

    artifact = Path.join(tmp_dir, "output-head")

    manifest =
      OutputHeadArtifact.extract!(artifact,
        fetch_json: fetch_json,
        fetch_range: fetch_range
      )

    assert manifest.parameter_count == @vocab * @hidden + @hidden
    assert manifest.parameter_bytes == (@vocab * @hidden + @hidden) * 2
    assert length(manifest.source_ranges) == 2

    head = ExtractedOutputHead.load!(artifact, Torchx.Backend)

    hidden =
      Nx.tensor(
        [
          [10.0, 10.0, 10.0, 10.0],
          [0.25, -0.5, 0.75, 1.0]
        ],
        type: :bf16
      )

    expected_raw = reference_raw_logits(hidden, embedding, norm)
    expected = softcap(expected_raw)
    {expected_raw_values, expected_indices} = Nx.top_k(expected_raw, k: 3)
    expected_values = softcap(expected_raw_values)
    result = ExtractedOutputHead.run(head, hidden, top_k: 3)

    assert_all_close(result.logits, expected, 1.0e-4)
    assert_all_close(result.raw_logits, expected_raw, 1.0e-4)
    assert_all_close(result.top_k_values, expected_values, 1.0e-4)
    assert_all_close(result.raw_top_k_values, expected_raw_values, 1.0e-4)
    assert Nx.to_flat_list(result.top_k_indices) == Nx.to_flat_list(expected_indices)
  end

  defp reference_raw_logits(hidden, embedding, norm) do
    hidden = Nx.slice_along_axis(hidden, 1, 1, axis: 0)
    hidden_f32 = Nx.as_type(hidden, :f32)

    normalized =
      hidden_f32
      |> Nx.pow(2)
      |> Nx.mean(axes: [1], keep_axes: true)
      |> Nx.add(@eps)
      |> Nx.pow(-0.5)
      |> Nx.multiply(hidden_f32)
      |> Nx.multiply(Nx.as_type(norm, :f32))
      |> Nx.as_type(:bf16)

    normalized
    |> Nx.dot(Nx.transpose(embedding))
    |> Nx.as_type(:f32)
    |> Nx.squeeze(axes: [0])
  end

  defp softcap(logits), do: Nx.multiply(Nx.tanh(Nx.divide(logits, @softcap)), @softcap)

  defp assert_all_close(left, right, tolerance) do
    max_difference =
      left
      |> Nx.as_type(:f32)
      |> Nx.subtract(Nx.as_type(right, :f32))
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    assert max_difference <= tolerance,
           "maximum absolute difference #{max_difference} exceeds #{tolerance}"
  end
end
