defmodule Gemma4MicTranscribe.RocmPreflight do
  @moduledoc false

  @gfx_regex ~r/\bgfx[0-9a-f]+\b/

  def check(opts \\ []) do
    with {:ok, gpu_targets} <- gpu_targets(opts),
         {:ok, xla_targets} <- xla_targets(opts) do
      missing = gpu_targets -- xla_targets

      cond do
        gpu_targets == [] ->
          {:error,
           "EXLA ROCm backend cannot start safely: unable to detect local ROCm GPU ISA with rocm_agent_enumerator."}

        xla_targets == [] ->
          {:error,
           "EXLA ROCm backend cannot start safely: unable to inspect compiled XLA extension ROCm offload bundles with llvm-objdump."}

        missing == [] ->
          :ok

        true ->
          {:error, incompatible_message(missing, xla_targets)}
      end
    end
  end

  def gpu_targets(opts \\ []) do
    with {:ok, output} <- rocm_agent_output(opts) do
      {:ok, parse_gfx_targets(output)}
    end
  end

  def xla_targets(opts \\ []) do
    path = Keyword.get_lazy(opts, :xla_extension_path, &default_xla_extension_path/0)

    cond do
      is_nil(path) ->
        {:ok, []}

      not File.regular?(path) ->
        {:ok, []}

      true ->
        with {:ok, output} <- llvm_objdump_output(path, opts) do
          {:ok, parse_offload_targets(output)}
        end
    end
  end

  def parse_gfx_targets(output) when is_binary(output) do
    output
    |> scan_gfx_targets()
    |> Enum.reject(&(&1 == "gfx000"))
  end

  def parse_offload_targets(output) when is_binary(output) do
    Regex.scan(~r/hipv\d+-amdgcn-amd-amdhsa--(gfx[0-9a-f]+)/, output, capture: :all_but_first)
    |> List.flatten()
    |> Enum.uniq()
    |> Enum.sort()
  end

  defp rocm_agent_output(opts) do
    executable =
      Keyword.get(opts, :rocm_agent_enumerator) ||
        System.find_executable("rocm_agent_enumerator") ||
        existing_path("/opt/rocm/bin/rocm_agent_enumerator")

    if executable do
      case System.cmd(executable, [], stderr_to_stdout: true) do
        {output, 0} -> {:ok, output}
        {_output, _status} -> {:ok, ""}
      end
    else
      {:ok, ""}
    end
  rescue
    _exception -> {:ok, ""}
  end

  defp llvm_objdump_output(path, opts) do
    executable =
      Keyword.get(opts, :llvm_objdump) ||
        System.find_executable("llvm-objdump") ||
        System.find_executable("llvm-objdump-18")

    if executable do
      case System.cmd(executable, ["--offloading", path], stderr_to_stdout: true) do
        {output, 0} -> {:ok, output}
        {_output, _status} -> {:ok, ""}
      end
    else
      {:ok, ""}
    end
  rescue
    _exception -> {:ok, ""}
  end

  defp default_xla_extension_path do
    case :code.priv_dir(:exla) do
      {:error, _reason} ->
        nil

      path ->
        path
        |> List.to_string()
        |> Path.join("xla_extension/lib/libxla_extension.so")
    end
  end

  defp existing_path(path) do
    if File.regular?(path), do: path
  end

  defp scan_gfx_targets(output) do
    @gfx_regex
    |> Regex.scan(output)
    |> List.flatten()
    |> Enum.uniq()
    |> Enum.sort()
  end

  defp incompatible_message(missing, xla_targets) do
    missing_text = Enum.join(missing, ", ")
    xla_text = if xla_targets == [], do: "<none detected>", else: Enum.join(xla_targets, ", ")

    "EXLA ROCm backend cannot start safely: local ROCm GPU ISA #{missing_text} is not " <>
      "present in the compiled XLA extension offload bundles. Built bundles: #{xla_text}. " <>
      "Rebuild XLA/EXLA with TF_ROCM_AMDGPU_TARGETS including #{missing_text}."
  end
end
