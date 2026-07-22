defmodule Gemma4MicTranscribe.RocmPreflight do
  @moduledoc false

  @gfx_regex ~r/\bgfx[0-9a-f]+\b/

  # gfx1151 workarounds, applied only when the flag is not already configured:
  # - autotuning crashes while loading HIP code objects (rocm-jax issue #234)
  # - HIP graph capture (XLA "command buffers") segfaults recursively inside
  #   libamdhip64 when the first fused executable runs, so it is disabled
  # - triton GEMM is not validated on RDNA3.5; force the rocBLAS path
  @gfx1151_flags [
    {"--xla_gpu_autotune_level", "--xla_gpu_autotune_level=0"},
    {"--xla_gpu_enable_command_buffer", "--xla_gpu_enable_command_buffer="},
    {"--xla_gpu_enable_triton_gemm", "--xla_gpu_enable_triton_gemm=false"}
  ]

  @default_min_free_bytes 24 * 1024 * 1024 * 1024

  def check(opts \\ []) do
    with {:ok, gpu_targets} <- gpu_targets(opts),
         {:ok, xla_inspection} <- xla_inspection(opts) do
      xla_targets = xla_inspection.targets
      missing = gpu_targets -- xla_targets

      cond do
        gpu_targets == [] ->
          {:error,
           "EXLA ROCm backend cannot start safely: unable to detect local ROCm GPU ISA with rocm_agent_enumerator."}

        xla_targets == [] ->
          {:error, xla_inspection_message(xla_inspection)}

        missing == [] ->
          if Keyword.get(opts, :skip_memory_budget, false), do: :ok, else: memory_budget(opts)

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
    with {:ok, inspection} <- xla_inspection(opts) do
      {:ok, inspection.targets}
    end
  end

  def xla_inspection(opts \\ []) do
    candidate_paths = xla_extension_paths(opts)

    inspections =
      candidate_paths
      |> Enum.filter(&File.regular?/1)
      |> Enum.map(fn path ->
        {:ok, output} = llvm_objdump_output(path, opts)
        %{path: path, targets: parse_offload_targets(output)}
      end)

    targets =
      inspections
      |> Enum.flat_map(& &1.targets)
      |> Enum.uniq()
      |> Enum.sort()

    {:ok,
     %{
       targets: targets,
       inspected_paths: Enum.map(inspections, & &1.path),
       candidate_paths: candidate_paths
     }}
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

  def apply_runtime_workarounds(opts \\ []) do
    with {:ok, gpu_targets} <- gpu_targets(opts) do
      {xla_flags, changed?} =
        runtime_workaround_flags(gpu_targets, System.get_env("XLA_FLAGS"))

      if changed? do
        System.put_env("XLA_FLAGS", xla_flags)
      end

      {:ok, changed?}
    end
  end

  def runtime_workaround_flags(gpu_targets, xla_flags \\ nil) when is_list(gpu_targets) do
    if "gfx1151" in gpu_targets do
      flags = String.trim(xla_flags || "")

      missing =
        @gfx1151_flags
        |> Enum.reject(fn {name, _flag} -> String.contains?(flags, name) end)
        |> Enum.map(fn {_name, flag} -> flag end)

      case missing do
        [] ->
          {xla_flags, false}

        missing ->
          {Enum.join(Enum.reject([flags | missing], &(&1 == "")), " "), true}
      end
    else
      {xla_flags, false}
    end
  end

  def gfx1151_flags, do: Enum.map(@gfx1151_flags, fn {_name, flag} -> flag end)

  def memory_info(opts \\ []) do
    with {:ok, output} <- rocm_smi_memory_output(opts) do
      parse_memory_info(output)
    end
  end

  def parse_memory_info(output) when is_binary(output) do
    with {:ok, json} <- extract_json_object(output),
         {:ok, decoded} <- Jason.decode(json),
         {_card, info} <- Enum.find(decoded, fn {_card, info} -> is_map(info) end),
         {:ok, total_text} <- Map.fetch(info, "VRAM Total Memory (B)"),
         {:ok, used_text} <- Map.fetch(info, "VRAM Total Used Memory (B)"),
         {total, ""} <- Integer.parse(total_text),
         {used, ""} <- Integer.parse(used_text) do
      {:ok, %{total: total, used: used, free: max(total - used, 0)}}
    else
      _ -> {:ok, nil}
    end
  end

  def memory_budget(opts \\ []) do
    min_free_bytes = Keyword.get(opts, :min_free_bytes, min_free_bytes())

    case memory_info(opts) do
      {:ok, nil} ->
        :ok

      {:ok, %{total: total, free: free}} when free >= min_free_bytes and total > 0 ->
        :ok

      {:ok, %{total: total, used: used, free: free}} ->
        {:error, memory_budget_message(total, used, free, min_free_bytes)}
    end
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

  defp rocm_smi_memory_output(opts) do
    case Keyword.fetch(opts, :rocm_smi_output) do
      {:ok, output} ->
        {:ok, output}

      :error ->
        executable =
          Keyword.get(opts, :rocm_smi) ||
            System.find_executable("rocm-smi") ||
            existing_path("/opt/rocm/bin/rocm-smi")

        if executable do
          case System.cmd(executable, ["--showmeminfo", "vram", "--json"], stderr_to_stdout: true) do
            {output, 0} -> {:ok, output}
            {_output, _status} -> {:ok, ""}
          end
        else
          {:ok, ""}
        end
    end
  rescue
    _exception -> {:ok, ""}
  end

  defp xla_extension_paths(opts) do
    cond do
      Keyword.has_key?(opts, :xla_extension_paths) ->
        Keyword.fetch!(opts, :xla_extension_paths)

      Keyword.has_key?(opts, :xla_extension_path) ->
        [Keyword.fetch!(opts, :xla_extension_path)]

      true ->
        default_xla_extension_paths()
    end
    |> Enum.reject(&is_nil/1)
    |> Enum.map(&Path.expand/1)
    |> Enum.uniq()
  end

  defp default_xla_extension_paths do
    case :code.priv_dir(:exla) do
      {:error, _reason} ->
        []

      path ->
        priv_dir = List.to_string(path)

        [
          Path.join(priv_dir, "xla_extension/lib/libxla_extension.so"),
          libexla_relative_xla_extension_path(Path.join(priv_dir, "libexla.so")),
          "vendor/exla/cache/xla_extension/lib/libxla_extension.so"
        ]
    end
  end

  defp libexla_relative_xla_extension_path(libexla_path) do
    if File.regular?(libexla_path) do
      libexla_path
      |> realpath()
      |> Path.dirname()
      |> Path.join("xla_extension/lib/libxla_extension.so")
    end
  end

  defp realpath(path) do
    case File.read_link(path) do
      {:ok, target} -> target |> Path.expand(Path.dirname(path)) |> realpath()
      {:error, _reason} -> Path.expand(path)
    end
  end

  defp xla_inspection_message(%{inspected_paths: []} = inspection) do
    paths = format_paths(inspection.candidate_paths)

    "EXLA ROCm backend cannot start safely: unable to find compiled XLA extension at #{paths}."
  end

  defp xla_inspection_message(%{inspected_paths: paths}) do
    "EXLA ROCm backend cannot start safely: no ROCm offload bundles were found in " <>
      "the inspected XLA extension(s): #{format_paths(paths)}. " <>
      "The active EXLA cache may be CUDA/CPU-only; rebuild or extract EXLA with XLA_TARGET=rocm."
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

  defp extract_json_object(output) do
    case :binary.match(output, "{") do
      {start, _length} ->
        {:ok, binary_part(output, start, byte_size(output) - start)}

      :nomatch ->
        {:error, :missing_json}
    end
  end

  defp min_free_bytes do
    case System.get_env("GEMMA_ROCM_MIN_FREE_GB") do
      nil ->
        @default_min_free_bytes

      value ->
        case Float.parse(value) do
          {gb, ""} when gb >= 0 -> round(gb * 1024 * 1024 * 1024)
          _ -> @default_min_free_bytes
        end
    end
  end

  defp format_gib(bytes) do
    bytes
    |> Kernel./(1024 * 1024 * 1024)
    |> :erlang.float_to_binary(decimals: 2)
  end

  defp memory_budget_message(total, used, free, min_free) do
    "EXLA ROCm backend cannot start safely: GPU VRAM headroom is too low. " <>
      "Total=#{format_gib(total)}GiB used=#{format_gib(used)}GiB free=#{format_gib(free)}GiB; " <>
      "requires at least #{format_gib(min_free)}GiB free before loading Gemma. " <>
      "Close other GPU workloads or lower GEMMA_ROCM_MIN_FREE_GB only if you accept the risk."
  end

  defp incompatible_message(missing, xla_targets) do
    missing_text = Enum.join(missing, ", ")
    xla_text = if xla_targets == [], do: "<none detected>", else: Enum.join(xla_targets, ", ")

    "EXLA ROCm backend cannot start safely: local ROCm GPU ISA #{missing_text} is not " <>
      "present in the compiled XLA extension offload bundles. Built bundles: #{xla_text}. " <>
      "Rebuild XLA/EXLA with TF_ROCM_AMDGPU_TARGETS including #{missing_text}."
  end

  defp format_paths([]), do: "<none>"
  defp format_paths(paths), do: Enum.join(paths, ", ")
end
