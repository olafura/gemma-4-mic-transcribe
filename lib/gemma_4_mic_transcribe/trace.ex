defmodule Gemma4MicTranscribe.Trace do
  @moduledoc false

  # BEAM-side call tracing via :dbg (runtime_tools). Unlike the bpftrace task
  # in scripts/trace/, this needs no root: it traces Elixir function calls and
  # logs per-call wall durations, which is enough to see where transcription
  # time goes on the BEAM side (tokenize, input build, prefill, decode steps).
  #
  # See https://www.erlang-solutions.com/blog/a-guide-to-tracing-in-elixir/

  @default_modules [
    Gemma4MicTranscribe.Audio,
    Gemma4MicTranscribe.Gemma4Unified.AudioFeatureExtractor,
    Gemma4MicTranscribe.Gemma4Unified.Input,
    Gemma4MicTranscribe.Gemma4Unified.Runtime,
    Gemma4MicTranscribe.Gemma4Unified.TokenSelection,
    Gemma4MicTranscribe.Gemma4Unified.Transcript,
    Gemma4MicTranscribe.SpeechGate,
    Gemma4MicTranscribe.StreamingSession,
    Gemma4MicTranscribe.Transcriber
  ]

  # Calls faster than this are dropped to keep the output readable.
  @default_min_us 100

  def default_modules, do: @default_modules

  def with_tracing(fun, opts \\ []) do
    :ok = enable(opts)

    try do
      fun.()
    after
      disable()
    end
  end

  def enable(opts \\ []) do
    modules = Keyword.get(opts, :modules, @default_modules)

    state = %{
      device: Keyword.get(opts, :device, :standard_error),
      min_us: Keyword.get(opts, :min_us, @default_min_us),
      stacks: %{}
    }

    {:ok, _tracer} = :dbg.tracer(:process, {&handle_event/2, state})
    {:ok, _} = :dbg.p(:all, [:call, :timestamp])

    for module <- modules do
      # tpl also traces private functions, not just the exported API.
      {:ok, _} = :dbg.tpl(module, :_, :_, [{:_, [], [{:return_trace}, {:exception_trace}]}])
    end

    :ok
  end

  def disable do
    :dbg.stop()
    :ok
  end

  defp handle_event({:trace_ts, pid, :call, {module, fun, args}, ts}, state) do
    key = {pid, module, fun, length(args)}
    update_in(state.stacks[key], &[ts | &1 || []])
  end

  defp handle_event({:trace_ts, pid, :return_from, mfa, _result, ts}, state) do
    log_return(state, pid, mfa, ts, "")
  end

  defp handle_event({:trace_ts, pid, :exception_from, mfa, _class_reason, ts}, state) do
    log_return(state, pid, mfa, ts, " (raised)")
  end

  defp handle_event(_event, state), do: state

  defp log_return(state, pid, {module, fun, arity}, ts, suffix) do
    key = {pid, module, fun, arity}

    case state.stacks[key] do
      [started_at | rest] ->
        elapsed_us = :timer.now_diff(ts, started_at)

        if elapsed_us >= state.min_us do
          IO.puts(
            state.device,
            "trace: #{inspect(pid)} #{inspect(module)}.#{fun}/#{arity} " <>
              "#{format_ms(elapsed_us)}ms#{suffix}"
          )
        end

        put_in(state.stacks[key], rest)

      _missing ->
        state
    end
  end

  defp format_ms(elapsed_us) do
    :io_lib.format("~.3f", [elapsed_us / 1000]) |> IO.iodata_to_binary()
  end
end
