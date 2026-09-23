# Confirms that probe-set rows built from your own requests were rendered
# exactly as `mix gemma.system_one route` serves them. The probe only works
# on the wording it was trained on, so a row whose `prompt` differs from
# `SystemOne.Router.direct_prompt/1` of its `request` would train it on a
# prompt the router never sends.
#
#     mix run --no-start scripts/system_one/check_router_render.exs data/system-one/router-probe-heldout.jsonl
#
# Rows without a `request` (the base set's twins, replay, GSM8K and ARC) are
# skipped. Exits 1 on the first mismatch, showing both renders.

alias Gemma4MicTranscribe.Gemma4.SystemOne.Router

files = System.argv()
if files == [], do: raise(ArgumentError, "usage: check_router_render.exs PROBE_SET.jsonl ...")

results =
  for file <- files,
      line <- File.stream!(file),
      String.trim(line) != "",
      row = Jason.decode!(line),
      request = row["request"],
      is_map(request) do
    {file, row["id"], row["prompt"], Router.direct_prompt(request)}
  end

case Enum.find(results, fn {_file, _id, built, served} -> built != served end) do
  nil ->
    IO.puts("render check: #{length(results)} rows match the router's direct prompt")

  {file, id, built, served} ->
    IO.puts(:stderr, "#{file}: #{id} was rendered differently from the router\n")
    IO.puts(:stderr, "--- built\n#{built}\n--- router\n#{served}")
    System.halt(1)
end
