"""Build the ask-back probe's training set in the router's direct prompt form.

Rows are `{"id", "label", "src", "prompt"}`; `label` is 1 when the state does
not settle the question. Sources:

  * twins from the round-3 training set (both halves of each sampled pair),
    rendered with their options and the one-line answer instruction;
  * replay prompts as they are (ordinary traffic, never ask);
  * GSM8K and ARC-Challenge test questions not in `--exclude`, in the direct
    form the router sends them (never ask).

    uv run --with pyarrow --with tokenizers python scripts/system_one/build_router_probe_set.py \\
        --train data/system-one/train-round3.jsonl \\
        --gsm8k gsm8k/main/test-00000-of-00001.parquet \\
        --arc arc/ARC-Challenge/test-00000-of-00001.parquet \\
        --exclude items60-direct.jsonl items200-direct.jsonl \\
        --output data/system-one/router-probe-train.jsonl

Prompts longer than `--max-tokens` are dropped: `cache` aborts on a prompt
longer than its largest bucket.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

OPTION = "Reply with only one line of the form 'Answer: <option name>' and nothing else."
NUMBER = "Reply with only one line of the form 'Answer: <number>' and nothing else."
LETTER = "Reply with only one line of the form 'Answer: <letter>' and nothing else."


def render_twin(row: dict) -> str:
    state = row["state"]
    state = state if isinstance(state, str) else json.dumps(state, separators=(",", ":"))
    options = "\n".join(f"- {name}: {text}" for name, text in row["options"].items())
    return f"State: {state.strip()}\n\n{row['question'].strip()}\n\nOptions:\n{options}\n\n{OPTION}"


def arc_body(row: dict) -> str:
    choices = "\n".join(f"{label}. {text}" for label, text in zip(row["choices"]["label"], row["choices"]["text"]))
    return row["question"].strip() + "\n\n" + choices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--arc", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[], help="JSONL files whose prompts must not be used")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=1200)
    parser.add_argument("--replay", type=int, default=400)
    parser.add_argument("--plain", type=int, default=150, help="GSM8K and ARC questions each")
    parser.add_argument("--max-tokens", type=int, default=470)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(line) for line in args.train.read_text().splitlines() if line.strip()]
    pairs = sorted({r["pair"] for r in rows if r["kind"] == "system_one"})
    keep = set(rng.sample(pairs, args.pairs))
    out = [
        {"id": r["id"], "label": 0 if r["decidable"] else 1, "src": "twin", "prompt": render_twin(r)}
        for r in rows
        if r["kind"] == "system_one" and r["pair"] in keep
    ]
    replay = [r for r in rows if r["kind"] == "replay"]
    out += [{"id": r["id"], "label": 0, "src": "replay", "prompt": r["prompt"]} for r in rng.sample(replay, args.replay)]

    used = set()
    for path in args.exclude:
        for line in path.read_text().splitlines():
            if line.strip():
                used.add(json.loads(line)["prompt"].split("\n\nReply with")[0].split("\n\nEnd your")[0])

    gsm = [r for r in pq.read_table(args.gsm8k).to_pylist() if r["question"].strip() not in used]
    for i, r in enumerate(rng.sample(gsm, args.plain)):
        out.append({"id": f"gt-{i:03d}", "label": 0, "src": "gsm8k", "prompt": r["question"].strip() + "\n\n" + NUMBER})
    arc = [
        r
        for r in pq.read_table(args.arc).to_pylist()
        if r["choices"]["label"] == ["A", "B", "C", "D"] and arc_body(r) not in used
    ]
    for i, r in enumerate(rng.sample(arc, args.plain)):
        out.append({"id": f"at-{i:03d}", "label": 0, "src": "arc", "prompt": arc_body(r) + "\n\n" + LETTER})

    tokenizer_path = glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/*/tokenizer.json")
    )[0]
    tokenizer = Tokenizer.from_file(tokenizer_path)
    kept = [r for r in out if len(tokenizer.encode(r["prompt"]).ids) <= args.max_tokens]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(kept), "dropped_long": len(out) - len(kept),
                      "sources": dict(Counter(f"{r['src']}:{r['label']}" for r in kept))}))


if __name__ == "__main__":
    main()
