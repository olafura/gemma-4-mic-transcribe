"""Score exported ask-back probes on a cached, labelled probe set, side by side.

    uv run --with scikit-learn --with safetensors --with numpy \\
        python scripts/system_one/eval_ask_probe.py \\
        --probe artifacts/system-one/ask-probe my-probe/ask-probe \\
        --cache data/system-one/cache-router-probe-heldout \\
        --labels data/system-one/router-probe-heldout.jsonl

`--labels` rows are the probe-set format of `build_router_probe_set.py`
(`label` 1 = the state does not settle the request); a row built from your own
request keeps it under `request`, whose `split` or `domain` break the table
down further. Each probe asks at its own threshold (`--threshold` overrides).
The numbers that matter for the router: unclear requests asked (recall),
needless asks on clear ones, and asks on plain questions (GSM8K, ARC), which
should stay at zero. CPU only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file
from sklearn.metrics import roc_auc_score


def features(cache: Path) -> tuple[list[str], np.ndarray]:
    manifest = json.loads((cache / "manifest.json").read_text())
    ids = [row["id"] for row in manifest["rows"]]
    x = np.stack([load_file(str(cache / row["file"]))["hidden_state"][0, -1].astype(np.float32)
                  for row in manifest["rows"]])
    return ids, x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", type=Path, nargs="+", required=True, help="exported probe directories")
    parser.add_argument("--cache", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", type=Path, nargs="+", required=True)
    parser.add_argument("--threshold", type=float, help="ask above this score instead of each probe's own")
    parser.add_argument("--by", default="src,split", help="row fields to break down by, default src,split")
    parser.add_argument("--output", type=Path, help="also write the table as JSON")
    args = parser.parse_args()

    rows = {}
    for path in args.labels:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    ids, parts = [], []
    for cache in args.cache:
        cache_ids, x = features(cache)
        ids += cache_ids
        parts.append(x)
    x = np.concatenate(parts)
    missing = [i for i in ids if i not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} cached rows have no label, e.g. {missing[0]}")
    y = np.array([rows[i]["label"] for i in ids])

    def field(i: str, name: str):
        row = rows[i]
        return row.get(name, (row.get("request") or {}).get(name))

    groups = {"all": np.ones(len(ids), bool)}
    for name in [f for f in args.by.split(",") if f]:
        values = defaultdict(list)
        for n, i in enumerate(ids):
            if field(i, name) is not None:
                values[field(i, name)].append(n)
        for value, members in sorted(values.items(), key=lambda kv: str(kv[0])):
            mask = np.zeros(len(ids), bool)
            mask[members] = True
            groups[f"{name}={value}"] = mask

    report = {}
    for probe_dir in args.probe:
        meta = json.loads((probe_dir / "probe.json").read_text())
        tensors = load_file(str(probe_dir / "probe.safetensors"))
        score = 1 / (1 + np.exp(-(x @ tensors["weight"] + tensors["bias"][0])))
        threshold = args.threshold if args.threshold is not None else meta["threshold"]
        ask = score > threshold
        print(f"\n{probe_dir}  (ask above {threshold}, trained on {meta.get('rows')} rows)")
        print(f"  {'group':24} {'rows':>5} {'AUROC':>6} {'unclear asked':>14} {'needless asks':>14}")
        table = {}
        for name, mask in groups.items():
            pos, neg = mask & (y == 1), mask & (y == 0)
            auroc = roc_auc_score(y[mask], score[mask]) if pos.any() and neg.any() else None
            entry = {"rows": int(mask.sum()), "auroc": auroc,
                     "unclear": int(pos.sum()), "unclear_asked": int((ask & pos).sum()),
                     "clear": int(neg.sum()), "needless_asks": int((ask & neg).sum())}
            table[name] = entry
            print(f"  {name:24} {entry['rows']:>5} {'-' if auroc is None else f'{auroc:.3f}':>6} "
                  f"{entry['unclear_asked']:>6}/{entry['unclear']:<7} {entry['needless_asks']:>6}/{entry['clear']:<7}")
        report[str(probe_dir)] = {"threshold": threshold, "groups": table}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
