"""Fit the router's ask-back probe and write it where `mix gemma.system_one route` reads it.

The probe is a logistic regression on the layer-45 input (the prefix output)
at the last prompt token, trained on prompts in the router's own direct form
("... Reply with only one line of the form 'Answer: <x>' and nothing else.").
It only works on the wording it was trained on: the same probe trained on the
System One template drops from AUROC 0.83 to 0.64 on direct-form prompts
(docs/system-one-expert-plan.md, "Route 1").

    uv run --with scikit-learn --with safetensors --with numpy \\
        python scripts/system_one/export_ask_probe.py \\
        --cache data/system-one/cache-router-probe \\
        --labels data/system-one/router-probe-train.jsonl \\
        --output artifacts/system-one/ask-probe

`--cache` is a `cache --last-prompt-token-only` run over `--labels`, whose
rows carry `label` (1 = the state does not settle the question). Both take
several paths, so a retrain on rows of your own reuses the base cache and
caches only what is new (`scripts/system_one/retrain_ask_probe.sh`). The
standardisation is folded into the weights, so serving is one dot product:
score = sigmoid(weight . x + bias). CPU only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def features(cache: Path) -> tuple[list[str], np.ndarray]:
    manifest = json.loads((cache / "manifest.json").read_text())
    ids, rows = [], []
    for row in manifest["rows"]:
        hidden = load_file(str(cache / row["file"]))["hidden_state"]
        # A last-prompt-token-only cache keeps exactly that one position.
        rows.append(hidden[0, -1].astype(np.float32))
        ids.append(row["id"])
    return ids, np.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--c", type=float, default=0.003, help="inverse L2 strength, default 0.003")
    parser.add_argument("--threshold", type=float, default=0.8, help="ask-back cutoff recorded with the probe")
    args = parser.parse_args()

    labels = {}
    for path in args.labels:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row["id"] in labels:
                    raise SystemExit(f"{path}: id {row['id']} is in more than one labels file")
                labels[row["id"]] = row
    ids, parts = [], []
    for cache in args.cache:
        cache_ids, x = features(cache)
        missing = [i for i in cache_ids if i not in labels]
        if missing:
            raise SystemExit(f"{cache}: {len(missing)} cached rows have no label, e.g. {missing[0]}")
        ids += cache_ids
        parts.append(x)
    x = np.concatenate(parts)
    y = np.array([labels[i]["label"] for i in ids])

    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=args.c, max_iter=5000).fit(scaler.transform(x), y)

    coef = model.coef_[0] / scaler.scale_
    weight = coef.astype(np.float32)
    bias = np.array([model.intercept_[0] - float(coef @ scaler.mean_)], dtype=np.float32)

    folded = 1 / (1 + np.exp(-(x @ weight + bias[0])))
    reference = model.predict_proba(scaler.transform(x))[:, 1]
    drift = float(np.abs(folded - reference).max())
    if drift > 1e-3:
        raise SystemExit(f"folded probe drifts from sklearn by {drift}")

    args.output.mkdir(parents=True, exist_ok=True)
    save_file({"weight": weight, "bias": bias}, str(args.output / "probe.safetensors"))
    meta = {
        "kind": "ask_probe",
        "input": "prefix output (layer-45 input) at the last prompt token",
        "hidden_size": int(x.shape[1]),
        "threshold": args.threshold,
        "c": args.c,
        "rows": len(ids),
        "positives": int(y.sum()),
        "sources": dict(Counter(f"{labels[i].get('src', '?')}:{labels[i]['label']}" for i in ids)),
        "train_fraction_over_threshold": {
            "label_0": float((folded[y == 0] > args.threshold).mean()),
            "label_1": float((folded[y == 1] > args.threshold).mean()),
        },
        "max_folding_drift": drift,
    }
    (args.output / "probe.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
