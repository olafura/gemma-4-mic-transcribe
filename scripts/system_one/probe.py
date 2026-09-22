"""Linear probe for the ask-or-answer decision at the layer-45 boundary.

Phase 0 of docs/system-one-expert-plan.md: the expert is only added to layers
45 to 47, so it can only carry the decision if the decision is already in the
hidden state entering layer 45. This reads the cache written by
`mix gemma.system_one cache` and asks how much of `decidable` a linear model
can recover from that vector, with the controls that say whether the answer
means anything.

    .venv-system-one/bin/python scripts/system_one/probe.py \\
        --cache data/system-one/prefix-cache \\
        --input data/system-one/seed-pairs.jsonl

CPU only: nothing here touches the GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# n is 400 and the vector is 3840 wide, so the fit is far into the
# overparameterised regime and the useful part of the grid is the strong end.
C_GRID = np.logspace(-6, 2, 9)
OUTER_FOLDS = 5
INNER_FOLDS = 3
BOOTSTRAP = 2000
SHUFFLE_REPEATS = 20
SEED = 20260921


@dataclass
class Rows:
    ids: list[str]
    x: np.ndarray
    y: np.ndarray
    pair: np.ndarray
    domain: np.ndarray
    lang: np.ndarray
    text: list[str]
    prompt_tokens: np.ndarray


def load(cache: Path, input_path: Path, tokenizer_dir: Path | None) -> Rows:
    manifest = json.loads((cache / "manifest.json").read_text())
    items = {}
    for line in input_path.read_text().splitlines():
        line = line.strip()
        if line:
            item = json.loads(line)
            items[item["id"]] = item

    decode = _decoder(tokenizer_dir)

    ids, vectors, labels, pairs, domains, langs, texts, lengths = [], [], [], [], [], [], [], []

    for entry in manifest["rows"]:
        item = items[entry["id"]]
        tensors = load_file(cache / entry["file"])
        hidden = tensors["hidden_state"].astype(np.float32)

        if hidden.shape[0] != 1 or hidden.shape[2] != manifest["hidden_size"]:
            raise SystemExit(f"{entry['id']}: unexpected hidden state shape {hidden.shape}")

        # A full-sequence cache keeps every position; the probe only ever reads
        # the last prompt one.
        position = 0 if hidden.shape[1] == 1 else entry["last_prompt_index"]
        vector = hidden[0, position]

        if not np.isfinite(vector).all():
            raise SystemExit(f"{entry['id']}: hidden state is not finite")

        prompt_ids = tensors["input_ids"][: entry["last_prompt_index"] + 1]

        ids.append(entry["id"])
        vectors.append(vector)
        labels.append(1 if item["decidable"] else 0)
        pairs.append(item["pair"])
        domains.append(item["domain"])
        langs.append(item["lang"])
        texts.append(decode(prompt_ids))
        lengths.append(len(prompt_ids))

    return Rows(
        ids=ids,
        x=np.stack(vectors),
        y=np.array(labels),
        pair=np.array(pairs),
        domain=np.array(domains),
        lang=np.array(langs),
        text=texts,
        prompt_tokens=np.array(lengths, dtype=np.float64),
    )


def _decoder(tokenizer_dir: Path | None):
    """Decodes the cached prompt ids back to text for the bag-of-words control.

    The cached ids are what the model actually read, so decoding them is the
    only rendering guaranteed to match; falling back to the ids as words still
    gives the control a fair shot at any lexical cue.
    """
    if tokenizer_dir is None:
        return lambda ids: " ".join(str(int(i)) for i in ids)

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    return lambda ids: tokenizer.decode([int(i) for i in ids], skip_special_tokens=False)


def pipeline(c: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=c, max_iter=5000)),
        ]
    )


def pick_c(x: np.ndarray, y: np.ndarray, groups: np.ndarray, make=pipeline) -> float:
    """Chooses C inside the training folds only, never on held-out rows."""
    folds = min(INNER_FOLDS, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=folds)
    best_c, best_score = C_GRID[0], -np.inf

    for c in C_GRID:
        scores = []
        for train, test in splitter.split(x, y, groups):
            model = make(c).fit(_take(x, train), y[train])
            scores.append(balanced_accuracy_score(y[test], model.predict(_take(x, test))))
        score = float(np.mean(scores))
        if score > best_score:
            best_c, best_score = c, score

    return best_c


def _take(x, index):
    if isinstance(x, list):
        return [x[i] for i in index]
    return x[index]


def cross_validate(x, y, groups, make=pipeline) -> dict:
    """Grouped 5-fold with C chosen inside each training split."""
    splitter = GroupKFold(n_splits=OUTER_FOLDS)
    scores = np.zeros(len(y), dtype=np.float64)
    chosen = []

    for train, test in splitter.split(x if not isinstance(x, list) else np.zeros(len(y)), y, groups):
        c = pick_c(_take(x, train), y[train], groups[train], make)
        chosen.append(c)
        model = make(c).fit(_take(x, train), y[train])
        scores[test] = model.decision_function(_take(x, test))

    return {"scores": scores, "c": chosen}


def summarise(y, scores, pairs, rng) -> dict:
    predictions = (scores > 0).astype(int)
    point = {
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "auc": roc_auc_score(y, scores),
        "within_pair": within_pair(y, scores, pairs),
    }
    interval = bootstrap(y, scores, pairs, rng)
    return {key: (value, interval[key]) for key, value in point.items()}


def within_pair(y, scores, pairs) -> float:
    """Does the probe score the decidable twin above its underspecified twin.

    Twins share domain, phrasing and nearly all of their text, so this is the
    number least able to be won by reading trivia.
    """
    wins, total = 0.0, 0
    for pair in np.unique(pairs):
        index = np.where(pairs == pair)[0]
        positive = index[y[index] == 1]
        negative = index[y[index] == 0]
        if len(positive) != 1 or len(negative) != 1:
            continue
        total += 1
        difference = scores[positive[0]] - scores[negative[0]]
        wins += 1.0 if difference > 0 else 0.5 if difference == 0 else 0.0
    return wins / total if total else float("nan")


def bootstrap(y, scores, pairs, rng) -> dict:
    """Resamples pairs, not rows, so twins move together."""
    unique = np.unique(pairs)
    index_of = {pair: np.where(pairs == pair)[0] for pair in unique}
    collected = {"balanced_accuracy": [], "auc": [], "within_pair": []}

    for _ in range(BOOTSTRAP):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([index_of[pair] for pair in drawn])
        if len(np.unique(y[index])) < 2:
            continue
        labels = np.concatenate(
            [np.full(len(index_of[pair]), i) for i, pair in enumerate(drawn)]
        )
        collected["balanced_accuracy"].append(
            balanced_accuracy_score(y[index], (scores[index] > 0).astype(int))
        )
        collected["auc"].append(roc_auc_score(y[index], scores[index]))
        collected["within_pair"].append(within_pair(y[index], scores[index], labels))

    return {
        key: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        for key, values in collected.items()
    }


def transfer(x_train, y_train, groups_train, x_test, y_test, pairs_test, rng, make=pipeline):
    c = pick_c(x_train, y_train, groups_train, make)
    model = make(c).fit(x_train, y_train)
    scores = model.decision_function(x_test)
    result = summarise(y_test, scores, pairs_test, rng)
    result["c"] = c
    return result


def shuffled_within_pairs(rows: Rows, c: float, rng) -> dict:
    """Swaps which twin is labelled decidable, which is the only label noise
    that leaves every other property of the data untouched."""
    accuracies, aucs = [], []

    for _ in range(SHUFFLE_REPEATS):
        y = rows.y.copy()
        for pair in np.unique(rows.pair):
            index = np.where(rows.pair == pair)[0]
            if rng.random() < 0.5:
                y[index] = 1 - y[index]

        splitter = GroupKFold(n_splits=OUTER_FOLDS)
        scores = np.zeros(len(y))
        for train, test in splitter.split(rows.x, y, rows.pair):
            model = pipeline(c).fit(rows.x[train], y[train])
            scores[test] = model.decision_function(rows.x[test])

        accuracies.append(balanced_accuracy_score(y, (scores > 0).astype(int)))
        aucs.append(roc_auc_score(y, scores))

    return {
        "balanced_accuracy": (float(np.mean(accuracies)), (float(np.min(accuracies)), float(np.max(accuracies)))),
        "auc": (float(np.mean(aucs)), (float(np.min(aucs)), float(np.max(aucs)))),
    }


def bag_of_words():
    def make(c: float) -> Pipeline:
        return Pipeline(
            [
                ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)),
                ("lr", LogisticRegression(C=c, max_iter=5000)),
            ]
        )

    return make


def report(name: str, result: dict) -> None:
    fields = []
    for key in ("balanced_accuracy", "auc", "within_pair"):
        if key in result:
            point, (low, high) = result[key]
            fields.append(f"{key}={point:.3f} [{low:.3f}, {high:.3f}]")
    print(f"{name:<34} " + "  ".join(fields), flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/gemma4-12b-packed-prefix-0-44/tokenizer"),
    )
    parser.add_argument("--json", type=Path, help="write the numbers here as well")
    options = parser.parse_args(argv)

    tokenizer = options.tokenizer if options.tokenizer.exists() else None
    rows = load(options.cache, options.input, tokenizer)
    rng = np.random.default_rng(SEED)

    print(
        f"rows={len(rows.y)} pairs={len(np.unique(rows.pair))} "
        f"decidable={int(rows.y.sum())} dim={rows.x.shape[1]}",
        flush=True,
    )

    results: dict[str, dict] = {}

    hidden = cross_validate(rows.x, rows.y, rows.pair)
    results["hidden_state"] = summarise(rows.y, hidden["scores"], rows.pair, rng)
    results["hidden_state"]["c"] = [float(c) for c in hidden["c"]]
    report("layer-45 hidden state", results["hidden_state"])

    length = cross_validate(rows.prompt_tokens.reshape(-1, 1), rows.y, rows.pair)
    results["prompt_length"] = summarise(rows.y, length["scores"], rows.pair, rng)
    report("control: prompt length", results["prompt_length"])

    words = cross_validate(rows.text, rows.y, rows.pair, make=bag_of_words())
    results["bag_of_words"] = summarise(rows.y, words["scores"], rows.pair, rng)
    report("control: bag of words", results["bag_of_words"])

    median_c = float(np.median(hidden["c"]))
    results["shuffled"] = shuffled_within_pairs(rows, median_c, rng)
    report("control: labels shuffled in pair", results["shuffled"])

    for domain in np.unique(rows.domain):
        held = rows.domain == domain
        results[f"domain:{domain}"] = transfer(
            rows.x[~held], rows.y[~held], rows.pair[~held],
            rows.x[held], rows.y[held], rows.pair[held], rng,
        )
        report(f"held-out domain: {domain}", results[f"domain:{domain}"])

    english = rows.lang == "en"
    results["en_to_other"] = transfer(
        rows.x[english], rows.y[english], rows.pair[english],
        rows.x[~english], rows.y[~english], rows.pair[~english], rng,
    )
    report(f"English -> other ({int((~english).sum())} rows)", results["en_to_other"])

    if options.json:
        options.json.write_text(json.dumps(results, indent=2, default=float))

    return 0


if __name__ == "__main__":
    sys.exit(main())
