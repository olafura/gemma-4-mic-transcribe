#!/usr/bin/env python3
"""Per-condition breakdowns of a scorecard run, and the expert-vs-baseline test.

`scorecard.py` gives one headline per condition. What decides whether a round
was worth running is the same number cut by the fields the held-out set was
built along (near/far, English/non-English, the two items flagged `debatable`)
and the paired difference between two conditions over the same items, with an
interval that says whether it is noise.

  breakdown.py heldout.jsonl NAME=scored/items.jsonl [NAME=... ...]
                             [--generated NAME=generated.jsonl ...]
                             [--against NAME] [--bootstrap 10000]

The items files are the `items.jsonl` a `scorecard.py --out-dir` run writes.
`--generated` supplies the generation JSONL of the same condition, which is
where tokens and ms live when the scored file has them as null.
"""
import argparse, json, random, statistics, sys

CUTS = [
    ("all", lambda it: True),
    ("decidable", lambda it: it["decidable"]),
    ("underspecified", lambda it: not it["decidable"]),
    ("near", lambda it: it["split"] == "near"),
    ("far", lambda it: it["split"] == "far"),
    ("english", lambda it: it.get("lang", "en") == "en"),
    ("non_english", lambda it: it.get("lang", "en") != "en"),
    ("no_debatable", lambda it: not it.get("debatable", False)),
]


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=3):
    return "-" if x is None else ("%.*f" % (nd, x))


def merge(heldout, scored, generated):
    """One row per held-out id, carrying the item's own fields and its score."""
    by_id = {r["id"]: r for r in scored}
    gen = {r["id"]: r for r in (generated or [])}
    rows = []
    for it in heldout:
        s = by_id.get(it["id"])
        if s is None:
            continue
        g = gen.get(it["id"], {})
        row = dict(it)
        row.update(s)
        for k in ("tokens", "ms", "gate", "reply"):
            if row.get(k) is None and g.get(k) is not None:
                row[k] = g[k]
        rows.append(row)
    return rows


def stats(rows):
    dec = [r for r in rows if r["decidable"]]
    und = [r for r in rows if not r["decidable"]]
    return {
        "n": len(rows),
        "headline": mean([r["score"] for r in rows]),
        "decidable_accuracy": mean([float(r["commits_to"] == r["gold"]) for r in dec]),
        "followup_rate_underspecified": mean([float(r["outcome"] == "followup") for r in und]),
        "committed_on_underspecified":
            mean([float(r["outcome"] == "committed_when_underspecified") for r in und]),
        "needless_ask_rate": mean([float(r["asks_question"]) for r in dec]),
        "mean_tokens": mean([r.get("tokens") for r in rows]),
        "mean_ms": mean([r.get("ms") for r in rows]),
    }


def bootstrap(a, b, n, seed=0):
    """Paired bootstrap over items of mean(a) - mean(b); a and b are aligned."""
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    k = len(d)
    point = sum(d) / k
    draws = []
    for _ in range(n):
        draws.append(sum(d[rng.randrange(k)] for _ in range(k)) / k)
    draws.sort()
    lo = draws[int(0.025 * n)]
    hi = draws[min(n - 1, int(0.975 * n))]
    return point, lo, hi, statistics.pstdev(draws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("heldout")
    ap.add_argument("conditions", nargs="+", help="NAME=path/to/items.jsonl")
    ap.add_argument("--generated", action="append", default=[], help="NAME=generated.jsonl")
    ap.add_argument("--against", default=None, help="condition every other one is compared with")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    heldout = load(args.heldout)
    gen = {}
    for spec in args.generated:
        name, _, path = spec.partition("=")
        gen[name] = load(path)

    conds = {}
    for spec in args.conditions:
        name, _, path = spec.partition("=")
        conds[name] = merge(heldout, load(path), gen.get(name))

    names = list(conds)
    out = {"cuts": {}, "against": args.against, "comparisons": {}}

    for cut, pred in CUTS:
        print("\n== %s ==" % cut)
        header = ("%-22s %5s %9s %9s %9s %9s %9s %8s %8s"
                  % ("condition", "n", "headline", "dec_acc", "fup_rate", "commit_u",
                     "needless", "tokens", "ms"))
        print(header)
        for name in names:
            rows = [r for r in conds[name] if pred(r)]
            s = stats(rows)
            out["cuts"].setdefault(cut, {})[name] = s
            print("%-22s %5d %9s %9s %9s %9s %9s %8s %8s"
                  % (name, s["n"], fmt(s["headline"]), fmt(s["decidable_accuracy"]),
                     fmt(s["followup_rate_underspecified"]), fmt(s["committed_on_underspecified"]),
                     fmt(s["needless_ask_rate"]), fmt(s["mean_tokens"], 1), fmt(s["mean_ms"], 0)))

    if args.against and args.against in conds:
        base = {r["id"]: r["score"] for r in conds[args.against]}
        print("\n== headline difference against %s (paired bootstrap, %d draws) =="
              % (args.against, args.bootstrap))
        for name in names:
            if name == args.against:
                continue
            ids = [r["id"] for r in conds[name] if r["id"] in base]
            a = [r["score"] for r in conds[name] if r["id"] in base]
            b = [base[i] for i in ids]
            point, lo, hi, sd = bootstrap(a, b, args.bootstrap)
            out["comparisons"][name] = {"n": len(ids), "difference": round(point, 4),
                                        "ci95": [round(lo, 4), round(hi, 4)],
                                        "bootstrap_sd": round(sd, 4)}
            print("%-22s n=%d  %+.4f  95%% CI [%+.4f, %+.4f]" % (name, len(ids), point, lo, hi))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
