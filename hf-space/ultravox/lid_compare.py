#!/usr/bin/env python3
"""Compare result JSONs (gemma_lid_eval / ultravox_lid) on a language subset with renormalised candidates."""
import json, sys, collections
def score(rows, langs, key="probabilities"):
    rows = [r for r in rows if r["language"] in langs]
    t1 = t3 = 0
    for r in rows:
        p = {l: v for l, v in r[key].items() if l in langs}
        rk = sorted(p, key=lambda l: -p[l])
        t1 += rk[0] == r["language"]; t3 += r["language"] in rk[:3]
    return len(rows), 100 * t1 / len(rows), 100 * t3 / len(rows)
def gen(rows, langs):
    rows = [r for r in rows if r["language"] in langs and r.get("generated") is not None]
    return len(rows), 100 * sum(r["generated_code"] == r["language"] for r in rows) / max(1, len(rows))
ALL = "en de fr es it nl pl pt cs sk sl sv-SE da fi et lv lt hu ro el bg".split()
NINE = "cs de en es fr nl pl pt sv-SE".split()
files = sys.argv[1:]
for name, langs in [("21 languages", ALL), ("9 languages", NINE)]:
    print(f"== {name}")
    for f in files:
        d = json.load(open(f)); rows = d["rows"]
        n, a1, a3 = score(rows, langs)
        extra = ""
        if rows and "generated" in rows[0]:
            gn, ga = gen(rows, langs); extra = f"  generated {ga:5.1f}% ({gn})"
        lat = sorted(r["ms"] for r in rows)
        print(f"{f.split('/')[-1]:32s} n={n:3d} top-1 {a1:5.1f}%  top-3 {a3:5.1f}%{extra}  p50 {lat[len(lat)//2]} ms")
