#!/usr/bin/env python3
"""Assemble the round-2 replay prompts from the hand-written batch files."""
import importlib.util
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "replay-prompts-r2.jsonl"

BATCHES = [
    "b1_config.py",
    "b2_mc_yesno.py",
    "b3_should_tool.py",
    "b4_classify_json.py",
    "b5_code.py",
    "b6_debug_maths.py",
    "b7_trans_summ_rewrite.py",
    "b8_creative_roleplay_factual.py",
    "b9_extract_transcript_short.py",
]


def load(name):
    spec = importlib.util.spec_from_file_location(name[:-3], HERE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ROWS


def systems():
    spec = importlib.util.spec_from_file_location("b10_systems", HERE / "b10_systems.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SYSTEMS


def main():
    rows = []
    for name in BATCHES:
        rows.extend(load(name))
    extra = systems()
    for i, r in enumerate(rows, 1):
        key = "rp2-%03d" % i
        if key in extra:
            cat, text = extra[key]
            assert r["cat"] == cat, "%s: expected %s, row is %s" % (key, cat, r["cat"])
            assert not r.get("sys"), "%s already has a system message" % key
            r["sys"] = text
    out = []
    for i, r in enumerate(rows, 1):
        sysmsg = r.get("sys")
        out.append(
            {
                "id": "rp2-%03d" % i,
                "category": r["cat"],
                "lang": r["lang"],
                "system": sysmsg.strip() if isinstance(sysmsg, str) else None,
                "prompt": r["prompt"].strip(),
            }
        )
    with OUT.open("w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote %d rows to %s" % (len(out), OUT))


if __name__ == "__main__":
    main()
