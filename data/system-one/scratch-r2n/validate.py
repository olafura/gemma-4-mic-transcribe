#!/usr/bin/env python3
"""Validate data/system-one/replay-prompts-r2.jsonl.

Checks: schema, unique ids, rough token length, no System One template shape,
3-gram Jaccard < 0.5 within the file and against replay-prompts.jsonl and
replay-fresh.jsonl. Prints category / language / system counts.
"""
import collections
import json
import pathlib
import re
import sys
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / "replay-prompts-r2.jsonl"
EXISTING = [ROOT / "replay-prompts.jsonl", ROOT / "replay-fresh.jsonl"]

FIELDS = {"id", "category", "lang", "system", "prompt"}
MAX_TOKENS = 200
JACCARD_LIMIT = 0.5
CJK = ("ja", "zh", "ko")

token_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)
word_re = re.compile(r"\w+", re.UNICODE)


def is_cjk_char(ch):
    if ch in "　、。":
        return True
    name = unicodedata.name(ch, "")
    return any(k in name for k in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL"))


def rough_tokens(text):
    """whitespace + punctuation token count x 1.3; CJK counted per character."""
    cjk = sum(1 for ch in text if is_cjk_char(ch))
    rest = "".join(" " if is_cjk_char(ch) else ch for ch in text)
    n = len(token_re.findall(rest)) + cjk
    return int(round(n * 1.3))


def grams(text, n=3):
    words = [w.lower() for w in word_re.findall(text)]
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    errors = []
    rows = [json.loads(l) for l in TARGET.open(encoding="utf-8")]

    # schema + ids
    seen = set()
    for i, r in enumerate(rows, 1):
        if set(r.keys()) != FIELDS:
            errors.append("%s: fields %s" % (r.get("id", i), sorted(r.keys())))
        rid = r.get("id")
        if rid != "rp2-%03d" % i:
            errors.append("%s: expected id rp2-%03d" % (rid, i))
        if rid in seen:
            errors.append("%s: duplicate id" % rid)
        seen.add(rid)
        if not isinstance(r.get("prompt"), str) or not r["prompt"].strip():
            errors.append("%s: empty prompt" % rid)
        if r.get("system") is not None and (
            not isinstance(r["system"], str) or not r["system"].strip()
        ):
            errors.append("%s: bad system" % rid)
        if not isinstance(r.get("category"), str) or not r["category"]:
            errors.append("%s: bad category" % rid)
        if not isinstance(r.get("lang"), str) or len(r["lang"]) != 2:
            errors.append("%s: bad lang" % rid)

        text = (r.get("system") or "") + "\n" + r.get("prompt", "")
        n = rough_tokens(text)
        if n > MAX_TOKENS:
            errors.append("%s: %d rough tokens (limit %d)" % (rid, n, MAX_TOKENS))

        # must not use the System One template shape
        p = r.get("prompt", "")
        if re.search(r"^\s*State:\s*[\{\[]", p, re.M) and re.search(
            r"^\s*Options:\s*\S", p, re.M
        ):
            errors.append("%s: looks like the System One template" % rid)

    # near-duplicate detection
    mine = [(r["id"], grams(r["prompt"])) for r in rows]
    for i in range(len(mine)):
        for j in range(i + 1, len(mine)):
            s = jaccard(mine[i][1], mine[j][1])
            if s >= JACCARD_LIMIT:
                errors.append(
                    "%s ~ %s: internal jaccard %.2f" % (mine[i][0], mine[j][0], s)
                )
    for path in EXISTING:
        other = [
            (json.loads(l)["id"], grams(json.loads(l)["prompt"]))
            for l in path.open(encoding="utf-8")
        ]
        for rid, g in mine:
            for oid, og in other:
                s = jaccard(g, og)
                if s >= JACCARD_LIMIT:
                    errors.append(
                        "%s ~ %s/%s: jaccard %.2f" % (rid, path.name, oid, s)
                    )

    cats = collections.Counter(r["category"] for r in rows)
    langs = collections.Counter(r["lang"] for r in rows)
    withsys = sum(1 for r in rows if r["system"])
    toks = [rough_tokens((r["system"] or "") + "\n" + r["prompt"]) for r in rows]

    print("rows: %d" % len(rows))
    print("with system: %d (%.1f%%)" % (withsys, 100.0 * withsys / len(rows)))
    non_en = len(rows) - langs["en"]
    print("non-english: %d (%.1f%%)" % (non_en, 100.0 * non_en / len(rows)))
    print("max rough tokens: %d, mean %.1f" % (max(toks), sum(toks) / len(toks)))
    print("\ncategories:")
    for c, n in sorted(cats.items(), key=lambda kv: -kv[1]):
        print("  %-18s %3d" % (c, n))
    print("\nlanguages:")
    for l, n in sorted(langs.items(), key=lambda kv: -kv[1]):
        print("  %-4s %3d" % (l, n))

    if errors:
        print("\nERRORS: %d" % len(errors))
        for e in errors[:60]:
            print("  " + e)
        sys.exit(1)
    print("\nvalidator errors: 0")


if __name__ == "__main__":
    main()
