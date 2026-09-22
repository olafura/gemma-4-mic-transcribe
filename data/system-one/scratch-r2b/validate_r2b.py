#!/usr/bin/env python3
"""Validator for data/system-one/train-r2-b.jsonl (round-2 batch B).

Checks schema, pairing, option/gold consistency, twin length parity,
language share, option-count spread, yes/no share, target punctuation,
and 3-gram Jaccard against the existing train files.

Usage: python3 validate_r2b.py [path ...]      (default: the real output file)
"""
import json
import re
import sys
import collections
import os

ROOT = "/home/olafura/dev/gemma-4-mic-transcribe/data/system-one"
DEFAULT = [os.path.join(ROOT, "train-r2-b.jsonl")]
EXISTING = [
    "train.jsonl",
    "train-lookup-mixed.jsonl",
    "train-scheduling-config.jsonl",
    "train-voice-support.jsonl",
]

DOMAINS = {
    "legal_compliance": 50,
    "manufacturing_qa": 50,
    "real_estate": 50,
    "media_publishing": 50,
    "agriculture": 50,
    "events_ticketing": 50,
}
LANGS = {"en", "de", "fr", "es", "it", "ja", "pt", "nl", "ko"}
FORBIDDEN = re.compile(
    r"\b(clinic|patient|nurse|doctor|prescription|shipment|courier|freight|pallet|"
    r"invoice reconcil|bank|iban|helpdesk|ticket sla|vlan|smart home|thermostat|"
    r"pupil|classroom|semester)\b",
    re.I,
)

D_KEYS = {"id", "pair", "domain", "lang", "state", "question", "options",
          "decidable", "gold", "missing", "target", "looks_ambiguous"}
U_KEYS = {"id", "pair", "domain", "lang", "state", "question", "options",
          "decidable", "gold", "missing", "target", "tempting", "blur_mode"}

YESNO = re.compile(
    r"^(is|are|does|do|must|can|may|was|were|did|has|have|should|will|shall|would)\b"
    r"|^(ist|sind|muss|muessen|darf|kann|war|hat|haben|wird|werden|gilt|greift)\b"
    r"|^(est-ce|faut-il|peut-on|doit-on|la penalit|est-il|sommes-nous)"
    r"|^¿(es|est|debe|puede|hay|se |sigue|seguimos|entra|cumple|supera|queda|aplica|hace)"
    r"|^(a reclama|o pedido|pode|deve|entra|cumpre|est)"
    r"|^(moet|mag|is |kan |geldt|voldoet)"
    r"|^(la penale|si puo|e obbligat|rientra|serve|deve)"
    r"|ますか。$|ますか？$|ますか$"
    r"|습니까\?$|됩니까\?$|합니까\?$",
    re.I,
)

WORD = re.compile(r"\w+", re.UNICODE)


def words(s):
    return WORD.findall(s.lower())


def flat(obj, prefix=""):
    """leaf path -> value"""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flat(v, prefix + "/" + str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flat(v, prefix + "/%d" % i))
    else:
        out[prefix] = obj
    return out


def row_text(r):
    return " ".join([
        json.dumps(r["state"], ensure_ascii=False),
        r["question"],
        " ".join(r["options"].keys()),
        " ".join(r["options"].values()),
        r["target"] or "",
    ])


def grams(text, n=3):
    w = words(text)
    return set(tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1)))


def main(paths):
    errors, warnings = [], []
    rows = []
    for p in paths:
        with open(p) as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    errors.append("%s:%d bad JSON: %s" % (p, ln, e))

    def err(rid, msg):
        errors.append("%s: %s" % (rid, msg))

    by_pair = collections.defaultdict(dict)
    for r in rows:
        rid = r.get("id", "?")
        m = re.fullmatch(r"r2b-(\d{3})-([du])", str(rid))
        if not m:
            err(rid, "id does not match r2b-NNN-[du]")
            continue
        n, side = m.group(1), m.group(2)
        if r.get("pair") != "r2b-" + n:
            err(rid, "pair field %r does not match id" % r.get("pair"))
        if side in by_pair[r["pair"]]:
            err(rid, "duplicate row for this side")
        by_pair[r["pair"]][side] = r

        keys = set(r.keys())
        want = D_KEYS if side == "d" else U_KEYS
        if keys != want:
            err(rid, "keys %s, expected %s" % (sorted(keys - want), sorted(want - keys)))

        if r.get("domain") not in DOMAINS:
            err(rid, "domain %r not allowed" % r.get("domain"))
        if r.get("lang") not in LANGS:
            err(rid, "lang %r not allowed" % r.get("lang"))

        st = r.get("state")
        if not isinstance(st, dict):
            err(rid, "state is not an object")
        else:
            if not (3 <= len(st) <= 9):
                err(rid, "state has %d top-level fields, want 3..9" % len(st))

        opts = r.get("options")
        if not isinstance(opts, dict):
            err(rid, "options is not an object")
        else:
            if not (2 <= len(opts) <= 5):
                err(rid, "%d options, want 2..5" % len(opts))
            for k, v in opts.items():
                if not isinstance(v, str) or not v:
                    err(rid, "option %r has no description" % k)
                elif len(words(v)) > 8:
                    err(rid, "option %r description is %d words (max 8)" % (k, len(words(v))))
                if not re.fullmatch(r"[a-z0-9_.:\-]+", k):
                    err(rid, "option key %r is not English snake_case" % k)

        tgt = r.get("target") or ""
        nw = len(words(tgt))
        if nw > 20:
            err(rid, "target is %d words (max 20)" % nw)
        if not tgt:
            err(rid, "empty target")
        low = tgt.lower()
        if low.startswith(("sorry", "i'm sorry", "unfortunately", "i apolog", "sure,", "okay", "ok,")):
            err(rid, "target has a preamble or apology")

        if side == "d":
            if r.get("decidable") is not True:
                err(rid, "decidable must be true on a -d row")
            if r.get("missing") is not None:
                err(rid, "missing must be null on a -d row")
            if not isinstance(r.get("looks_ambiguous"), bool):
                err(rid, "looks_ambiguous must be a boolean")
            if isinstance(opts, dict) and r.get("gold") not in opts:
                err(rid, "gold %r is not an option key" % r.get("gold"))
            if tgt.rstrip().endswith(("?", "？", "か", "か。")):
                err(rid, "-d target ends like a question")
        else:
            if r.get("decidable") is not False:
                err(rid, "decidable must be false on a -u row")
            if r.get("gold") is not None:
                err(rid, "gold must be null on a -u row")
            if not isinstance(r.get("missing"), str) or len(r["missing"]) < 10:
                err(rid, "missing must name the deciding fact")
            if not isinstance(r.get("tempting"), bool):
                err(rid, "tempting must be a boolean")
            if r.get("blur_mode") not in ("blurred", "removed"):
                err(rid, "blur_mode %r not in blurred/removed" % r.get("blur_mode"))
            t = tgt.rstrip("。 ")
            if not (t.endswith("?") or t.endswith("？") or t.endswith("か")):
                err(rid, "-u target does not end in a question mark")

        blob = row_text(r)
        hit = FORBIDDEN.search(blob)
        if hit:
            err(rid, "held-out domain vocabulary: %r" % hit.group(0))

    # pairing
    nums = sorted(by_pair)
    for pr in nums:
        sides = by_pair[pr]
        if set(sides) != {"d", "u"}:
            errors.append("%s: missing side %s" % (pr, {"d", "u"} - set(sides)))
            continue
        d, u = sides["d"], sides["u"]
        for f in ("domain", "lang", "question"):
            if d.get(f) != u.get(f):
                errors.append("%s: twins differ in %s" % (pr, f))
        if d.get("options") != u.get("options"):
            errors.append("%s: twins have different options" % pr)
        fd, fu = flat(d.get("state", {})), flat(u.get("state", {}))
        changed = [k for k in set(fd) | set(fu) if fd.get(k) != fu.get(k)]
        if not changed:
            errors.append("%s: twin states are identical" % pr)
        elif len(changed) > 2:
            errors.append("%s: twins differ at %d leaves (%s)" % (pr, len(changed), changed[:4]))
        ld = len(json.dumps(d.get("state"), ensure_ascii=False)) + len(d.get("question", ""))
        lu = len(json.dumps(u.get("state"), ensure_ascii=False)) + len(u.get("question", ""))
        ratio = ld / max(1, lu)
        if not (0.75 <= ratio <= 1.34):
            errors.append("%s: twin length ratio %.2f (want 0.75..1.34)" % (pr, ratio))

    expected = sum(DOMAINS.values())
    if len(nums) != expected:
        errors.append("%d pairs, expected %d" % (len(nums), expected))
    want_ids = set("r2b-%03d" % i for i in range(1, expected + 1))
    if set(nums) != want_ids:
        errors.append("pair ids are not r2b-001..r2b-%03d (missing %s, extra %s)"
                      % (expected, sorted(want_ids - set(nums))[:5], sorted(set(nums) - want_ids)[:5]))

    # aggregates
    dom = collections.Counter(by_pair[p]["d"]["domain"] for p in nums if "d" in by_pair[p])
    for k, v in DOMAINS.items():
        if dom.get(k, 0) != v:
            errors.append("domain %s has %d pairs, expected %d" % (k, dom.get(k, 0), v))

    langs = collections.Counter(r["lang"] for r in rows)
    non_en = sum(v for k, v in langs.items() if k != "en")
    share = non_en / max(1, len(rows))
    if not (0.15 <= share <= 0.21):
        errors.append("non-English share %.3f, want 0.15..0.21" % share)

    nopts = collections.Counter(len(by_pair[p]["d"]["options"]) for p in nums if "d" in by_pair[p])
    two = nopts.get(2, 0) / max(1, len(nums))
    if two > 0.50:
        errors.append("two-option pairs are %.1f%% of the set (max 50%%)" % (100 * two))
    if len(nopts) < 3:
        errors.append("option counts are not spread: %s" % dict(nopts))

    pos = collections.Counter()
    for p in nums:
        d = by_pair[p].get("d")
        if not d or d.get("gold") not in d.get("options", {}):
            continue
        pos[list(d["options"]).index(d["gold"]) + 1] += 1
    top = max(pos.values()) / max(1, sum(pos.values())) if pos else 1
    if top > 0.45:
        errors.append("gold clusters in one option position: %s" % dict(pos))

    yn = [p for p in nums if "d" in by_pair[p] and YESNO.search(by_pair[p]["d"]["question"].strip())]
    if len(yn) / max(1, len(nums)) < 0.25:
        errors.append("yes/no-style items are %d of %d pairs (want >=25%%)" % (len(yn), len(nums)))

    la = collections.Counter(by_pair[p]["d"].get("looks_ambiguous") for p in nums if "d" in by_pair[p])
    if la.get(True, 0) < 0.35 * len(nums):
        errors.append("only %d -d rows are looks_ambiguous (want >=35%%)" % la.get(True, 0))
    blur = collections.Counter(by_pair[p]["u"].get("blur_mode") for p in nums if "u" in by_pair[p])

    # near-duplicate check, 3-gram Jaccard
    new_g = {r["id"]: grams(row_text(r)) for r in rows if "state" in r}
    old = []
    for f in EXISTING:
        path = os.path.join(ROOT, f)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                o = json.loads(line)
                old.append((o["id"], grams(row_text(o))))
    index = collections.defaultdict(list)
    for i, (oid, g) in enumerate(old):
        for gr in g:
            index[gr].append(i)
    dup = []
    for rid, g in new_g.items():
        counts = collections.Counter()
        for gr in g:
            for i in index.get(gr, ()):
                counts[i] += 1
        for i, c in counts.most_common(5):
            j = c / (len(g) + len(old[i][1]) - c)
            if j >= 0.5:
                dup.append((rid, old[i][0], round(j, 3)))
                break
    ids = list(new_g)
    inner = collections.defaultdict(list)
    for i, rid in enumerate(ids):
        for gr in new_g[rid]:
            inner[gr].append(i)
    seen = set()
    for i, rid in enumerate(ids):
        counts = collections.Counter()
        for gr in new_g[rid]:
            for j in inner.get(gr, ()):
                if j != i and ids[j][:7] != rid[:7]:
                    counts[j] += 1
        for j, c in counts.most_common(3):
            g1, g2 = new_g[rid], new_g[ids[j]]
            jac = c / (len(g1) + len(g2) - c)
            if jac >= 0.5 and (ids[j], rid) not in seen:
                seen.add((rid, ids[j]))
                dup.append((rid, ids[j], round(jac, 3)))
                break
    for a, b, j in dup:
        errors.append("near-duplicate: %s vs %s (3-gram Jaccard %.2f)" % (a, b, j))

    print("rows            : %d (%d pairs)" % (len(rows), len(nums)))
    print("domains         : %s" % dict(dom))
    print("langs           : %s  non-English share %.3f" % (dict(langs), share))
    print("options per item: %s  two-option share %.2f" % (dict(sorted(nopts.items())), two))
    print("gold position   : %s" % dict(sorted(pos.items())))
    print("yes/no style    : %d pairs (%.0f%%)" % (len(yn), 100 * len(yn) / max(1, len(nums))))
    print("looks_ambiguous : %s" % dict(la))
    print("blur_mode       : %s" % dict(blur))
    print("dup pairs >=0.5 : %d" % len(dup))
    print("")
    for w in warnings:
        print("WARN  " + w)
    for e in errors:
        print("ERROR " + e)
    print("\n%d errors, %d warnings" % (len(errors), len(warnings)))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or DEFAULT))
