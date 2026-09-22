#!/usr/bin/env python3
"""Validator for data/system-one/train-r2-a.jsonl (and partial batches).

Usage: python3 validate.py FILE [FILE...]
Exits non-zero if any ERROR is found. WARN lines are advisory.
"""
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict

D_KEYS = {"id", "pair", "domain", "lang", "state", "question", "options",
          "decidable", "gold", "missing", "target", "looks_ambiguous"}
U_KEYS = {"id", "pair", "domain", "lang", "state", "question", "options",
          "decidable", "gold", "missing", "target", "tempting", "blur_mode"}

DOMAINS = {"hr_payroll", "travel", "retail_inventory", "insurance",
           "devops_ci", "hospitality"}
FORBIDDEN = {"clinic", "medical", "logistics", "shipping", "finance",
             "banking", "helpdesk", "smarthome", "smart_home", "education"}
LANGS = {"en", "de", "fr", "es", "it", "ja", "pt", "nl", "ko"}
CJK = {"ja", "ko"}

KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# yes/no-style compliance question openers, per language
YESNO = {
    "en": r"^(is|are|was|were|does|do|did|can|could|may|must|will|would|should|has|have|am)\b",
    "de": r"^(ist|sind|war|waren|darf|dürfen|muss|müssen|kann|können|wird|werden|hat|haben|greift|gilt|liegt|besteht|reicht|zahlt|deckt|erfüllt)\b",
    "fr": r"^(est-ce|peut-on|faut-il|doit-on|la prime est-elle|y a-t-il)\b",
    "es": r"^¿(cumple|se puede|procede|es |está|puede|debe|tiene|hay|entra|queda|aplica|cubre|corresponde)",
    "it": r"^(è |si può|rientra|deve|può|va |serve|copre|basta|vale|spetta)",
    "pt": r"^(cumpre|é |pode|deve|está|tem|há|entra|aplica|cobre|fica)",
    "nl": r"^(is|mag|voldoet|kan|moet|heeft|wordt|geldt|dekt|valt|komt)\b",
}
WH_JA = ("何", "どちら", "どの", "いくら", "いつ", "どう", "誰", "だれ", "どこ", "いくつ")
WH_KO = ("무엇", "어느", "어떤", "얼마", "언제", "누구", "어떻게", "며칠", "몇")


def words(s):
    return [w for w in re.split(r"\s+", s.strip()) if w]


def tlen(s, lang):
    """length of a target, in words for latin scripts, characters for cjk"""
    if lang in CJK:
        return len([c for c in s if not c.isspace()])
    return len(words(s))


def tmax(lang):
    return 40 if lang in CJK else 20


def norm_tokens(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return [t for t in re.split(r"[^0-9a-zÀ-ɏ぀-ヿ一-鿿가-힯]+", text) if t]


def trigrams(row):
    parts = [row["question"], json.dumps(row["state"], ensure_ascii=False)]
    toks = norm_tokens(" ".join(parts))
    return {tuple(toks[i:i + 3]) for i in range(len(toks) - 2)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_question(target, lang):
    t = target.rstrip()
    if lang in CJK:
        return t.endswith("？") or t.endswith("?") or t.endswith("か")
    return t.endswith("?")


def is_compliance(row):
    q = unicodedata.normalize("NFKC", row["question"]).strip()
    lang = row["lang"]
    if lang == "ja":
        return q.rstrip("。？?").endswith("か") and not any(w in q for w in WH_JA)
    if lang == "ko":
        return q.rstrip("?？.").endswith(("나요", "합니까", "됩니까", "입니까", "있나요", "하나요")) \
            and not any(w in q for w in WH_KO)
    pat = YESNO.get(lang)
    if not pat:
        return False
    return re.match(pat, q.lower()) is not None


def shape(state):
    nested = any(isinstance(v, dict) for v in state.values())
    listed = any(isinstance(v, list) for v in state.values())
    nulled = any(v is None for v in state.values())
    return nested, listed, nulled


def main(paths):
    errors, warns = [], []
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{p}:{n} bad json: {exc}")
    if errors:
        for e in errors:
            print("ERROR", e)
        return 1

    def err(rid, msg):
        errors.append(f"{rid}: {msg}")

    ids = Counter(r.get("id") for r in rows)
    for rid, c in ids.items():
        if c > 1:
            err(rid, f"duplicate id ({c}x)")

    pairs = defaultdict(dict)
    for r in rows:
        rid = r.get("id", "<no id>")
        m = re.match(r"^(r2a-(\d{3}))-(d|u)$", rid or "")
        if not m:
            err(rid, "id does not match r2a-NNN-d/u")
            continue
        pair, _, side = m.groups()
        if r.get("pair") != pair:
            err(rid, f"pair field {r.get('pair')!r} != {pair!r}")
        if side in pairs[pair]:
            err(rid, "two rows on the same side")
        pairs[pair][side] = r

        want = D_KEYS if side == "d" else U_KEYS
        if set(r) != want:
            err(rid, f"key set mismatch: extra={sorted(set(r) - want)} missing={sorted(want - set(r))}")
            continue

        if r["domain"] not in DOMAINS:
            err(rid, f"domain {r['domain']!r} not in the six assigned domains")
        if r["lang"] not in LANGS:
            err(rid, f"lang {r['lang']!r} not allowed")

        st = r["state"]
        if not isinstance(st, dict):
            err(rid, "state is not an object")
            continue
        if not 3 <= len(st) <= 9:
            err(rid, f"state has {len(st)} top-level fields, want 3-9")

        opts = r["options"]
        if not isinstance(opts, dict) or not 2 <= len(opts) <= 5:
            err(rid, f"options must be an object of 2-5 entries, got {len(opts) if isinstance(opts, dict) else type(opts)}")
            continue
        for k, v in opts.items():
            if not KEY_RE.match(k):
                err(rid, f"option key {k!r} is not english snake_case")
            if not isinstance(v, str) or not 1 <= len(words(v)) <= 8:
                err(rid, f"option description {v!r} must be 1-8 words")
        if len(set(opts.values())) != len(opts):
            err(rid, "two options share a description")

        tgt = r["target"]
        if not isinstance(tgt, str) or not tgt.strip():
            err(rid, "empty target")
            continue
        n = tlen(tgt, r["lang"])
        if n > tmax(r["lang"]):
            err(rid, f"target is {n} {'chars' if r['lang'] in CJK else 'words'}, max {tmax(r['lang'])}")
        low = tgt.lower()
        for bad in ("i'm sorry", "sorry", "as an ai", "i cannot", "let me", "sure,", "of course"):
            if low.startswith(bad):
                err(rid, f"target opens with preamble/apology {bad!r}")

        if side == "d":
            if r["decidable"] is not True:
                err(rid, "decidable must be true on a -d row")
            if r["missing"] is not None:
                err(rid, "missing must be null on a -d row")
            if r["gold"] not in opts:
                err(rid, f"gold {r['gold']!r} is not an option key")
            if not isinstance(r["looks_ambiguous"], bool):
                err(rid, "looks_ambiguous must be a boolean")
            if is_question(tgt, r["lang"]):
                err(rid, "a -d target must not be a question")
        else:
            if r["decidable"] is not False:
                err(rid, "decidable must be false on a -u row")
            if r["gold"] is not None:
                err(rid, "gold must be null on a -u row")
            min_missing = 8 if r["lang"] in CJK else 3
            if not isinstance(r["missing"], str) or tlen(r["missing"], r["lang"]) < min_missing:
                err(rid, "missing must be a phrase naming the absent fact")
            if not isinstance(r["tempting"], bool):
                err(rid, "tempting must be a boolean")
            if r["blur_mode"] not in ("blurred", "removed"):
                err(rid, f"blur_mode {r.get('blur_mode')!r} must be blurred or removed")
            if not is_question(tgt, r["lang"]):
                err(rid, "a -u target must end in a question mark (or か/？)")
            if tgt.count("?") + tgt.count("？") > 1:
                err(rid, "a -u target must ask exactly one question")

    # forbidden held-out domains anywhere in the text
    for r in rows:
        blob = json.dumps(r, ensure_ascii=False).lower()
        for w in FORBIDDEN:
            if re.search(r"\b" + w + r"\b", blob):
                warns.append(f"{r.get('id')}: mentions held-out domain word {w!r}")

    # pairing
    for pair, sides in sorted(pairs.items()):
        if set(sides) != {"d", "u"}:
            errors.append(f"{pair}: incomplete pair, has {sorted(sides)}")
            continue
        d, u = sides["d"], sides["u"]
        if d["question"] != u["question"]:
            errors.append(f"{pair}: twins ask different questions")
        if d["options"] != u["options"]:
            errors.append(f"{pair}: twins offer different options")
        if d["domain"] != u["domain"] or d["lang"] != u["lang"]:
            errors.append(f"{pair}: twins differ in domain or lang")
        if set(d["state"]) != set(u["state"]):
            errors.append(f"{pair}: twins have different state fields")
        ds = json.dumps(d["state"], ensure_ascii=False)
        us = json.dumps(u["state"], ensure_ascii=False)
        if max(len(ds), len(us)) > 1.35 * min(len(ds), len(us)):
            errors.append(f"{pair}: twin states differ too much in length ({len(ds)} vs {len(us)})")
        if ds == us:
            errors.append(f"{pair}: twin states are identical")
        dt, ut = tlen(d["target"], d["lang"]), tlen(u["target"], u["lang"])
        limit = 20 if d["lang"] in CJK else 10
        if abs(dt - ut) > limit:
            errors.append(f"{pair}: twin targets differ by {abs(dt - ut)} (max {limit})")
        # how many state fields changed
        changed = [k for k in d["state"] if k not in u["state"] or d["state"][k] != u["state"][k]]
        if len(changed) != 1:
            errors.append(f"{pair}: {len(changed)} state fields differ, want exactly 1: {changed}")

    npairs = len(pairs)
    print(f"rows={len(rows)} pairs={npairs}")

    # -- distributions -------------------------------------------------
    nopt = Counter(len(r["options"]) for r in rows if isinstance(r.get("options"), dict))
    two_share = nopt[2] / max(1, sum(nopt.values()))
    print("option counts:", dict(sorted(nopt.items())), f"two-option share={two_share:.1%}")
    if two_share > 0.50:
        errors.append(f"two-option items are {two_share:.1%} of rows, max 50%")

    golds = Counter()
    for r in rows:
        if r.get("decidable") and r.get("gold") in (r.get("options") or {}):
            golds[list(r["options"]).index(r["gold"])] += 1
    tot = max(1, sum(golds.values()))
    print("gold position:", {k: f"{v} ({v / tot:.0%})" for k, v in sorted(golds.items())})
    if golds and max(golds.values()) / tot > 0.45:
        errors.append(f"gold sits at one position {max(golds.values()) / tot:.0%} of the time, max 45%")
    if len([k for k in golds if golds[k] / tot >= 0.05]) < 3:
        errors.append("gold does not reach at least three option positions")

    langs = Counter(r["lang"] for r in rows)
    non_en = sum(v for k, v in langs.items() if k != "en")
    share = non_en / max(1, len(rows))
    print("langs:", dict(langs.most_common()), f"non-english={share:.1%}")
    if npairs >= 300 and not 0.15 <= share <= 0.21:
        errors.append(f"non-english share {share:.1%} outside 15-21%")

    comp_pairs = sorted({r["pair"] for r in rows if is_compliance(r)})
    cshare = len(comp_pairs) / max(1, npairs)
    print(f"yes/no compliance pairs: {len(comp_pairs)} ({cshare:.1%})")
    if npairs >= 300 and cshare < 0.25:
        errors.append(f"yes/no compliance items are {cshare:.1%} of pairs, need 25%")

    doms = Counter(r["domain"] for r in rows)
    print("domains:", dict(doms.most_common()))

    nested = sum(1 for r in rows if shape(r["state"])[0])
    listed = sum(1 for r in rows if shape(r["state"])[1])
    nulled = sum(1 for r in rows if shape(r["state"])[2])
    fields = Counter(len(r["state"]) for r in rows)
    print(f"state shapes: nested={nested} with-list={listed} with-null={nulled}",
          "fieldcounts:", dict(sorted(fields.items())))
    if npairs >= 300:
        if nested < 0.3 * len(rows):
            errors.append("fewer than 30% of states use nesting")
        if listed < 0.12 * len(rows):
            errors.append("fewer than 12% of states use a list")
        if len(fields) < 4:
            errors.append("state field counts are not varied enough")

    blur = Counter(r.get("blur_mode") for r in rows if r.get("blur_mode"))
    print("blur modes:", dict(blur))
    la = Counter(r.get("looks_ambiguous") for r in rows if "looks_ambiguous" in r)
    print("looks_ambiguous:", dict(la))
    tp = Counter(r.get("tempting") for r in rows if "tempting" in r)
    print("tempting:", dict(tp))
    if npairs >= 300:
        if la[True] < 0.5 * npairs:
            errors.append("fewer than half the -d rows are looks_ambiguous")
        if tp[True] < 0.6 * npairs:
            errors.append("fewer than 60% of -u rows are tempting")

    # -- duplicate check ----------------------------------------------
    mine = {r["id"]: trigrams(r) for r in rows if "question" in r and "state" in r}
    worst_ext = (0.0, None, None)
    try:
        with open("data/system-one/train.jsonl", encoding="utf-8") as fh:
            ext = []
            for line in fh:
                line = line.strip()
                if line:
                    o = json.loads(line)
                    if "question" in o and "state" in o:
                        ext.append((o.get("id"), trigrams(o)))
    except FileNotFoundError:
        ext = []
        warns.append("train.jsonl not found, external duplicate check skipped")
    for rid, tg in mine.items():
        for oid, otg in ext:
            j = jaccard(tg, otg)
            if j > worst_ext[0]:
                worst_ext = (j, rid, oid)
            if j >= 0.5:
                errors.append(f"{rid}: 3-gram Jaccard {j:.2f} against train.jsonl {oid}")
    print(f"max jaccard vs train.jsonl: {worst_ext[0]:.2f} ({worst_ext[1]} ~ {worst_ext[2]})")

    items = sorted(mine.items())
    worst_int = (0.0, None, None)
    for i in range(len(items)):
        rid, tg = items[i]
        for j2 in range(i + 1, len(items)):
            oid, otg = items[j2]
            if rid[:-2] == oid[:-2]:
                continue  # twins are supposed to be near-identical
            j = jaccard(tg, otg)
            if j > worst_int[0]:
                worst_int = (j, rid, oid)
            if j >= 0.5:
                errors.append(f"{rid}: 3-gram Jaccard {j:.2f} against sibling {oid}")
    print(f"max jaccard inside this file: {worst_int[0]:.2f} ({worst_int[1]} ~ {worst_int[2]})")

    for w in warns:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print(f"\n{len(errors)} error(s), {len(warns)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
