#!/usr/bin/env python3
"""Does the cascade judge agree with a human labeller? (plan section 2, the 85% gate)

Input is the labelled JSONL joined to the seed pairs it was written against:

  judge-calibration.jsonl   id, source_id, domain, decidable, reply, category,
                            label_commits_to (option / "asks_followup" / "none"),
                            label_asks_question
  seed-pairs.jsonl          id, lang, state, question, options, gold

Stage A is the deterministic rule, reported alone and assisted by Laya's `noul` head. Stage B
is Laya's `choice` over the item's real options, swept over the `none` threshold, both
phrasings and every checkpoint, then broken down by domain and by language. A question-shaped
reply that still names an option confidently is a `confirmation`, which has its own, stricter
threshold. Agreement is always the raw argmax: a scalar temperature cannot move an argmax, so
temperature is reported for NLL and ECE only and has no bearing on the gate.

Usage:
  judge_calibration.py data/system-one/judge-calibration.jsonl \
      --seeds data/system-one/seed-pairs.jsonl
"""
import argparse, collections, gc, json, os, sys

from scorecard import (ASKS, JUDGE_Q, NONE, asks_question_rule, cap_threads, load_agent,
                       requests_for, run_batch, score_item)

MARGINS = [0.10, 0.20, 0.30, 0.40]
THRESHOLDS = [0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
CONFIRM = [0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
PHRASINGS = ("reply", "reply_question")
GRID = [0.25 + 0.05 * i for i in range(76)]


def pct(x):
    return "%5.1f%%" % (100 * x)


def load_items(path, seeds_path):
    seeds = {r["id"]: r for r in map(json.loads, open(seeds_path))}
    items = []
    for r in map(json.loads, open(path)):
        s = seeds[r["source_id"]]
        it = dict(r)
        it.update(state=s["state"], question=s["question"], options=s["options"],
                  gold=s.get("gold"), lang=s.get("lang", "en"))
        if it["label_commits_to"] not in (ASKS, NONE) and it["label_commits_to"] not in it["options"]:
            raise ValueError("%s: label_commits_to=%r is not an option" % (it["id"], it["label_commits_to"]))
        items.append(it)
    return items


def agreement(pred, lab):
    return sum(int(p == y) for p, y in zip(pred, lab)) / max(1, len(lab))


def by_group(pred, lab, groups):
    out = {}
    for g in sorted(set(groups)):
        sel = [i for i, x in enumerate(groups) if x == g]
        out[g] = (len(sel), agreement([pred[i] for i in sel], [lab[i] for i in sel]))
    return out


# ---------------------------------------------------------------- stages

def stage_a(items, nouls, gate):
    lab = [bool(it["label_asks_question"]) for it in items]
    rule = [asks_question_rule(it["reply"]) for it in items]
    print("\n-- stage A: asks_question (rule is deterministic, identical for every checkpoint)")
    print("  rule only            %s" % pct(agreement(rule, lab)))
    for name, nl in nouls.items():
        laya = [p >= 0.5 for p in nl]
        print("  %-20s %s  (noul alone)" % (name, pct(agreement(laya, lab))))
        for m in MARGINS:
            pred = [(l if (l != r and abs(p - 0.5) >= m) else r) for r, l, p in zip(rule, laya, nl)]
            print("    + rule, margin %.2f  %s  (%d overrides)"
                  % (m, pct(agreement(pred, lab)), sum(int(p != r) for p, r in zip(pred, rule))))
    tp = sum(int(p and y) for p, y in zip(rule, lab)); fp = sum(int(p and not y) for p, y in zip(rule, lab))
    fn = sum(int((not p) and y) for p, y in zip(rule, lab)); tn = len(lab) - tp - fp - fn
    acc = agreement(rule, lab)
    print("  rule confusion: question %d caught / %d missed | answer %d ok / %d false alarm" % (tp, fn, tn, fp))
    print("  by domain  %s" % "  ".join("%s %s(%d)" % (g, pct(a), n)
                                        for g, (n, a) in by_group(rule, lab, [i["domain"] for i in items]).items()))
    print("  by lang    %s" % "  ".join("%s %s(%d)" % (g, pct(a), n)
                                        for g, (n, a) in by_group(rule, lab, [i["lang"] for i in items]).items()))
    print("  STAGE A: %s [%s]" % (pct(acc), "PASS" if acc >= gate else "FAIL"))
    for it, r, y in zip(items, rule, lab):
        if r != y:
            print("    miss %-8s rule=%-5s label=%-5s  %r" % (it["id"], r, y, it["reply"][:72]))
    return acc


def b_predict(answers, items, keep, th):
    """Laya's stage-B call on the kept rows, with the `none` escape hatch applied."""
    pred = []
    for i in keep:
        p = answers[i][JUDGE_Q]["probabilities"]
        top = max(p, key=p.get)
        pred.append(top if p[top] >= th else NONE)
    return pred


def stage_b(items, runs, gate):
    """`runs` maps (checkpoint, phrasing) -> per-item answers over all 200 rows."""
    keep = [i for i, it in enumerate(items) if not it["label_asks_question"]]
    lab = [items[i]["label_commits_to"] for i in keep]
    print("\n-- stage B: commits_to  (%d of %d rows are not questions; %d labelled none)"
          % (len(keep), len(items), sum(1 for l in lab if l == NONE)))

    best = (None, None, -1.0, None)
    for (ckpt, phrasing), answers in runs.items():
        cells = [(th, agreement(b_predict(answers, items, keep, th), lab)) for th in THRESHOLDS]
        print("  %-16s %-14s %s" % (ckpt, phrasing, " ".join("th%.2f %s" % (t, pct(a)) for t, a in cells)))
        for th, a in cells:
            if a > best[2] + 1e-9:
                best = (ckpt, phrasing, a, th)

    ckpt, phrasing, acc, th = best
    answers = runs[(ckpt, phrasing)]
    pred = b_predict(answers, items, keep, th)
    print("\n  best: %s / %s / th=%.2f -> %s [%s]"
          % (ckpt, phrasing, th, pct(acc), "PASS" if acc >= gate else "FAIL"))

    tab = collections.defaultdict(lambda: [0, 0, 0])
    for y, p in zip(lab, pred):
        tab["none" if y == NONE else "option"][0 if p == y else (2 if p == NONE else 1)] += 1
    print("  confusion                 predicted:   match  other-option   none")
    for k in sorted(tab):
        print("    labelled %-8s                  %5d %13d %6d" % (k, tab[k][0], tab[k][1], tab[k][2]))
    print("  by domain  %s" % "  ".join(
        "%s %s(%d)" % (g, pct(a), n) for g, (n, a) in
        by_group(pred, lab, [items[i]["domain"] for i in keep]).items()))
    print("  by lang    %s" % "  ".join(
        "%s %s(%d)" % (g, pct(a), n) for g, (n, a) in
        by_group(pred, lab, [items[i]["lang"] for i in keep]).items()))

    # does routing non-English rows to the multilingual checkpoint beat one checkpoint for all?
    nonen = [j for j, i in enumerate(keep) if items[i]["lang"] != "en"]
    if nonen and ("multilingual", phrasing) in runs:
        ml = b_predict(runs[("multilingual", phrasing)], items, keep, th)
        sub = lambda p, s: [p[j] for j in s]
        print("\n  non-English rows (%d): %s %s | multilingual %s"
              % (len(nonen), ckpt, pct(agreement(sub(pred, nonen), sub(lab, nonen))),
                 pct(agreement(sub(ml, nonen), sub(lab, nonen)))))
        routed = [ml[j] if items[keep[j]]["lang"] != "en" else pred[j] for j in range(len(keep))]
        ra = agreement(routed, lab)
        print("  router (multilingual off English): %s overall, vs %s for %s alone"
              % (pct(ra), pct(acc), ckpt))
        if ra > acc + 1e-9:
            print("  -> routing wins")
            return ckpt, phrasing, th, ra, True
    return ckpt, phrasing, th, acc, False


def stage_confirm(items, answers, idx, none_th, th_confirm_grid):
    """The replies that commit and also ask.

    Swept on end-to-end headline error, not on a class count: a false confirmation on an
    underspecified item costs 1.5 points and a missed one on a decidable item costs 1.25, so
    counting classes would not weigh the two the same way the headline does.
    """
    from scorecard import score_all
    conf = [i for i, it in enumerate(items)
            if it["label_asks_question"] and it["label_commits_to"] not in (ASKS, NONE)]
    clean = [i for i, it in enumerate(items) if it["label_commits_to"] == ASKS]
    cand = [i for i in range(len(items)) if "confirm" in idx.get(i, {})]
    print("\n-- confirmations: %d rows commit and ask; %d are clean follow-ups" % (len(conf), len(clean)))
    print("  the prefix rule puts %d rows forward for stage C, %d of the %d confirmations among them"
          % (len(cand), len(set(cand) & set(conf)), len(conf)))

    def pre(i):
        a = answers[idx[i]["confirm"]][JUDGE_Q]
        return a["choice"], a["probabilities"][a["choice"]]

    lab = label_rows(items)
    best = (None, float("inf"))
    for th in th_confirm_grid:
        named = sum(1 for i in conf if i in cand and pre(i)[1] >= th
                    and pre(i)[0] == items[i]["label_commits_to"])
        false = sum(1 for i in clean if i in cand and pre(i)[1] >= th)
        cas = score_all(items, answers, idx, none_th, th)
        mad = sum(abs(c["score"] - l["score"]) for c, l in zip(cas, lab)) / len(cas)
        print("  th=%.2f  %d/%d confirmations caught correctly, %d/%d clean follow-ups wrongly "
              "confirmed, headline |error| %.4f" % (th, named, len(conf), false, len(clean), mad))
        if mad < best[1] - 1e-9:
            best = (th, mad)
    for i in cand:
        c, p = pre(i)
        print("    %-8s %-12s prefix p=%.2f names=%-22s label=%-22s %r"
              % (items[i]["id"], "confirmation" if i in conf else "follow-up", p, c[:22],
                 items[i]["label_commits_to"][:22], items[i]["reply"][:52]))
    print("  best confirm threshold by headline error: %.2f" % best[0])
    return best[0]


def label_rows(items):
    """The headline the labels imply, so the cascade can be compared against it row by row."""
    out = []
    for it in items:
        c = it["label_commits_to"]
        confirmation = bool(it["label_asks_question"]) and c not in (ASKS, NONE)
        score, outcome = score_item(it["decidable"], it.get("gold"), c, confirmation)
        out.append({"id": it["id"], "score": score, "outcome": outcome})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--checkpoints", default="english,typed-decisions,multilingual")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--gate", type=float, default=0.85)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None, help="reuse encoder outputs across analysis runs")
    args = ap.parse_args()
    cap_threads(args.threads)

    items = load_items(args.items, args.seeds)
    print("%d labelled rows: %d questions (%d of them also name an option), %d commit, %d none"
          % (len(items), sum(1 for i in items if i["label_asks_question"]),
             sum(1 for i in items if i["label_asks_question"] and i["label_commits_to"] not in (ASKS, NONE)),
             sum(1 for i in items if i["label_commits_to"] not in (ASKS, NONE)),
             sum(1 for i in items if i["label_commits_to"] == NONE)))
    print("domains %s" % dict(collections.Counter(i["domain"] for i in items)))
    print("langs   %s" % dict(collections.Counter(i["lang"] for i in items)))

    # one checkpoint resident at a time: a GPU job is running beside this
    runs, nouls, truncated = {}, {}, 0
    if args.cache and os.path.exists(args.cache):
        blob = json.load(open(args.cache))
        runs = {tuple(k.split("|")): (v["answers"], {int(i): e for i, e in v["idx"].items()})
                for k, v in blob["runs"].items()}
        nouls = blob["nouls"]
        print("loaded encoder outputs from %s" % args.cache)
    for ckpt in [c.strip() for c in args.checkpoints.split(",")]:
        if ckpt in nouls:
            continue
        agent = load_agent(ckpt, args.device)
        nouls[ckpt] = [a["asks_question"]["noul"] for a in run_batch(
            agent, [({"reply": it["reply"]},
                     {"asks_question": {"type": "noul",
                                        "instructions": "Does `reply` ask the user a question "
                                                        "instead of answering?"}}) for it in items],
            args.batch)]
        for phrasing in PHRASINGS:
            reqs, idx, cut = requests_for(agent, items, phrasing)
            truncated += cut
            runs[(ckpt, phrasing)] = (run_batch(agent, reqs, args.batch), idx)
        del agent
        gc.collect()
        print("  ran %s" % ckpt)
    if args.cache:
        json.dump({"runs": {"|".join(k): {"answers": a, "idx": i} for k, (a, i) in runs.items()},
                   "nouls": nouls}, open(args.cache, "w"))
    if truncated:
        print("note: %d option sets were shortened to fit the option budget" % truncated)

    # stage B looks at one answer per item, so hand it the stage-B call of each run
    aligned = {k: [a[i[j][JUDGE_Q]] for j in range(len(items))] for k, (a, i) in runs.items()}
    a_acc = stage_a(items, nouls, args.gate)
    ckpt, phrasing, th, b_acc, routed = stage_b(items, aligned, args.gate)
    answers, idx = runs[(ckpt, phrasing)]
    th_confirm = stage_confirm(items, answers, idx, th, CONFIRM)

    # end to end: the cascade's headline against the labels' headline, row by row
    from scorecard import score_all
    cas = score_all(items, answers, idx, th, th_confirm)
    lab = label_rows(items)
    diff = [(c, l) for c, l in zip(cas, lab) if abs(c["score"] - l["score"]) > 1e-9]
    mad = sum(abs(c["score"] - l["score"]) for c, l in zip(cas, lab)) / len(cas)
    print("\n-- end to end (%s / %s / none=%.2f / confirm=%.2f)" % (ckpt, phrasing, th, th_confirm))
    print("  headline from cascade %.4f   from labels %.4f   mean |diff| %.4f   rows differing %d/%d"
          % (sum(c["score"] for c in cas) / len(cas), sum(l["score"] for l in lab) / len(lab), mad, len(diff), len(cas)))
    for c, l in diff:
        print("    %-8s cascade %-30s %5.2f   labels %-30s %5.2f" % (c["id"], c["outcome"], c["score"],
                                                                     l["outcome"], l["score"]))
    print("\nGATE (%.0f%%): stage A %s [%s]   stage B %s [%s]"
          % (100 * args.gate, pct(a_acc), "PASS" if a_acc >= args.gate else "FAIL",
             pct(b_acc), "PASS" if b_acc >= args.gate else "FAIL"))

    if args.out:
        json.dump({"checkpoint": ckpt, "phrasing": phrasing, "none_threshold": th,
                   "confirm_threshold": th_confirm, "stage_a": a_acc, "stage_b": b_acc,
                   "routed": routed}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    sys.exit(main())
