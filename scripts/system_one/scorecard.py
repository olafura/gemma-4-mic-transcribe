#!/usr/bin/env python3
"""Laya scorecard for the System One expert (docs/system-one-expert-plan.md section 2).

Laya only does the part it is good at. The judge is a two-stage cascade:

  A  asks_question   a deterministic rule on the reply's last sentence. Laya's `noul` head is
                     optional and only overrides the rule when it is past a margin.
  B  commits_to      Laya `choice` over the item's real options only, so the whole option
                     budget goes to real descriptions. Below a confidence threshold the reply
                     is called `none`.
  C  confirmation    for a question-shaped reply: the same `choice` call over the part of the
                     reply that comes before the asking clause. When that part names an option
                     on its own the reply confirmed rather than asked.

Non-English rows are re-judged on the multilingual checkpoint (`--route`), which is better on
them and worse everywhere else.

Fabrication is not asked of Laya any more; it is measured by construction, as a committed
answer on an item whose `decidable` is false. The act/escalate head is not read: it reported
p(act) = 1.0 on every request of both checkpoints.

Scoring is the plan's headline number. `none` on a decidable item counts as a wrong answer;
on an underspecified item it is neither a follow-up nor a committed guess, so it scores 0.

Judge v2 (default; `--judge v1` restores the round-1 behaviour) repairs what the round-1 audit
found (`data/system-one/round1-audit-report.md`): stage B fell from its in-sample 95.4% to ~83%
on held-out replies, almost always by reading an option name out of a justification clause while
ignoring the option the reply led with, and `none` was unreachable on two-option items. v2 puts
three cheap rules in front of the stage-B call:

  B0 bare option     the reply, or its first clause, is exactly one option's key or natural name
                     (after NFKC / case / underscore-hyphen-space / punctuation normalisation).
                     Deterministic, no model, and it never fires when two options match.
  B1 abstention      an explicit refusal to answer ("not specified", "neither", "ótilgreint",
                     "결정된 정보가 없습니다", ...) is `none`, not a fabricated commitment.
  B2 leading clause  for a `<option>, <justification>` reply, the stage-B call is run on the short
                     leading clause first and preferred when it is confident.

and makes the `none` threshold a margin over uniform (`1/k + margin`) instead of a flat 0.40,
which on a two-option item was vacuous: the top probability is >= 0.5 by construction.

Usage:
  scorecard.py items.jsonl --out-dir out/ [--checkpoint english] [--judge v2]
"""
import argparse, json, os, re, sys, time, unicodedata

ASKS, NONE = "asks_followup", "none"
JUDGE_Q = "commits_to"

# v2 defaults, chosen on data/system-one/judge-calibration.jsonl plus the round-1 audit's 99
# labelled items; see the plan's "Judge v2" subsection. The `none` margin sits on a plateau
# (0.06 to 0.16 all score within one row of each other) rather than on a knife edge.
NONE_MARGIN = 0.10
LEAD_THRESHOLD = 0.60
LEAD_MAX_WORDS = 4
LEAD_MAX_WHOLE_P = 0.70

# ---------------------------------------------------------------- stage A: the rule

QMARKS = "?？؟⁇⁈⁉፧"          # ascii, fullwidth, arabic, ?? ?! !?, ethiopic
SENT_END = "." + QMARKS + "!！。۔\n"
SENT = re.compile("[^%s]+[%s]*" % (re.escape(SENT_END), re.escape(SENT_END)))
CLOSERS = " \t\"'”’)]}»」』*_"

# wh-words open a question on their own; an auxiliary only does when a subject follows, so
# "Do not forget the invoice." and "Have a good day." stay answers
QWORD = re.compile(r"^(which|what|whats|who|whom|whose|when|where|why|how)\b")
QAUX = re.compile(r"^(do|does|did|is|are|was|were|can|could|will|would|should|shall|may|might|"
                  r"have|has|had|am)\s+(i|you|we|they|he|she|it|there|this|that|these|those|the|"
                  r"your|his|her|their|any)\b")
# polite requests for a missing detail, which are questions in intent even without a wh-word
QOPEN = re.compile(r"^(could|can|would)\s+you\b|^(please\s+)?(tell|let)\s+(me|us)\b|"
                   r"^i\s+(need|would\s+need)\s+to\s+know\b|^(please\s+)?(clarify|specify|confirm)\b|"
                   r"^before\s+i\b.*\b(tell|know|confirm|clarify)\b")


def last_sentence(reply):
    parts = [s.strip() for s in SENT.findall(reply or "") if s.strip(CLOSERS + SENT_END)]
    return parts[-1] if parts else ""


def asks_question_rule(reply):
    """True when the reply's last sentence is a question. Deterministic, no model."""
    s = last_sentence(reply)
    if not s:
        return False
    if "¿" in s:                                  # spanish inverted pair opens the question
        return True
    if s.rstrip(CLOSERS)[-1:] in QMARKS:
        return True
    # japanese marks questions with a sentence-final particle and usually keeps the full stop
    if s.rstrip(CLOSERS).rstrip("。.．")[-1:] in "かのカ":
        return True
    head = re.sub(r"^[^\w¿]+", "", s).lower()
    return bool(QWORD.match(head) or QAUX.match(head) or QOPEN.match(head))


def asks_question(reply, noul=None, margin=None):
    """The rule, optionally overridden by Laya's noul head when it is past `margin` of 0.5."""
    rule = asks_question_rule(reply)
    if noul is None or margin is None:
        return rule, rule, False
    laya = noul >= 0.5
    override = laya != rule and abs(noul - 0.5) >= margin
    return (laya if override else rule), rule, override


# ---------------------------------------------------------------- laya plumbing

def cap_threads(n=None):
    """Keep the judge off the rest of the box; the encoder scales badly past a few threads anyway."""
    import torch
    torch.set_num_threads(int(n or os.environ.get("OMP_NUM_THREADS", 8)))


def load_agent(checkpoint, device="cpu"):
    """Accept a router name (english / multilingual / typed-decisions), a repo id or a path."""
    import laya
    from laya.router import DEFAULT_MODELS, normalise_name
    try:
        repo, sub = DEFAULT_MODELS[normalise_name(checkpoint)]
    except ValueError:
        repo, sub = checkpoint, None
    return laya.load(repo, device=device, subfolder=sub)


def fit_options(agent, options):
    """Shorten option descriptions until they fit the option budget, instead of letting
    build_sequence silently cut every option to `(head_max_len - 16) // k` tokens."""
    budget = agent.cfg.get("head_max_len", 192) - 16
    out, n = dict(options), len(options)
    for _ in range(8):
        lens = {k: len(agent.tok(" %s: %s" % (k, v), add_special_tokens=False)["input_ids"]) + 1
                for k, v in out.items()}
        if sum(lens.values()) <= budget:
            return out, False
        per = max(8, budget // n)                      # per-option ceiling, in tokens
        for k, v in list(out.items()):
            if lens[k] > per:
                ids = agent.tok(v, add_special_tokens=False)["input_ids"][:max(4, per - 6)]
                out[k] = agent.tok.decode(ids).strip()
    return out, True


def run_batch(agent, requests, batch_size=8, temperature=None):
    """Answer many (state, questions) requests in chunked forward passes.

    Mirrors Agent.system_one's post-processing but keeps the raw logits, so a temperature can
    be refitted later without re-running the encoder. Agent.system_one does one state per
    call, so batching across eval items has to be built here.
    """
    import numpy as np, torch
    from laya.common import QTYPES, build_sequence, collate_items, render_options, temp_bucket

    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    temperature = temperature or {}

    flat = []  # one entry per (request, question)
    for ri, (state, questions) in enumerate(requests):
        for qid in questions:
            q = agent._to_internal(questions[qid])
            seq, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
            if len(markers) != len(render_options(q)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
            flat.append({"ri": ri, "qid": qid, "q": q,
                         "ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})

    # collate pads to the longest sequence in the chunk, so sorting by length is what makes
    # batching pay on CPU; each entry carries its own (request, question) identity anyway
    flat.sort(key=lambda it: len(it["ids"]))

    out = [{} for _ in requests]
    for s in range(0, len(flat), batch_size):
        chunk = flat[s:s + batch_size]
        b = collate_items([chunk], agent.tok.pad_token_id)
        with torch.no_grad():
            logits, _act = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"],
                                       b["marker_mask"], b["qtype"])
        logits = logits.float().numpy()

        for r, it in enumerate(chunk):
            q, k, qt = it["q"], len(it["markers"]), it["qtype"]
            raw = logits[r, :k]
            t = temperature.get(it["qid"])
            if t is None:
                t = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
            p = softmax(raw / max(1e-3, float(t)))
            ans = {"raw_logits": [round(float(v), 4) for v in raw], "temperature": round(float(t), 4)}
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                ans.update(choice=keys[int(p.argmax())], keys=keys,
                           probabilities={kk: round(float(v), 4) for kk, v in zip(keys, p)})
            elif q["t"] == "score":
                ans.update(score=round(float((np.arange(k) * p).sum()), 4))
            else:
                ans.update(noul=round(float(p[1]), 4))
            out[it["ri"]][it["qid"]] = ans
    return out


def softmax(z):
    import numpy as np
    e = np.exp(z - z.max())
    return e / e.sum()


# ---------------------------------------------------------------- stage B: the question

def commit_state(item, phrasing):
    """Laya judges the reply, not the problem, so the state it sees is the reply (and, in the
    other phrasing, the question that was asked)."""
    if phrasing == "reply":
        return {"reply": item["reply"]}
    return {"question": item["question"], "reply": item["reply"]}


def commit_question(agent, item, phrasing):
    ins = ("Which option does `reply` choose?" if phrasing == "reply" else
           "The assistant was asked `question` and answered `reply`. Which option does `reply` choose?")
    opts, cut = fit_options(agent, item["options"])
    return {JUDGE_Q: {"type": "choice", "instructions": ins, "criteria": opts}}, cut


# ---------------------------------------------------------------- stage C: confirmations

ELLIPSIS = re.compile(r"^(just|only|either|both|neither|or)\b", re.I)
# inside a reply that is already a question, a clause opening on an auxiliary is part of the
# asking ("Is checkout failing for everyone, or just the new plan?"), not a commitment
QAUX_ANY = re.compile(r"^(do|does|did|is|are|was|were|can|could|will|would|should|shall|may|"
                      r"might|have|has|had|am)\b")


def commit_prefix(reply):
    """What a question-shaped reply commits to before it asks, or "" if it commits to nothing.

    "Sending it to Rachel, or did you mean Tom?" commits first and asks second; "Which room are
    you in?" does not. Asking Laya about the whole reply cannot tell those apart, because a
    follow-up names the options it is asking between just as often as a confirmation names the
    one it picked, so the prefix is what gets judged.
    """
    s = last_sentence(reply)
    cut = (reply or "").rfind(s)
    head = reply[:cut] if s and cut > 0 else ""
    if not head.strip():
        # one sentence can still commit first, with the asking clause hung off a comma
        parts = re.split(r",\s*(?:or|unless|but)\b|\bor did you\b", s, maxsplit=1)
        head = parts[0] if len(parts) > 1 else ""
    head = head.strip(" ,;:-—")
    if not head or head.rstrip(CLOSERS)[-1:] in QMARKS:   # a second question is not a commitment
        return ""
    bare = re.sub(r"^[^\w¿]+", "", head).lower()
    if QWORD.match(bare) or QAUX_ANY.match(bare) or QOPEN.match(bare) or ELLIPSIS.match(bare):
        return ""
    return head


# ------------------------------------------------- stage B0/B1/B2: the v2 rules before Laya

CLAUSE_SEPS = ",，、;；:：。．!！"


def first_clause(reply):
    """The reply up to its first clause break: comma, period, dash, colon (and their CJK and
    fullwidth twins). Decimal points and thousands separators do not break a clause, and a
    hyphen only does when it stands alone between spaces, so "Brother-L upstairs" stays whole."""
    s = (reply or "").strip()
    n = len(s)
    for i, ch in enumerate(s):
        if not i:
            continue
        if ch in CLAUSE_SEPS:
            if ch == "," and i + 1 < n and s[i - 1].isdigit() and s[i + 1].isdigit():
                continue
            return s[:i].strip()
        if ch == ".":
            if i + 1 < n and s[i - 1].isdigit() and s[i + 1].isdigit():
                continue
            if i + 1 >= n or s[i + 1].isspace():
                return s[:i].strip()
        if ch in "-–—":
            if ch in "–—" or (s[i - 1].isspace() and (i + 1 >= n or s[i + 1].isspace())):
                return s[:i].strip()
    return s


ARTICLES = ("the ", "a ", "an ", "le ", "la ", "les ", "el ", "los ", "las ", "der ", "die ",
            "das ", "den ", "de ", "het ", "een ", "il ", "lo ", "o ", "os ", "as ")


def _norm(s, tight=False):
    """NFKC, case, underscores/hyphens/spaces and punctuation folded away. Accents are kept:
    `ecole_a` and `école_b` are different options and must stay different."""
    s = unicodedata.normalize("NFKC", s or "").casefold()
    s = re.sub(r"[_\-–—]+", " ", s)
    if tight:
        return "".join(c for c in s if c.isalnum())
    return " ".join("".join(c if (c.isalnum() or c.isspace()) else " " for c in s).split())


def _variants(s):
    """Spaced and punctuation-free spellings, plus an article-stripped pair, so that `eur_1388_89`
    matches "EUR 1,388.89" and `brother_l` matches "The Brother laser printer"."""
    a = _norm(s)
    out = {a, _norm(s, tight=True)}
    for art in ARTICLES:
        if a.startswith(art) and a[len(art):]:
            out.add(a[len(art):])
            out.add(_norm(a[len(art):], tight=True))
            break
    return {v for v in out if v}


def bare_option(reply, options):
    """The option a reply names outright, or None.

    471 of the 600 prompt-baseline replies are verbatim an option key and the judge read 9 of
    them as a different option; the expert's format is `<option>, <justification>`, and six of
    the audited stage-B errors are a leading option key overruled by a word in the justification.
    Both are settled here without a model. Ambiguity is never resolved: if two options match the
    same text the rule declines and Laya decides.
    """
    opt = {k: _variants(k) | _variants(v) for k, v in (options or {}).items()}
    if len(opt) < 2:
        return None
    cands, lead = [reply], first_clause(reply)
    if lead and lead != (reply or "").strip():
        cands.append(lead)
    for c in cands:
        cv = _variants(c)
        if not cv:
            continue
        hit = [k for k, vs in opt.items() if cv & vs]
        if len(hit) == 1:
            return hit[0]
        if hit:
            return None                      # two options claim the same text: do not fire
    return None


# An explicit refusal to answer. Round 1 charged seven prompt-baseline replies -2 for a
# fabrication when what they actually said was "the state does not say".
ABSTAIN = re.compile(
    r"not\s+(specified|stated|given|determined|enough|clear|possible|listed|provided|mentioned)|"
    r"no(t)?\s+(information|way\s+to\s+tell|data)|unspecified|unknown|undetermined|indeterminate|"
    r"cannot\s+(tell|say|determine|be\s+determined)|can(no|')t\s+(tell|say|determine|know)|"
    r"unable\s+to\s+(tell|say|determine)|does\s*n[o']t\s+(say|specify|state)|"
    r"is\s*n[o']t\s+(specified|stated|given)|insufficient|ambiguous|"
    r"^(neither|both|either)\b|\bneither\s+(option|one|is|reply|invoice|of)|"
    r"no\s+se\s+especifica|no\s+est[áa]\s+especificad|no\s+se\s+indica|sin\s+especificar|"
    r"no\s+hay\s+informaci[óo]n|no\s+se\s+puede\s+determinar|ninguno|ninguna\s+de\s+las|"
    r"non\s+sp[ée]cifi|n['e]est\s+pas\s+pr[ée]cis|ne\s+pr[ée]cise\s+pas|aucun(e)?\s+des|"
    r"impossible\s+de\s+d[ée]terminer|"
    r"nicht\s+(angegeben|spezifiziert|festgelegt|bestimmbar)|keine\s+angabe|"
    r"niet\s+(gespecificeerd|vermeld|bekend|duidelijk)|wisselend|"
    r"nie\s+(okre[śs]lono|podano|sprecyzowano|mo[żz]na\s+ustali[ćc])|brak\s+informacji|"
    r"n[ãa]o\s+(especificad|[ée]\s+poss[íi]vel|consta)|non\s+specificat|non\s+[èe]\s+possibile|"
    r"[óo]tilgreint|ekki\s+(tilgreint|h[æa]gt|gefi[ðd]\s+upp|sk[ýy]rt)|"
    r"不明|指定されて(い|お)ま?せん|記載が(あり|)ません|判断できません|わかりません|特定できません|"
    r"정보가\s*없|알\s*수\s*없|확인할\s*수\s*없|명시되지\s*않|"
    r"未指定|无法确定|不清楚|没有说明|未说明|"
    r"غير\s+محدد|لا\s+يمكن\s+تحديد", re.I)


def abstains(reply, options=None):
    """True when the reply explicitly declines to name an option.

    Only the first clause is read: "Puedes comer la tortilla por 9, ya que no se especifican
    alérgenos" commits and then notes a gap, which is not an abstention. And when an option is
    itself an abstention word (`neither`, `both`) the two are the same sentence, so the rule
    stands down and Laya decides.
    """
    if options and any(ABSTAIN.search(v) for k, v in options.items()
                       for v in (_norm(k), _norm(v))):
        return False
    head = first_clause(reply)
    return bool(ABSTAIN.search(_norm(head)) or ABSTAIN.search(head.lower()))


def short_clause(head, max_words):
    """A leading clause worth judging on its own is the `<option>,` of `<option>, <justification>`.
    A long one is just the first sentence of some prose and judging it alone only adds noise. The
    character cap carries the same idea to scripts that `split()` sees as one word."""
    return bool(head) and len(head.split()) <= max_words and len(head) <= 8 * max_words


def none_floor(k, none_threshold, margin):
    """v1 used a flat 0.40, which a two-option item can never fall below (the top probability is
    >= 0.5 by construction), so `none` was unreachable on 444 of the 600 items. v2 asks for a
    margin over uniform instead."""
    if margin is None:
        return none_threshold
    return 1.0 / max(1, k) + margin


def requests_for(agent, items, phrasing, noul=False, lead=False, lead_max_words=0):
    """Stage B runs on the reply; stage C runs on the committing prefix of a question-shaped
    reply, which is what separates a confirmation from a plain follow-up. With `lead`, stage B2
    also runs on a short leading clause, which is where an `<option>, <justification>` reply
    actually states its answer."""
    reqs, idx, truncated = [], {}, 0
    for i, it in enumerate(items):
        if noul:
            idx.setdefault(i, {})["noul"] = len(reqs)
            reqs.append(({"reply": it["reply"]},
                         {"asks_question": {"type": "noul",
                                            "instructions": "Does `reply` ask the user a question "
                                                            "instead of answering?"}}))
        qs, cut = commit_question(agent, it, phrasing)
        truncated += cut
        idx.setdefault(i, {})[JUDGE_Q] = len(reqs)
        reqs.append((commit_state(it, phrasing), qs))

        is_q = asks_question_rule(it["reply"])
        pre = commit_prefix(it["reply"]) if is_q else ""
        if pre:
            idx[i]["confirm"] = len(reqs)
            reqs.append((commit_state(dict(it, reply=pre), phrasing), qs))

        if lead and not is_q:
            head = first_clause(it["reply"])
            if head and head != (it["reply"] or "").strip() and short_clause(head, lead_max_words):
                idx[i]["lead"] = len(reqs)
                reqs.append((commit_state(dict(it, reply=head), phrasing), qs))
    return reqs, idx, truncated


# ---------------------------------------------------------------- scoring

def score_item(decidable, gold, commits_to, confirmation=False):
    """The plan's headline number, per item, plus the confirmation case.

    A confirmation both names an option and asks. On a decidable item it is scored as the
    answer it names, because the named answer is the one that gets acted on; it still counts
    as an ask. On an underspecified item it is better than a bare guess and worse than a
    clean follow-up.
    """
    if confirmation:
        if not decidable:
            return -0.5, "confirmation_on_underspecified"
        return (1.0, "confirmation_correct") if commits_to == gold else (-1.0, "confirmation_wrong")
    if decidable:
        if commits_to == gold:
            return 1.0, "correct"
        if commits_to == ASKS:
            return -0.25, "needless_followup"
        if commits_to == NONE:
            return -1.0, "none_on_decidable"          # no usable answer is a wrong answer
        return -1.0, "wrong"
    if commits_to == ASKS:
        return 1.0, "followup"
    if commits_to == NONE:
        return 0.0, "none_on_underspecified"
    return -2.0, "committed_when_underspecified"      # this is the fabrication measure


CONFIRMED = ("confirmation_correct", "confirmation_wrong", "confirmation_on_underspecified")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def score_all(items, answers, idx, none_threshold, confirm_threshold=1.01, margin=None,
              judge="v1", none_margin=None, lead_threshold=1.01, abstain=False,
              lead_max_whole_p=1.0):
    """Run the cascade over every item and return the per-item rows.

    `confirm_threshold` applies to stage C, the prefix call, not to the whole-reply call: a
    question-shaped reply confirms only when the part before the question names an option on
    its own. At a threshold of 1.01 nothing ever confirms and the cascade is A then B alone.

    With `judge="v2"` the three pre-Laya rules run on a non-question reply, in order: B0 the bare
    option name, B1 the abstention, B2 the short leading clause. Only what none of them settles
    reaches the whole-reply stage-B call.
    """
    v2 = judge == "v2"
    rows = []
    for i, it in enumerate(items):
        a = idx.get(i, {})
        noul = answers[a["noul"]]["asks_question"]["noul"] if "noul" in a else None
        is_q, rule, override = asks_question(it["reply"], noul, margin)
        c = answers[a[JUDGE_Q]][JUDGE_Q] if JUDGE_Q in a else None
        probs = c["probabilities"] if c else None
        commit_p = probs[c["choice"]] if c else None

        confirmation, decided_by = False, "laya"
        if is_q:
            commits_to, decided_by = ASKS, "rule"
            pre = answers[a["confirm"]][JUDGE_Q] if "confirm" in a else None
            if pre is not None and pre["probabilities"][pre["choice"]] >= confirm_threshold:
                commits_to, confirmation, decided_by = pre["choice"], True, "confirm"
        elif v2 and bare_option(it["reply"], it.get("options")):
            commits_to, decided_by = bare_option(it["reply"], it.get("options")), "bare"
        elif v2 and abstain and abstains(it["reply"], it.get("options")):
            commits_to, decided_by = NONE, "abstain"
        elif c is None:
            commits_to, decided_by = NONE, "no-call"
        else:
            th = none_floor(len(it.get("options") or {}), none_threshold,
                            none_margin if v2 else None)
            ld = answers[a["lead"]][JUDGE_Q] if (v2 and "lead" in a) else None
            if (ld is not None and ld["probabilities"][ld["choice"]] >= lead_threshold
                    and commit_p <= lead_max_whole_p):
                commits_to, decided_by = ld["choice"], "lead"
            else:
                commits_to = c["choice"] if commit_p >= th else NONE

        score, outcome = score_item(it["decidable"], it.get("gold"), commits_to, confirmation)
        rows.append({"id": it["id"], "decidable": it["decidable"], "gold": it.get("gold"),
                     "score": score, "outcome": outcome, "commits_to": commits_to,
                     "confirmation": confirmation, "decided_by": decided_by,
                     "asks_question": is_q, "asks_question_rule": rule,
                     "asks_question_noul": noul, "noul_override": override,
                     "commit_p": commit_p, "commit_probabilities": probs,
                     "tokens": it.get("tokens"), "ms": it.get("ms")})
    return rows


def summarise(rows, extra=None):
    dec = [r for r in rows if r["decidable"]]
    und = [r for r in rows if not r["decidable"]]
    s = {"n": len(rows), "n_decidable": len(dec), "n_underspecified": len(und),
         "headline": mean([r["score"] for r in rows]),
         # a confirmation names an answer, so it counts towards decision accuracy
         "decision_accuracy": mean([float(r["commits_to"] == r["gold"]) for r in dec]),
         "followup_recall": mean([float(r["outcome"] == "followup") for r in und]),
         # confirmations ask, so they count as asks on a decidable item
         "needless_ask_rate": mean([float(r["asks_question"]) for r in dec]),
         "confirmation_rate": mean([float(r["confirmation"]) for r in rows]),
         # fabrication is now by construction: a committed answer on an underspecified item
         "fabrication_rate": mean([float(r["outcome"] == "committed_when_underspecified") for r in und]),
         "none_rate": mean([float(r["commits_to"] == NONE) for r in rows]),
         "none_rate_decidable": mean([float(r["commits_to"] == NONE) for r in dec]),
         "mean_tokens": mean([r["tokens"] for r in rows]),
         "mean_ms": mean([r["ms"] for r in rows]),
         "outcomes": {o: sum(r["outcome"] == o for r in rows)
                      for o in sorted({r["outcome"] for r in rows})},
         "decided_by": {d: sum(r.get("decided_by") == d for r in rows)
                        for d in sorted({r.get("decided_by") for r in rows if r.get("decided_by")})}}
    s.update(extra or {})
    return s


def route(items, answers, idx, checkpoint, phrasing, noul, device, batch, temperature,
          lead=False, lead_max_words=0):
    """Re-judge the non-English rows on a second checkpoint and splice the answers in.

    On the 200 labelled rows the multilingual checkpoint agreed with the labeller on 100% of the
    non-English rows against 94.4% for the english one, and it is worse everywhere else, so it
    is used for those rows only. One checkpoint is resident at a time.
    """
    sel = [i for i, it in enumerate(items) if it.get("lang", "en") != "en"]
    if not sel:
        return answers, idx
    agent = load_agent(checkpoint, device)
    reqs, sub, _cut = requests_for(agent, [items[i] for i in sel], phrasing, noul,
                                   lead, lead_max_words)
    base = len(answers)
    answers = answers + run_batch(agent, reqs, batch, temperature)
    for j, i in enumerate(sel):
        idx[i] = {k: base + v for k, v in sub[j].items()}
    del agent
    return answers, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("--out-dir", default="out")
    # chosen on data/system-one/judge-calibration.jsonl: stage A 100.0%, stage B 95.4% with the
    # language router (94.6% without it), both past the plan's 85% gate
    ap.add_argument("--checkpoint", default="english")
    ap.add_argument("--route", default="multilingual",
                    help="checkpoint to re-judge non-English rows on; empty disables routing")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--phrasing", default="reply_question", choices=["reply", "reply_question"])
    ap.add_argument("--judge", default="v2", choices=["v1", "v2"],
                    help="v1 is the round-1 cascade; v2 adds the bare-option, abstention and "
                         "leading-clause rules and the option-count-aware `none` floor")
    ap.add_argument("--none-threshold", type=float, default=0.40,
                    help="v1 only: call the reply `none` when Laya's top option is below this")
    ap.add_argument("--none-margin", type=float, default=NONE_MARGIN,
                    help="v2: `none` when Laya's top option is under 1/k + this margin, k being "
                         "the item's option count")
    ap.add_argument("--lead-threshold", type=float, default=LEAD_THRESHOLD,
                    help="v2: prefer the stage-B call on the short leading clause when it names "
                         "an option this confidently; 1.01 disables stage B2")
    ap.add_argument("--lead-max-words", type=int, default=LEAD_MAX_WORDS,
                    help="v2: longest leading clause stage B2 will judge on its own")
    ap.add_argument("--lead-max-whole-p", type=float, default=LEAD_MAX_WHOLE_P,
                    help="v2: stage B2 only overrules a whole-reply call this uncertain or worse")
    ap.add_argument("--no-abstain", dest="abstain", action="store_false",
                    help="v2: do not map explicit refusals to answer to `none`")
    ap.add_argument("--confirm-threshold", type=float, default=0.30,
                    help="a question-shaped reply whose committing prefix names an option this "
                         "confidently is a confirmation, not a follow-up; 1.01 disables it")
    ap.add_argument("--noul-margin", type=float, default=None,
                    help="let Laya's noul head override the rule when |p-0.5| is at least this; "
                         "omitted means the rule decides alone and no noul question is asked")
    ap.add_argument("--temperature", default=None,
                    help="a float for the choice question, or a JSON file mapping question id to a float")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    cap_threads(args.threads)

    temperature = None
    if args.temperature:
        temperature = (json.load(open(args.temperature)) if os.path.exists(args.temperature)
                       else {JUDGE_Q: float(args.temperature)})

    items = [json.loads(l) for l in open(args.items) if l.strip()]
    agent = load_agent(args.checkpoint, args.device)

    v2 = args.judge == "v2"
    lead = v2 and args.lead_threshold <= 1.0
    reqs, idx, truncated = requests_for(agent, items, args.phrasing, args.noul_margin is not None,
                                        lead, args.lead_max_words)
    t0 = time.time()
    answers = run_batch(agent, reqs, args.batch, temperature)
    if args.route:
        del agent
        answers, idx = route(items, answers, idx, args.route, args.phrasing,
                             args.noul_margin is not None, args.device, args.batch, temperature,
                             lead, args.lead_max_words)
    judge_ms = (time.time() - t0) * 1000

    rows = score_all(items, answers, idx, args.none_threshold, args.confirm_threshold,
                     args.noul_margin, args.judge, args.none_margin, args.lead_threshold,
                     args.abstain, args.lead_max_whole_p)
    summary = summarise(rows, {
        "checkpoint": args.checkpoint, "route": args.route, "phrasing": args.phrasing,
        "judge": args.judge, "confirm_threshold": args.confirm_threshold,
        "none_threshold": args.none_threshold if not v2 else None,
        "none_margin": args.none_margin if v2 else None,
        "lead_threshold": args.lead_threshold if v2 else None,
        "lead_max_words": args.lead_max_words if v2 else None,
        "lead_max_whole_p": args.lead_max_whole_p if v2 else None,
        "abstain": args.abstain if v2 else None,
        "noul_margin": args.noul_margin,
        "options_truncated": truncated,
        "laya_requests": len(reqs), "judge_ms_per_item": round(judge_ms / max(1, len(rows)), 1)})

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "items.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    for r in rows:
        print("%-8s %-28s %6.2f  commits=%-18s p=%-6s asks=%d(rule %d)"
              % (r["id"], r["outcome"], r["score"], r["commits_to"],
                 "%.2f" % r["commit_p"] if r["commit_p"] is not None else "-",
                 r["asks_question"], r["asks_question_rule"]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
