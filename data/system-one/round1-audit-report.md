# Round 1 audit of the Laya scorecard

Held-out check of the judge itself, as required by
`docs/system-one-expert-plan.md` section 2 ("Every round Opus audits 50 random
scored items. If audited agreement drops under 80% the scorecard is repaired
before the next round."). The scorecard's thresholds were tuned on the same 200
rows they were measured on; nothing below was.

Auditor: read the state, question, options, gold and the reply, decide for each
reply whether it commits to one option (and which), asks a follow-up, does both
(confirmation) or neither (`none`), then compare with
`judged_commits_to` / `judged_asks_question`. Replies were capped at 64 tokens;
truncation was taken into account and never changed a call.

## 1. Agreement on the 50 random items

`data/system-one/round1-audit50.jsonl` — 50 items, all from the **expert-r1**
condition (verified against `heldout-expert-r1.jsonl`: 50/50 replies match).

| measure | result |
| --- | --- |
| agreement, reply class (commit-to-X / ask / both / none) | **44/50 = 88.0%**, Wilson 95% **[76.2%, 94.4%]** |
| agreement, score-affecting only | 45/50 = 90.0% [78.6%, 95.7%] |
| stage A (ask rule) alone | 49/50 |
| stage B (which option a committed reply named) alone | 24/29 ≈ 83% |

Above the plan's 80% repair gate, but the lower end of the interval touches it,
and stage B is well under the 95.4% it reported on the calibration rows. The
gap is the expected in-sample optimism: stage B was tuned on those rows.

## 2. Agreement on the 50 hard (hedged) items

Pool: `judge_commit_p` in [0.35, 0.65] **or** `outcome` starting with `none`
(expert-r1: 177/600 in pool; prompt: 87/600). 25 drawn per condition with
`random.Random(20260922).sample`.

| condition | agreement | Wilson 95% |
| --- | --- | --- |
| expert-r1, hedged 25 | 18/25 = 72.0% | [52.4%, 85.7%] |
| prompt, hedged 25 | 23/25 = 92.0% | [75.0%, 97.8%] |
| both, hedged 50 | **41/50 = 82.0%** | [69.2%, 90.2%] |
| both, score-affecting only | 44/50 = 88.0% | [76.2%, 94.4%] |

All 100 audited replies together: 85/100 = 85.0% [76.7%, 90.7%].
The ask rule (stage A) made one error in 100 replies: 99/100 [94.6%, 99.8%].
**Every disagreement below is a stage-B (option matching) error except
`ho-sho21-d`.**

The hedged pool is much larger for the expert (29.5% of items) than for the
prompt baseline (14.5%), and the judge is markedly worse inside it on the
expert, because the expert writes prose and the baseline writes a bare option
key.

## 3. Every disagreement

`Δ` is my score minus the judge's score for that item (the headline is a mean
of per-item scores, so Δ is the per-item headline error).

### Random 50 (expert-r1)

| id | reply | judge says | I say | why | Δ |
| --- | --- | --- | --- | --- | --- |
| ho-edu17-d | `Kyoka, 17 + 2 = 19, which is over the limit of 18.` | commits `fuka` → correct +1 | commits `kyoka` → wrong | The reply leads with the option key `kyoka` (認める, allow) and then gives reasoning that contradicts it. The model's format everywhere else is `<option>, <justification>`, so the named option is the commitment. Self-contradictory reply; the judge read the justification, not the answer. | −2.0 |
| ho-lkp13-d | `Dwie opcje przesyłki są tu rozważane, ale przesyłka do Gdańska mieści się w przesyłce polskiej.` | commits `zl_0` (free) → wrong −1 | commits `zl_15` → correct | "Shipping to Gdańsk falls under Polish (domestic) shipping" is the 15 zł band. Free shipping is what it does *not* say (order is 180 zł, free from 200 zł). | +2.0 |
| ho-log02-u | `The carrier charges the greater of the actual weight and the volumetric weight.` | commits `kg_18_4` → fabrication −2 | none | It restates the rule and names no figure. `kg_18_4` is the *combined* weight, which "the greater of" explicitly rules out. | +2.0 |
| ho-mix24-d | `The sixth is Friday 13, the seventh is Friday 20.` | commits `feb_06` → wrong −1 | none (incoherent) | No option is selected; the reply lists dates. Scores the same either way (`none_on_decidable` = −1). A generous reading ("Friday 13") would make it correct, i.e. up to +2 more. | 0.0 |
| ho-sho07-d | `Washer runs at delay, it is now peak hours.` | commits `now_run` → wrong −1 | commits `delay` → correct | The reply literally says "runs at delay". The justification is false (21:30 is not peak) but the commitment is the gold option. | +2.0 |
| ho-sho21-d | `Which house arm itself, the house arms itself in arm away mode.` | asks → needless_followup −0.25 | commits `arm_away` → correct | Garbled disfluent opening starting with "Which"; no question mark and the clause that follows states the answer. The one stage-A false positive in 100 replies. | +1.25 |

Net over the random 50: **+5.25 / 50 = +0.105 per item**, i.e. the judge
*under*-scores the expert.

### Hedged 25, expert-r1

| id | reply | judge says | I say | why | Δ |
| --- | --- | --- | --- | --- | --- |
| ho-edu12-u | `It can go, the pupil has signed the form and the payment is expected.` | commits `hold` | commits `goes` | Named option is wrong; both are fabrications on an underspecified item, so the score is unchanged. | 0.0 |
| ho-fin08-d | `The invoice is for 1,500 USD.` | commits `eur_1388_89` → correct +1 | none | The reply never states a euro amount. The judge inferred the arithmetic the reply did not do. Textbook hedged error (p = 0.5545). | −2.0 |
| ho-fin18-u | `The second was reimbursed,` ×13 (64-token loop) | commits `full` → −2 | none | Degenerate repetition, no amount stated. (If "the second" means the second option, the judge's reading is defensible; a reader gets no answer either way.) | +2.0 |
| ho-fin20-d | `The second of the three months is the contract start date, so the contract runs from 1 April to 31 December.` | commits `eur_36000` → wrong −1 | none | No amount named. Same score either way. | 0.0 |
| ho-hlp20-d | `VPN, it works at the office and over VPN.` | commits `office` → wrong −1 | commits `vpn` → correct | Reply opens with "VPN"; the judge latched on to "office" inside the justification (p 0.449 vs 0.432). | +2.0 |
| ho-sho19-d | `Wait, it charges at seven.` | commits `charge_now` → wrong −1 | commits `wait` → correct | Same family: leading option word ignored, justification clause matched instead. | +2.0 |
| ho-sup11-u | `The third fee is calculated as follows, the fee is added to the disputed amount, and the refund is subtracted.` | commits `eur_165` → −2 | none | Restates the formula, gives no number. | +2.0 |

Net over these 25 hedged expert items: **+6.0**.

### Hedged 25, prompt

| id | reply | judge says | I say | why | Δ |
| --- | --- | --- | --- | --- | --- |
| ho-cfg07-u | `ok` | commits `violates` | commits `ok` | The reply *is* the option key. Both are commitments on an underspecified item, so the score is unchanged. | 0.0 |
| ho-mix16-d | `n_16` | commits `n_12` → wrong −1 | commits `n_16` → correct | The reply is verbatim the gold option key and the judge chose another option at p = 0.58. | +2.0 |

Net over these 25 hedged prompt items: **+2.0**.

## 4. Do the errors favour a condition or a direction?

**Direction: overwhelmingly deflating.** Of the 15 disagreements, 9 cost the
model points it had earned, 2 gave it points it had not, and 4 were
score-neutral (the judge named the wrong option but the outcome bucket was the
same). The judge is biased against the models, not for them, so the round-1
numbers are conservative.

**Condition:** the errors do not manufacture the expert's advantage — they
suppress it.

Two whole-population sweeps back this up beyond the 100 sampled replies.

*Prompt baseline, near-exhaustive.* 471 of its 600 replies are verbatim an
option key; a string match found 9 the judge read as a different option. The
other 21 non-question replies were read by hand. Corrections:

- judge too harsh (+2 each): `ho-sch03-d`, `ho-cfg21-d`, `ho-mix16-d`,
  `ho-edu15-d` (`école_b` judged `ecole_a`), plus 7 explicit abstentions forced
  into a fabrication: `ho-sup13-u` ("Neither reply stands…"), `ho-sch20-u`
  ("결정된 정보가 없습니다"), `ho-voi09-u` ("No se especifica…"), `ho-mix04-u`
  ("Neither invoice is smaller…"), `ho-cli19-u` ("Both p1 and p2 are
  confirmed."), `ho-hlp05-u` ("ótilgreint"), `ho-lkp16-u` ("Wisselende tijden")
- judge too kind (−2): `ho-sup08-d` (`wystaw_nowa` judged as the gold
  `wyslij_ponownie`)

Net **+20 / 600 = +0.033**; prompt headline −0.1392 → ≈ **−0.106**.

*Expert-r1, partial.* A sweep for "first six words name exactly one option, the
judge chose another" (plus the hand-read items) confirms 8 deflating errors
(`ho-cfg07-d`, `ho-cli12-d`, `ho-log12-d`, `ho-sho07-d`, `ho-sho19-d`,
`ho-edu12-d`, `ho-hlp20-d`, `ho-lkp13-d`), 3 rule-restatements scored as
fabrications, and 4 inflating ones (`ho-fin02-d` "Post it" judged `reject`,
`ho-sup01-d` "Credit it" judged `refund`, `ho-edu17-d`, `ho-fin08-d`). That is
a **lower bound** of +14/600 = +0.023, because prose replies cannot be swept
exhaustively. The unbiased random-50 estimate is +0.105 per item.

| headline | as judged | corrected (lower bound) | corrected (random-50 estimate) |
| --- | --- | --- | --- |
| expert-r1 | 0.075 | ≈ 0.098 | ≈ 0.18 |
| prompt | −0.139 | ≈ −0.106 | ≈ −0.106 |
| gap | 0.214 | 0.204 | ≈ 0.29 |

The gap survives every correction. Two further reasons to trust it:

1. The gap is driven by `followup_recall` (0.63 vs 0.34) and `fabrication_rate`
   (0.37 vs 0.66), both decided by the **deterministic ask rule**, which made
   1 error in 100 audited replies — and that one error was against the expert.
2. The judge's weakness is prose, and only the expert writes prose. Any
   residual bias therefore runs against the expert's headline, not for it.

The number that is *not* safe is the expert's `decision_accuracy` (0.533 =
160/300). The sweep found 8 correct answers marked wrong and 3 wrong answers
marked correct, so the figure is at least 165/300 ≈ 0.55, and higher once the
un-sweepable prose errors are counted.

## 5. Scorecard defects to repair before round 2

1. **`none` is unreachable on two-option items** — 444 of 600 items have two
   options, where the top probability is ≥ 0.50 by construction and
   `--none-threshold` is 0.40. Measured minimum `commit_p` on two-option items:
   0.5031 (expert), 0.5019 (prompt), 0.5001 (base); every `none` in the whole
   run lands on a three-option item. A reply that answers nothing is therefore
   *forced* into a commitment, which is −2 on an underspecified item. Make the
   threshold a function of the option count (e.g. `0.5 + m` for k = 2) or add an
   explicit abstain option to the choice call.
2. **Match a bare option key deterministically before calling Laya.** 471/600
   prompt replies are exactly an option key and the judge got 9 wrong. Free
   accuracy, and it removes the baseline's largest error source.
3. **Prefer the leading clause for `<option>, <justification>` replies.** Six
   of the confirmed expert errors are a reply that opens with the right option
   and is judged by a word in its justification. Stage C already computes a
   committing prefix; run the same prefix call on non-question replies and
   prefer it when it is confident.
4. **Detect abstentions** ("not specified", "neither", "unknown", "ótilgreint",
   "결정된 정보가 없습니다", "no se especifica") and map them to `none`. Seven
   prompt items were charged −2 for correctly refusing to guess.

## 6. Suspect or debatable item labels

Flagged from the 100 items read. None of these were `debatable: true`.

| id | label | why it is suspect |
| --- | --- | --- |
| `ho-fin05-u` | `decidable: false`, missing "翌月初旬が何日までを指すのか" | 初旬 conventionally means the first ten days of the month, and the submission is 8月4日 — inside it. The item is arguably decidable with gold `shounin`, which would flip its `-u` twin's whole purpose. Strongest candidate for a wrong label. |
| `ho-cli02-u` | `decidable: false` | Of the two options only `t_1100` can ever fit (a new patient needs 30–45 min; the 10:00 slot is 15 min). A model answering `t_1100` takes −2 for naming the only survivable option. Needs a "neither" option or a third slot. |
| `ho-edu10-u` | `decidable: false`, missing "whether the evidence is on file" | The state says `evidence: "report requested"`, which reads as *not yet on file* — i.e. `refuse` / hold is determined. |
| `ho-cli13-u` | `decidable: false`, missing "how many identifiers 'several' means" | "Several" is normally ≥ 3 and the patient has 2 of 3, so `turn_away` is arguably determined. |
| `ho-voi14-d` | `decidable: true`, gold `upstairs` | Only reachable via "downstairs is already at 20, so 'set it to twenty' must mean upstairs". Pronoun items of the same shape are labelled underspecified elsewhere (`ho-voi16-u`, `ho-sho04-u`). |
| `ho-fin22-u` | options `return_70` / `nothing` | The missing figure is "almost 500" against a 500 advance, so "return seventy euros" contradicts the state whichever way the missing detail resolves; only `nothing` is even plausible. The option set, not the flag, is the problem. |
| `ho-mix18-d` | gold `l_6_9` | 42.5 L / 615 km = 6.91, so a reply of "7.0" is right to two significant figures but is not an option; the prompt baseline answered `l_7_0` and was credited with gold by the judge. Option granularity is finer than the task. |
| `ho-hlp05-d` | gold `sjalfvirkt` | Requires a two-hop lookup (access `lesandi` → group `jira_lesandi` → "sjálfvirkt"). The baseline replied `lesandi` — the intermediate, not an option — and was scored correct. |

## Reproduction

```
data/system-one/round1-audit50.jsonl                 # the 50 random items (expert-r1)
data/system-one/scored/{expert-r1,prompt}/items.jsonl  # judge output
data/system-one/heldout{,-expert-r1,-prompt}.jsonl     # items and replies
```

Hedged sample: filter `0.35 <= commit_p <= 0.65 or outcome.startswith("none")`,
then `random.Random(20260922).sample(pool, 25)` per condition.
