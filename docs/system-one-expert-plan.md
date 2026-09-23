# System One expert for Gemma 4 12B

Goal: give Gemma 4 12B a routed "quick decision" expert. Given an input it
answers directly, with no reasoning channel, in free text (no typed-output
constraint as in Laya or Jev). When the input does not determine the answer it
asks one follow-up question instead of inventing one. Everything Gemma 4
already does must keep working.

References: [Laya](https://huggingface.co/convaiinnovations/laya) (421M
ModernBERT encoder, `choice` / `score` / `noul` questions, calibrated
probabilities, act/escalate head, Apache 2.0, `pip install laya` 0.3.4) and
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (closed
API, RLCD training, typed outputs only). We keep their idea of a fast
calibrated decision and drop the typed-output restriction.

## 1. What the expert is

Gemma 4 12B is dense: 48 layers, each with one always-on FFN
(`down(act(gate(x)) * up(x))`, 3840 to 15360). There is no router to retune, so
the expert and its router are added and nothing existing is modified:

```text
ffn_out(x) = base_ffn(x) + g(x) * expert(x)        layers 45, 46, 47
expert(x)  = down_e(act(gate_e(x)) * up_e(x))      3840 -> 2048 -> 3840
g(x)       = sigmoid(w . x + b)                    one scalar per token per layer
```

`x` is the output of the existing `pre_ffn_norm` (`Model.decoder_block/8`,
`gemma4_unified/model.ex:602-611`), so the router needs no norm of its own. The
sum happens before `post_ffn_norm` and `layer_scalar`; placed after them a
zero expert would no longer be bit-identical. The expert is a sibling subgraph
of `gated_ffn`, so the packed `Q4Gemv` / `Q4DualGemv` custom calls keep their
operand shapes. It is enabled by a `spec.system_one_layers` field that defaults
to `[]`.

- All Gemma weights stay frozen and byte-identical. About 71M new parameters
  (23.6M per layer plus three 3841-parameter routers), 142 MB in bf16.
- `down_e` starts at zero, so step 0 is exactly Gemma 4.
- At inference `g < 0.05` is clamped to 0 and the expert is skipped. With the
  router closed the output is bit-identical to today's, which is testable.
- Layers 45 to 47 because the repo already splits there
  (`Gemma4.extract_decoder_pipeline(runtime, 45..47)`,
  `artifacts/gemma4-12b-packed-prefix-0-44`, `...-packed-tail-45-47`). The
  prefix is untouched by the expert, so its output for a teacher-forced
  sequence is a constant that can be computed once and cached. Training then
  only ever holds a 3-layer tail, which is what makes this safe to run beside
  real work.

Risk: three late layers may be able to change style (terse, no reasoning) but
not the ask-or-answer decision. Phase 0 measures that with a linear probe on
the layer-45 input before any training is paid for (section 5).

## 2. Scoring with Laya

Laya never sees gradients. It is the scorecard, run on the CPU (about 1.7 GB).
Each eval item is a `state`, a typed question with gold option, and a flag
saying whether the state determines the answer. The expert answers in free
text, then Laya is asked about the pair:

| Laya question | type | measures |
| --- | --- | --- |
| which option does the reply commit to (options + `asks_followup` + `none`) | choice | decision accuracy against gold |
| does the reply ask the user a question | noul | follow-up recall on underspecified items, needless-ask rate on decidable ones |
| does the reply state a detail absent from the state | noul | fabrication rate |
| Laya's own answer on the bare state, with its probability and act/escalate | choice | a second opinion on which items are really ambiguous |

Headline number, per item: +1 correct answer on a decidable item, +1 follow-up
on an underspecified item, -0.25 needless follow-up, -1 wrong answer, -2
committed answer on an underspecified item. Reported beside it: mean generated
tokens and wall time, and the Brier score of the router's gate against the
ask/answer flag.

First contact with the real package (12-item smoke fixture, zero-shot, both
checkpoints) already moved this design. The act/escalate head returned
p(act) = 1.0 on every request, so it carries no signal. The `fabricates`
question sat at chance. Neither checkpoint ever chose `asks_followup` out of a
5-way choice, and five options overflow the 192-token option budget. Peak RSS
is 3 GB and an item costs about 0.6 s on 8 threads. The scorecard is therefore
a cascade: a punctuation and phrasing rule (optionally backed by Laya's
`noul`) decides whether the reply is a question; only then does Laya's
`choice`, over the item's real options alone, say which option a committed
reply picked, with low confidence mapping to `none`. Fabrication is no longer
judged at all, it is read off the data: a committed answer on an item built to
be undecidable. If the cascade still misses 85% agreement on the 200 labelled
rows, the fallback is fine-tuning the Laya judge on Opus labels.

Measured on the 200 labelled rows (`scripts/system_one/fixtures/calibration200.txt`):
the question rule agrees with the labels on 200 of 200; Laya's option matching
agrees on 95.4% of the 130 committed replies, using the English checkpoint
with `laya-multilingual` routed in for non-English rows, the reply plus the
question as state, and `none` under 0.40 confidence. Replies that commit and
ask at once ("Sending it to Rachel, or did you mean Tom?") are their own
`confirmation` outcome, found by judging only the clause before the question:
scored as the named answer on a decidable item but counted as a needless ask,
and -0.5 on an underspecified one. The headline computed from the cascade
differs from the one computed from the labels on 4 rows of 200. Both stages
pass the 85% gate, so Laya is not fine-tuned. The thresholds were chosen on
these same rows, so the figures are optimistic; the per-round audit below is
the held-out check. Hedged replies (confidence 0.52 to 0.58) are where the
judge still errs.

Laya is a noisy judge (its card reports 0.362 zero-shot and 0.766 fine-tuned
on typed decisions, ECE 0.213 before temperature scaling). So:

- Phase 0 has Opus label 200 (reply, question) pairs and we keep only the judge
  questions where Laya agrees at least 85% of the time, picking the better of
  `laya` and `laya-typed-decisions`, with temperature fitted on that set.
- Ask/answer gold comes from how the data is built (section 3), not from Laya.
- Every round Opus audits 50 random scored items. If audited agreement drops
  under 80% the scorecard is repaired before the next round.

Baselines on the same 600 held-out items: Gemma with the reasoning channel,
Gemma with the channel skipped, and Gemma with a system prompt that asks for
brevity and follow-ups. If the prompt-only baseline is within noise of the
expert, the expert is not worth shipping and the plan stops there.

The prompt-only baseline's system message, fixed once and reused for every
round, is:

> Answer in one short line. Name exactly one of the options you are given. If
> the state you are given does not determine the answer, do not guess: ask one
> short question for the missing detail instead.

The reasoning-on baseline is dropped from round 1 on. The shakedown measured it
as indistinguishable from the closed channel in reply length and wall time at a
64-token budget, so it buys no information for an hour of GPU time. It is not
rerun here, and the round-1 table has no reasoning-on column.

### Judge v2

The round-1 audit (`data/system-one/round1-audit-report.md`) found stage B at
about 83% on held-out replies against the 95.4% it reported on the rows its
thresholds were tuned on, almost always by reading an option name out of a
justification clause while ignoring the option the reply led with, and `none`
unreachable on the 444 two-option items. Judge v2 is the repair. It is the
default; `--judge v1` reproduces the round-1 cascade bit for bit (checked:
0 of 600 rows differ on the prompt baseline).

Three rules run in front of the stage-B call on a non-question reply, and the
first one that fires decides:

- **B0, bare option.** The reply, or its first clause, is exactly one option's
  key or natural name after NFKC, case, underscore/hyphen/space and punctuation
  normalisation. It never fires when two options match. No model: it settles
  475 of the 600 prompt-baseline replies and 97 of the expert's.
- **B1, abstention.** An explicit refusal to answer — "not specified",
  "neither", "no se especifica", "ótilgreint", "결정된 정보가 없습니다" and their
  kin in ten languages — is `none`, not a fabricated commitment. Only the first
  clause is read, and the rule stands down when an option is itself an
  abstention word (`neither`, `both`), where abstaining and choosing are the
  same sentence.
- **B2, leading clause.** For a `<option>, <justification>` reply the stage-B
  call is also run on a leading clause of at most four words and preferred when
  it names an option at p >= 0.60 *and* the whole-reply call is at most 0.70 —
  the gate matters, because on a confident whole-reply call the justification is
  usually carrying real information ("It will succeed, but it locks the table").

The `none` threshold is now a margin over uniform, `1/k + 0.10`, instead of a
flat 0.40 that a two-option item could never fall below. That one change is the
largest single win, and it sits on a plateau: every margin from 0.06 to 0.16
scores within one row of it on the tuning set.

Tuned on the 200 calibration rows plus the 99 distinct items the round-1 auditor
labelled (`data/system-one/judge2-dev99.jsonl`, the 15 disagreements and the
84 agreements, reconstructed from the audit report). Held out: 60 replies the
auditor never saw, 30 from the expert condition and 30 from prompt-only, half
decidable, drawn under a fixed seed and labelled by hand *before* the new judge
was run on them (`data/system-one/judge2-fresh60.jsonl`).

| set | stage B, v1 | stage B, v2 | reply class, v1 | reply class, v2 |
| --- | --- | --- | --- | --- |
| calibration 200 (tuning) | 95.4% | 96.2% | 97.0% | 97.5% |
| audit 99 (tuning) | 66.7% | **86.7%** | 84.8% | 93.9% |
| fresh 60 (held out) | 87.0% | **91.3%** [79.7, 96.6] | 90.0% | **93.3%** [84.1, 97.4] |

Wilson 95%. Stage A is untouched and made the same one error in 60 as before.
The fresh sample is an unbiased draw, so it is easier than the audit's, half of
which was drawn from the hedged band; the like-for-like comparison is v1's 90.0%
against v2's 93.3% on the same 60 rows. Of the 24 items the audit named with an
explicit corrected label, v2 now reads 18 the auditor's way.

Rescored into `data/system-one/scored-v2/`, same inputs and same checkpoints:

| condition | headline | dec_acc | fup_rate | commit_u | needless | tokens | ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | -0.657 | 0.673 | 0.000 | 0.830 | 0.000 | 45.6 | 4949 |
| prompt-only | -0.106 | 0.853 | 0.340 | 0.637 | 0.020 | 3.9 | 1566 |
| expert | +0.078 | 0.527 | 0.630 | 0.360 | 0.263 | 16.1 | 1866 |
| expert+sys | +0.373 | 0.300 | 0.877 | 0.117 | 0.673 | 9.7 | 1668 |

Paired bootstrap against prompt-only, 10,000 draws over the 600 items: expert
**+0.1837 [+0.0592, +0.3033]**, expert+sys +0.4792 [+0.3488, +0.6071], base
-0.5508 [-0.6533, -0.4492]. Split by decidability: base +0.347 / -1.660,
prompt-only +0.722 / -0.933, expert +0.246 / -0.090, expert+sys +0.105 / +0.642.

Dropping the audit's eight suspect items plus the `ho-edu24` and `ho-log16`
pairs (588 items): base -0.648, prompt-only -0.089, expert +0.083, expert+sys
+0.367; expert against prompt-only +0.1726 [+0.0514, +0.2976], expert+sys
+0.4562 [+0.3261, +0.5901], base -0.5587 [-0.6620, -0.4583].

Nothing in the round-1 reading changes. The interval still excludes zero, the
gap narrows from 0.214 to 0.184 because the baseline gained more from the repair
than the expert did (its headline moves to -0.106, which is exactly what the
audit predicted from its own hand sweep), and the expert still buys its headline
on underspecified items while giving back decision accuracy. `none` is reachable
now: 46 / 7 / 7 / 1 two-option items across the four conditions, against 0 under
v1.

What v2 does not repair: a reply that restates a rule and names no figure
("the greater of the actual weight and the volumetric weight") is still read as
a commitment, and stage A still takes a disfluent `Which ...` opening with no
question mark for a question. Both want round-2 work.

## 3. Data

About 6,000 training items and 600 held-out, written by Opus workers, as JSONL.
Domains: transcribed voice commands and journal-style utterances (this repo's
real traffic), support triage, scheduling, form and config checks, short
factual lookups over a given context.

- Decidable: state plus question with one defensible answer. Target is a one
  line answer. Where base Gemma with reasoning enabled reaches the gold answer,
  its own final line is the target, so the expert learns Gemma's System 2
  result in Gemma's voice rather than Opus prose.
- Underspecified: the same item with the deciding field removed or made
  ambiguous. Target is one specific follow-up question naming the missing
  thing. Pairing makes the ask/answer label true by construction.
- Replay, 25% of every batch: ordinary chat, transcription and reasoning
  prompts, where the target is base Gemma's own output and the router target
  is closed.

Held-out items come from templates and domains the training workers never saw,
and are written by a separate worker.

## 4. Training

Elixir, Axon and Polaris on `exla:rocm`, reusing from
`language_id/finetune.ex` the optimizer chain (global-norm clipping, Adam,
`guard_non_finite/1`), the trainer skeleton and the index-shuffle batching
over one cached tensor, and from `gemma4_e4b/audio_encoder.ex` the custom-grad
`stable_sigmoid`. Not copied: `batches/4` hands `Axon.Loop.run` a
re-enumerable stream that already spans every epoch, so with `epochs: n` it
trains about n squared epochs.

1. Cache: run the packed prefix once over every teacher-forced sequence and
   store the layer-45 input as f16 (about 2 MB per 256-token item, 15 GB
   total, deleted at the end). Packed, because that is the deployed prefix.
2. Tail: dequantize the packed tail to f32 with `Q4Gemv.dequantize/3` (the
   artifact layout; `CompressedTensors.linear_kernel/1` is for the raw
   checkpoint layout and would give the wrong matrix) so training sees the
   deployed weights and gradients can flow through plain `Nx.dot`; the q4
   custom call has no gradient. Measured from the safetensors header: 6.4 GB,
   of which 3.75 GB is the vocabulary matrix that is already f32 and 2.6 GB
   the three dequantized layers.
3. Loss: cross-entropy on response tokens only, with the first
   `--lead-tokens` (4) of a System One reply weighted `--lead-weight` (3x),
   because those tokens are the decision and everything after them is
   committed to it; plus KL to the cached base logits where they exist; plus
   a binary cross-entropy on the router, whose rule is *open only while
   producing the response to a System One prompt*. Logits are computed for
   response positions only, since the vocabulary is 262k wide.
4. Adam, lr 1e-4 expert and 1e-3 router, batch 8 x 256 tokens, checkpoint
   every 200 steps. One round is about 2 epochs.
5. Step: not `Axon.Loop`, which threads the whole parameter map through the
   step and hands back a copy of it, and not one program either. Three jitted
   programs run per step, with the frozen tensors as arguments and only the
   expert, the optimizer state and scalars as outputs: the tail forward over
   the cached hidden states, the vocabulary head on the gathered response
   positions alone (its own program, which returns the loss terms and
   `dloss/dhidden`, so the 262144 x 3840 kernel never enters the tail's
   backward pass and exists in exactly the two layouts its two GEMMs want),
   and the tail's gradient against that cotangent. The tail forward therefore
   runs twice per step, which is three layers' worth of work.

A round: train, export `artifacts/system-one/round-N`, generate on the 600
held-out items with the packed pipeline plus expert, run the Laya scorecard,
have Opus analysts read the worst 100 items and write up to 1,000 targeted
items for the next round. At most 3 rounds; stop early when the headline
number moves less than 2 points.

Non-regression gate, every round, must pass before an artifact is kept:

- router forced closed: `journal1.wav` gives token ids
  `[712, 81686, 3124, 8178, 586, 5756, 506, 5597, 2214]` and the seed-42
  single-word gate matches `single-word-packed-native-seed42.json` exactly;
- router learned: on 300 replay prompts the gate opens on under 1% of tokens
  and greedy outputs match base Gemma on at least 99% of prompts.

## 5. Phases

| phase | work | heavy? |
| --- | --- | --- |
| 0 | install Laya in a scratch venv and run it; add a text-only path (there is none today: `Input.build_text/2`, a `:thought_channel` option through `Prompt.build/4` and `DecoderPipeline.generate_prepared/3`, and a prefix-capture command that loads the packed prefix artifact); write the job wrapper; Laya-vs-Opus judge calibration; linear probe for ask/answer on cached layer-45 inputs of 400 items | one short GPU job |
| 1 | data authoring and held-out set; `Gemma4.SystemOne` expert layer, router, artifact format, `mix gemma.system_one cache/train/eval`; scorecard script | CPU only |
| 2 | baselines, hidden-state cache, round 1 | GPU |
| 3 | rounds 2 and 3, regression gate, README section | GPU |

Phase 0 decides between going ahead as written, or, if the probe cannot
separate ask from answer at layer 45 (under 80% balanced accuracy), moving the
experts earlier (for example layers 24, 32, 40). That needs the whole bf16
model in the backward pass, about 30 GB, which does not fit beside daily work
on this machine, so that variant would train on an HF A100 job with the
existing EXLA CUDA image and only evaluate here.

### Phase 0 result

Probe on the layer-45 input at the last prompt token, 200 twin pairs, 5 folds
grouped by pair (`scripts/system_one/probe.py`):

| | balanced accuracy | within-pair |
| --- | --- | --- |
| layer-45 hidden state | 0.807 [0.770, 0.843] | 0.930 [0.890, 0.965] |
| prompt length only | 0.507 | 0.535 |
| bag of words | 0.573 | 0.635 |
| labels shuffled within pair | 0.499 | |
| held-out domain, worst to best | 0.675 to 0.975 | 0.850 to 1.000 |
| English to non-English | 0.683 | 0.900 |
| prompt without the closed thought channel | 0.772 | 0.880 |

The headline clears 80% only just, and its interval straddles it. The
within-pair number is the one acted on: twins differ in nothing but the
deciding fact and the probe ranks them correctly 93% of the time, while the
text-only controls sit near chance. The ask/answer direction transfers across
domains and languages; the threshold along it does not, which is an argument
for a learned nonlinear expert rather than a fixed probe. Linear readability
is necessary, not sufficient. Decision: train the tail here. The training
cache keeps the closed thought channel and pads every row to one 256-token
bucket, because padding to different buckets shifts the hidden state by a
rounding-level amount that correlates with prompt length.

Measured cost of the cache job: 1.06 rows/s, 10.6 GiB VRAM, 7.0 GiB host RAM,
`MemAvailable` never under 23.7 GiB.

Round 1 is smaller than section 3 describes: about 2,800 training rows and
600 held-out, with Opus-written targets. Using Gemma's own reasoning-mode
answers as targets would cost roughly 15 hours of generation on this machine,
so it waits until round 1 shows the approach works. Replay targets (base
Gemma, about 300 prompts, 64 tokens) are cheap and stay in.

### Round 1 result

Trained on `data/system-one/cache-round1-256`: 3,056 rows (2,756 system-one +
300 replay), one 256-token bucket, 0 truncated, batch 8, 2 epochs, 764 steps,
0 skipped, 50.6 minutes. CE fell 2.13 to 1.03. `gate_mean_system_one` rose
0.49 to 0.84; `gate_mean_replay` fell 0.341 to about 0.02, epoch-2 mean 0.037
against epoch-1 0.063, with isolated spikes to 0.11 and 0.13 at steps 220 and
690. The KL term is inert this round: the cache carries no `base_top_k_values`,
so every step logs `kl: 0.0` and replay behaviour is held only by the CE and
the gate-closed penalty. Artifact: `artifacts/system-one/round1`.

Held out: 600 items, 300 decidable and 300 underspecified twins. Every
condition uses the closed thought channel and a 64-token budget. `expert+sys`
is the round-1 expert given the same prompt-only system message.

| condition | headline | dec_acc | fup_rate | commit_u | needless | tokens | ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | -0.733 | 0.723 | 0.000 | 0.957 | 0.000 | 45.6 | 4949 |
| prompt-only | -0.139 | 0.843 | 0.340 | 0.660 | 0.020 | 3.9 | 1566 |
| expert | +0.075 | 0.533 | 0.630 | 0.370 | 0.263 | 16.1 | 1866 |
| expert+sys | +0.363 | 0.293 | 0.877 | 0.120 | 0.673 | 9.7 | 1668 |

Paired bootstrap of the headline against prompt-only, 10,000 draws over the
600 items: expert **+0.2137 [+0.0887, +0.3367]**, expert+sys +0.5025 [+0.3700,
+0.6338], base -0.5942 [-0.6958, -0.4946].

The interval excludes zero, so the expert beats the prompt-only baseline by
more than noise on the headline. The headline is the wrong thing to read
alone. Split by whether the item is decidable:

| condition | decidable | underspecified |
| --- | --- | --- |
| base | +0.447 | -1.913 |
| prompt-only | +0.702 | -0.980 |
| expert | +0.259 | -0.110 |
| expert+sys | +0.092 | +0.635 |

All of the expert's gain is on underspecified items (+0.870 over prompt-only)
and it gives back -0.443 on decidable ones, where accuracy drops 0.843 to
0.533 and the needless-follow-up rate rises 0.020 to 0.263. Round 1 did not
teach the model when to ask; it taught it to ask more. The scorecard's weights
pay +1 for a follow-up and charge only -0.25 for a needless one, so asking
almost always is a winning strategy against this metric — which is what
`expert+sys` does, asking on 67% of decidable items for the best headline in
the table and the worst decision accuracy. Treat the headline as gameable and
read the decidable column beside it.

Other cuts move together and reveal nothing the split above does not: near
+0.093 against far +0.056, English +0.110 against non-English -0.069, and
dropping the two debatable items (`ho-edu24`, `ho-log16`) moves the expert
from +0.075 to +0.077. Non-English is the one cut where the expert does not
beat prompt-only (-0.069 against -0.083, overlapping).

**The non-regression gate fails.** The forced-closed check passes: with the
router clamped to 0 the expert reproduces base Gemma token for token on
`journal1.wav` (`base_match: true`). The learned router does not close on its
own. Regenerating the 300 replay prompts with the live router gives replies
identical to the step-1 base outputs on only **135 of 300 (45.0%)** against a
required 99%, and the router opens on **27.9%** of prompt positions against a
required under 1%; all 300 rows have at least one open position. Forty fresh
general prompts written for this check (`data/system-one/replay-fresh.jsonl`)
behave the same: 19 of 40 identical, 29.1% of positions open. The drift is
mild in kind — mean reply length is unchanged at 61.4 against 61.5 tokens and
the diffs are paraphrases, not derailments — but mild is not the gate. The
mean gate on replay prompts is 0.055, just over the 0.05 inference floor,
which is why so many positions clear it. Round 1 is not shippable as it
stands.

`reference_match: false` in the regress output is stale bookkeeping, not a
round-1 regression: the shakedown runs print the same token ids with the same
flag. The reference ids recorded in this document need refreshing.

Worst 100 scored items are in `data/system-one/round1-worst100.jsonl` (all 100
are `committed_when_underspecified`, score -2.0) and a random 50 in
`data/system-one/round1-audit50.jsonl`, both unaudited.

### Round 2, part 1: how the router is supervised

Round 1's router never learned to close because it was never asked to. The
open target was the whole token mask of a System One row, so the loss pushed
the gate open on the chat template and the question as well as the reply —
and those positions are exactly what an ordinary request is made of. The
closed term was not a cross-entropy at all but a linear penalty on the mean
gate, which has a vanishing gradient nowhere and settles wherever its weight
balances the rest of the loss: in round 1 that was 0.04-0.06, straddling the
0.05 inference floor. 28% open positions on replay is what those two choices
predict.

What the router is trained on now:

- **Open** on a System One row from the position that predicts the first
  response token (`response_start - 1`) to the end of the reply, and
  **closed** on every prompt position before it.
- **Closed** on every real position of a replay row, prompt and response
  alike.
- Both directions are a binary cross-entropy computed from the router's
  pre-sigmoid logit through `softplus`, because read back off the sigmoid one
  of `log g` and `log (1 - g)` is always the log of a rounded-off zero. The
  two are **averaged separately** and weighted 0.5 (open) and 1.0 (closed): a
  batch has an order of magnitude more closed positions than open ones, and a
  single mean over both would be the closed half with a rounding error on it.
- Logged per step beside `gate_mean_system_one` / `gate_mean_replay`:
  `gate_false_open`, the fraction of closed-target positions with a gate at
  or above 0.5, and `gate_false_closed`, the fraction of open-target
  positions below it. Those two are what the inference floor actually reads;
  the means are not.

The expert's own cross-entropy gradient still flows through `g` on the
response positions — the gate multiplies the expert in the main graph and
nothing about that changed — so the router is still free to learn that a
larger gate helps the reply, and the BCE only says where it may do so.

**Inference floor.** A newly trained artifact records `gate_floor: 0.5` in its
manifest (`--gate-floor` to override). The gate is now supervised as a binary
decision, so half is the decision boundary rather than a tuned number. Round
1's artifact keeps the 0.05 it was saved with and still loads; `generate` and
`regress` take `--gate-floor F` to evaluate any artifact at another floor.

**Starting point.** `train --init-from <artifact>` begins from another
artifact's weights with a fresh optimizer state and the new run's gate floor,
so round 2 need not rediscover round 1's expert.

**What the round-1 expert does at a floor of 0.5.** Raising the floor alone
fixes the regression side and costs the behaviour side, which is the trade
round 2 has to train its way out of rather than tune.

| 40 fresh general prompts | identical to base | positions open | rows with any open position | mean gate |
| --- | --- | --- | --- | --- |
| floor 0.05 (round 1) | 19/40 (47.5%) | 29.1% | 40 | 0.051 |
| floor 0.50 | **40/40 (100%)** | 0.16% | 4 | 0.001 |

| first 200 held-out items | headline | decidable | underspecified | dec. acc. | follow-up | committed-u | tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| floor 0.05 (round 1) | +0.010 | +0.130 | -0.110 | 0.46 | 0.63 | 37 | 16.7 |
| floor 0.50 | -0.155 | +0.185 | -0.495 | 0.51 | 0.49 | 49 | 19.7 |

The router is not indiscriminate: on System One prompts it is open on 90% of
positions at a floor of 0.5 with a mean gate of 0.85, and on fresh general
prompts it is open on 0.16%. What it lacks is the *margin* — the round-1
gates on ordinary prompts sit just above 0.05, so the only floor that
separates the two populations also clips a third of the expert's effect on
the population it was trained for, and the reply drifts back towards base:
follow-up recall 0.49 against 0.63, committed-when-underspecified 49 against
37 out of 100. Both sets of 200 were judged with one frozen copy of
`scorecard.py`, which reproduces the round-1 judge's outcome on all 200 of
the floor-0.05 replies.

**KL is left alone.** The top-k KL path is unchanged. Under the floor a gate
is clamped to exactly 0, so a closed position's block output is
bit-identical to base Gemma — not close, identical — and no KL term can
improve on that. What protects replay is therefore the closed half of the
gate loss, not the KL; a KL that fires only where the gate is already open is
measuring the decision the gate loss is supposed to prevent. (Round 1's KL
was inert anyway: the cache carries no `base_top_k_values`, so `kl` was 0.0
every step.)

**The bf16 head is strictly worse here and the usage text was wrong.**
`--head-type` documented bf16 as the default; the default is and was f32. bf16
is not an option worth taking: ROCm has no bf16 GEMM for the 262144 x 3840
shape with the autotuner and Triton off, so XLA converts the kernel back to
f32 for the dot and the step then holds both types — a measured 3.75 GiB of
bf16 buffers on top of the f32 ones they were meant to replace, and a
4.13 GiB transient that killed a run at step 250. The flag stays, with the
text corrected.

### Round 2, part 2: data and what counts as success

Round 1 won the headline by asking more, and its router could not tell a
System One prompt from an ordinary one. Round 2 changes the data to match the
new gate loss:

- **1,188 new System One rows** (594 twin pairs) in twelve domains round 1
  never saw (`train-r2-a.jsonl`, `train-r2-b.jsonl`, minus six pairs the
  authors flagged as debatable). They lean on the two things round 1 got
  wrong: decidable items that *look* ambiguous, so the expert has to commit,
  and underspecified twins with a tempting option. Highest 3-gram Jaccard
  against the held-out set is 0.08.
- **700 new replay prompts** (`replay-prompts-r2.jsonl`), 250 of them hard
  negatives: ordinary requests that mention options, policies or decisions
  without being a System One item. Base Gemma's own replies are the targets,
  and every position of a replay row is a closed-gate target.
- Training starts from the round-1 expert (`--init-from`), with fresh
  optimizer state, the BCE gate loss, lead weight 3.0 on the first four
  response tokens, and the gate floor at 0.5.

Success is read on the split, not the headline:

1. Regression gate: forced-closed output token-identical to base; with the
   live router at floor 0.5, at least 99% of ordinary prompts (replay and an
   unseen set) token-identical and under 1% of their positions open.
2. Decidable accuracy within 5 points of prompt-only (0.853), not 33 below it.
3. Committed-when-underspecified well under prompt-only's 0.637, needless
   follow-ups under 0.10.
4. Paired bootstrap against prompt-only on the headline, with and without the
   suspect held-out labels.

If 1 fails the round is not usable whatever the score says.

### Round 2 result

Trained 2026-09-22 from the round-1 weights on 4,944 rows (3,944 System One,
1,000 replay): 1,236 steps, 75.5 min, 3.59 s/step, no skipped steps, VRAM
flat at 37.5 GiB. Replay false-opens during training stayed near 1% and
false-closes went to zero. Judge v2 on the 600 held-out items:

| condition | headline | dec_acc | fup_rate | commit_u | needless | tokens |
|---|---|---|---|---|---|---|
| base | −0.657 | 0.673 | 0.000 | 0.830 | 0.000 | 45.6 |
| prompt-only | −0.106 | 0.853 | 0.340 | 0.637 | 0.020 | 3.9 |
| expert-r1 | +0.078 | 0.527 | 0.630 | 0.360 | 0.263 | 16.1 |
| expert-r2 | +0.205 | 0.403 | 0.727 | 0.243 | 0.500 | 13.6 |
| expert-sys-r2 | +0.407 | 0.377 | 0.870 | 0.120 | 0.590 | 9.8 |

Split (headline on decidable / underspecified): prompt-only +0.72 / −0.93,
expert-r1 +0.25 / −0.09, expert-r2 +0.17 / +0.24. Paired bootstrap of
expert-r2 against prompt-only: +0.311 [+0.187, +0.439] on all 600, +0.296
[+0.173, +0.424] without the 12 suspect labels. Near and far are the same
(+0.21 / +0.20). English +0.32, the 120 non-English items −0.24.

**Criterion 1 fails.** Forced-closed output is token-identical to base. With
the live router at floor 0.5 the prompt-side gate is essentially closed
(0.03% of prompt positions open on the 300 replay prompts, 0.1% on the 60
unseen prompts, 0% on the 25 hard negatives), but the replies are not
identical: 26/40 on the fresh 40, 229/300 on the replay prompts, 46/60 on
the unseen set (22/25 on the hard negatives). The divergence is late in the
reply: median position 39–44 tokens, only 33 of 99 differing rows diverge
before token 32. That is where supervision stops: 700 of the 1,000 replay
targets are 32 tokens long, so the closed-gate loss never saw positions 32–64
of an ordinary reply, and the gate opens there. The `gate` field written by
`generate` is a prefill probe and cannot see this; token identity is the only
measure of the reply side. The epoch-1 checkpoint (step 600) does not have
the leak: 58/60 identical on the unseen set, 25/25 on the hard negatives.
The training log agrees, with replay false-opens rising from 0.5% at the end
of epoch 1 to 1.5% at the end of epoch 2; the second epoch is what opened the
gate on late reply positions.

**Criteria 2 and 3 fail, and in the wrong direction.** Decidable accuracy
fell 0.53 → 0.40 while needless follow-ups rose 0.26 → 0.50. On the 300
decidable items the expert now asks 147 times, answers correctly 121 times
and wrongly 24. The questions it asks are restatements ("Who gets the free
place, the person named ines or the person named kofi?", "What does the air
conditioner do for the room?"): it has learned the form of a follow-up, not
the discrimination. The likely mechanism is entropy: a question opener is a
small vocabulary and cheap under cross-entropy, an option name costs the
rule application, and the lead weight of 3.0 on the first four response
tokens rewards exactly the cheap path. The headline gain (+0.13 over round
1) is the scorecard's known gameability, not progress on the goal.

What round 2 did deliver: the prompt-side router now separates ordinary
prompts from System One items (round 1 opened on 28% of ordinary prompt
positions; round 2 on 0.03%), committed-when-underspecified fell to 0.24, and
the expert answers in the item's language.

### Round 3: the gate is the decision

Round 2 asked one 70M-parameter expert to do two things at once: decide
whether an item is decidable, and compute the answer when it is. It
learned the cheap half of that (ask) and not the expensive half (apply the
rule). Base Gemma with the round-2 system message already scores 0.853 on
the decidable items and its failure is the other one: it commits on 64% of
the underspecified items. So the two capabilities are already split across
the two networks, and the router is the right place to put the decision.

`train --gate-mode classifier` changes what the gate is trained to mean.
Under the round-2 mode (`response`) the gate target is 1 on every System
One row from the last prompt position through the response. Under
`classifier` only the underspecified rows are open-gate targets; a
decidable row is treated exactly like a replay row — closed-gate target on
every position, zero cross-entropy weight — so the expert sees
cross-entropy only for follow-up questions and the router is trained on a
single question, "is something missing", with 1,972 positives (`-u`) and
2,972 negatives (1,972 `-d` plus 1,000 replay rows). The rows carry a
`decidable` flag; the id suffix is the fallback.

What that buys, if the router can learn the discrimination:

- decidable items are answered by base Gemma with the gate shut, so
  `dec_acc ≈ 0.853 × P(closed | decidable)`;
- underspecified items get the expert, so
  `commit_u ≈ 0.64 × P(closed | underspecified)`;
- the lead-weight question of round 2 goes away: there is no option name
  to compete against in the expert's cross-entropy, only questions;
- served configuration is expert + the round-2 system message, since that
  is what the decidable half now relies on.

The risk is capacity. The router is one linear probe per layer on the
residual stream of layers 45–47, and "decidable" is a property of the rule
and the state together, not of surface form. If the probe cannot separate
the two, the failure shows up as `gate_false_open` on decidable rows
staying high while `gate_false_open_replay` goes to zero — the discrimination
that is learnable from surface form only.

Two other changes:

- The 700 round-2 replay targets were generated at 32 new tokens and 650
  of them hit that budget, so during training positions 32–64 of an
  ordinary reply were never a closed-gate target, and round 2's divergence
  on ordinary prompts sat at a median of ~40 tokens in. They are
  regenerated at 64 (every prompt still fits the 256 bucket) and re-cached;
  the other 4,244 rows are reused by hardlink.
- Training starts from the round-2 weights with a fresh optimizer, 2 epochs
  with checkpoints every 200 steps, and the served checkpoint is chosen by
  the regression gate (token identity on the 40 + 300 + 60 ordinary
  prompts), not by the training loss: round 2's epoch-1 checkpoint was
  clean (58/60) where its final was not (46/60).

Success is the round-2 criteria unchanged: ≥99% token-identical on ordinary
prompts; `dec_acc` within 5 points of 0.853; `commit_u` well under 0.637
with `needless` under 0.10; paired bootstrap against prompt-only.

### Round 3 result

Trained 2026-09-22 from the round-2 weights with `--gate-mode classifier`
on the same 4,944 rows, the 700 round-2 replay targets regenerated at 64
tokens (577 of them use the full budget; teacher-forced length max 234):
1,236 steps, 75.4 min, 3.59 s/step, VRAM flat. Training-set gate metrics at
the end: false-open 1.5% of closed-target positions (replay rows alone
0.3%), false-closed 0.7% of open-target positions, cross-entropy on the
questions 0.3–0.4. Those per-position averages are dominated by prompt
positions and say little about the one position that matters; see below.

Judge v2 on the 600 held-out items. `expert-sys-r3` is the served
configuration at the artifact's floor of 0.5; `-f80` is the same expert at
floor 0.8:

| condition | headline | dec_acc | fup_rate | commit_u | needless | tokens |
|---|---|---|---|---|---|---|
| prompt-only | −0.106 | 0.853 | 0.340 | 0.637 | 0.020 | 3.9 |
| expert-sys-r2 | +0.407 | 0.377 | 0.870 | 0.120 | 0.590 | 9.8 |
| expert-r3 (no system message) | +0.240 | 0.230 | 0.817 | 0.147 | 0.677 | 17.5 |
| expert-sys-r3 | **+0.434** | 0.540 | 0.823 | 0.170 | 0.407 | 7.7 |
| expert-sys-r3-f80 | +0.386 | 0.663 | 0.743 | 0.247 | 0.263 | 6.5 |

Split (decidable / underspecified): expert-sys-r2 +0.19 / +0.63,
expert-sys-r3 +0.385 / +0.48, expert-sys-r3-f80 +0.52 / +0.25. Paired
bootstrap against prompt-only: expert-sys-r3 +0.540 [+0.425, +0.655] (+0.527
without the suspect labels), f80 +0.492 [+0.390, +0.598]. Near and far are
the same (+0.46 / +0.41). Non-English is no longer negative: es +0.79, fr
+0.66, de +0.42, pl +0.42, is +0.19, ja +0.08.

This is the first round where the decidable side improved (0.377 → 0.540 →
0.663) without the underspecified side collapsing, so the classifier mode
does what it was meant to: with the gate shut base Gemma answers, and the
expert no longer competes with the option name. The question is now purely
whether the router can tell the two classes apart.

**The router's decision.** `generate` now also writes `gate_last`, the gate
at the final prompt position (the one whose hidden state predicts the first
reply token), averaged over the three layers. A one-token run at floor 0
over the held-out set gives, for decidable items, quantiles 10/25/50/75/90
of 0.16 / 0.32 / 0.59 / 0.74 / 0.87, and for underspecified items 0.52 /
0.67 / 0.79 / 0.90 / 0.95. The classes overlap heavily: on unseen items the
linear router opens on 60% of decidable and 91% of underspecified items at
floor 0.5, 17% and 48% at floor 0.8. It opens more on decidable items the
authors flagged `looks_ambiguous` (median 0.63 against 0.39), i.e. it reads
surface cues, not whether the state settles the rule. Emulating the floor
sweep offline (open item → the expert's judged outcome, closed item →
prompt-only's) predicts the trade-off well on the underspecified side and is
optimistic by about ten points on the decidable side, because the gate also
opens mid-reply on decidable items whose decision position was shut. No
floor satisfies criteria 2 and 3 together: `dec_acc ≥ 0.80` needs a floor of
about 0.9, where `commit_u` is back at 0.51.

**Criterion 1.** Forced-closed output is token-identical to base, and the
live router at floor 0.5 now passes the single reference prompt round 2
failed. At floor 0.5 the ordinary sets are 38/40, 266/300 and 52/60
identical (round 2: 26/40, 229/300, 46/60), every divergence a decode-step
opening on decision-flavoured text ("policy", "Option A", "decide on") 13–61
tokens in; 18/25 on the hard negatives. At floor 0.8 the 60 unseen prompts
are 60/60 identical including all 25 hard negatives, the fresh 40 are 40/40
(no open prompt position on any row), and the replay 300 are 290/300
(96.7%; round 2: 229/300). All ten replay divergences are decode-step
openings the prefill probe cannot see (`gate_last` 0.0 on every one): six
start 18–51 tokens in, four in the first two reply tokens — a summary that
begins "Based on your friend's notes" instead of "Based on the notes your
friend took", an Icelandic explanation that takes a different first
sentence — same meaning, different wording, no question asked. Criterion 1
(≥ 99% on the replay set) is therefore still not met at floor 0.8: 96.7% is
the best this router does, and the residual 3% is the same decode-step
leak, just rarer.

**Verdict after three rounds.** The expert as designed — a 70M FFN behind a
linear router on layers 45–47 — is a working "ask instead of guess"
mechanism whose limit is the router: one linear probe of the residual stream
cannot separate "the state determines the answer" from "it does not" on
unseen items (median gate 0.59 against 0.79). Served at floor 0.8 it gives
Gemma a real follow-up behaviour on underspecified requests (commit_u 0.637
→ 0.247) at the cost of asking needlessly on a quarter of decidable ones
(dec_acc 0.853 → 0.663), and leaves ordinary prompts untouched on 97–100%
of rows (the rest reword, they do not start asking). Round 3
moved the headline 2.7 points and the split by 20, so the stop rule does not
trigger on its own, but the remaining gain is in the router, not in more
data or epochs: the next step, if there is one, is a router with a hidden
layer (or the expert's own down-projection as a feature) trained in
classifier mode, which is a change to `SystemOne.gate_nodes`, not to the
data.

### Route 1: a router in front of Gemma

After round 3 the direction changed: instead of an expert inside the tail,
put a Laya-like router in front of unmodified Gemma that decides per request
whether to **answer now**, **reason first** or **ask back**. No weights
change and no Laya-shaped JSON: the router only picks which of Gemma's own
modes runs. Everything below was measured on 2026-09-23 with the stock
12B (q4), greedy decoding; routing was simulated offline from recorded runs.

**Native thinking is not usable here.** `generate --think` renders the 12B
template's `<|think|>` system turn (no newline after it, unlike E4B). The
packed 12B then spells `<thought` as text instead of emitting `<|channel>`;
with the channel pre-opened it reaches the answer and loops "let me
double-check" without ever closing the channel, even at 1,024 tokens. It
needs sampling, and the decoder is greedy-only. Normal mode already reasons
visibly (≈ 250 tokens), so "reason" means an ordinary reply ending in an
`Answer:` line.

**Answer now or reason: answer confidence.** The direct prompt asks for one
line, `Answer: <x>`. `generate --scores` records each chosen token's
log-probability (full-vocabulary log-softmax, suppressed tokens excluded;
picks are token-identical to a run without it), and the confidence is the
minimum probability over the answer tokens. Below a cutoff the request is
re-run in normal mode. On 200 fresh GSM8K + ARC-Challenge items (100 each,
disjoint from the 60 used to pick the method) direct answers are right on
139 (GSM 45, ARC 94), confidence separates right from wrong at AUROC 0.960,
and reasoning costs 29.9 s per item:

| cutoff | reasoned | correct | s/q |
|---|---|---|---|
| 0.7 | 44 | 176/200 | 8.1 |
| 0.9 | 70 | 189/200 | 12.1 |
| 0.99 | 93 | 194/200 | 15.1 |

At 0.9 the router is 2.5× faster than reasoning on everything; the six
wrong answers it keeps all had confidence 0.93–0.97. (On the first 60,
always-reason was 57/60 at 26 s/q.)

**Ask back: a probe, not confidence.** Answer confidence cannot see that
something is missing: on 100 held-out twin pairs (the decidable item and
its blurred twin, in the direct form) it separates them at AUROC 0.585.
Adding an `ask` option to the prompt gets 0.740, but at the cost of the
decidable answers (67/100 right). A logistic probe (standardised features,
C = 0.003) on the layer-44 hidden state at the last prompt token — computed
anyway during prefill, so free at serve time — gets 0.83–0.86, **but only on
the prompt wording it was trained on**: trained on the System One template
from the round-3 cache it scores 0.825 on the held-out twins in that
template and 0.637 on the same twins in the router's direct form. Retrained
on 3,089 router-form prompts (2,400 training twins, 400 replay prompts, 300
unseen GSM/ARC), it scores 0.857 on the held-out twins (near 0.886, far
0.847) and never fires on the 200 GSM/ARC items (highest score 0.29).

**The whole router** (ask if probe > 0.8, else answer now if confidence ≥
the cutoff, else reason). Reasoning helps decidable twins it is sent (43/45
right against 36/45 direct) but never asks: it committed to an option on
16 of 17 underspecified twins, so asking is the probe's job alone.

| on 100 + 100 held-out twins | decidable right | needless asks | underspecified asked | s/q |
|---|---|---|---|---|
| direct answer only | 87 | 0 | 0 | 1.5 |
| `ask` option in the prompt | 67 | 9 | 54 | 1.5 |
| router, cutoff 0.9 | 85 | 8 | 64 | 4.5 |
| router, cutoff 0.99 | 88 | 8 | 64 | 7.9 |

On GSM/ARC the router gives exactly the confidence-router numbers above,
since the probe never asks there. Compared with the round-3 expert at floor
0.8 (needless asks on 26% of decidable items, a quarter of the unclear
ones still guessed, and 3% of ordinary replies reworded), the router asks
less needlessly, catches more of the unclear requests, and cannot change an
ordinary reply at all: it only chooses which unmodified mode runs.

## 6. Opus workers

Spawned with the Agent tool, `model: opus`. Two lanes:

- Light lane, up to 4 in parallel: data authors, held-out author, judge
  labeller, failure analysts, code for the expert layer and scorecard. No
  model loads, no `mix` jobs longer than a test run.
- Heavy lane, exactly one worker at a time, the only one allowed to load
  Gemma: cache, train, eval, regression gate.

## 7. Keeping the machine usable

Measured at planning time: 62 GB RAM with 24 GB available and 45 GB already in
swap, 64 GB VRAM carve-out with 22 GB in use, 110 GB free disk.

Every heavy job goes through one wrapper, `scripts/system_one_job.sh`:

- `flock` on one lockfile, so two model jobs cannot overlap (two 12B GPU
  processes have segfaulted this box before);
- preflight refuses to start under 14 GB `MemAvailable`, under 24 GB free
  VRAM, or under 60 GB free disk;
- `systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=2G -p
  CPUQuota=800% --collect --setenv=XLA_FLAGS`, under `nice -n 19 ionice -c3`,
  so an OOM kills the job and nothing else, and it takes 8 of 32 threads;
- a watchdog stops the job if `MemAvailable` stays under 6 GB for 30 s;
  training resumes from the last 200-step checkpoint;
- measured peak, 60 steps at batch 8 x 256 on the seed cache: 35.7 GB VRAM
  over the 22.4 GB idle baseline, 6.4 GB in the systemd scope, `MemAvailable`
  never under 23.1 GB. A frozen matrix is not one buffer: XLA keeps a layout
  per GEMM and the tail enters two programs, so the three dequantized layers
  cost about 21.5 GB of the total and the head, which now runs as its own
  program, exactly 8.1 GB. That leaves about 2 GB under the BFC limit the
  rocm client is configured for (`preallocate: false`, `memory_fraction:
  0.55`, so 35.2 GiB), which is why batch 8 x 256 is the ceiling and a bigger
  batch needs the memory fraction raised first;
- about 3.56 s per step in steady state, plus about two minutes to load the
  tail and compile the three programs: roughly 45 minutes for 2 epochs over
  3,056 rows;
- toolchain as usual: `mise x erlang@29.0.3 elixir@1.20.2-otp-29 -- mix ...`
  with `XLA_FLAGS` exported before launch.
