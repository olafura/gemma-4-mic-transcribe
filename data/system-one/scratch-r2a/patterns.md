# Round-2A failure patterns (read from round1-worst100 + expert-r1 items)

All 100 worst items are `committed_when_underspecified`. 79 of 300 decidable
items drew a needless follow-up (26.3%). So the batch has to push in both
directions.

## -u side: why the model commits on an undecidable state

- **U1 rule_present_value_missing** — the threshold/rule is spelled out in the
  state and the *measured* value is the thing that is gone. The model recites
  the rule and treats it as the answer. (ho-hlp06 "three years" vs a 36-month
  line, ho-fin14 "400+" vs a 400 km line, ho-cli12 "about 15" vs 15 km.)
- **U2 vague_quantifier** — "most", "a few", "about 100", "around 16", "well
  over". The model rounds the hedge into a number. (ho-edu07, ho-cli17,
  ho-cfg20, ho-edu22.)
- **U3 ambiguous_referent** — two rows/entities match the query equally.
  The model picks the first one. (ho-lkp01 X1 vs X1 Pro, ho-lkp04 S3 vs S5,
  ho-lkp19 two 08:20 flights, ho-cli08 two free couch rooms.)
- **U4 missing_unit_or_format** — number present, unit/scale/format absent:
  "30000 no unit", "1,234", "03.04.2026", a bare cron in "server local",
  an area "45" with no km2. (ho-cfg19, ho-cfg10, ho-lkp10, ho-mix08.)
- **U5 fuzzy_date_or_time** — the date that sets a window is a season or a
  half-month: "mid February", "late September", "Frühjahr 2026", "14時すぎ".
  (ho-edu16, ho-edu24, ho-cli10, ho-cli05.)
- **U6 category_not_in_rule** — the entity's category is not one the rule
  table covers: "hybrid" worker, "prod-like" env, "editor" role where only
  viewer/admin are listed, "coastal" where the rule says island/mainland.
  (ho-hlp15, ho-cfg07, ho-hlp01, ho-log23.)
- **U7 cap_parameterised_elsewhere** — the rule exists but its number lives
  somewhere the state does not show: "a fair amount a day", "minimum set by
  team", "divisor varies by contract", "ab einem Schwellwert".
  (ho-fin18, ho-cfg05, ho-log02, ho-cfg04.)
- **U8 status_not_settled** — the state gives a step in a process, not its
  outcome: "payment expected", "referral asked", "pencilled", "signed
  partial", "1 of 3 retries failed". (ho-edu12, ho-cli20, ho-cfg16, ho-log22.)

## -d side: decidable rows that must not draw a follow-up

- **D1 one_step_arithmetic** — a sum, a difference, a proportion, a count of
  months, compared against a stated line.
- **D2 two_field_join** — a value in one field is the key into a table in
  another field.
- **D3 distractor_fields** — extra fields that look load-bearing, including a
  `null` or an "unknown" in a field the question does not touch.
- **D4 explicit_precedence** — looks like a conflict, but the state names the
  tie-breaker ("the highest applicable rate", "first match wins").
- **D5 exception_not_triggered** — a conditional whose condition is plainly
  false, so the ordinary branch applies.
