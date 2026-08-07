# S1-d — sinks decomposition: is the effect the WINDOW or the SINK?

*Registered 2026-08-07, BEFORE the runs, with a QUANTITATIVE
prediction rather than a directional one. Named as a follow-up in the
S1-b document ("the sinks-vs-window decomposition") before S1-c's
result was known.*

## Why this cell exists

Everything measured so far used `swa(w=64, sinks=1)`. The sink keeps
token 0 globally visible to every query (StreamingLLM-style). So every
claim in `S1b-window-monotone` and `S1c-budget-reversal` is really a
claim about **window + sink together**, and neither cell can say which
component carries the effect.

Two independent reasons to resolve it:

1. **Scientific.** If the sink is doing the work, then "locality" was
   never the story even descriptively, and the monotone-in-window
   result needs re-reading.
2. **Structural.** coalfire.cpp has NO sink support (verified: no
   `sink` symbol in `src/` or `include/`; see `tools/coalfire_spec.py`,
   which refuses to translate any spec with sinks > 0). So the
   `sinks=0` lane is the ONLY configuration that is cross-engine
   comparable. Until it exists, no replication of any S1 result on a
   second engine is possible.

## The prediction is QUANTITATIVE, and that makes the null informative

The capacity story (`H-BUDGET`, supported by `S1c-budget-reversal`)
says loss tracks how much attention mass the model can use. So it
predicts the sink's effect from its DENSITY CONTRIBUTION ALONE, and
that contribution is tiny:

| config | visible entries (T=256) | density |
|---|---|---|
| w=64, sinks=1 | 14,560 | 44.28% |
| w=64, sinks=0 | 14,368 | 43.68% |
| difference | 192 | **0.60 pp** |

Calibrating off the S1-b slope (w=128 → w=64 moved loss −0.042 over
−30.3 pp, i.e. ≈ −0.0014 nats per pp), the capacity model predicts:

> **P1 — expected |Δ(sinks=0 − sinks=1)| ≈ 0.0008 nats at 400 steps.**

Our resolution at n=5 paired is roughly 0.021 nats (2.776 × SE ≈
0.0075). **The predicted effect is ~25× below what we can detect.** So
the capacity model predicts a NULL — and that is the point: a null here
is a successful prediction, not an absence of information.

- **P1 (capacity model complete):** |Δ| < 0.021 at 400 steps, i.e.
  indistinguishable. Sink contributes only its density and nothing
  structural.
- **P2 (sink does something beyond density):** |Δ| ≥ 0.021 — an effect
  at least 25× the density prediction. This would mean global
  visibility of token 0 is doing structural work (attention-sink
  literature says it acts as a no-op attractor absorbing surplus
  softmax mass), and the capacity story is incomplete.
- **P3 (direction, committed):** if P2 fires, we predict sinks=0 is
  **BETTER** (Δ < 0), because S1-b established loss falls monotonically
  as attention narrows and removing the sink narrows it further. A
  result where sinks=0 is WORSE would contradict the monotone finding
  at a rung it did not test, and would be the most interesting outcome
  available here.
- **P4 (budget):** the reversal reproduces without sinks — Δ(swa
  sinks=0 − exact) is negative at 400 and positive at 1200, matching
  `S1c-budget-reversal`. If the reversal only happens WITH sinks, then
  S1-c is a claim about sinks, not about sparsity.

## Design

New runs: `swa(w=64, sinks=0)` at **steps ∈ {400, 1200}** × seeds 1-5
= **10 runs** (~5.7h at one worker, nice 19).

No new control runs needed — all four comparison lanes are already on
disk as committed receipts at the same seeds:
`experiments/sparse_s1/atlas_rows.jsonl` (exact@400, swa-sinks1@400)
and `experiments/sparse_s1_budget/atlas_rows.jsonl` (exact@1200,
swa-sinks1@1200). Every comparison is therefore paired by seed.

## Decision rules

- **P1 holds (null):** register it as a supported null — the sink is
  inert beyond its density contribution, the capacity model survives a
  quantitative test, and the `sinks=0` lane is certified as the
  cross-engine vehicle. Proceed to cross-engine replication.
- **P2 + P3 (sinks=0 better, detectably):** the monotone story extends
  and strengthens; amend `S1b-window-monotone` to note the sink rung.
- **P2 + NOT P3 (sinks=0 WORSE):** the sink is doing structural work
  that the density model cannot explain. Register as a positive
  mechanism finding — the most interesting outcome on the table — and
  it becomes a candidate mechanism to study rather than a nuisance
  parameter.
- **P4 fails (reversal needs the sink):** `S1c-budget-reversal` gets a
  scope amendment restricting it to sinked configurations.

## Threats

- n=5 at df=4. Powered for ~0.021 nats and NOT for anything smaller;
  a result between 0.008 and 0.021 is INCONCLUSIVE and must be reported
  that way rather than as a null. The quantitative prediction above is
  what makes this honest — we said the detection floor before looking.
- The density arithmetic assumes the S1-b slope is locally linear in
  density. It is a calibration, not a law; if the measured effect lands
  between 2× and 10× the prediction, that is ambiguous rather than
  decisive, and will be reported as such.
- One sink at T=256 is a single configuration. A null here does not
  license "sinks don't matter" in general — only at this scale, this
  window, this budget.
