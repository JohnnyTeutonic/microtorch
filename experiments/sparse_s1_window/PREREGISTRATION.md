# S1-b — window sweep: is there an optimal locality scale?

*Registered 2026-08-05, BEFORE the runs. Direction pre-specified this
time — the S1 lesson: the n=3 cut had to be downgraded partly because
the manifest expected match-or-cost and the observed direction was a
surprise, which forbids a one-tailed reading. Predictions below are
committed; RESULTS get appended, never edited in.*

## The question

`S1-swa-beats-exact` (registry, supported): sliding window w=64 + 1
sink beats full attention at T=256, paired t=-10.35 over 10 seeds. The
registered claim deliberately stops there — the MECHANISM is not
established. Two explanations survive, and they make different
predictions about the shape of loss-vs-window:

- **H-PRIOR (locality is a favorable inductive bias):** attending to
  recent context is the right prior for this data, and attending to
  everything wastes capacity on distant tokens that carry little
  signal. Predicts an INTERIOR OPTIMUM: too small a window starves the
  model of context, too large re-admits the noise.
- **H-BUDGET (the win is a regularization/optimization artifact):**
  restricting attention reduces effective capacity, which helps at 400
  steps on a small corpus for the same reason any regularizer helps a
  short run. Predicts MONOTONE improvement as w shrinks (down to where
  the model can no longer represent the task at all), and predicts the
  advantage SHRINKS OR REVERSES at longer budgets.

## Design

One factor, `arch.custom.window` ∈ {16, 32, 128, 256}, 5 seeds each
(20 runs). Everything else identical to the S1 cell — gpt2 2-block,
d=128, T=256, TinyStories, 400 steps, AdamW@1e-3, batch 4, sinks=1.
The w=64 (n=10) and exact (n=10) rows from S1 are the anchors; seeds
1-5 overlap them, so those comparisons are PAIRED.

Density at T=256 (fraction of full causal entries computed):
w=16 → 12.3% · w=32 → 22.4% · w=64 → 44.3% · w=128 → 74.6% · w=256 →
100%.

## Predictions (committed before running)

- **P1 — HARNESS CHECK, must pass or nothing else counts.** w=256 with
  sinks=1 reduces to the full causal range by construction (the sink
  interval is empty once win_lo = b0), so swa@256 must reproduce the
  exact lane BITWISE at matched seeds: |best_val(swa,256,s) −
  best_val(exact,s)| < 1e-6 for all 5 shared seeds. This is the
  equivalence pin verified in-situ, inside real training runs rather
  than a unit test. FAILURE VOIDS THE CELL.
- **P2 — H-PRIOR predicts an interior optimum.** best_val(16) >
  best_val(64) (too little context hurts) AND best_val(64) <
  best_val(256) (already established). Direction pre-specified:
  w=16 WORSE than w=64.
- **P3 — H-BUDGET predicts monotone.** best_val(16) ≤ best_val(32) ≤
  best_val(64) — smaller is better all the way down.
- **P4 — the near-full rung.** best_val(128) sits between best_val(64)
  and best_val(256); i.e. the advantage decays smoothly as the window
  approaches T rather than switching on at one particular scale.

P2 and P3 are mutually exclusive at the w=16 rung, which is the point:
this cell distinguishes the two live explanations rather than
decorating the existing result.

## Decision rules

- P1 fails → cell void, harness bug, nothing is interpreted.
- P2 holds (interior optimum) → register a mechanism-constraining
  finding: the win is locality-as-prior, and the optimal scale is a
  measurable property of the data, not an artifact of shrinking
  capacity. Next question becomes whether the optimum moves with T.
- P3 holds (monotone down to 16) → H-PRIOR is WRONG at this scale, the
  S1 win is likely a budget/capacity artifact, and `S1-swa-beats-exact`
  gets a scope amendment saying so. A negative that reinterprets our
  own positive is exactly the kind we publish.
- Mixed/flat (all rungs within seed noise) → report as uninformative;
  the window is not a live knob at this scale and the S1 win comes from
  something else entirely (sinks? the two-range structure?).

## Threats

- 400 steps is short; both hypotheses' predictions could change at
  longer budgets. The longer-budget run is already queued as a separate
  cell and is the discriminating follow-up either way.
- Seeds 1-5 overlap the S1 anchors, so anchor comparisons are paired
  and powerful, but the w=16/32/128 rungs are n=5 (df=4, two-tailed
  crit 2.776) — adequately powered for effects of the S1 size (~0.048
  nats at sd ~0.015) and NOT for effects three times smaller. If a rung
  lands inside noise, that is reported as inconclusive, not as a null.
