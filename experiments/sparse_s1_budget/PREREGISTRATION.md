# S1-c — budget cell: does the sparse advantage survive longer training?

*Registered 2026-08-06, BEFORE the runs, direction committed. This
cell was named as the discriminating follow-up inside the S1-b
pre-registration BEFORE S1-b's result was known — it is not a
post-hoc rescue of a story that just broke.*

## Why this cell exists

S1-b (`experiments/sparse_s1_window/`) ran the window sweep and
returned **P3**: loss falls MONOTONICALLY as the window shrinks —
w=16 (3.785) < w=32 (3.822) < w=64 (3.852) < w=128 (3.894) < w=256 =
exact (3.899), with every adjacent contrast clearing its paired test.
No interior optimum exists down to w=16 (12.3% of full causal
entries).

That kills **H-PRIOR** at this scale: if locality were a favorable
inductive bias matched to the data, a window too small to carry the
relevant context would hurt, and none did. The surviving explanation
is **H-BUDGET** — restricting attention reduces effective capacity,
which helps a short run on a small corpus for the same reason any
regularizer helps a short run. `S1-swa-beats-exact` has been amended
accordingly: the empirical fact stands, the mechanism story does not.

H-BUDGET makes one sharp prediction that H-PRIOR does not, and it is
the reason this cell is worth compute.

## Prediction (committed)

- **P1 — the advantage SHRINKS at longer budget.** Let Δ(steps) =
  best_val(swa@64) − best_val(exact), paired by seed. S1 measured
  Δ(400) = −0.0479 (t = −10.35, n=10). H-BUDGET predicts
  **Δ(1200) > Δ(400)** — i.e. strictly less negative. Direction
  committed: the gap closes.
- **P2 — the strong form.** H-BUDGET's strongest version predicts the
  advantage eventually REVERSES (Δ > 0) once the exact model has
  budget enough to use the capacity the window was denying it. Not
  required for P1 to hold; recorded so that a reversal counts as
  confirmation rather than surprise.
- **P3 — the falsifier for H-BUDGET.** If Δ(1200) ≈ Δ(400) or grows
  MORE negative, H-BUDGET is wrong too, and the S1 effect is neither
  locality-as-prior nor a short-run regularization artifact. That
  outcome would leave the effect UNEXPLAINED and would be registered
  as such — an open question, not a story.

## Design

Two lanes, `arch.custom.attention` ∈ {exact, swa} with window=64,
sinks=1 — identical to the S1 cell in every other respect except
`train.steps` 400 → **1200** (3× budget). 5 seeds (1-5), paired.
Seeds match the S1 rows, so Δ(400) and Δ(1200) are measured on the
same seeds.

n=5 paired, df=4, two-tailed crit 2.776. Powered for effects of the
S1 size; a shrinkage smaller than roughly half the S1 effect will land
inside noise and MUST be reported as inconclusive rather than as
support for P1.

## Decision rules

- **P1 holds (gap narrows, significantly):** register the
  budget-conditionality finding and amend `S1-swa-beats-exact` a
  second time to state that its effect is budget-dependent and decays.
  H-BUDGET becomes the supported mechanism.
- **P2 holds (reversal):** as above, plus the sparse advantage is
  formally scoped to short-budget regimes — a strong constraint on
  every sparse-attention comparison run at small scale, including the
  ones in the literature that stop early.
- **P3 holds (no shrinkage):** H-BUDGET falsified. Register the effect
  as UNEXPLAINED with both candidate mechanisms eliminated, and design
  a third cell (sinks-vs-window decomposition — is the win coming from
  the sink rather than the window?).
- **Inside noise:** report inconclusive, state the power limitation,
  do not spin it as support for any of the three.

## Threats

- 1200 steps may still be far from convergence; "longer" is relative
  and a null here does not establish behaviour at 10× or 100×. The
  claim registered must say 1200, not "long budgets".
- Early-stop patience interacts with budget. Patience is left at the
  S1 default and identical across lanes, so it cannot differ BETWEEN
  lanes, but it may cap both.
- The corpus is small; at some budget both lanes overfit and the
  comparison changes character. If best_val stops improving well
  before step 1200 in both lanes, say so and treat the cell as
  measuring the overfitting regime rather than the converged one.
