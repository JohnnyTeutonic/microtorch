"""Pre-registered analysis for S1-d (experiments/sparse_s1_sinks/PREREGISTRATION.md).

Committed BEFORE the runs (e0999e5):
  P1  |delta(sinks0 - sinks1)| < 0.021 at 400 steps  [capacity model: predicts
      ~0.0008, i.e. ~25x below the detection floor -> a NULL is the prediction]
  P2  |delta| >= 0.021                               [sink does something structural]
  P3  if P2 fires, direction is sinks=0 BETTER       [S1-b monotone extends]
  P4  reversal reproduces without sinks (needs the 1200 cell)
  Pre-committed: 0.008 < |delta| < 0.021 = INCONCLUSIVE, not null.
"""
import json
import os
import statistics as st

os.chdir("/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch")
SEEDS = [1, 2, 3, 4, 5]
CRIT4 = 2.776
FLOOR = 0.021
GREY = 0.008


def load_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def by_seed(rows, pred):
    return {r["seed"]: r["best_val"] for r in rows
            if r.get("complete") and pred(r)}


# new lane: sinks=0, split by budget
sink0 = load_rows("/home/jonat/sparse_s1_sinks/atlas_rows.jsonl")
s0_400 = by_seed(sink0, lambda r: int(r.get("factor.train.steps", r.get("steps", 0))) == 400)
s0_1200 = by_seed(sink0, lambda r: int(r.get("factor.train.steps", r.get("steps", 0))) == 1200)

# anchors already on disk, matched seeds
a400 = load_rows("experiments/sparse_s1/atlas_rows.jsonl")
s1_400 = by_seed(a400, lambda r: r["factor.arch.custom.attention"] == "swa")
ex_400 = by_seed(a400, lambda r: r["factor.arch.custom.attention"] == "exact")

a1200 = load_rows("experiments/sparse_s1_budget/atlas_rows.jsonl")
s1_1200 = by_seed(a1200, lambda r: r["factor.arch.custom.attention"] == "swa")
ex_1200 = by_seed(a1200, lambda r: r["factor.arch.custom.attention"] == "exact")


def paired(a, b, label):
    ss = sorted(s for s in SEEDS if s in a and s in b)
    if len(ss) < 2:
        return None
    d = [a[s] - b[s] for s in ss]
    m, sd = st.fmean(d), st.stdev(d)
    se = sd / len(d) ** 0.5
    t = m / se if se else float("inf")
    print(f"  {label}: n={len(d)}  mean {m:+.5f}  se {se:.5f}  t={t:+.2f}")
    print(f"    diffs {[f'{x:+.4f}' for x in d]}")
    return m, se, t, len(d)


print("=" * 72)
print("S1-d PRE-REGISTERED ANALYSIS  (predictions committed in e0999e5)")
print("=" * 72)

print(f"\n400-step cell  (sinks=0 n={len(s0_400)}, anchors n={len(s1_400)}/{len(ex_400)})")
if s0_400:
    print(f"  means: sinks0 {st.fmean([s0_400[s] for s in sorted(s0_400)]):.4f}  "
          f"sinks1 {st.fmean([s1_400[s] for s in SEEDS]):.4f}  "
          f"exact {st.fmean([ex_400[s] for s in SEEDS]):.4f}")

print("\n" + "-" * 72)
print("PRIMARY TEST — P1 vs P2/P3: sinks=0 minus sinks=1 at 400 steps")
print(f"  capacity model predicts |delta| ~ 0.0008; detection floor {FLOOR}")
r = paired(s0_400, s1_400, "sinks0 - sinks1 @400")
if r:
    m, se, t, n = r
    a = abs(m)
    if a < GREY:
        v = ("P1 SUPPORTED — null, and the capacity model's QUANTITATIVE "
             "prediction survives (|delta| below even the grey zone)")
    elif a < FLOOR:
        v = (f"INCONCLUSIVE by pre-commitment — |delta|={a:.4f} sits in the "
             f"grey zone ({GREY}-{FLOOR}). NOT reportable as a null.")
    elif m < 0:
        v = ("P2+P3 SUPPORTED — sinks=0 detectably BETTER; the monotone story "
             "extends to the sink rung")
    else:
        v = ("P2 SUPPORTED, P3 CONTRADICTED — sinks=0 WORSE. The sink is doing "
             "structural work the density model cannot explain. This was "
             "flagged in advance as the most interesting outcome available.")
    print(f"  significance: |t|={abs(t):.2f} vs crit {CRIT4} -> "
          f"{'detectable' if abs(t) > CRIT4 else 'not detectable'}")
    print(f"  VERDICT: {v}")

print("\n" + "-" * 72)
print("SECONDARY — is the sparse advantage still there without sinks?")
paired(s0_400, ex_400, "sinks0 - exact @400 ")

if len(s0_1200) >= 2:
    print("\n" + "-" * 72)
    print(f"P4 (partial, n={len(s0_1200)}/5) — reversal without sinks at 1200")
    paired(s0_1200, ex_1200, "sinks0 - exact @1200")
    paired(s0_1200, s1_1200, "sinks0 - sinks1 @1200")
else:
    print("\n(1200 cell incomplete — P4 deferred)")
