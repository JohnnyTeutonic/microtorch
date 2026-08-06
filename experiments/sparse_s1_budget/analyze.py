"""Pre-registered analysis for S1-c (experiments/sparse_s1_budget/PREREGISTRATION.md).

Committed predictions:
  P1  Delta(1200) > Delta(400)          [H-BUDGET: the gap closes]
  P2  Delta(1200) > 0                   [strong form: reversal]
  P3  Delta(1200) ~= or < Delta(400)    [H-BUDGET falsified]
Plus the stated threat check: did both lanes stop improving well before
step 1200 (i.e. are we measuring the overfitting regime)?
"""
import json
import os
import statistics as st

os.chdir("/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch")
CRIT = {4: 2.776, 9: 2.262}
SEEDS = [1, 2, 3, 4, 5]


def load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("complete"):
            out.setdefault(r["factor.arch.custom.attention"], {})[r["seed"]] = r
    return out


b = load("/tmp/sparse_s1_budget/atlas_rows.jsonl")
s1 = load("experiments/sparse_s1/atlas_rows.jsonl")


def paired_delta(cell, seeds):
    d = [cell["swa"][s]["best_val"] - cell["exact"][s]["best_val"]
         for s in seeds if s in cell.get("swa", {}) and s in cell.get("exact", {})]
    n = len(d)
    m, sd = st.fmean(d), st.stdev(d)
    return n, m, sd, m / (sd / n ** 0.5), d


print("=" * 72)
print("S1-c PRE-REGISTERED ANALYSIS (predictions committed in 8a01a92)")
print("=" * 72)

n4, m4, sd4, t4, d4 = paired_delta(s1, SEEDS)
n12, m12, sd12, t12, d12 = paired_delta(b, SEEDS)

print(f"\nDelta(400)  = {m4:+.5f}  (n={n4}, t={t4:.2f})   [S1, same seeds]")
print(f"  diffs: {[f'{x:+.4f}' for x in d4]}")
print(f"Delta(1200) = {m12:+.5f}  (n={n12}, t={t12:.2f})")
print(f"  diffs: {[f'{x:+.4f}' for x in d12]}")

print("\nlane means at 1200 steps:")
for lane in ("exact", "swa"):
    v = [b[lane][s]["best_val"] for s in SEEDS]
    print(f"  {lane:>5}: {st.fmean(v):.4f} (sd {st.stdev(v):.4f})")

# --- P1: did the gap shrink? paired on the SAME seeds -------------------
shrink = [d12[i] - d4[i] for i in range(min(len(d4), len(d12)))]
ms, sds = st.fmean(shrink), st.stdev(shrink)
ts = ms / (sds / len(shrink) ** 0.5)
crit = CRIT[len(shrink) - 1]
print("\n" + "-" * 72)
print("P1 (H-BUDGET): Delta(1200) > Delta(400) -- the advantage shrinks")
print(f"  per-seed change in Delta: {[f'{x:+.4f}' for x in shrink]}")
print(f"  mean change {ms:+.5f}   t = {ts:.2f}   df = {len(shrink)-1}   crit {crit}")
sig = abs(ts) > crit
if not sig:
    v1 = "INSIDE NOISE -- inconclusive, report as such (pre-reg decision rule)"
elif ms > 0:
    v1 = "P1 SUPPORTED: the advantage shrinks with budget -> H-BUDGET holds"
else:
    v1 = "P3 SUPPORTED: advantage GREW -> H-BUDGET FALSIFIED, effect unexplained"
print(f"  VERDICT: {v1}")

print("\nP2 (strong form): reversal, Delta(1200) > 0")
print(f"  Delta(1200) = {m12:+.5f} -> {'REVERSED' if m12 > 0 else 'no reversal'}"
      f" (swa still better in {sum(1 for x in d12 if x < 0)}/{len(d12)} seeds)")

# --- threat check: are we in the overfitting regime? --------------------
print("\n" + "-" * 72)
print("THREAT CHECK (pre-registered): did best_val stall well before 1200?")
for lane in ("exact", "swa"):
    fs = [b[lane][s].get("final_step") for s in SEEDS]
    es = [b[lane][s].get("early_stopped") for s in SEEDS]
    ne = [b[lane][s].get("n_evals") for s in SEEDS]
    print(f"  {lane:>5}: final_step {fs}  early_stopped {es}  n_evals {ne}")
print("  (if both lanes early-stopped far below 1200, the cell measures the "
      "overfitting regime, not the converged one -- say so in the row)")
