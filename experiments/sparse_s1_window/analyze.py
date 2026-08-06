"""Pre-registered analysis for S1-b (experiments/sparse_s1_window/PREREGISTRATION.md).

Runs ONLY the committed tests: P1 (harness/bitwise), P2 vs P3 (the
discriminating w=16 rung), P4 (near-full rung). Nothing else.
"""
import json
import os
import statistics as st

os.chdir("/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch")

CRIT = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262}


def paired(a, b, seeds):
    """a, b are {seed: best_val}. Returns (n, mean, t, crit, verdict)."""
    ss = sorted(s for s in seeds if s in a and s in b)
    d = [a[s] - b[s] for s in ss]
    if len(d) < 2:
        return None
    m, sd = st.fmean(d), st.stdev(d)
    t = m / (sd / len(d) ** 0.5) if sd > 0 else float("inf")
    crit = CRIT.get(len(d) - 1, 1.96)
    return len(d), m, t, crit, abs(t) > crit, d


# --- load both cells ------------------------------------------------------
win = {}   # window -> {seed: best_val}
for line in open("/tmp/sparse_s1_window/atlas_rows.jsonl"):
    r = json.loads(line)
    if not r.get("complete"):
        continue
    win.setdefault(int(r["factor.arch.custom.window"]), {})[r["seed"]] = r["best_val"]

s1 = {}    # attention -> {seed: best_val}   (w=64 anchor + exact anchor)
for line in open("experiments/sparse_s1/atlas_rows.jsonl"):
    r = json.loads(line)
    if not r.get("complete"):
        continue
    s1.setdefault(r["factor.arch.custom.attention"], {})[r["seed"]] = r["best_val"]

W64, EXACT = s1["swa"], s1["exact"]
SEEDS = [1, 2, 3, 4, 5]

print("=" * 70)
print("S1-b PRE-REGISTERED ANALYSIS  (predictions committed in b2f44e4)")
print("=" * 70)

print("\nmeans by window (seeds 1-5):")
for w in sorted(win):
    v = [win[w][s] for s in SEEDS if s in win[w]]
    print(f"  w={w:>3}  {st.fmean(v):.4f}  (n={len(v)}, sd {st.stdev(v):.4f})")
v64 = [W64[s] for s in SEEDS]
vex = [EXACT[s] for s in SEEDS]
print(f"  w= 64  {st.fmean(v64):.4f}  (n=5, sd {st.stdev(v64):.4f})   [S1 anchor]")
print(f"  exact  {st.fmean(vex):.4f}  (n=5, sd {st.stdev(vex):.4f})   [S1 anchor]")

# --- P1: harness check ----------------------------------------------------
print("\n" + "-" * 70)
print("P1 (HARNESS, must pass or cell is VOID): swa@w=256 == exact bitwise")
if 256 in win:
    diffs = [(s, win[256][s] - EXACT[s]) for s in SEEDS if s in win[256]]
    worst = max(abs(d) for _, d in diffs)
    for s, d in diffs:
        print(f"  seed {s}: swa256 {win[256][s]:.6f}  exact {EXACT[s]:.6f}  diff {d:+.2e}")
    print(f"  worst |diff| = {worst:.2e}   threshold 1e-6")
    p1 = worst < 1e-6
    print(f"  P1: {'PASS' if p1 else 'FAIL — CELL VOID'}")
else:
    p1 = False
    print("  P1: no w=256 rows — CELL VOID")

# --- P2 vs P3: the discriminating rung ------------------------------------
print("\n" + "-" * 70)
print("P2 (H-PRIOR: w=16 WORSE than w=64)  vs  P3 (H-BUDGET: w=16 <= w=64)")
r = paired(win[16], W64, SEEDS)
n, m, t, crit, sig, d = r
print(f"  paired diffs (w16 - w64): {[f'{x:+.4f}' for x in d]}")
print(f"  mean {m:+.5f}   t = {t:.2f}   df = {n-1}   two-tailed crit {crit}")
if not sig:
    verdict = "INCONCLUSIVE (inside noise) — neither P2 nor P3 supported"
elif m > 0:
    verdict = "P2 SUPPORTED (w=16 worse) -> H-PRIOR, interior optimum"
else:
    verdict = "P3 SUPPORTED (w=16 better) -> H-BUDGET, our S1 row needs amending"
print(f"  VERDICT: {verdict}")

# w=32 as the intermediate rung of the same contrast
r32 = paired(win[32], W64, SEEDS)
print(f"  (w=32 - w=64): mean {r32[1]:+.5f}  t {r32[2]:.2f}  "
      f"{'signal' if r32[4] else 'inside noise'}")

# --- P4: near-full rung ---------------------------------------------------
print("\n" + "-" * 70)
print("P4: best_val(128) sits between best_val(64) and best_val(256)")
m128 = st.fmean([win[128][s] for s in SEEDS])
m64, m256 = st.fmean(v64), st.fmean([win[256][s] for s in SEEDS])
print(f"  w=64 {m64:.4f}  <  w=128 {m128:.4f}  <  w=256 {m256:.4f} ?")
print(f"  P4: {'HOLDS' if m64 < m128 < m256 else 'DOES NOT HOLD'}")
r128 = paired(win[128], W64, SEEDS)
print(f"  (w=128 - w=64): mean {r128[1]:+.5f}  t {r128[2]:.2f}  "
      f"{'signal' if r128[4] else 'inside noise'}")
