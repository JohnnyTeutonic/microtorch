# Sparse Attention Research Program

The flagship research phase. Goal: an original, *natively trainable* efficient-attention
mechanism validated in microtorch — not another commodity reimplementation.

Survey date: 2026-07-30 (live arXiv sweep; refresh before each design round).

## 1. The landscape, mapped to our opening questions

### Q: "Split dot-product attention into blocks?"
Two distinct lineages, often confused:
- **Flash Attention** — blocked but **exact**. An IO-aware tiling of the same math.
  Not sparsity; nothing is skipped.
- **Block-sparse** — skip most blocks. Sparse Transformers/BigBird (2019-20, fixed
  patterns) → the modern trainable era: **NSA** (arXiv:2502.11089, compressed +
  selected + sliding tiers, end-to-end trainable), **MoBA** (top-k block routing via
  MoE-style gate), **InfLLM-V2** (arXiv:2509.24663, dense↔sparse switchable).
  Verdict: *whether blocks* is settled; **how to cheaply score blocks** is not
  (XAttention, arXiv:2503.16428, scores blocks by antidiagonal sums — the field is
  still guessing at proxies).

### Q: "Hot cache of frequently used tokens / speculative mechanism?"
- Inference-side is crowded: H2O heavy-hitters, StreamingLLM attention sinks,
  HashEvict (arXiv:2412.16187, SimHash Hamming distance as pre-attention proxy),
  Expected Attention (arXiv:2510.00636, estimates future-query attention).
- **Trainable** cache residency — the model *learning* what stays hot during
  pretraining — is much thinner. Our V1 is this idea in disguise: the surprise gate
  is a learned residency policy.

### Q: "LSH / SimHash as cheap dot-product approximation?"
Reformer (2020) used LSH as the attention itself: trains poorly (bucket imbalance,
causality pain). The revival (HashEvict) uses hashes as *selectors*, not replacements.
**Lesson: cheap approximations survive as routers, not as the attention.**

### Q: "Traditional algorithms not yet exploited?" (the Q2 vein)
- **Fast Multipole / H-matrices**: FMA (arXiv:2310.11960), Multipole Attention
  (arXiv:2506.13059). Exists but thin; mostly not natively trained at scale.
  N-body framing (near-field exact, far-field summarized) remains underexploited.
- **Leverage scores** (randomized NLA): Compactor (arXiv:2507.08143), pre-scoring
  (arXiv:2505.11040) — inference-side only so far. Training-time leverage-score
  routing: open.
- **Matrix sketching (Frequent Directions / Co-occurring Directions)**: FD is
  15 years of streaming theory with deterministic guarantees; CoD sketches the
  product of two matrix streams — which is literally linear attention's K^T V
  state. **No attention paper found using it.** → V2.
- **Discrepancy theory**: streaming attention approximation exists in theory
  (arXiv:2502.07861); no practical trained variant seen.

### Q: "Diagonalisation methods?"
Partially mined, from three directions: Performer's FAVOR+ (random-feature
factorization of the softmax kernel), Nyströmformer (low-rank spectral
approximation via landmarks), FNet (fixed Fourier mixing — diagonalization of
circulant structure), and S4/Mamba (state matrices *literally* run diagonalized,
DPLR/S4D). Open-ish corner: maintaining a cheap streaming eigen-sketch of the
attention Gram matrix and routing by spectral leverage — connects to the
leverage-score vein above. Keep on the shortlist; check for 2026 papers before
investing.

### Q: "Synthesis risk — degrading training?"
The central lesson of NSA: **sparsity must be trainable end-to-end**. Post-hoc
sparsification of a densely-trained model degrades; discrete/non-differentiable
routers starve gradients. Every microtorch variant must therefore:
1. keep gradients flowing through the router (soft gates, straight-through, or
   aux losses),
2. pass finite-difference gradcheck on every new op,
3. match the Kimi-linear baseline's training curve on TinyStories before claiming
   anything (the graduation gate).

## 2. Our three shots

### V1 — Surprise-Routed Density (SRD)  ← IN PROGRESS
NSA/MoBA route blocks by learned top-k affinity scores. We route **per-query
density by prediction residual** — the cerebellum mechanism already in-repo:

    predicted = RoutinePredictor(x)              (small MLP, trained in-loop)
    g[t]      = sigmoid(4 * rms(x[t] - predicted[t]) - 1)      in (0,1)
    out[t]    = g[t] * ExactAttention(x)[t] + (1-g[t]) * KimiLinear(x)[t]

Routine tokens (predictable) ride the O(n·d²) linear path; surprising tokens get
exact O(n²) attention. Q and K/V projections are SHARED between paths, so the
mechanism difference is isolated. Training is soft (both paths computed, fully
differentiable); inference hardens the gate (g > τ → exact for that query only),
giving O(ρ·n²·d + n·d²) with ρ = surprise rate. An aux loss mean(g) lets the
caller price density.

Novelty claim (as of the 2026-07 sweep): routing by *prediction error* rather
than by attention-affinity scores appears untried. Closest neighbors:
InfLLM-V2 (global dense/sparse switch, not per-token), MoE-style MoBA gates
(affinity-based), SSA (arXiv:2511.20102, aligns sparse & full outputs — a
useful *evaluation* idea for us).

Risks: (a) gate collapse (always open / always closed) — monitor gate histogram,
price with aux loss; (b) the residual signal may proxy token frequency rather
than contextual novelty — probe with synthetic sequences; (c) soft training
cost is exact+linear together — acceptable at research scale.

### V2 — Sketch-State Attention (CoD/FD)
Replace linear attention's unbounded K^T V accumulation with a Co-occurring
Directions sketch: fixed sketch size ℓ, deterministic error bounds, streaming-
native, O(n·ℓ·d). The deterministic-guarantee counterpart to Performer's random
features. No prior attention use found. Design question: differentiating through
the SVD-based shrink step (options: straight-through the shrink, periodic
detached re-sketch, or replace SVD with a differentiable power-iteration).

### V3 — Cheap-proxy block-scoring bake-off
Head-to-head, trained end-to-end NSA-lite: PQ codebook lookup vs SimHash Hamming
vs antidiagonal sums vs leverage-score sampling as the block router. Least novel
(Online VQ Attention, arXiv:2602.03922, is adjacent and *current* — move fast if
we pick this), but fastest to a publishable ablation and it builds the block
harness V1's hardened-inference mode needs anyway.

## 3. Protocol

- **Scale ladder**: (1) op-level gradchecks → (2) 2-layer model, TinyStories
  word-level (transformer_cpp corpus + tinyllama.cpp for inference sanity) →
  (3) Colab T4 for anything bigger ([[colab discipline: no local GPU]]).
- **Metrics**: perplexity-vs-FLOPs frontier against exact + Kimi-linear + NSA-lite
  baselines; needle-in-haystack retrieval for long context; gate statistics
  (V1); gradient norms through the router (the q4 guard).
- **Falsifiers first**: each variant ships with the experiment that would kill it
  (V1: shuffle the predictor's inputs — if quality holds, the gate wasn't using
  surprise; V2: sketch size sweep — if ℓ≈n needed, no win; V3: random router
  baseline — if proxies don't beat random block choice, stop).
- Venue discipline: journals only, strength over speed.

## 4. Bibliography (survey snapshot 2026-07-30)

- NSA: Native Sparse Attention — arXiv:2502.11089
- MoBA optimization — arXiv:2511.11571
- InfLLM-V2 dense-sparse switchable — arXiv:2509.24663
- XAttention antidiagonal block scoring — arXiv:2503.16428
- SSA: aligning sparse & full outputs — arXiv:2511.20102
- HashEvict (SimHash eviction) — arXiv:2412.16187
- Expected Attention — arXiv:2510.00636
- Online VQ Attention — arXiv:2602.03922
- Compactor (leverage-score KV compression) — arXiv:2507.08143
- Efficient Attention via Pre-Scoring — arXiv:2505.11040
- Fast Multipole Attention — arXiv:2310.11960
- Multipole Attention for long-context reasoning — arXiv:2506.13059
- Frequent Directions — arXiv:1501.01711
- Streaming attention via discrepancy theory — arXiv:2502.07861
