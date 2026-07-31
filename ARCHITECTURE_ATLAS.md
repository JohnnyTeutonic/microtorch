# The Architecture Atlas

**Status: design note, 2026-07-31. Nothing here is implemented.** The idea is
Jonathan's; the sections marked *Engineering note* are additions from working
through feasibility against the existing codebase, and some of them change the
plan rather than merely costing it.

---

## 0. The intellectual move

> An architecture should not be represented only by its source code or its
> final benchmark score. It should be represented by the pattern of behaviours
> and dependencies it exhibits under controlled interventions.

That is the whole thesis. It turns ablation from a one-paper diagnostic into a
comparative empirical science of model architecture.

A leaderboard says *architecture A scores 0.7 better than B*. This says
*architecture A is high-performing because one component carries it, while B is
high-performing because six components each contribute a little — and those are
different kinds of object even when the scores match.*

---

## 1. The three representations

Every architecture in the corpus gets three vectors.

### 1.1 Structural features — what the model *is*

`x_i ∈ R^d`, from the spec itself: depth; width; parameter count; attention-head
count; head dimension; MLP expansion ratio; normalisation type and placement;
positional encoding; activation function; residual topology; convolutional vs
attention components; recurrence; parameter sharing; sparsity; routing / MoE
structure; local vs global attention; estimated FLOPs and memory.

Entries are numeric, categorical, or graph-derived.

*Engineering note.* microtorch already has most of this for free — a run **is** a
spec file, and a spec is very nearly `x_i` already. The categorical fields
(`arch.preset`, `arch.custom.attention`) and the numeric ones (`d`, `layers`,
`heads`, `T`) are exactly the factor coordinates. What is missing is derived
quantities: parameter count, estimated FLOPs, and a graph encoding of residual
topology. Parameter count is one line over `named_parameters()`; FLOPs is a
short analytic pass over the module tree.

### 1.2 Behavioural profile — how the model *behaves*

`y_i ∈ R^m`: validation loss; convergence speed; sample efficiency; compute
efficiency; memory efficiency; sensitivity to learning rate; gradient
instability; variance across seeds; robustness to distribution shift;
long-context degradation; calibration; performance by task family.

Two architectures may be structurally near-identical and behaviourally distant.
Conversely, syntactically different architectures may occupy the same
behavioural region. That mismatch is one of the more interesting things the
Atlas could measure.

*Engineering note.* The `events.jsonl` stream already carries most of the raw
material: per-step loss, `grad_norm`, per-module gradient norms, eval points,
early-stop triggers. Convergence speed, gradient instability and the loss curve
shape are post-hoc reductions of a file that today is thrown away after the run.
`live_variables()` (added with checkpointing) gives the memory axis. The missing
pieces are held-out task families and a distribution-shift eval set.

### 1.3 Ablation signature — what the model *depends on*

For architecture `i` and component `j`:

```
Δ_ij = M(A_i) − M(A_i^−j)
a_i  = (Δ_i1, Δ_i2, …, Δ_ik)
```

`M` is validation loss, accuracy, compute-adjusted performance, stability, or
another metric. `a_i` is the **ablation signature**: not how well the
architecture performs, but what its performance rests on.

*Engineering note — the substitution lattice.* "Removing" a component is usually
undefined. You cannot delete RoPE from a Llama and still have a model; you
replace it with NoPE or learned positional embeddings. So `Δ` is a
**substitution effect**, not a removal effect, and the corpus needs an explicit
lattice declaring, for each component slot, the legal alternatives:

```
norm       : {RMSNorm, LayerNorm, none}
position   : {RoPE, learned, sinusoidal, NoPE}
activation : {SwiGLU, GELU-MLP, ReLU-MLP}
attention  : {exact, kimi-linear, SRD, sliding-window}
head tying : {tied, untied}
residual   : {pre-norm, post-norm}
```

This matters for interpretation: `Δ` is only meaningful relative to a named
baseline substitute, and the doc/report must always say which one.

---

## 2. Clustering, and what each space reveals

Candidate methods: k-means; hierarchical; Gaussian mixtures; spectral; HDBSCAN;
or a k-NN graph followed by community detection.

**Structural clustering** on `x_i` recovers architectural families:
deep-and-narrow, shallow-and-wide, attention-heavy, MLP-heavy, recurrent
hybrids, parameter-sharing, sparse-routing. The most obvious analysis and
probably the least surprising.

**Behavioural clustering** on `y_i` gives groups like: fast learners that
plateau early; slow but high-ceiling; efficient small-data models; unstable
high-capacity models; strong long-context models; robust but less accurate
models. This can reveal that conventional architectural labels conceal more
meaningful empirical groupings.

**Ablation-signature clustering** on `a_i` is where the project becomes
original. Groups might be: architectures highly dependent on positional
encoding; architectures where MLP design matters more than attention design;
architectures robust to normalisation substitution; architectures with strong
component redundancy; architectures where the gain lives entirely in
interactions.

Two models with identical benchmark scores can have completely different causal
dependency profiles. One is carried by a single component; the other spreads its
dependence evenly. That distinction is invisible on a leaderboard, and it is the
thing a practitioner most needs to know before borrowing an idea.

*Engineering note — cluster stability is mandatory.* With `d` features and `N`
architectures, clustering will find "families" in noise whenever `N` is not
comfortably larger than `d`. Two defences, both cheap, both non-negotiable
given this project's discipline:

1. **Bootstrap the clustering.** Resample architectures, re-cluster, and report
   the co-assignment frequency for each pair. A "family" that survives 90% of
   resamples is a finding; one that survives 55% is a picture.
2. **Pre-register the protocol** — metric, method, `k` selection rule — before
   looking at the embedding. Otherwise cluster count becomes a researcher
   degree of freedom, and the SRD replication failure (SPARSE_ATTENTION.md,
   2026-07-31) is the standing reminder of what that costs.

---

## 3. Neighbours become meaningful

For any selected architecture the Atlas reports four distinct notions of
proximity:

- **Structural neighbours** — built most similarly.
- **Behavioural neighbours** — most similar learning and generalisation profile.
- **Ablation neighbours** — depend on their components in similar ways.
- **Counterfactual neighbours** — differ by one or two features yet behave
  substantially differently.

That last class is the scientifically valuable one. It supports queries like
*what is the nearest architecture that performs much better?* and *which
minimally different architecture is substantially more stable?* — which turns
nearest-neighbour lookup into automated comparative experimentation.

---

## 4. Interaction effects

Single-component ablation misleads, because architectural innovations interact.
For components R (RMSNorm), P (RoPE), S (SwiGLU), measure not only `Δ_R, Δ_P,
Δ_S` but

```
I_R,P = M(R,P) − M(R) − M(P) + M(∅)
```

which asks whether the pair contributes more or less than the sum of its parts.
That distinguishes:

- **synergy** — the two work unusually well together;
- **redundancy** — either alone is sufficient;
- **suppression** — one reduces the value of the other;
- **conditional dependence** — a component matters only within a family.

The output is architectural knowledge rather than benchmark reporting. Not
"SwiGLU adds 0.7 points" but "SwiGLU helps wide models, contributes little to
narrow ones, and is especially beneficial with pre-normalisation."

### 4.1 Engineering note — this is a design-of-experiments problem, and that changes the plan

Enumerating ablations per architecture explodes: `k` components is `O(k)` runs
for main effects, `O(k²)` for pairs, `2^k` for the full Shapley decomposition.
At `k = 8` that is 8, 28 and 256 configurations respectively — **per
architecture, per seed**. Multiply by a corpus of 100 and it is dead on arrival.

The fix is to stop thinking "corpus of architectures, each ablated" and start
thinking **factorial design over the architectural factor space**. Treat each
component slot as a factor; then:

- A **full factorial** at `k = 8` binary factors is 256 runs and gives every
  main effect and every interaction to 8th order — far more than anyone needs.
- A **resolution-V fractional factorial** (2^(8−2) = **64 runs**) gives all 8
  main effects and all 28 two-way interactions unconfounded with each other.
  That is the entire content of section 4 for a quarter of the cost.
- A **Plackett–Burman screening design** (12 runs for up to 11 factors) gives
  main effects only, and is the right first pass to find which factors are worth
  the resolution-V budget at all.

So the recommended shape is a two-stage design: screen with Plackett–Burman,
then spend the real budget on a resolution-V design restricted to the factors
that screening flagged. This is standard industrial DoE and it maps onto the
spec system exactly — **a design matrix row is a spec file**.

This also reframes the ablation signature: `a_i` for an *individual*
architecture is then estimated from the fitted model (section 7) rather than
measured directly for every architecture, which is both cheaper and less noisy.

---

## 5. The corpus

Six sources, deliberately mixed:

1. Canonical architectures extracted from papers (`papers/fetch.py` is already
   this).
2. Controlled variants generated by microtorch.
3. Ablated / substituted versions of canonical models.
4. Interpolations between known architectures.
5. Random but valid architectures sampled from a constrained grammar.
6. User-created models contributed through the Studio.

**Why this matters more than it looks.** Published models are heavily selected —
they are the architectures researchers chose to report. Analyse only those and
you learn *what researchers tend to publish*, not *how architectural features
behave*. Sources 2–5 are the correction for that selection bias, and they are
the reason this cannot be done by scraping a leaderboard.

*Engineering note.* Source 5 needs a **constrained grammar** that only emits
valid, trainable configurations (divisibility of `d` by heads, legal norm
placements, compatible position encodings). That grammar is a small piece of
work and is worth building early, because it is also what makes the spec builder
in the Studio impossible to misconfigure.

---

## 6. Confounding, and the protocol

Architecture performance is massively confounded by parameter count, token
count, optimiser, LR schedule, regularisation, data quality, batch size,
training duration, implementation quality, and **tuning effort**. You cannot
conclude "A beats B" if A got ten times the tuning.

Minimum protocol: fixed compute budgets; matched parameter counts; matched
datasets; common optimiser search spaces; multiple seeds; identical evaluation
pipelines; explicit confidence intervals.

Several comparison regimes, maintained in parallel:

| Regime | Held constant |
|---|---|
| Parameter-matched | approximate parameter count |
| FLOP-matched | training compute |
| Latency-matched | inference-time constraint |
| Memory-matched | hardware memory budget |
| Frontier | nothing — compare on Pareto fronts |

The frontier regime matters because architectural strength is rarely
one-dimensional, and collapsing it to a scalar is how leaderboards mislead.

*Engineering note — tuning effort is the nastiest confound because it is
unobservable after the fact.* The only real defence is to make it a fixed part
of the protocol: every architecture gets the **same LR search budget** (say a
fixed 5-point grid over the same range), and the budget is recorded in the
result row alongside the score. An architecture that would have won with more
tuning simply loses under the stated protocol, and the protocol is published.
That is honest and reproducible; "we tuned until we were satisfied" is neither.

*Engineering note — seeds are not optional here.* On 2026-07-31 a single seed in
the SRD needle experiment produced what looked like a clean phase-change result;
three replication seeds returned 0/3 on both pre-registered criteria, and the
falsifier lane outperformed the mechanism in two of them. Every cell in this
corpus needs ≥3 seeds, and **seed variance is itself one of the behavioural
features** (`y_i` already lists it). The Atlas would have caught that result
automatically, which is a decent argument for building it.

---

## 7. Statistical modelling

Clustering is not the end. With a large enough corpus, fit models:

```
Y = β0 + β1(RMSNorm) + β2(RoPE) + β3(depth) + β4(width)
       + β5(RMSNorm × depth) + ε
```

estimating average effects while controlling for measured features. More
flexible options: random forests; gradient-boosted trees; Gaussian processes;
graph neural networks over computational graphs; Bayesian hierarchical models;
mixed-effects models across datasets and seeds.

A **hierarchical model** is the right default because the data are nested — runs
within architectures, architectures within families, datasets within task types:

```
performance_{i,d,s} = α + u_i + v_d + β' x_i + γ'(x_i × d) + ε_{i,d,s}
```

which directly answers: *which architectural properties have general effects,
and which are task-dependent?*

*Engineering note.* The mixed-effects model is also what makes the fractional
factorial (section 4.1) pay off: the design gives clean, unconfounded estimates
for `β` and the interaction terms, and the random effects `u_i, v_d` absorb the
architecture- and dataset-level noise that would otherwise be read as signal.
Design and model belong together; picking one without the other wastes the
compute.

---

## 8. Architectural fingerprints

The clearest framing of the output. Each model gets a fingerprint composed of
structure, learning dynamics, task strengths, robustness, efficiency, ablation
dependence, and interaction effects — rendered as a report:

> **Architecture profile**
> - Structurally nearest to GPT-style decoder transformers.
> - Behaviourally nearest to recurrent-attention hybrids.
> - Strongest relative advantage: long-context retention.
> - Primary dependency: rotary positional encoding.
> - Most redundant component: gated MLP.
> - Failure mode: high variance under low-data training.
> - Distinguishing feature: depth contributes more than width relative to its
>   nearest peers.

Substantially more informative than a scalar score, and — importantly — every
line of it is a claim traceable to specific runs in the corpus.

---

## 9. The research questions

- Do structurally similar architectures behave similarly?
- Are there multiple architectural routes to the same performance profile?
- Which components are robustly beneficial across model families?
- Which innovations work only through interactions with other innovations?
- Which architectures are over-engineered? Which components are redundant?
- Are published architectural families genuine behavioural families, or mostly
  historical labels?
- Do certain structures systematically produce stable training?
- Can ablation signatures predict generalisation?
- Can an unseen architecture's performance be estimated from its neighbours?
- Can we identify underexplored regions of architecture space?
- Can architecture recommendations be conditioned on a desired capability
  profile?

---

## 10. Connection to the paper pipeline

The paper-analysis feature becomes the entry point. microtorch reads a paper and
extracts claimed innovations, baseline architecture, reported metrics and
proposed causal explanations, then compares against the corpus:

> **Paper claim:** RMSNorm improves training stability.
> **Corpus evidence:** RMSNorm is associated with lower gradient variance in 71%
> of matched transformer comparisons, but the effect is concentrated in models
> deeper than 24 layers.
> **Recommended experiment:** compare LayerNorm and RMSNorm at depths 12, 24 and
> 48, holding compute constant.
> **Reason:** existing evidence suggests a depth-dependent interaction.

That is much stronger than auto-generating a generic ablation grid: the system
is using accumulated evidence to propose **high-information** experiments — the
ones whose outcome the corpus cannot already predict.

*Engineering note.* "High-information" can be made precise rather than
rhetorical: rank candidate experiments by expected reduction in posterior
variance of the coefficient in question (Bayesian optimal experimental design).
The corpus gives the prior; the design gives the expected information gain. This
is the feature that would make the tool genuinely worth using rather than
merely interesting.

---

## 11. Product shape

**Option A — microtorch Research Mode.** A feature inside the existing
ecosystem: parse a paper, reproduce the architecture, generate ablations,
compare against a corpus, produce an evidence report.

**Option B — a separate architecture observatory.** Its own identity and
repository: standardised experiment database, architecture embeddings,
similarity search, statistical meta-analysis, interactive maps, hypothesis
generation, experiment recommendation — sharing microtorch's execution engine.

Both are viable and they are not exclusive: A is the on-ramp to B, and A is
worth building even if B never happens.

*On naming.* Of the candidates — Architecture Observatory, Model Atlas, ArchLab
— note that an **observatory observes** and this system's entire epistemic
advantage is that it **intervenes**. Controlled substitution under a fixed
protocol is what buys causal traction over scraped leaderboards. "Lab" or
"Atlas" carry that better than "Observatory"; **Model Atlas** additionally
matches the map-and-neighbourhood metaphor the interface is built on.

---

## 12. Feasibility: what this actually costs

Concrete numbers, using measured microtorch throughput after the 2026-07-31
performance work (llama-tiny, d=128, 2 layers, T=128, batch=4: ~1.03 s/step, so
a 3,000-step run is ~52 minutes).

| Scope | Runs | Serial CPU | 4 concurrent sessions |
|---|---|---|---|
| Plackett–Burman screen, 11 factors, 3 seeds | 36 | ~31 h | **~8 h** |
| Resolution-V design, 8 factors, 3 seeds | 192 | ~166 h | **~42 h** |
| Naive per-architecture ablation, 100 arch × 6 comps × 3 seeds | 1,800 | ~1,560 h | ~390 h |

The screening pass is an overnight job. The resolution-V design is a long
weekend. The naive approach is sixteen days and buys strictly less. That gap is
the entire argument for section 4.1.

**What already exists** and does not need building: the execution engine, the
declarative spec (a design-matrix row *is* a spec), checkpoint/resume for long
sweeps, the JSONL event stream as the measurement record, early stopping,
multi-seed CLI support (`srd_needle` already takes a seed argument), and the
falsifier/pre-registration discipline this whole thing depends on.

**What needs building**, in dependency order:
1. Derived structural features (parameter count, FLOP estimate) — hours.
2. A constrained architecture grammar + design-matrix → specs generator — days.
3. A sweep runner and a results store (JSONL is sufficient; a database is not
   required until the corpus is large) — days.
4. Behavioural feature extraction from `events.jsonl` — days.
5. The statistical layer (mixed-effects fit, interaction estimates) — Python,
   `statsmodels`/`pymc`, days.
6. The Studio map view (embedding + click-through) — the visible payoff, and the
   piece that should be built last because everything above determines what it
   displays.

---

## 13. Risks, stated honestly

**The central external-validity threat: scale.** These experiments run at
d=128, 2 layers. Many architectural effects are scale-dependent, and some famous
ones reverse. An Atlas built entirely at tiny scale risks producing a beautiful,
well-clustered, internally-consistent map of a regime nobody deploys in.

The mitigation is not to pretend otherwise but to **make scale an axis rather
than a constant**: measure every effect at three model sizes and report the
*trend* rather than the point estimate. A component whose benefit grows with
depth is a different finding from one that shrinks, and the trend is often
extrapolatable where the point estimate is not. Any claim not measured across
the ladder should be explicitly labelled as small-scale-only.

**Second risk: the corpus measures microtorch, not architectures.** Effects
could be artefacts of one implementation. Partial defence: the GPT-2 and Qwen
parity checks already establish that this implementation agrees with the
reference on known models, and that argument should be cited whenever the corpus
makes a claim.

**Third risk: garden of forking paths.** A rich enough feature space plus
flexible clustering will always yield an interesting-looking story.
Pre-registration of the analysis, bootstrap stability for clusters, and holding
out a validation slice of the corpus are the standard defences and should be
protocol from run one, not retrofitted.

**Fourth risk: scope.** This is a research programme, not a feature. Option A
(Research Mode) is a bounded deliverable; Option B is a multi-year project. The
staged plan below is deliberately arranged so that each stage is independently
useful if the next never happens.

---

## 14. Staged plan

**Stage 0 — instrument (days).** Derived structural features; behavioural
extraction from `events.jsonl`; persist run records instead of discarding them.
*Independently useful:* every future microtorch run becomes a data point.

**Stage 1 — the grammar and the sweep runner (days).** Constrained architecture
grammar; design matrix → specs; a runner that executes a sweep with
checkpoint/resume and writes one result row per run.
*Independently useful:* multi-seed sweeps become one command, which the SRD line
needs anyway.

**Stage 2 — the screening experiment (one night).** Plackett–Burman over 11
architectural factors, 3 seeds, one dataset, parameter-matched. Report main
effects with confidence intervals.
*Independently useful:* this alone is a publishable negative-or-positive result
about which architectural factors matter at small scale.

**Stage 3 — the resolution-V design (a weekend).** All main effects and pairwise
interactions for the factors screening flagged. Fit the mixed-effects model.
*Independently useful:* the interaction table is the scientific core of the
whole idea.

**Stage 4 — the scale ladder (weeks).** Repeat stages 2–3 at three model sizes;
report trends.
*This is what converts small-scale findings into defensible claims.*

**Stage 5 — the Atlas surface (weeks).** Embedding, map view, four kinds of
neighbour, fingerprint reports, paper-vs-corpus comparison.
*The product.*

---

## 15. Why this fits microtorch specifically

Most people cannot build this. It requires an execution engine you control
end-to-end, a declarative run format, cheap experiments, and — most rarely — a
research culture that publishes negatives and pre-registers criteria. microtorch
has all four, and the last one is the scarce ingredient: an atlas assembled by
someone willing to report that their own mechanism failed to replicate is worth
considerably more than one assembled by someone who is not.
