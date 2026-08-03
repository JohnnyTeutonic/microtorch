# Atlas findings registry

12 claims; every row carries its receipts. Retractions are rows, not deletions.

| id | status | claim | evidence | scope |
|---|---|---|---|---|
| S2-muon | ✅✅ replicated | Muon (hybrid per-head) improves final val loss and whole-run loss AUC over AdamW at tiny scale | t=-6.21 (n=36) | llama family, 0.95-2.7M params, TinyStories, 400 steps |
| S2-lr-null | ↻ superseded | Learning rate (1e-3 vs 3e-3) has no effect on final val loss | t=+1.20 (n=36) | llama family, tiny scale, 400 steps |
| S2-ctx | ❌ retracted | Longer context (T=256 vs 128) improves final val loss | t=-2.77 (n=36) | llama family, tiny scale, FIXED STEP COUNT |
| S2-heads-null | ✅ supported | Head count (4 vs 8) has no effect on any measured metric at tiny scale | t=-0.31 (n=36) | llama family, tiny scale, 400 steps; null on all 5 metrics |
| S2-spike-metric | ❌ retracted | Larger batch increases gradient spikes | t=+3.13 (n=36) | llama family, tiny scale |
| S3-lrxopt | ✅ supported | lr x optimizer is a real interaction: 3e-3 is the best lr under Muon and the worst under AdamW (orthogonalized updates widen the stable lr range) | t=-3.12 (n=48) | llama family, tiny scale, 3x S2 token budget; also spikes t=-4.01, AUC t=-2.69 |
| S3-muon | ✅ supported | Muon's val-loss advantage replicates at 3x token budget | t=-7.38 (n=48) | llama family, tiny scale |
| S3-ctx-null | ✅ supported | With tokens held equal, context length (T=128/1200steps vs T=256/600steps) has no val-loss effect | t=-1.48 (n=48) | llama family, tiny scale, 614400 tokens both levels |
| S3-d-unpaid | ✅ supported | Width d=192 does not improve val loss over d=128 even at 3x token budget (crossover, if any, lies above this scale) | t=+1.10 (n=48) | llama family, tiny scale; throughput cost -209 tok/s (t=-6.6) is significant |
| SRD-recall | ❌ retracted | Surprise-routed density improves needle recall over its falsifier lane |  | 2-block parity model, synthetic needle task |
| SRD-gate-conc | ⏳ pending | The SRD gate concentrates on needle positions (interpretation OPEN: retrieval router vs novelty detector) |  | 2-block parity model; needle tokens are distributionally distinct BY CONSTRUCTION in this design |
| EX-scorer | ✅ supported | Contribution-vs-mention scoring separates used from mentioned architecture flavors better than first-match, with zero wrong assertions | 0.882 [CI 0.75–1.0] | arXiv LaTeX sources; 47/65 verdicts correct, 0 wrong, abstention-first |

Reproduce any experimental row: `python tools/reproduce.py <id>`.
