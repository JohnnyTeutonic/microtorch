# SRD 2×2 — pre-registration: novelty detector or retrieval router?

*Registered 2026-08-03, BEFORE any harness code or runs. This document
is the commitment: hypotheses, design, metrics, seeds, and decision
rules are fixed here; results get appended below the line, never edited
above it. Lineage: SPARSE_ATTENTION.md (the recall-claim retraction and
the 5× gate-concentration replication this experiment interrogates).*

## The question

`SurpriseRoutedAttention`'s gate concentrates on needle positions
(replicated 5×) and its placement provably depends on predictor
information (shuffle falsifier, 5–6σ, twice). But the recall claim
failed replication. Two live explanations:

- **H-N (novelty detector):** the gate fires on locally surprising
  spans. Needles in our benchmark are out-of-distribution BY
  CONSTRUCTION, so concentration-on-needles is guaranteed by the
  benchmark's construct, not by retrieval semantics.
- **H-R (retrieval router):** the gate tracks retrieval-relevant
  structure beyond mere surprisal.

These are not exhaustive (both can be partly true); the design below
separates them. Independently, Arm B tests the re-theorized pay