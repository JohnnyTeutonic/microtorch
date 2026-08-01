#!/usr/bin/env python3
"""Contribution-vs-mention benchmark for the flavor scorer (AUROC).

    python papers/flavor_bench.py            # download (cached) + evaluate
    python papers/flavor_bench.py --offline  # cached papers only

A paper MENTIONS many flavor alternatives; it USES one. This benchmark
measures whether score_flavors() separates the two, on real papers whose
architectures are public knowledge. For every (paper, field) with ground
truth, every candidate the text contains becomes one (score, label) pair
— label 1 iff the candidate is what the paper actually uses. AUROC over
those pairs is the discrimination number; the naive baseline is the old
first-match rule (score = match priority, what fetch.py did before).

Papers where the true value is NOT in our lattice (Primer: squared ReLU)
contribute only negative labels — the scorer's job there is to assert
nothing. Verdict accuracy is reported alongside AUROC: top-1 correctness
of what extract() would actually apply ("used"/"contested" fields).

Ground truth sources: the papers themselves (each architecture statement
is quotable) — Vaswani §3, LLaMA §2, PaLM §2, BLOOM §3.1, GPT-NeoX §2.1,
Pythia §2.1, RoFormer, ALiBi §3, Primer §4 (squared ReLU), T5 §2.1.

SAMPLE-SIZE CAVEAT (STUDIO_PLAN.md §13.1): 10 papers / 31 pairs is
enough to kill the observed failure modes, not enough for a stable
AUROC estimate — grow TRUTH toward 30-50 papers and report bootstrap
CIs (resampling papers, the independent unit) before quoting these
numbers anywhere load-bearing.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from fetch import FLAVOR, extract, fetch_source, score_flavors, split_sections

CACHE = pathlib.Path(__file__).parent / ".cache"

# paper -> field -> the value this paper USES (None = the true value is
# outside our lattice, so every in-lattice candidate is a negative).
TRUTH: dict[str, dict[str, str | None]] = {
    "1706.03762": {"norm": "layernorm", "activation": "relu",
                   "positional": "sinusoidal"},          # Transformer
    "2302.13971": {"norm": "rmsnorm", "activation": "swiglu",
                   "positional": "rope"},                # LLaMA
    "2104.09864": {"positional": "rope"},                # RoFormer
    "2108.12409": {"positional": "alibi"},               # ALiBi
    "2109.08668": {"norm": "rmsnorm",
                   "activation": None},                  # Primer (squared ReLU)
    "2204.02311": {"activation": "swiglu",
                   "positional": "rope"},                # PaLM
    "2211.05100": {"norm": "layernorm", "activation": "gelu",
                   "positional": "alibi"},               # BLOOM
    "2204.06745": {"norm": "layernorm", "activation": "gelu",
                   "positional": "rope"},                # GPT-NeoX
    "2304.01373": {"norm": "layernorm",
                   "positional": "rope"},                # Pythia
    "1910.10683": {"activation": "relu"},                # T5
}


def get_tex(arxiv_id: str, offline: bool) -> str | None:
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{arxiv_id}.tex"
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace")
    if offline:
        return None
    try:
        tex = fetch_source(arxiv_id)
    except Exception as e:                                # noqa: BLE001
        print(f"  [skip] {arxiv_id}: {e}")
        return None
    p.write_text(tex, encoding="utf-8", errors="replace")
    return tex


def auroc(pairs: list[tuple[float, int]]) -> float | None:
    """Rank-based AUROC with tie handling (average ranks)."""
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for pv in pos:
        for nv in neg:
            wins += 1.0 if pv > nv else 0.5 if pv == nv else 0.0
    return wins / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    scored: list[tuple[float, int]] = []   # the scorer
    naive: list[tuple[float, int]] = []    # old first-match priority
    per_field: dict[str, list[tuple[float, int]]] = {}
    # Grouped = one AUROC per (paper, field), averaged. This is the
    # DEPLOYED decision ("in this paper, does the used flavor outrank
    # the mentioned ones?"); pooled additionally measures cross-paper
    # score calibration, which the used/contested threshold relies on.
    grouped: list[tuple[list[tuple[float, int]], list[tuple[float, int]]]] = []
    verdict_ok, verdict_bad, contested = 0, [], 0
    n_papers = 0

    for arxiv_id, truths in TRUTH.items():
        tex = get_tex(arxiv_id, args.offline)
        if tex is None:
            continue
        n_papers += 1
        cands_by_field = score_flavors(split_sections(tex))
        arch = extract(arxiv_id, tex)
        for fieldname, true_val in truths.items():
            cands = cands_by_field.get(fieldname, [])
            prio = {v: -i for i, (v, _) in enumerate(FLAVOR[fieldname])}
            g_new, g_old = [], []
            for c in cands:
                label = 1 if c["value"] == true_val else 0
                scored.append((c["score"], label))
                naive.append((prio[c["value"]], label))
                per_field.setdefault(fieldname, []).append((c["score"], label))
                g_new.append((c["score"], label))
                g_old.append((prio[c["value"]], label))
            grouped.append((g_new, g_old))
            if args.verbose:
                pretty = ", ".join(f"{c['value']}:{c['score']}" for c in cands)
                print(f"  {arxiv_id} {fieldname:11} truth={true_val}  [{pretty}]")
            # verdict accuracy: what extract() would actually apply
            f = arch.fields.get(fieldname)
            if f is None or f.verdict is None:
                applied = None
            elif f.verdict == "contested":
                contested += 1
                applied = None
            else:
                applied = f.value
            if applied == true_val or (applied is None and true_val is None):
                verdict_ok += 1
            elif applied is None and true_val is not None:
                # abstained where a truth existed: counted separately —
                # honest but not wrong
                verdict_ok += 0
            else:
                verdict_bad.append((arxiv_id, fieldname, applied, true_val))

    print(f"\n== flavor benchmark: {n_papers} papers, "
          f"{len(scored)} (candidate,label) pairs ==")
    ga_new = [auroc(g) for g, _ in grouped]
    ga_old = [auroc(g) for _, g in grouped]
    ga_new = [a for a in ga_new if a is not None]
    ga_old = [a for a in ga_old if a is not None]
    g_new = sum(ga_new) / len(ga_new) if ga_new else None
    g_old = sum(ga_old) / len(ga_old) if ga_old else None
    a_new, a_old = auroc(scored), auroc(naive)
    print(f"grouped AUROC (per paper+field, the deployed decision): "
          f"scorer {g_new:.3f}   naive {g_old:.3f}   "
          f"({len(ga_new)} groups)")
    print(f"pooled AUROC (cross-paper calibration):                "
          f"scorer {a_new:.3f}   naive {a_old:.3f}")
    for fieldname, pairs in sorted(per_field.items()):
        a = auroc(pairs)
        print(f"  {fieldname:11} pooled "
              f"{'n/a (single class)' if a is None else f'{a:.3f}'} "
              f"({len(pairs)} pairs)")
    total = sum(len(t) for aid, t in TRUTH.items()
                if (CACHE / f"{aid}.tex").exists())
    print(f"verdicts: {verdict_ok}/{total} correct, {contested} contested, "
          f"{len(verdict_bad)} WRONG")
    for aid, fieldname, applied, true_val in verdict_bad:
        print(f"  WRONG: {aid} {fieldname}: applied {applied}, truth {true_val}")
    if g_new is not None and g_old is not None and g_new < g_old:
        print("REGRESSION: grouped AUROC under naive baseline")
        return 1
    if verdict_bad:
        print("FAIL: wrong assertions exist (worse than abstaining)")
        return 1
    print("BENCH-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
