#!/usr/bin/env python3
"""Atlas stage 2 analysis: main effects from a screening sweep.

Reads an mtsweep out_root (atlas_rows.jsonl) plus the sweep description
and estimates, for every factor, the MAIN EFFECT on each metric:

    effect = mean(metric | factor at high level)
           - mean(metric | factor at low  level)

with a standard error from the run-level spread (for a balanced
orthogonal screen like PB12 the level groups are equal-sized, and
SE = sqrt(s+^2/n+ + s-^2/n-) over runs). |t| >= 2 is flagged as a
SCREEN SIGNAL — the factors worth a resolution-V budget; everything
else is reported, not hidden, per the Atlas discipline.

    python tools/atlas_analyze.py SWEEP.json [--metric best_val ...]
                                  [--md report.md]
    python tools/atlas_analyze.py --selftest

Metrics default to best_val plus the behavioural set that
atlas_extract computes. Sign convention: for loss-like metrics a
NEGATIVE effect means the high level is better; the report says so
in words next to every number.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

LOSS_LIKE = {"best_val", "final_train_loss", "loss_auc_norm",
             "loss_tail_std", "grad_spike_count"}
DEFAULT_METRICS = ["best_val", "loss_auc_norm", "steps_to_half_gap",
                   "grad_spike_count", "tokens_per_second"]


def load_rows(out_root):
    rows = []
    with open(os.path.join(out_root, "atlas_rows.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main_effects(rows, factors, metric):
    """Per-factor: (effect, se, t, n_hi, n_lo, better_word)."""
    out = {}
    for fac, levels in factors.items():
        lo, hi = levels[0], levels[1]
        key = f"factor.{fac}"
        ys_hi = [r[metric] for r in rows
                 if r.get(key) == hi and r.get(metric) is not None]
        ys_lo = [r[metric] for r in rows
                 if r.get(key) == lo and r.get(metric) is not None]
        if len(ys_hi) < 2 or len(ys_lo) < 2:
            continue
        eff = statistics.fmean(ys_hi) - statistics.fmean(ys_lo)
        v_hi = statistics.variance(ys_hi)
        v_lo = statistics.variance(ys_lo)
        se = math.sqrt(v_hi / len(ys_hi) + v_lo / len(ys_lo))
        t = eff / se if se > 0 else float("inf") if eff else 0.0
        if metric in LOSS_LIKE:
            better = str(hi) if eff < 0 else str(lo)
        else:
            better = str(hi) if eff > 0 else str(lo)
        out[fac] = {"effect": eff, "se": se, "t": t,
                    "n": len(ys_hi) + len(ys_lo),
                    "levels": [lo, hi], "better": better,
                    "signal": abs(t) >= 2.0}
    return out


def report(sweep, rows, metrics):
    factors = sweep["factors"]
    # strip the dotted prefixes for display but keep full keys for lookup
    lines = [f"# Screening analysis — {len(rows)} runs, "
             f"{len(factors)} factors, seeds {sweep.get('seeds')}", ""]
    complete = [r for r in rows if r.get("complete")]
    lines.append(f"complete runs: {len(complete)}/{len(rows)}")
    for metric in metrics:
        effs = main_effects(complete, factors, metric)
        if not effs:
            continue
        lines += ["", f"## {metric}", "",
                  "| factor | low → high | effect | SE | t | better | signal |",
                  "|---|---|---|---|---|---|---|"]
        for fac, e in sorted(effs.items(), key=lambda kv: -abs(kv[1]["t"])):
            name = fac.split(".")[-1]
            sig = "**YES**" if e["signal"] else "no"
            lines.append(
                f"| {name} | {e['levels'][0]} → {e['levels'][1]} "
                f"| {e['effect']:+.4g} | {e['se']:.3g} | {e['t']:+.2f} "
                f"| {e['better']} | {sig} |")
    lines += ["", "Screen signals (|t| >= 2) are candidates for the",
              "resolution-V budget; a screen estimates MAIN effects only —",
              "interactions are deliberately aliased and unmeasured here."]
    return "\n".join(lines)


def selftest():
    import random
    rng = random.Random(0)
    # Synthetic: factor A has a planted -0.5 effect on y, B has none.
    rows = []
    for _ in range(60):
        a = rng.choice([0, 1])
        b = rng.choice([0, 1])
        y = 5.0 - 0.5 * a + rng.gauss(0, 0.1)
        rows.append({"factor.train.a": a, "factor.train.b": b,
                     "best_val": y, "complete": True})
    effs = main_effects(rows, {"train.a": [0, 1], "train.b": [0, 1]},
                        "best_val")
    ea, eb = effs["train.a"], effs["train.b"]
    assert abs(ea["effect"] + 0.5) < 0.1, ea
    assert ea["signal"] and ea["better"] == "1", ea
    assert not eb["signal"], eb
    print(f"SELFTEST OK: planted effect recovered "
          f"({ea['effect']:+.3f} ~ -0.5, t={ea['t']:+.1f}); "
          f"null factor stayed null (t={eb['t']:+.2f})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", nargs="?")
    ap.add_argument("--metric", action="append")
    ap.add_argument("--md")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.sweep:
        ap.error("give the sweep.json (or --selftest)")
    with open(args.sweep, encoding="utf-8") as f:
        sweep = json.load(f)
    rows = load_rows(sweep["out_root"])
    text = report(sweep, rows, args.metric or DEFAULT_METRICS)
    print(text)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\nwrote {args.md}")


if __name__ == "__main__":
    main()
