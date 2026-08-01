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


def canon_level(level):
    """LINKED factors (dict levels) are stringified in the rows by
    mtsweep; canonicalize manifest levels the same way so comparisons
    hold. Without this, a linked factor silently vanishes from the
    analysis."""
    return json.dumps(level, sort_keys=True) if isinstance(level, dict) else level


def level_label(level):
    if isinstance(level, dict):
        return ",".join(f"{k.split('.')[-1]}={v}" for k, v in sorted(level.items()))
    return str(level)


def main_effects(rows, factors, metric):
    """Per-factor: (effect, se, t, n_hi, n_lo, better_word)."""
    out = {}
    for fac, levels in factors.items():
        lo, hi = canon_level(levels[0]), canon_level(levels[1])
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
        lo_lab, hi_lab = level_label(levels[0]), level_label(levels[1])
        if metric in LOSS_LIKE:
            better = hi_lab if eff < 0 else lo_lab
        else:
            better = hi_lab if eff > 0 else lo_lab
        out[fac] = {"effect": eff, "se": se, "t": t,
                    "n": len(ys_hi) + len(ys_lo),
                    "levels": [lo_lab, hi_lab], "better": better,
                    "signal": abs(t) >= 2.0}
    return out


def interactions(rows, factors, metric):
    """Two-way interaction effects for a FULL factorial (every factor
    at 2 levels). With ±1 coding sA, sB per row, the AB interaction is

        effect_AB = mean(y | sA·sB = +1) − mean(y | sA·sB = −1)

    — the same two-group contrast shape as a main effect, so the same
    seed-spread SE applies. Only meaningful when the design crosses all
    pairs (grid); PB12 aliases interactions and must not report them."""
    names = list(factors.keys())
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            fa, fb = names[i], names[j]
            hi_a = canon_level(factors[fa][1])
            hi_b = canon_level(factors[fb][1])
            ka, kb = f"factor.{fa}", f"factor.{fb}"
            ys_pos, ys_neg = [], []
            for r in rows:
                if r.get(metric) is None or ka not in r or kb not in r:
                    continue
                sa = 1 if r[ka] == hi_a else -1
                sb = 1 if r[kb] == hi_b else -1
                (ys_pos if sa * sb > 0 else ys_neg).append(r[metric])
            if len(ys_pos) < 2 or len(ys_neg) < 2:
                continue
            eff = statistics.fmean(ys_pos) - statistics.fmean(ys_neg)
            se = math.sqrt(statistics.variance(ys_pos) / len(ys_pos) +
                           statistics.variance(ys_neg) / len(ys_neg))
            t = eff / se if se > 0 else float("inf") if eff else 0.0
            pair = f"{fa.split('.')[-1]} × {fb.split('.')[-1]}"
            out[pair] = {"effect": eff, "se": se, "t": t,
                         "n": len(ys_pos) + len(ys_neg),
                         "signal": abs(t) >= 2.0}
    return out


def is_full_factorial(sweep):
    return (sweep.get("design", "grid") == "grid" and
            all(len(v) == 2 for v in sweep.get("factors", {}).values()))


def report(sweep, rows, metrics):
    factors = sweep["factors"]
    full = is_full_factorial(sweep)
    kind = "Full-factorial" if full else "Screening"
    # strip the dotted prefixes for display but keep full keys for lookup
    lines = [f"# {kind} analysis — {len(rows)} runs, "
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
        if full:
            ints = interactions(complete, factors, metric)
            if ints:
                lines += ["", f"### {metric} — two-way interactions", "",
                          "| pair | effect | SE | t | signal |",
                          "|---|---|---|---|---|"]
                for pair, e in sorted(ints.items(),
                                      key=lambda kv: -abs(kv[1]["t"])):
                    sig = "**YES**" if e["signal"] else "no"
                    lines.append(
                        f"| {pair} | {e['effect']:+.4g} | {e['se']:.3g} "
                        f"| {e['t']:+.2f} | {sig} |")
    if full:
        lines += ["", "Full factorial: every two-way interaction above is",
                  "unconfounded. A signal interaction means the factors'",
                  "effects are NOT additive — read the pair's cell means",
                  "before acting on either main effect alone."]
    else:
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
    # Interactions: plant y = 5 - 0.5a - 0.3*sa*sb (+noise) on a full
    # 2^3 grid; recover AB ~ -0.6, keep AC null.
    rows2 = []
    for _ in range(120):
        a, b, c = (rng.choice([0, 1]) for _ in range(3))
        sa, sb = 2 * a - 1, 2 * b - 1
        y = 5.0 - 0.5 * a - 0.3 * sa * sb + rng.gauss(0, 0.1)
        rows2.append({"factor.a": a, "factor.b": b, "factor.c": c,
                      "best_val": y, "complete": True})
    ints = interactions(rows2, {"a": [0, 1], "b": [0, 1], "c": [0, 1]},
                        "best_val")
    iab, iac = ints["a × b"], ints["a × c"]
    assert abs(iab["effect"] + 0.6) < 0.1 and iab["signal"], iab
    assert not iac["signal"], iac
    # Linked (dict) levels: manifest carries dicts, rows carry the
    # stringified form — canonicalization must reconnect them.
    import json as _json
    lv = [{"data.T": 128, "train.steps": 1200},
          {"data.T": 256, "train.steps": 600}]
    rows3 = []
    for _ in range(40):
        pick = rng.choice([0, 1])
        y = 4.0 - 0.4 * pick + rng.gauss(0, 0.05)
        rows3.append({"factor.ctx": _json.dumps(lv[pick], sort_keys=True),
                      "best_val": y, "complete": True})
    ectx = main_effects(rows3, {"ctx": lv}, "best_val")["ctx"]
    assert abs(ectx["effect"] + 0.4) < 0.08 and ectx["signal"], ectx
    assert ectx["levels"][1] == "T=256,steps=600", ectx
    print(f"SELFTEST OK: planted main effect ({ea['effect']:+.3f} ~ -0.5, "
          f"t={ea['t']:+.1f}); null factor null (t={eb['t']:+.2f}); "
          f"planted interaction ({iab['effect']:+.3f} ~ -0.6, "
          f"t={iab['t']:+.1f}); null pair null (t={iac['t']:+.2f}); "
          f"linked dict levels canonicalized ({ectx['effect']:+.3f} ~ -0.4)")
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
