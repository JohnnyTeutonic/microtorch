#!/usr/bin/env python3
"""The Atlas findings registry — architecture claims as DATA.

    python tools/atlas_findings.py validate          # schema + link check
    python tools/atlas_findings.py render [--md F]   # human-readable table
    python tools/atlas_findings.py advise "QUERY"    # cite-or-refuse advisor
    python tools/atlas_findings.py --selftest

atlas/findings.jsonl is the cumulative, machine-readable evidence base:
one row per claim, carrying its effect/SE/t, design, scope, STATUS
(supported | replicated | superseded | retracted | pending) and receipt
paths into the repo. Retractions and supersessions are first-class rows,
never deletions — a registry that visibly corrects itself is the point.

The advisor answers config questions FROM the registry with citations,
and REFUSES questions the evidence does not cover. It is deliberately
dumb: keyword-matched retrieval over the claims, no generation, no
interpolation beyond scope strings. An honest recommender beats a
fluent one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(ROOT, "atlas", "findings.jsonl")

STATUSES = {"supported", "replicated", "superseded", "retracted", "pending"}
REQUIRED = {"id", "date", "claim", "design", "scope", "status", "receipts"}


def load(path=REGISTRY):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SystemExit(f"line {i}: bad JSON: {e}")
    return rows


def validate(rows, root=ROOT):
    errs = []
    ids = set()
    for r in rows:
        missing = REQUIRED - set(r)
        if missing:
            errs.append(f"{r.get('id', '?')}: missing {sorted(missing)}")
        if r.get("status") not in STATUSES:
            errs.append(f"{r.get('id')}: bad status {r.get('status')!r}")
        if r.get("id") in ids:
            errs.append(f"duplicate id {r['id']}")
        ids.add(r.get("id"))
        for p in r.get("receipts", []):
            if not os.path.exists(os.path.join(root, p)):
                errs.append(f"{r['id']}: receipt missing on disk: {p}")
        m = r.get("manifest")
        if m and not os.path.exists(os.path.join(root, m)):
            errs.append(f"{r['id']}: manifest missing on disk: {m}")
    for r in rows:
        for key in ("superseded_by", "replicates"):
            v = r.get(key)
            if v and v not in ids:
                errs.append(f"{r['id']}: {key} -> unknown id {v}")
        for v in r.get("replicated_by", []):
            if v not in ids:
                errs.append(f"{r['id']}: replicated_by -> unknown id {v}")
        # A superseded/retracted row must point at what replaced it OR
        # explain itself: accountability is the schema.
        if r["status"] in ("superseded",) and not r.get("superseded_by"):
            errs.append(f"{r['id']}: superseded without superseded_by")
        if r["status"] == "retracted" and not (r.get("note") or
                                               r.get("superseded_by")):
            errs.append(f"{r['id']}: retracted without note")
    return errs


BADGE = {"supported": "✅", "replicated": "✅✅", "superseded": "↻",
         "retracted": "❌", "pending": "⏳"}


def render(rows):
    lines = ["# Atlas findings registry", "",
             f"{len(rows)} claims; every row carries its receipts. "
             "Retractions are rows, not deletions.", "",
             "| id | status | claim | evidence | scope |", "|---|---|---|---|---|"]
    for r in rows:
        ev = ""
        if "t" in r:
            ev = f"t={r['t']:+.2f} (n={r.get('runs', '?')})"
        elif "ci" in r:
            ev = f"{r.get('effect')} [CI {r['ci'][0]}–{r['ci'][1]}]"
        lines.append(f"| {r['id']} | {BADGE[r['status']]} {r['status']} | "
                     f"{r['claim']} | {ev} | {r['scope']} |")
    lines += ["", "Reproduce any experimental row: "
              "`python tools/reproduce.py <id>`."]
    return "\n".join(lines)


def advise(rows, query):
    """Keyword retrieval over live (non-retracted, non-superseded) claims;
    retracted/superseded rows are surfaced as WARNINGS when they match,
    because knowing what was corrected is part of the answer."""
    q = set(w.lower().strip("?,.") for w in query.split())
    scored = []
    for r in rows:
        text = (r["claim"] + " " + r["scope"] + " " + r.get("note", "")).lower()
        hits = sum(1 for w in q if len(w) > 2 and w in text)
        if hits:
            scored.append((hits, r))
    scored.sort(key=lambda x: -x[0])
    live = [r for _, r in scored if r["status"] in ("supported", "replicated")]
    dead = [r for _, r in scored if r["status"] in ("retracted", "superseded")]
    pend = [r for _, r in scored if r["status"] == "pending"]
    out = []
    if not scored:
        out.append("NO EVIDENCE in the registry for this question. The "
                   "registry covers: tiny-scale llama-family training "
                   "factors, SRD needle behaviour, and arXiv flavor "
                   "extraction. Absence of evidence is the honest answer.")
        return "\n".join(out)
    for r in live[:3]:
        ev = f" [t={r['t']:+.2f}, {r.get('runs', '?')} runs]" if "t" in r else ""
        out.append(f"• {r['claim']}{ev}  ({r['id']}; scope: {r['scope']})")
    for r in pend[:2]:
        out.append(f"⏳ OPEN: {r['claim']} ({r['id']} — experiment running/queued)")
    for r in dead[:2]:
        out.append(f"⚠ CORRECTED RECORD: '{r['claim']}' was {r['status']} — "
                   f"{r.get('note', 'see ' + str(r.get('superseded_by')))[:140]}")
    if live:
        out.append("Every claim above is scoped: nothing here licenses "
                   "extrapolation beyond the stated params/budget.")
    return "\n".join(out)


def selftest():
    rows = load()
    errs = validate(rows)
    assert not errs, errs
    assert any(r["status"] == "retracted" for r in rows)
    assert any(r["status"] == "pending" for r in rows)
    # advisor: cites for a covered question
    a = advise(rows, "which optimizer should I use at tiny scale?")
    assert "S3-muon" in a or "S2-muon" in a, a
    # surfaces the corrected record for the context question
    a2 = advise(rows, "does longer context help val loss?")
    assert "CORRECTED" in a2 or "S3-ctx-null" in a2, a2
    # refuses what it cannot know
    a3 = advise(rows, "quantization deployment latency on mobile GPUs")
    assert "NO EVIDENCE" in a3, a3
    print(f"SELFTEST OK: {len(rows)} findings validate (links + receipts on "
          "disk), advisor cites, surfaces corrections, and refuses")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["validate", "render", "advise"])
    ap.add_argument("query", nargs="?")
    ap.add_argument("--md")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    rows = load()
    if args.cmd == "validate":
        errs = validate(rows)
        for e in errs:
            print("INVALID:", e)
        print("OK" if not errs else f"{len(errs)} problems")
        sys.exit(1 if errs else 0)
    if args.cmd == "render":
        text = render(rows)
        print(text)
        if args.md:
            with open(args.md, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        return
    if args.cmd == "advise":
        if not args.query:
            ap.error("advise needs a QUERY")
        print(advise(rows, args.query))
        return
    ap.error("pick validate | render | advise (or --selftest)")


if __name__ == "__main__":
    main()
