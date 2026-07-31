#!/usr/bin/env python3
"""Atlas stage 1: the component taxonomy and the constrained grammar
(ARCHITECTURE_ATLAS.md sections 7 and 12).

The taxonomy declares, for each component SLOT, the legal alternatives
(with aliases), and the compatibility CONSTRAINTS between slots. It is
one shared object with three consumers:
  - corpus generation: sample() draws random VALID architectures
    (corpus source 5 — the correction for publication selection bias)
  - ablation definition: alternatives(slot) is the substitution lattice
    a Delta is measured against
  - validation: validate(spec) refuses illegal configurations before
    compute is spent on them (mtsweep calls this on every materialized
    spec)

Scoped HONESTLY to what the spec system can express today. Slots the
Atlas doc names that are not yet spec-switchable (norm, position,
activation, residual) are declared PLANNED with their lattice recorded,
so the taxonomy is ready the day the spec grows the knob — but sample()
and validate() only touch implemented slots.

    python tools/atlas_taxonomy.py --selftest
    python tools/atlas_taxonomy.py --sample 8 [--seed 3]
    python tools/atlas_taxonomy.py --validate spec.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys

# --------------------------------------------------------------------------
# The taxonomy. status: "implemented" slots are spec-expressible today;
# "planned" slots record the lattice for when the spec grows the knob.
TAXONOMY = {
    "family": {
        "status": "implemented",
        "path": None,  # chosen via arch.preset; see PRESET_OF_FAMILY
        "alternatives": ["gpt2", "llama"],
        "aliases": {"gpt-2": "gpt2", "decoder": "gpt2", "llama2": "llama"},
    },
    "attention": {
        "status": "implemented",
        "path": "arch.custom.attention",
        "alternatives": ["exact", "kimi", "srd", "attnres"],
        "aliases": {"softmax": "exact", "full": "exact",
                    "linear": "kimi", "kimi-linear": "kimi",
                    "surprise": "srd", "surprise-routed": "srd",
                    "attention-residuals": "attnres"},
    },
    "optimizer": {
        "status": "implemented",
        "path": "train.optimizer",
        "alternatives": ["adamw", "muon"],
        "aliases": {"adam": "adamw"},
    },
    "d": {"status": "implemented", "path": "arch.custom.d",
          "range": [64, 96, 128, 192, 256]},
    "layers": {"status": "implemented", "path": "arch.custom.layers",
               "range": [2, 3, 4, 6]},
    "heads": {"status": "implemented", "path": "arch.custom.heads",
              "range": [2, 4, 8]},
    "T": {"status": "implemented", "path": "data.T",
          "range": [64, 128, 256]},
    "lr": {"status": "implemented", "path": "train.lr",
           "range": [1e-3, 2e-3, 3e-3]},
    "batch": {"status": "implemented", "path": "train.batch",
              "range": [1, 2, 4]},
    # ---- planned lattices (ARCHITECTURE_ATLAS section 12) ----
    "norm": {"status": "planned",
             "alternatives": ["rmsnorm", "layernorm", "none"],
             "aliases": {"rms": "rmsnorm", "ln": "layernorm"}},
    "position": {"status": "planned",
                 "alternatives": ["rope", "learned", "sinusoidal", "nope"],
                 "aliases": {"rotary": "rope", "absolute": "learned"}},
    "activation": {"status": "planned",
                   "alternatives": ["swiglu", "gelu-mlp", "relu-mlp"],
                   "aliases": {"swish-glu": "swiglu", "gelu": "gelu-mlp"}},
    "residual": {"status": "planned",
                 "alternatives": ["pre-norm", "post-norm", "attnres"],
                 "aliases": {"preln": "pre-norm", "postln": "post-norm"}},
}

PRESET_OF_FAMILY = {"gpt2": "gpt2-nano", "llama": "llama-tiny"}

# Compatibility constraints, each a (predicate, message) over a flat
# {slot: value} assignment. The grammar's whole job is that sample()
# can never emit an assignment violating one of these.
CONSTRAINTS = [
    (lambda a: a["d"] % a["heads"] == 0,
     "d must divide by heads"),
    (lambda a: a["family"] != "llama" or a["attention"] == "exact",
     "llama family is exact-attention only (kimi/srd are gpt2-family lanes)"),
    (lambda a: a["family"] != "gpt2" or a["layers"] == 2,
     "gpt2 family is the 2-block parity model"),
    (lambda a: a["d"] // a["heads"] >= 8,
     "head_dim below 8 is degenerate at these scales"),
]


def canonical(slot, value):
    """Resolve an alias to its canonical alternative name."""
    t = TAXONOMY[slot]
    if isinstance(value, str):
        value = t.get("aliases", {}).get(value, value)
    return value


def alternatives(slot):
    """The substitution lattice for a slot (what a Delta is measured
    against)."""
    t = TAXONOMY[slot]
    return t.get("alternatives") or t.get("range")


def violations(assignment):
    out = []
    for pred, msg in CONSTRAINTS:
        try:
            if not pred(assignment):
                out.append(msg)
        except KeyError:
            pass  # slot not present in this assignment; nothing to check
    return out


def sample(n, seed=0):
    """Draw n random VALID assignments over the implemented slots —
    corpus source 5 (random valid architectures from the grammar)."""
    rng = random.Random(seed)
    impl = {s: t for s, t in TAXONOMY.items()
            if t["status"] == "implemented"}
    out = []
    while len(out) < n:
        a = {s: rng.choice(alternatives(s)) for s in impl}
        if not violations(a):
            out.append(a)
    return out


def assignment_to_spec(a, base=None):
    """Turn a sampled assignment into an mtstudio spec fragment."""
    spec = json.loads(json.dumps(base)) if base else {}
    spec.setdefault("arch", {})["preset"] = PRESET_OF_FAMILY[a["family"]]
    for slot, val in a.items():
        path = TAXONOMY[slot].get("path")
        if not path:
            continue
        d = spec
        keys = path.split(".")
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val
    return spec


def spec_assignment(spec):
    """Extract the implemented-slot assignment from a spec dict (for
    validate)."""
    def get(path, default=None):
        d = spec
        for k in path.split("."):
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d
    preset = get("arch.preset", "")
    family = "llama" if str(preset).startswith("llama") else "gpt2"
    a = {"family": family}
    defaults = {"d": 128, "layers": 2, "heads": 4, "T": 128,
                "lr": 3e-3, "batch": 1, "attention": "exact",
                "optimizer": "adamw"}
    if str(preset).startswith("kimi"):
        defaults["attention"] = "kimi"
    if str(preset).startswith("srd"):
        defaults["attention"] = "srd"
    for slot, dv in defaults.items():
        path = TAXONOMY[slot]["path"]
        a[slot] = canonical(slot, get(path, dv)) if path else dv
    return a


def validate_spec_file(path):
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    return violations(spec_assignment(spec))


def selftest():
    # canonicalisation
    assert canonical("attention", "softmax") == "exact"
    assert canonical("norm", "rms") == "rmsnorm"
    # violations caught
    bad = {"family": "llama", "attention": "kimi", "d": 100, "heads": 8,
           "layers": 2, "T": 128, "lr": 3e-3, "batch": 1, "optimizer": "adamw"}
    v = violations(bad)
    assert any("divide" in m for m in v), v
    assert any("llama" in m for m in v), v
    bad2 = dict(bad, family="gpt2", attention="kimi", d=128, layers=4)
    assert any("2-block" in m for m in violations(bad2))
    # sampler: everything valid, decent diversity
    s = sample(40, seed=3)
    assert len(s) == 40
    assert all(not violations(a) for a in s)
    assert len({json.dumps(a, sort_keys=True) for a in s}) > 25
    # round-trip: sampled assignment -> spec -> assignment
    a = s[0]
    spec = assignment_to_spec(a, base={"data": {"T": 999}})
    back = spec_assignment(spec)
    for slot in ("family", "attention", "optimizer", "d", "layers",
                 "heads", "T", "lr", "batch"):
        assert back[slot] == a[slot], (slot, back[slot], a[slot])
    # planned slots carry their lattices
    assert "nope" in alternatives("position")
    print("SELFTEST OK: aliases, constraints, sampler validity+diversity, "
          "spec round-trip, planned lattices")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validate")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.sample:
        for a in sample(args.sample, args.seed):
            print(json.dumps(a, sort_keys=True))
        return
    if args.validate:
        v = validate_spec_file(args.validate)
        if v:
            for m in v:
                print(f"VIOLATION: {m}")
            sys.exit(1)
        print("spec valid")
        return
    ap.error("pick one of --selftest / --sample N / --validate spec.json")


if __name__ == "__main__":
    main()
