# SRD rung 2 — pre-registered analysis

| cell | n | exact acc (control) | RSI | DCI | concentration | srd acc |
|---|---|---|---|---|---|---|
| distinct decoys=0 | 3 | 0.000 | +0.0095 ± 0.0046 | — | +0.5892 ± 0.0420 | 0.000 |
| distinct decoys=2 | 3 | 0.000 | -0.0008 ± 0.0029 | 0.971 | +0.6008 ± 0.0122 | 0.010 |
| indist decoys=0 | 3 | 0.000 | +0.0012 ± 0.0012 | — | -0.0032 ± 0.0011 | 0.010 |
| indist decoys=2 | 3 | 0.000 | +0.0043 ± 0.0055 | 1.889 | -0.0113 ± 0.0009 | 0.000 |

## Pre-registered predictions

- **HOLDS** — P1 RSI>0 in distinct/no-decoy: +0.0095 ± 0.0046
- **FAILS** — P2 RSI survives in-distribution: +0.0012 ± 0.0012
- **FAILS** — P3 concentration survives in-distribution: -0.0032 ± 0.0011
- **FAILS** — P4 decoys ignored (distinct): DCI 0.971 (<0.5 = retrieval router; ~1 = novelty detector)
- **FAILS** — P4 decoys ignored (indist): DCI 1.889 (<0.5 = retrieval router; ~1 = novelty detector)

## P5 — matched-density quality (accuracy, mean ± SE over all cells/seeds)

| policy | ρ=0.10 | ρ=0.25 |
|---|---|---|
| srd_top | 0.003 ± 0.003 | 0.003 ± 0.003 |
| random | 0.000 ± 0.000 | 0.003 ± 0.003 |
| positional | 0.005 ± 0.004 | 0.005 ± 0.004 |
| exact_ref (bound) | 0.005 | |
| linear_ref (bound) | 0.000 | |

- **FAILS** — P5 SRD beats BOTH baselines at BOTH densities (beating random but losing to positional is a negative result and gets published as one)

## Control-first check

- ⚠ distinct decoys=0: exact lane at 0.000 accuracy — cell is UNINFORMATIVE about SRD (pre-registered threat to validity).
- ⚠ distinct decoys=2: exact lane at 0.000 accuracy — cell is UNINFORMATIVE about SRD (pre-registered threat to validity).
- ⚠ indist decoys=0: exact lane at 0.000 accuracy — cell is UNINFORMATIVE about SRD (pre-registered threat to validity).
- ⚠ indist decoys=2: exact lane at 0.000 accuracy — cell is UNINFORMATIVE about SRD (pre-registered threat to validity).
