# Screening analysis — 36 runs, 7 factors, seeds [1, 2, 3]

complete runs: 36/36

## grad_spike_count

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| lr | 0.001 → 0.003 | +1.222 | 0.5 | +2.45 | 0.001 | **YES** |
| T | 128 → 256 | +1.111 | 0.507 | +2.19 | 128 | **YES** |
| d | 128 → 192 | +1 | 0.514 | +1.94 | 128 | no |
| optimizer | adamw → muon | -0.8889 | 0.52 | -1.71 | muon | no |
| heads | 4 → 8 | -0.6667 | 0.53 | -1.26 | 8 | no |
| layers | 2 → 4 | +0.5556 | 0.534 | +1.04 | 2 | no |
| batch | 2 → 4 | +0 | 0.542 | +0.00 | 2 | no |

## grad_init_transient

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| lr | 0.001 → 0.003 | +1.341 | 0.422 | +3.18 | 0.001 | **YES** |
| d | 128 → 192 | +1.266 | 0.428 | +2.95 | 128 | **YES** |
| batch | 2 → 4 | +1.045 | 0.445 | +2.35 | 2 | **YES** |
| layers | 2 → 4 | +0.9562 | 0.451 | +2.12 | 2 | **YES** |
| T | 128 → 256 | +0.8566 | 0.457 | +1.87 | 128 | no |
| heads | 4 → 8 | -0.1924 | 0.479 | -0.40 | 8 | no |
| optimizer | adamw → muon | -0.1204 | 0.48 | -0.25 | muon | no |

Screen signals (|t| >= 2) are candidates for the
resolution-V budget; a screen estimates MAIN effects only —
interactions are deliberately aliased and unmeasured here.
