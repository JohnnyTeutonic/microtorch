# Screening analysis — 36 runs, 7 factors, seeds [1, 2, 3]

complete runs: 36/36

## best_val

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| optimizer | adamw → muon | -0.3411 | 0.0549 | -6.21 | muon | **YES** |
| T | 128 → 256 | -0.2008 | 0.0725 | -2.77 | 256 | **YES** |
| batch | 2 → 4 | -0.1333 | 0.0769 | -1.73 | 4 | no |
| d | 128 → 192 | +0.1226 | 0.0774 | +1.58 | 128 | no |
| lr | 0.001 → 0.003 | +0.09413 | 0.0786 | +1.20 | 0.001 | no |
| layers | 2 → 4 | +0.04449 | 0.0799 | +0.56 | 2 | no |
| heads | 4 → 8 | -0.02494 | 0.0801 | -0.31 | 8 | no |

## loss_auc_norm

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| optimizer | adamw → muon | -0.05338 | 0.00496 | -10.76 | muon | **YES** |
| batch | 2 → 4 | -0.01774 | 0.00996 | -1.78 | 4 | no |
| T | 128 → 256 | -0.01656 | 0.01 | -1.65 | 256 | no |
| heads | 4 → 8 | -0.004859 | 0.0104 | -0.47 | 8 | no |
| layers | 2 → 4 | +0.003096 | 0.0104 | +0.30 | 2 | no |
| d | 128 → 192 | +0.002462 | 0.0104 | +0.24 | 128 | no |
| lr | 0.001 → 0.003 | +0.0007616 | 0.0104 | +0.07 | 0.001 | no |

## steps_to_half_gap

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| lr | 0.001 → 0.003 | -7.444 | 0.842 | -8.84 | 0.001 | **YES** |
| d | 128 → 192 | -4 | 1.37 | -2.93 | 128 | **YES** |
| T | 128 → 256 | +1.222 | 1.52 | +0.81 | 256 | no |
| optimizer | adamw → muon | +1.111 | 1.52 | +0.73 | muon | no |
| layers | 2 → 4 | -0.8889 | 1.52 | -0.58 | 2 | no |
| heads | 4 → 8 | -0.6667 | 1.53 | -0.44 | 4 | no |
| batch | 2 → 4 | -0.2222 | 1.53 | -0.15 | 2 | no |

## grad_spike_count

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| lr | 0.001 → 0.003 | +2.778 | 0.796 | +3.49 | 0.001 | **YES** |
| batch | 2 → 4 | +2.556 | 0.817 | +3.13 | 2 | **YES** |
| d | 128 → 192 | +1.222 | 0.903 | +1.35 | 128 | no |
| T | 128 → 256 | +1.222 | 0.903 | +1.35 | 128 | no |
| optimizer | adamw → muon | +1.111 | 0.908 | +1.22 | adamw | no |
| layers | 2 → 4 | -0.1111 | 0.927 | -0.12 | 4 | no |
| heads | 4 → 8 | +0 | 0.927 | +0.00 | 4 | no |

## tokens_per_second

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| d | 128 → 192 | -280 | 45.1 | -6.20 | 128 | **YES** |
| layers | 2 → 4 | -198.7 | 56.4 | -3.52 | 2 | **YES** |
| optimizer | adamw → muon | -148 | 60.8 | -2.43 | adamw | **YES** |
| T | 128 → 256 | -44.99 | 65.5 | -0.69 | 128 | no |
| lr | 0.001 → 0.003 | -33.9 | 65.6 | -0.52 | 0.001 | no |
| heads | 4 → 8 | -19.62 | 65.8 | -0.30 | 4 | no |
| batch | 2 → 4 | +8.273 | 65.9 | +0.13 | 4 | no |

Screen signals (|t| >= 2) are candidates for the
resolution-V budget; a screen estimates MAIN effects only —
interactions are deliberately aliased and unmeasured here.
