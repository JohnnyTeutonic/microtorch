# Full-factorial analysis — 48 runs, 4 factors, seeds [1, 2, 3]

complete runs: 48/48

## best_val

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| optimizer | adamw → muon | -0.3085 | 0.0418 | -7.38 | muon | **YES** |
| lr | 0.001 → 0.003 | +0.1339 | 0.0585 | +2.29 | 0.001 | **YES** |
| ctx | T=128,steps=1200 → T=256,steps=600 | -0.08913 | 0.0604 | -1.48 | T=256,steps=600 | no |
| d | 128 → 192 | +0.06737 | 0.061 | +1.10 | 128 | no |

### best_val — two-way interactions

| pair | effect | SE | t | signal |
|---|---|---|---|---|
| lr × optimizer | -0.1752 | 0.0561 | -3.12 | **YES** |
| d × optimizer | -0.09843 | 0.06 | -1.64 | no |
| optimizer × ctx | -0.06157 | 0.0611 | -1.01 | no |
| lr × ctx | -0.01782 | 0.0617 | -0.29 | no |
| d × ctx | +0.008693 | 0.0618 | +0.14 | no |
| d × lr | +0.001792 | 0.0618 | +0.03 | no |

## loss_auc_norm

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| optimizer | adamw → muon | -0.05284 | 0.00425 | -12.43 | muon | **YES** |
| ctx | T=128,steps=1200 → T=256,steps=600 | +0.008355 | 0.00879 | +0.95 | T=128,steps=1200 | no |
| lr | 0.001 → 0.003 | +0.006286 | 0.00883 | +0.71 | 0.001 | no |
| d | 128 → 192 | +0.001479 | 0.00887 | +0.17 | 128 | no |

### loss_auc_norm — two-way interactions

| pair | effect | SE | t | signal |
|---|---|---|---|---|
| lr × optimizer | -0.02217 | 0.00825 | -2.69 | **YES** |
| d × optimizer | -0.01059 | 0.00874 | -1.21 | no |
| optimizer × ctx | -0.006076 | 0.00883 | -0.69 | no |
| lr × ctx | -0.004596 | 0.00885 | -0.52 | no |
| d × ctx | -0.002494 | 0.00887 | -0.28 | no |
| d × lr | +0.001919 | 0.00887 | +0.22 | no |

## grad_spike_count

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| ctx | T=128,steps=1200 → T=256,steps=600 | -3.875 | 1.4 | -2.78 | T=256,steps=600 | **YES** |
| d | 128 → 192 | +3.125 | 1.44 | +2.18 | 128 | **YES** |
| lr | 0.001 → 0.003 | +1.708 | 1.49 | +1.15 | 0.001 | no |
| optimizer | adamw → muon | -1.208 | 1.5 | -0.81 | muon | no |

### grad_spike_count — two-way interactions

| pair | effect | SE | t | signal |
|---|---|---|---|---|
| lr × optimizer | -5.208 | 1.3 | -4.01 | **YES** |
| d × ctx | -2.125 | 1.48 | -1.44 | no |
| d × optimizer | -1.458 | 1.49 | -0.98 | no |
| d × lr | +1.292 | 1.5 | +0.86 | no |
| optimizer × ctx | -0.7917 | 1.5 | -0.53 | no |
| lr × ctx | -0.2083 | 1.51 | -0.14 | no |

## grad_init_transient

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| lr | 0.001 → 0.003 | +1.971 | 0.388 | +5.07 | 0.001 | **YES** |
| d | 128 → 192 | +1.556 | 0.427 | +3.64 | 128 | **YES** |
| ctx | T=128,steps=1200 → T=256,steps=600 | +0.3729 | 0.482 | +0.77 | T=128,steps=1200 | no |
| optimizer | adamw → muon | -0.3129 | 0.483 | -0.65 | muon | no |

### grad_init_transient — two-way interactions

| pair | effect | SE | t | signal |
|---|---|---|---|---|
| optimizer × ctx | +0.8002 | 0.47 | +1.70 | no |
| lr × optimizer | -0.2043 | 0.484 | -0.42 | no |
| lr × ctx | -0.1698 | 0.484 | -0.35 | no |
| d × lr | -0.1591 | 0.484 | -0.33 | no |
| d × ctx | -0.1103 | 0.485 | -0.23 | no |
| d × optimizer | +0.00584 | 0.485 | +0.01 | no |

## tokens_per_second

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| d | 128 → 192 | -208.7 | 31.7 | -6.59 | 128 | **YES** |
| optimizer | adamw → muon | -68.3 | 43 | -1.59 | adamw | no |
| ctx | T=128,steps=1200 → T=256,steps=600 | -61.35 | 43.2 | -1.42 | T=128,steps=1200 | no |
| lr | 0.001 → 0.003 | -55.75 | 43.4 | -1.29 | 0.001 | no |

### tokens_per_second — two-way interactions

| pair | effect | SE | t | signal |
|---|---|---|---|---|
| d × lr | +117.8 | 40.6 | +2.90 | **YES** |
| lr × optimizer | -78.01 | 42.6 | -1.83 | no |
| optimizer × ctx | +59.66 | 43.3 | +1.38 | no |
| d × ctx | +51.83 | 43.5 | +1.19 | no |
| d × optimizer | +39.53 | 43.8 | +0.90 | no |
| lr × ctx | -16.26 | 44.1 | -0.37 | no |

Full factorial: every two-way interaction above is
unconfounded. A signal interaction means the factors'
effects are NOT additive — read the pair's cell means
before acting on either main effect alone.
