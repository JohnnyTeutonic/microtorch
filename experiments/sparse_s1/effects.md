# Full-factorial analysis — 6 runs, 1 factors, seeds [1, 2, 3]

complete runs: 6/6

## best_val

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| attention | exact → swa | -0.05074 | 0.0149 | -3.41 | swa | **YES** |

## loss_auc_norm

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| attention | exact → swa | -0.003554 | 0.0017 | -2.10 | swa | **YES** |

## grad_spike_rate

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| attention | exact → swa | -2.632 | 2.15 | -1.22 | swa | no |

## tokens_per_second

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| attention | exact → swa | -85.44 | 71.3 | -1.20 | exact | no |

Full factorial: every two-way interaction above is
unconfounded. A signal interaction means the factors'
effects are NOT additive — read the pair's cell means
before acting on either main effect alone.
