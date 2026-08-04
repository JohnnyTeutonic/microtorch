# Full-factorial analysis — 20 runs, 1 factors, seeds [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

complete runs: 20/20

## best_val

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| attention | exact → swa | -0.04791 | 0.0114 | -4.19 | swa | **YES** |

## loss_auc_norm

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| attention | exact → swa | -0.003674 | 0.000825 | -4.45 | swa | **YES** |

## grad_spike_rate

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| attention | exact → swa | -1.053 | 0.832 | -1.26 | swa | no |

## tokens_per_second

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| attention | exact → swa | +169.8 | 41.1 | +4.13 | swa | **YES** |

Full factorial: every two-way interaction above is
unconfounded. A signal interaction means the factors'
effects are NOT additive — read the pair's cell means
before acting on either main effect alone.
