# Full-factorial analysis — 10 runs, 1 factors, seeds [1, 2, 3, 4, 5]

complete runs: 10/10

## best_val

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| attention | exact → swa | +0.01229 | 0.0123 | +1.00 | exact | no |

## loss_auc_norm

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| attention | exact → swa | -0.004812 | 0.000331 | -14.56 | swa | **YES** |

Full factorial: every two-way interaction above is
unconfounded. A signal interaction means the factors'
effects are NOT additive — read the pair's cell means
before acting on either main effect alone.
