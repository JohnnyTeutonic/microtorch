# Screening analysis — 20 runs, 1 factors, seeds [1, 2, 3, 4, 5]

complete runs: 20/20

## best_val

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| window | 16 → 32 | +0.03681 | 0.0147 | +2.50 | 16 | **YES** |

## loss_auc_norm

| factor | low → high | effect | SE | t | better | signal |
|---|---|---|---|---|---|---|
| window | 16 → 32 | +0.002869 | 0.00278 | +1.03 | 16 | no |

Screen signals (|t| >= 2) are candidates for the
resolution-V budget; a screen estimates MAIN effects only —
interactions are deliberately aliased and unmeasured here.
