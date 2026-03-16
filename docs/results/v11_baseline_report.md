# V11 Baseline Report

**Date:** 2026-03-15
**Status:** Template (run `run_phase0_validation.py all --hands 5000` to fill)
**Blueprint:** v9-B0, 13-action grid, 100M iterations
**Mapping:** confidence_nearest (now default)
**OpponentProfile:** Disabled (retired v11)

---

## Baseline Configuration

| Parameter | Value |
|-----------|-------|
| Strategy file | `experiments/best/v9_B0_100M_allbots_positive.json` |
| Action grid | 13 (auto-detected) |
| Mapping | confidence_nearest |
| OpponentProfile | Disabled |
| Hands per bot | 5,000 |
| Seeds | 42, 123, 456 |
| Exploitability samples | 500 |

---

## How to Generate

```bash
# Full baseline (5k hands, 3 seeds, ~15 min)
venv/bin/python run_phase0_validation.py all --hands 5000

# Or manual gauntlet with v11 defaults (confidence_nearest, no OP)
venv/bin/python run_eval_harness.py \
    --gauntlet --hands 5000 --seeds 42,123,456 \
    --strategy experiments/best/v9_B0_100M_allbots_positive.json
```

Note: Since v11 defaults are now confidence_nearest and no-opponent-model,
the standard eval harness will automatically use the correct configuration.

---

## Expected Results (from v10 corrected baseline)

| Bot | confidence_nearest (v10) | Expected v11 |
|-----|-------------------------|-------------|
| NitBot | +200.3 | ~same |
| AggroBot | +444.0 | ~same |
| OverfoldBot | -3.9 | ~same |
| CallStationBot | +73.0 | ~same |
| DonkBot | +717.6 | ~same |
| WeirdSizingBot | -202.8 | ~same |
| PerturbBot | +347.2 | ~same |
| **Average** | **+225.1** | **~225** |

The v11 baseline should match v10 confidence_nearest results since no
blueprint retraining has occurred. Any differences indicate infrastructure
regressions from the v11 code changes.

---

## Exploitability (from v10)

| Metric | Value |
|--------|-------|
| Mean (3-seed) | 1.2211 bb/100 |
| Preflop | 1.0204 |
| CI (95%) | ±0.04 |

---

## V11 Infrastructure Checklist

- [x] Action grid auto-detected correctly (should be 13)
- [x] confidence_nearest is default (no explicit flag needed)
- [x] OpponentProfile disabled by default
- [x] Strategy hit rate logged
- [x] Grid size logged
- [x] Mapping mode logged

---

## Post-Baseline Experiments

Once baseline is locked, run these v11 experiments:

1. **B1: Polynomial weighting** — `train_gto.py --weight-schedule polynomial --weight-param 1.5 --action-grid 13`
2. **B2: Scheduled DCFR** — `train_gto.py --weight-schedule scheduled --weight-param 2.0 --action-grid 13`
3. **C2: Local refinement** — `run_eval_harness.py --gauntlet` with GTOAgent mapping="refine"
4. **D1: Selective action** — `add_selective_action('river', 'facing_bet', Action.BET_THREE_QUARTER_POT)` then retrain
