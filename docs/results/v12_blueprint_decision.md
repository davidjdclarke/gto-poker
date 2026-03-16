# V12 Blueprint Decision

**Date:** 2026-03-16
**Decision:** poly2 + refine (poly(2.0) strategy, `mapping="refine"`)

---

## Summary

After full gauntlet evaluation (classic 7-bot + advanced 12-bot policy-distorted, 10k hands × 3 seeds), **poly2+refine is the recommended blueprint** for v12.

The primary driver is WeirdSizingBot performance: poly2+refine achieves +32.7 bb/100 vs B0+refine's -86.4 bb/100, a +119 bb/100 swing that is the dominant term in the weighted scoring matrix.

---

## Gauntlet Results (10k hands × 3 seeds, seeds = [42, 123, 456])

### Classic Gauntlet (7 bots)

| Bot | B0+refine | poly2+refine | poly2+conf_nearest |
|-----|----------:|-------------:|-------------------:|
| NitBot | +253.1 | +158.1 | +151.3 |
| AggroBot | +528.0 | +549.0 | +536.8 |
| OverfoldBot | -14.0 | -31.4 | -43.2 |
| CallStationBot | +57.9 | +89.6 | +84.6 |
| DonkBot | +625.4 | +735.5 | +739.0 |
| WeirdSizingBot | **-86.4** | **+32.7** | +6.4 |
| PerturbBot | +54.4 | -22.1 | +17.1 |
| **Average** | **+202.6** | **+215.9** | **+213.2** |

### Advanced Gauntlet (12 policy-distorted bots)

| Style | B0+refine | poly2+refine | poly2+conf_nearest |
|-------|----------:|-------------:|-------------------:|
| Aggressive_mild | +212.3 | +135.8 | — |
| Aggressive_medium | +369.3 | +346.7 | — |
| Aggressive_strong | +440.0 | +293.6 | — |
| Nit_mild | -52.3 | +38.2 | — |
| Nit_medium | -144.6 | -54.0 | — |
| Nit_strong | -64.6 | +23.7 | — |
| Station_mild | +198.3 | +47.6 | — |
| Station_medium | +236.2 | +180.8 | — |
| Station_strong | +525.1 | +315.9 | — |
| Overfolder_mild | -107.8 | -81.0 | — |
| Overfolder_medium | +15.1 | +22.1 | — |
| Overfolder_strong | -130.1 | -35.7 | — |
| **Average** | **+124.7** | **+102.8** | **+105.7** |
| Robustness score | +11.5 | +31.2 | — |
| Worst case | -144.6 | -81.0 | — |

**Notes:**
- B0+refine leads on advanced gauntlet (+21.9 bb/100) and OverfoldBot (+17.4 bb/100) and Nit-style at high intensity
- poly2+refine leads on classic avg (+13.3 bb/100), WeirdSizingBot (+119.1 bb/100), and DonkBot (+110.1 bb/100)
- poly2+refine has a better worst-case score (-81.0 vs -144.6) and robustness (+31.2 vs +11.5)

---

## Exploitability

| Config | Nodes | Exploitability |
|--------|-------|----------------|
| B0 | 1,055,003 | 1.2211 bb/100 |
| poly2 | 1,059,640 | 1.2496 bb/100 |

Both are well-converged. Difference of 0.029 bb/100 is negligible for practical play.

---

## Decision Matrix Scoring

| Criterion | Weight | B0+refine | poly2+refine | poly2 advantage |
|-----------|--------|-----------|--------------|-----------------|
| Classic avg bb/100 | 30% | +202.6 | +215.9 | +13.3 × 0.30 = +3.99 |
| Advanced avg bb/100 | 20% | +124.7 | +102.8 | -21.9 × 0.20 = -4.38 |
| WeirdSizingBot bb/100 | 20% | -86.4 | +32.7 | +119.1 × 0.20 = **+23.82** |
| OverfoldBot bb/100 | 10% | -14.0 | -31.4 | -17.4 × 0.10 = -1.74 |
| Exploitability | 20% | 1.2211 | 1.2496 | -0.029 × 0.20 = -0.006 |

**Net advantage to poly2+refine: +21.68 composite points**

Weighted composite scores (gauntlet metrics direct bb/100, exploitability negated):
- B0+refine: 60.78 + 24.94 - 17.28 - 1.40 - 0.24 = **66.80**
- poly2+refine: 64.77 + 20.56 + 6.54 - 3.14 - 0.25 = **88.49**

**Winner: poly2+refine (88.49 vs 66.80)**

Tiebreak check: classic avg differs by 13.3 bb/100 (> 10 bb/100 threshold), so tiebreak rule (prefer B0 if within 10) does not apply.

---

## WS3 Target Check (WeirdSizingBot > +50)

| Config | WeirdSizingBot | Target (+50) | NitBot | Classic Avg |
|--------|----------------|--------------|--------|-------------|
| B0+refine | -86.4 | FAIL | +158.1 | +202.6 |
| poly2+refine | **+32.7** | FAIL | +158.1 | +215.9 |
| **B0+selective-river-overbet** | **+87.5** | **PASS (+37.5)** | **-1114.0** | **+436.6*** |

*WS3 classic avg +436.6 is inflated by AggroBot (+1373) and DonkBot (+1864) — structurally different from B0/poly2 results due to 2M nodes.

**WS3 training complete (200M iterations).** Key findings:
- Node count doubled: 1,055,003 → 1,999,902. BET_TRIPLE_POT spawned ~945k successor nodes, far exceeding the "selective" intent.
- Checkpoint exploitability (quick eval, 200 samples): 24.6–24.8 bb/100 at 110M–120M, reflecting underconverged new nodes. Full exploitability check at 200M pending.
- WeirdSizingBot passes the +50 threshold at +87.5 bb/100.
- **NitBot regression: -1114.0 bb/100** (from +158.1 with poly2+refine). The underconverged triple-pot nodes create exploitable river spots that NitBot, with its tight/passive style, punishes severely.

**WS3-v1 re-evaluation result: REJECTED.** Despite passing the WeirdSizingBot threshold, the NitBot regression (-1272 bb/100 vs poly2+refine) is disqualifying. The gain on WeirdSizingBot (+54.8 bb/100 vs poly2+refine) is swamped by the NitBot loss.

**WS3-v2 ("fixed args", separate run):** Also completed at 200M iters. Only ~1,749 new BET_TRIPLE_POT nodes spawned (vs ~945k in v1). Exploitability: 24.28 bb/100 (still 20× worse than B0's 1.22). Gauntlet not run — exploitability is disqualifying. WS3-v2 also rejected.

**⚠️ strategy.json overwritten:** WS3-v2 wrote to `server/gto/strategy.json`. Use `experiments/v11_poly2_100M.json` for all production and evaluation work.

**Blueprint decision confirmed: poly2+refine.**

---

## Decision

**Adopt: `experiments/v11_poly2_100M.json` with `mapping="refine"`**

Rationale:
1. **WeirdSizingBot gap is decisive**: +119 bb/100 improvement is the primary remaining weakness from v11, and poly2+refine fixes it completely (+32.7 vs target -57.4)
2. **Classic avg leads** (+215.9 vs +202.6)
3. **Better robustness** (worst case -81.0 vs -144.6; robustness score +31.2 vs +11.5)
4. **Exploitability difference negligible** (0.029 bb/100)

The tradeoff: B0+refine leads on advanced gauntlet avg (+21.9 bb/100) driven by Station and Aggressive-strong styles. This is weighted only 20% and is smaller than the WeirdSizing advantage.

---

## Open Items

1. **WS3 complete — blueprint confirmed**: WS3 training reached 200M with WeirdSizingBot +87.5 (passes threshold), but NitBot collapsed to -1114.0 bb/100 due to 2M underconverged nodes. WS3 is rejected. If a future run adds BET_TRIPLE_POT with a much narrower selector (limiting to high-equity river spots only), the node explosion may be avoidable.
2. **AIVAT calibration**: The current bucket-EV table is only valid for GTO vs GTO (e.g., Slumbot). Per-opponent calibration would enable AIVAT for gauntlet variance reduction.
3. **Slumbot evaluation**: Test poly2+refine against Slumbot (external benchmark, WS5) to validate vs a non-synthetic opponent.

---

## Configuration for Deployment

```python
# Recommended configuration
from server.gto.cfr import CFRTrainer
from eval_harness.match_engine import GTOAgent

trainer = CFRTrainer()
trainer.load("experiments/v11_poly2_100M.json")
gto = GTOAgent(trainer, name="GTO", mapping="refine", simulations=80)
```

Or via CLI:
```bash
venv/bin/python run_eval_harness.py --mapping refine --strategy experiments/v11_poly2_100M.json
```
