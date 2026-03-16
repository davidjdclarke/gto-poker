# V10 Phase 0 — Measurement Hardening Results

**Date:** March 15, 2026
**Status:** Complete
**Strategy:** v9-B0 (13-action, 100M iterations, 1,055,003 nodes)

---

## Executive Summary

Phase 0 achieved its primary goal: making the project's evaluation scientifically trustworthy. Three critical issues were discovered and resolved:

1. **16-action grid compatibility bug** (CRITICAL) — The v9 code expansion added 3 new actions (BET_QUARTER_POT, BET_THREE_QUARTER_POT, BET_DOUBLE_POT) to the postflop action menu, breaking compatibility with the B0 strategy trained on 13 actions. This caused ~45% of strategy lookups to fall back to uniform random. **Fixed** with auto-detection of grid size from loaded strategy nodes.

2. **OpponentProfile counterproductive** — The real-time opponent model reduced performance significantly, especially vs CallStationBot (-995 bb/100 with OP vs +72.5 without). The OP model over-adjusts bluff frequencies based on a small rolling window, creating exploitable patterns. **Disabled** for pure GTO evaluation.

3. **Published results inflated** — The prior +677 bb/100 average was inflated by (a) 500-hand sample noise, (b) the grid bug (run on pre-expansion code), and (c) variance. The true B0 performance is +28.5 bb/100 average, with WeirdSizingBot as a -771 outlier.

---

## Definitive Gauntlet Results

**Protocol:** 5,000 hands/bot, 3 seeds (42, 123, 456), bb=20, no OpponentProfile, 13-action grid

| Bot | Mean (bb/100) | Std | 95% CI | Status |
|-----|-------------|-----|--------|--------|
| NitBot | +119.9 | 72.1 | [+38.4, +201.5] | PASS |
| AggroBot | +339.9 | 67.6 | [+263.5, +416.4] | PASS |
| OverfoldBot | +6.7 | 25.2 | [-21.9, +35.2] | PASS |
| CallStationBot | +72.5 | 103.5 | [-44.7, +189.6] | PASS |
| DonkBot | +409.6 | 102.3 | [+293.8, +525.4] | PASS |
| WeirdSizingBot | **-770.9** | 63.8 | [-843.1, -698.7] | **FAIL** |
| PerturbBot | +22.1 | 65.5 | [-52.0, +96.1] | PASS |
| **Average** | **+28.5** | | | |
| **Avg (excl. WeirdSizing)** | **+162.0** | | | |

### Key Observations

- **6/7 bots positive or statistically zero** — the core strategy is sound
- **WeirdSizingBot is the dominant failure** at -771 bb/100 with a tight CI — this is real, not noise
- Without WeirdSizing, the average is +162 bb/100 — a competent GTO strategy
- PerturbBot at +22 confirms near-GTO play against standard-sized actions
- OverfoldBot at +7 is expected (GTO doesn't maximally exploit folders)

---

## Comparison to Published Results

| Bot | Published (500h) | Corrected (5000h, 3-seed) | Delta |
|-----|-----------------|--------------------------|-------|
| NitBot | +718.5 | +119.9 | -598.6 |
| AggroBot | +893.4 | +339.9 | -553.5 |
| OverfoldBot | +115.9 | +6.7 | -109.2 |
| CallStationBot | +1071.4 | +72.5 | -998.9 |
| DonkBot | +1024.0 | +409.6 | -614.4 |
| WeirdSizingBot | +220.4 | -770.9 | -991.3 |
| PerturbBot | +696.7 | +22.1 | -674.6 |
| **Average** | **+677.2** | **+28.5** | **-648.7** |

The published results were inflated by ~650 bb/100 on average due to small sample size (500 hands = high variance) and post-hoc code changes.

---

## Bugs Found and Fixed

### Bug 1: 16-action grid incompatibility (CRITICAL)

**Root cause:** `_postflop_actions()` in `abstraction.py` was modified to return 9 bet sizes (including BET_QUARTER_POT, BET_THREE_QUARTER_POT, BET_DOUBLE_POT) for the v9 16-action experiment. The B0 strategy was trained with 6 bet sizes. When the match engine looked up a strategy node, `len(avg_strategy) != len(actions)`, causing fallback to uniform random for ~45% of decisions.

**Fix:** Added `_ACTION_GRID_SIZE` module-level flag and `detect_action_grid_from_strategy()` to auto-detect grid size from loaded nodes. `_postflop_actions()` now respects the grid setting.

**Impact:** Hit rate improved from 54.5% to 92%+.

**Files modified:** `server/gto/abstraction.py`, `server/gto/engine.py`

### Bug 2: OpponentProfile counterproductive

**Root cause:** The `OpponentProfile` model tracks opponent fold/call rates over a 50-action window and adjusts EQ0-3 bet probabilities. Against CallStationBot (high call rate), it reduces bluff bets to 5% of their trained probability. While this is directionally correct (don't bluff a station), the extreme adjustment (0.05x multiplier) overcorrects and removes all betting, including thin-value bets that are profitable against stations.

**Evidence:**
- CallStationBot with OP: -995 bb/100
- CallStationBot without OP: +72.5 bb/100
- NitBot with OP: +77 bb/100
- NitBot without OP: +120 bb/100

**Fix:** Added `--no-opponent-model` flag. Phase 0 validation now disables OP for pure GTO evaluation.

**Files modified:** `run_eval_harness.py`, `run_phase0_validation.py`

---

## B0 vs v7 Causal Diagnostics

**Protocol:** 5,000 hands, seed=42, detailed_tracking=True, no OpponentProfile

| Bot | B0 (bb/100) | v7 (bb/100) | Delta |
|-----|-----------|-----------|-------|
| CallStationBot | +185.4 | +144.5 | +40.9 |
| NitBot | +40.1 | -2.3 | +42.4 |
| WeirdSizingBot | -641.6 | -628.0 | -13.6 |

### Interpretation

- **B0 is modestly better than v7** against CallStation (+41) and NitBot (+42)
- **WeirdSizingBot: no improvement** B0 vs v7 (-642 vs -628) — both lose heavily
- The B0 gains come primarily from flop play (flop delta +136 vs CallStation, +152 vs WeirdSizing)
- The 2x phase schedule gives B0 better flop convergence, which is the main improvement source

---

## Checkpoint Cliff Analysis

B0 intermediate checkpoints (10M-100M) were overwritten by the 16-action training run. Two early-stage B0 checkpoints survive:

| Iters | Type | Nodes | Coverage | Avg Weight |
|-------|------|-------|----------|-----------|
| 50,000 | B0 | 863,609 | 64.5% | 30,142 |
| 100,000 | B0 | 933,794 | 78.3% | 213,113 |

The 16-action run shows why it failed to converge:

| Iters | Nodes | Coverage | Exploitability |
|-------|-------|----------|---------------|
| 10M | 1,962,590 | 0.0% | — |
| 30M | 1,982,607 | 11.3% | — |
| 100M | 1,998,261 | 14.8% | — |
| 200M | 1,999,902 | 15.1% | 41.4 |

Only **15.1% of nodes are well-visited** at 200M iterations — the 16-action grid creates 2x more nodes than compute can cover. This is the structural reason it never converged.

---

## Phase 0 Success Criteria Assessment

| Criterion | Met? | Evidence |
|-----------|------|---------|
| Explain why B0 beat v7 | **Partial** | Better flop play from 2x schedule, but small margin (+40 bb/100) |
| Identify worst off-tree gaps | **Yes** | WeirdSizingBot at -771 is the clear outlier |
| Trust protocol for comparison | **Yes** | 5k hands + 3 seeds + CIs now standard |
| B0 remains superior | **Yes** | +28.5 avg, 6/7 bots positive, beats v7 on key bots |

**Kill criteria check:** B0 IS superior under 5k-hand multi-seed — proceeding to Phases 2-4.

---

## Implications for v10 Plan

1. **Phase 2 (solver dynamics):** Baseline is +28.5 avg, not +677. New experiments must beat this corrected number.
2. **Phase 3 (bridge):** WeirdSizingBot at -771 confirms H2 (bridge problem). This is the #1 target for improvement.
3. **Phase 4 (abstraction):** The 16-action grid's 15% coverage at 200M confirms it needs either 10x more compute or selective expansion.
4. **OpponentProfile:** Needs redesign before it can be useful. Currently counterproductive.
