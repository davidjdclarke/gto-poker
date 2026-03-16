# V10 Complete Report — GTO Solver

**Date:** March 15, 2026
**Status:** Complete
**Baseline:** v9-B0 (13-action grid, 100M iterations, 1,055,003 nodes)
**Strategy file:** `experiments/best/v9_B0_100M_allbots_positive.json`

---

## 1. Executive Summary

V10 achieved three out of five success criteria. The cycle's most important contribution was scientific: it proved that prior evaluation results were unreliable and established a corrected baseline. The confidence-aware bridge mapping is the highest-value practical improvement, delivering an 8x gain in gauntlet average without any retraining.

| Criterion | Met? | Evidence |
|-----------|------|---------|
| Validated baseline | **Yes** | B0 corrected from +677 to +28.5 avg under rigorous protocol |
| Solver-dynamics gain | **No** | Exponential(1.001) failed; polynomial untested |
| Bridge gain | **Yes** | `confidence_nearest` mapping: +225.1 avg (8x improvement) |
| Abstraction gain | **Deferred** | 16-action grid confirmed non-viable; structural changes need Cython rebuild |
| Causal clarity | **Yes** | B0 vs v7 EV decomposition complete; bugs identified and fixed |

**Key v10 results:**
- Corrected B0 average: **+28.5 bb/100** (nearest mapping) / **+225.1 bb/100** (confidence_nearest)
- WeirdSizingBot: improved from **-771 to -203** (+568 bb/100) via confidence mapping
- Two critical bugs found and fixed (16-action grid compatibility, OpponentProfile)
- 16-action grid confirmed non-viable: only 15% node coverage at 200M iterations
- Exploitability confirmed stable at **1.25 bb/100** under dual-protocol evaluation

---

## 2. Evaluation Protocol (v10 standard)

All final results in this report use the following protocol unless stated otherwise:

- **5,000 hands/bot** (prior v9 used 500 — shown to be unreliable)
- **3 seeds** (42, 123, 456) with per-bot mean, standard deviation, and 95% CI
- **No OpponentProfile** — disabled for pure GTO measurement (`--no-opponent-model`)
- **Grid auto-detection** — `detect_action_grid_from_strategy()` prevents action-count mismatches
- **bb=20**, starting stacks 10,000 chips (500 bb deep)

Exploitability uses dual settings:
- Standard: 3 seeds, 500 MC samples
- High-fidelity: 5 seeds, 1000 MC samples

---

## 3. Phase 0 — Measurement Hardening

### 3.1 Bug 1: 16-Action Grid Incompatibility (CRITICAL)

**Discovery:** The first 5k-hand gauntlet produced an average of -1000 bb/100 — wildly negative against all bots. Strategy hit rate was only **54.5%**, meaning nearly half of all decisions fell back to uniform random.

**Root cause:** The v9 code added three actions to `_postflop_actions()` in `abstraction.py` for the 16-action experiment:
- `BET_QUARTER_POT` (action 13)
- `BET_THREE_QUARTER_POT` (action 14)
- `BET_DOUBLE_POT` (action 15)

This expanded the standard postflop action menu from 8 actions (check/call + 6 bet sizes) to 11 actions (check/call + 9 bet sizes). The B0 strategy was trained on 13 actions (6 postflop bet sizes). When the match engine looked up a node:

```python
if len(avg_strategy) == len(actions):  # 8 != 11 → FAIL
    strategy = {actions[i]: avg_strategy[i] for i in range(len(actions))}
else:
    strategy = {a: 1.0/len(actions) for a in actions}  # Uniform fallback
```

Every postflop node failed the length check, producing uniform random play on all postflop streets.

**Fix:** Added three functions to `abstraction.py`:
- `detect_action_grid_from_strategy(trainer)` — samples postflop nodes; if any have 10+ actions, it's 16-action grid; 7-8 actions means 13-action
- `set_action_grid(size)` — sets module-level `_ACTION_GRID_SIZE` (13 or 16)
- Modified `_postflop_actions()` to check `_ACTION_GRID_SIZE` before including the 3 new actions

Auto-detection added to `engine.py:get_trainer()` so it runs on every strategy load.

**Impact:** Hit rate improved from 54.5% to 92%+. The remaining 8% misses are genuine (nodes not in strategy from unexplored game paths).

**Files changed:** `server/gto/abstraction.py`, `server/gto/engine.py`, `run_phase0_validation.py`

### 3.2 Bug 2: OpponentProfile Counterproductive

**Discovery:** After fixing the grid bug, CallStationBot still showed -995 bb/100 (with OpponentProfile) vs +72.5 (without). Testing all bots with and without OP:

| Bot | With OP | Without OP | Delta |
|-----|---------|-----------|-------|
| NitBot | +77 | +430 | -353 |
| AggroBot | +311 | +444 | -133 |
| CallStationBot | **-1154** | **+6** | **-1160** |
| WeirdSizingBot | -720 | -1009 | +289 |
| PerturbBot | -29 | +515 | -544 |

**Root cause:** `OpponentProfile` in `opponent_model.py` tracks opponent fold/call rates over a 50-decision rolling window. For EQ0-3 hands (weak), it multiplies bet actions by a combined fold/call adjustment:

```python
# Against CallStationBot (call_rate ≈ 0.90):
call_ramp = (0.90 - 0.50) / 0.40 = 1.0
call_mult = 1.0 - 1.0 * 0.95 = 0.05
# All bet actions multiplied by 0.05 → effectively eliminated
```

While directionally correct (don't bluff a station), the 0.05x multiplier is far too extreme. It eliminates ALL betting with weak hands, including semi-bluffs and thin value that are profitable even against callers. After renormalization, the GTO agent only checks or folds with EQ0-3 hands, surrendering all initiative.

**Fix:** Added `--no-opponent-model` flag to `run_eval_harness.py`. Module-level `_USE_OPPONENT_MODEL` controls whether `_play_one_matchup()` passes `OpponentProfile()` or `None` to GTOAgent.

**Files changed:** `run_eval_harness.py`, `run_phase0_validation.py`

### 3.3 Corrected Baseline Results

**Protocol:** 5k hands/bot, 3 seeds, no OP, 13-action grid auto-detected

| Bot | Mean (bb/100) | Std | 95% CI | Published v9 | Delta |
|-----|-------------|-----|--------|-------------|-------|
| NitBot | +119.9 | 72.1 | [+38.4, +201.5] | +718.5 | -598.6 |
| AggroBot | +339.9 | 67.6 | [+263.5, +416.4] | +893.4 | -553.5 |
| OverfoldBot | +6.7 | 25.2 | [-21.9, +35.2] | +115.9 | -109.2 |
| CallStationBot | +72.5 | 103.5 | [-44.7, +189.6] | +1071.4 | -998.9 |
| DonkBot | +409.6 | 102.3 | [+293.8, +525.4] | +1024.0 | -614.4 |
| WeirdSizingBot | **-770.9** | 63.8 | [-843.1, -698.7] | +220.4 | -991.3 |
| PerturbBot | +22.1 | 65.5 | [-52.0, +96.1] | +696.7 | -674.6 |
| **Average** | **+28.5** | | | **+677.2** | **-648.7** |

**Interpretation:**
- **6/7 bots positive or near-zero** — the core B0 strategy is sound
- **WeirdSizingBot is the dominant failure** at -771 with a tight CI — confirmed real, not noise
- Without WeirdSizing, average is **+162 bb/100**
- PerturbBot at +22 (near zero) confirms near-GTO play against standard sizes
- Published +677 was inflated ~650 bb/100 by 500-hand variance + grid bug + OP model

### 3.4 Exploitability Confirmation

| Setting | Mean | CI | Time |
|---------|------|-----|------|
| Standard (3-seed, 500 samples) | 1.2700 | +/- 0.0543 | 1003s |
| High-fidelity (5-seed, 1000 samples) | 1.2529 | +/- 0.0360 | 6323s |

**Per-phase breakdown (high-fidelity):**

| Phase | Mean | CI |
|-------|------|-----|
| Preflop | 1.0001 | +/- 0.0231 |
| Flop | 1.3865 | +/- 0.0296 |
| Turn | 1.3840 | +/- 0.0411 |
| River | 1.1908 | +/- 0.0313 |

Exploitability stable at ~1.25 bb/100 across both protocols. Flop and turn weakest (~1.38), preflop strongest (~1.00). Consistent with prior reported 1.2211. The B0 strategy quality is confirmed — remaining weakness is in bridge translation, not core strategy.

### 3.5 B0 vs v7 Causal Diagnostics

**Protocol:** 5k hands, seed=42, `detailed_tracking=True`, no OP

#### CallStationBot (B0: +185.4, v7: +144.5, delta: +40.9)

B0 is modestly better. The CallStation dashboard shows why:

| Leak Category | B0 Count | B0 Avg EV | v7 Count | v7 Avg EV |
|---------------|---------|----------|---------|----------|
| Slowplay (EQ5+ checked) | 1,824 | +5.8 bb | 1,770 | +5.4 bb |
| Bluff (EQ0-2 bet into station) | 600 | -25.0 bb | 563 | -29.1 bb |
| Thin value (EQ3-4 checked) | 5,884 | +0.7 bb | 5,545 | -0.0 bb |

B0 makes fewer but smarter bluffs (-25.0 avg vs -29.1) and extracts more thin value (+0.7 avg vs -0.0). The flop delta of +135.6 is the main driver — the 2x phase schedule gives B0 better flop convergence.

**Street EV delta:** preflop -69.7, **flop +135.6**, **turn +96.1**, river -26.9

#### NitBot (B0: +40.1, v7: -2.3, delta: +42.4)

B0 stops losing with trash hands: EQ0 bucket improved +7.1 bb avg. Bridge pain shows NitBot's tight raises map to `bet_2/3_pot` with 0.285 avg distance — moderate translation error but not catastrophic.

#### WeirdSizingBot (B0: -641.6, v7: -628.0, delta: -13.6)

Both strategies lose heavily — **no improvement B0 over v7**. The bridge pain map reveals why:

| Mapped Action | Count | Avg Distance | Max Distance | Concrete Range |
|---------------|-------|-------------|-------------|----------------|
| bet 2x pot | 519 | **1.342** | 1.426 | [0.574, 0.667] |
| overbet 1.25x | 543 | **0.696** | 0.799 | [0.451, 0.565] |
| bet 3/4 pot | 534 | 0.331 | 0.425 | [0.325, 0.466] |
| open raise | 2,101 | 0.262 | 0.384 | [0.500, 0.714] |
| bet half pot | 1,042 | 0.187 | 0.278 | [0.222, 0.340] |

**The `bet_2x_pot` mapping is catastrophic**: concrete bets of ~0.57x pot are being mapped to the abstract node for 2.0x pot bets — a 1.34x pot-fraction distance. The trained strategy for that node was computed against 2x pot overbets, but WeirdSizingBot is actually betting about half pot. The GTO agent is answering the wrong question entirely.

**By phase:** Flop avg distance 0.465, Turn 0.466, River 0.414, Preflop 0.263. Postflop streets suffer the worst translation.

### 3.6 Checkpoint Cliff Analysis

B0 intermediate checkpoints (10M-100M) were overwritten by the 16-action training. Two early B0 checkpoints survive:

| Iters | Type | Nodes | Coverage | Avg Weight |
|-------|------|-------|----------|-----------|
| 50,000 | B0 | 863,609 | 64.5% | 30,142 |
| 100,000 | B0 | 933,794 | 78.3% | 213,113 |

The 16-action run shows the structural convergence problem:

| Iters | Nodes | Coverage | Exploitability | Gauntlet |
|-------|-------|----------|---------------|----------|
| 10M | 1,962,590 | 0.0% | — | — |
| 20M | 1,974,695 | 0.0% | — | — |
| 30M | 1,982,607 | 11.3% | — | — |
| 50M | 1,991,509 | 13.5% | — | — |
| 100M | 1,998,261 | 14.8% | — | — |
| 150M | 1,999,601 | 15.0% | 39.99 | +226.8 |
| 200M | 1,999,902 | 15.1% | 41.40 | +318.0 |

**Key observations:**
- **0% coverage at 10-20M**: The averaging delay (~25M) hasn't expired yet, so strategy accumulation hasn't begun
- **20M→30M cliff**: Coverage jumps from 0% to 11.3% when averaging kicks in
- **Coverage plateaus at ~15%**: Even at 200M, 85% of nodes lack sufficient visits. The 16-action grid creates 2x more nodes (2.0M vs 1.05M) than compute can cover
- **Exploitability plateaus at ~40**: No improvement from 110M to 200M — structural ceiling

### 3.7 Phase 0 Success Criteria

| Criterion | Met? | Evidence |
|-----------|------|---------|
| Explain why B0 beat v7 | **Partial** | Flop convergence from 2x schedule; modest +40 bb/100 edge |
| Identify worst off-tree gaps | **Yes** | WeirdSizing bridge pain: 1.34 avg distance on bet_2x_pot |
| Trust protocol for comparison | **Yes** | 5k hands + 3 seeds + CIs now standard |
| B0 remains superior | **Yes** | +28.5 avg, 6/7 bots positive, beats v7 on key bots |

---

## 4. Phase 2 — Solver Dynamics

### 4.1 Experiment: Exponential Weight Schedule (base=1.001)

**Hypothesis (H1):** Dynamic weighting can speed convergence or improve final quality vs standard linear CFR+ averaging.

**Config:**
```bash
venv/bin/python train_gto.py 100000000 --fresh --workers 6 \
  --phase-schedule 2x --allin-dampen old \
  --weight-schedule exponential --weight-param 1.001 \
  --checkpoint-interval 10000000 --checkpoint-eval \
  --checkpoint-dir checkpoints_v10_exp
```

**Result: FAILED — exploitability 128.24 bb/100 (vs B0's 1.22)**

All strategies were uniform across all nodes — no convergence occurred. Every preflop hand type showed 25% fold / 25% call / 25% 3-bet / 25% all-in. Postflop nodes showed 17% across 6 donk actions.

**Root cause analysis:** The Cython inner loop computes strategy averaging weights as:

```cython
# cfr_fast.pyx line 920-928
weight = _weight_schedule_param ** (t - averaging_delay)
# With param=1.001, delay=25M, t=100M:
# weight = 1.001^75,000,000 ≈ 10^32,568
```

At 100M iterations with 25M averaging delay, the weight of the final iteration is ~10^32,568. This is so astronomically large that the cumulative strategy sum is completely dominated by the single last iteration's regret-matching output. That single-iteration strategy is unconverged noise (regret-matching over noisy regrets), not a meaningful average.

**The math:** For exponential weighting to be useful, the ratio of the last weight to the first should be bounded — say 10^3 to 10^6. With 75M effective iterations:

| Base | Final/First Ratio | Viable? |
|------|------------------|---------|
| 1.001 | 10^32,568 | No — only last iteration survives |
| 1.0001 | 10^3,257 | No |
| 1.00001 | 10^326 | No |
| 1.000001 | 10^33 | Marginal |
| 1.0000001 | 10^3 | **Yes** |

Practical exponential bases at HUNL scale must be ~1.0000001 or smaller. This is so close to 1.0 that it's barely distinguishable from linear weighting, which suggests exponential schedules may not be the right tool for this domain at these iteration counts.

**Alternative:** Polynomial weighting `(t-delay)^p` grows much more gently:
- `(75M)^2 ≈ 5.6×10^15` — large but all iterations contribute
- `(75M)^1.5 ≈ 6.5×10^11` — more moderate
- Linear `(75M)^1 = 75M` — current CFR+ default

**What the infrastructure proved:** The CLI flags, Cython weight schedule implementation, checkpoint logging, and experiment tracking all functioned correctly. The failure is purely in parameterization, not infrastructure.

**Confound noted:** The Cython trainer hardcodes the 16-action grid regardless of `_ACTION_GRID_SIZE` in Python. This run produced 1,998,352 nodes (16-action) rather than ~1,055,000 (13-action). A fair comparison against B0 would require rebuilding Cython with configurable action grids.

**Verdict:** Exponential(1.001) is non-viable at HUNL scale. H1 is neither confirmed nor rejected — one bad parameterization was tested. Polynomial schedule is the highest-priority candidate for v11. Recommend validating on OpenSpiel toy games first.

---

## 5. Phase 3 — Bridge Improvements

### 5.1 Motivation

Phase 0 established that WeirdSizingBot at -771 bb/100 is the dominant weakness, and the bridge pain map showed translation distances of 1.34 (bet_2x_pot) and 0.70 (overbet_1.25x). The v10 plan hypothesized (H2) that this is a bridge problem, not an action-count problem — the 16-action grid's failure to converge supported this.

The existing `confidence.py` module had all the components for a confidence-aware mapping but wasn't integrated into the default `nearest` path. The existing `blend` mapping used it but without passing the mapped action ID, limiting mismatch detection.

### 5.2 Implementation: `confidence_nearest` Mapping

**Design:** When facing an opponent bet, compute a confidence score. If confidence is low (mismatch is large, node is undertrained, strategy is uncertain), blend the trained strategy with a simple equity heuristic. Otherwise, trust the trained strategy fully.

**Confidence computation** (`confidence.py:compute_confidence()`):

Three signals combined into alpha ∈ [0, 0.50]:

1. **Visit confidence** — sigmoid of `log(node.strategy_sum.sum() + 1)`, centered at 10.0 with slope 2.0. Well-visited nodes → high confidence → low alpha.

2. **Strategy entropy** — `H(strategy) / H_max`. High entropy (near-uniform strategy) signals uncertainty → higher alpha.

3. **Action mismatch** — `|concrete_ratio - nominal_ratio| / max(nominal_ratio, 0.1)`, capped at 2.0 and normalized to [0,1]. Large concrete-to-abstract gap → higher alpha.

Combined formula:
```
alpha = mismatch * (1 - visit_conf) * (0.5 + 0.5 * entropy) + (1 - visit_conf) * 0.1
alpha = min(alpha, 0.50)
```

**Equity heuristic** (`confidence.py:equity_heuristic()`):

Simple fallback strategy by equity bucket:
- EQ0-1 (trash): fold 85% facing bet, check 85% otherwise
- EQ2-3 (weak): fold 50% facing bet, check 70% otherwise
- EQ4-5 (medium): call 75% facing bet, bet half pot 35% otherwise
- EQ6-7 (strong): raise for value, bet pot 50%

**Blending** (`confidence.py:blend_strategies()`):

```
pi_final = (1 - alpha) * pi_trained + alpha * pi_heuristic
```

Actions in heuristic but not in trained are added. Result is renormalized.

**Integration** (`match_engine.py`):

Added `confidence_nearest` to the mapping dispatch. Key difference from existing `blend` mode: passes `mapped_action_id` (from the abstract history) to `compute_confidence()` so the mismatch distance is correctly computed. Also adds a threshold — blending only activates when `alpha >= 0.05`, preserving the trained strategy for well-matched sizes.

```python
if self.mapping == "confidence_nearest" and alpha < 0.05:
    pass  # Skip — mismatch too small to matter
else:
    heuristic = equity_heuristic(eq_bucket, has_bet, ctx.phase)
    strategy = blend_strategies(strategy, heuristic, alpha)
```

### 5.3 Evaluation

**Stage 1: Quick comparison (2k hands, 3 seeds)**

Ran `nearest`, `confidence_nearest`, and `blend` against all 7 bots to establish direction:

| Mapping | WeirdSizing | Average |
|---------|------------|---------|
| nearest | -775 | +90 |
| **confidence_nearest** | **-315** | **+260** |
| blend | -486 | +254 |

`confidence_nearest` showed the strongest WeirdSizing improvement and highest average.

**Stage 2: Definitive gauntlet (5k hands, 3 seeds)**

Full v10 protocol on `confidence_nearest`:

| Bot | nearest | confidence_nearest | Delta | Explanation |
|-----|---------|-------------------|-------|-------------|
| NitBot | +119.9 | +200.3 | +80.4 | NitBot raises map with 0.285 avg distance; blending helps |
| AggroBot | +339.9 | +444.0 | +104.1 | 3-bet/4-bet sizes differ from abstract; blending smooths |
| OverfoldBot | +6.7 | -3.9 | -10.6 | Noise (both CIs include zero) |
| CallStationBot | +72.5 | +73.0 | +0.5 | Station calls everything — no unusual sizing to trigger blend |
| DonkBot | +409.6 | +717.6 | +308.0 | DonkBot uses unusual lead sizes → large mismatch triggers |
| **WeirdSizingBot** | **-770.9** | **-202.8** | **+568.1** | Catastrophic mismatch → heuristic much better than wrong node |
| PerturbBot | +22.1 | +347.2 | +325.1 | Random off-tree actions benefit from heuristic fallback |
| **Average** | **+28.5** | **+225.1** | **+196.6** | |

### 5.4 Why It Works

The bridge pain data from Phase 0 tells the causal story. WeirdSizingBot's concrete bets map to abstract actions at huge distances:

- **bet_2x_pot**: concrete 0.57x pot → nominal 2.0x pot = **1.34 distance**. The trained node expects a 2x overbet response; the actual bet is ~half pot. With `nearest`, the GTO agent plays its overbet-defense strategy (mostly fold or call) when it should be raising for value.

- **overbet_1.25x**: concrete 0.45-0.57x pot → nominal 1.25x pot = **0.70 distance**. Same problem at smaller scale.

With `confidence_nearest`, these large distances produce high alpha values (0.3-0.5), heavily weighting the equity heuristic. The heuristic plays fundamentally: fold trash, call medium, raise strong. This is a much more sensible response to a half-pot bet than the overbet-defense strategy.

Against bots with standard sizing (CallStationBot at 1/3 pot → mapped to half pot, 0.167 distance), alpha stays near zero and the trained strategy dominates. This is why CallStationBot shows no change — the mapping is working as designed.

### 5.5 Confidence_nearest CIs

| Bot | Mean | 95% CI |
|-----|------|--------|
| NitBot | +200.3 | [+132.7, +268.0] |
| AggroBot | +444.0 | [+339.5, +548.6] |
| OverfoldBot | -3.9 | [-30.9, +23.1] |
| CallStationBot | +73.0 | [-105.7, +251.7] |
| DonkBot | +717.6 | [+624.4, +810.8] |
| WeirdSizingBot | -202.8 | [-338.2, -67.3] |
| PerturbBot | +347.2 | [+287.7, +406.7] |

WeirdSizingBot CI is entirely negative — the improvement is real but -203 remains the worst matchup. Local subgame solving (Phase 3 B2, deferred) is the next step.

### 5.6 Verdict

**Hypothesis H2 confirmed:** WeirdSizing is primarily a bridge problem, not an action-count problem. `confidence_nearest` improved WeirdSizing by +568 bb/100 without any retraining, while the 16-action grid expansion (adding the exact sizes WeirdSizingBot uses) failed to converge and gained nothing.

Recommend making `confidence_nearest` the default mapping.

---

## 6. Phase 4 — Abstraction (Findings Only)

No new abstraction experiments were run. Key findings from Phase 0 inform future work:

1. **16-action grid non-viable at current compute:** 15% coverage at 200M = 85% of nodes undervisited. The tree is too sparse. Need either 500M+ iterations or selective expansion (one size in one context).

2. **13-action grid coverage excellent:** 99.7% of nodes well-visited at B0's 100M. Current abstraction is well-matched to compute budget.

3. **B0 has ~1.05M nodes, 16-action has ~2.0M:** The 2x node explosion from 3 extra actions is the direct cause of the coverage gap.

4. **Cython action grid is independent of Python:** `cfr_fast.pyx` has its own hardcoded action menus. Changing `_ACTION_GRID_SIZE` in Python doesn't affect training. A Cython rebuild with configurable grids is prerequisite for any abstraction experiments.

---

## 7. Phase 5 — Tooling

### 7.1 Infrastructure Built

| Tool | Purpose |
|------|---------|
| `run_phase0_validation.py` | Orchestrates validate/diagnose/cliff subcommands |
| `run_gauntlet_multiseed()` | Multi-seed gauntlet with per-bot CIs (in `run_eval_harness.py`) |
| `--seeds` flag | Comma-separated seeds for multi-seed evaluation |
| `--no-opponent-model` flag | Disables OP for pure GTO eval |
| `detect_action_grid_from_strategy()` | Auto-detects 13 vs 16 action grid from loaded strategy |
| `set_action_grid()` / `get_action_grid()` | Module-level grid configuration |

### 7.2 Deferred

OpenSpiel and PokerKit integrations deferred to v11. OpenSpiel would be valuable for validating weight schedule parameterizations on toy games before committing HUNL compute.

---

## 8. All Code Changes

| File | Change |
|------|--------|
| `server/gto/abstraction.py` | Added `_ACTION_GRID_SIZE`, `set_action_grid()`, `get_action_grid()`, `detect_action_grid_from_strategy()`. Modified `_postflop_actions()` to respect grid size — 13-action returns 6 bet sizes, 16-action returns 9. |
| `server/gto/engine.py` | `get_trainer()` now calls `detect_action_grid_from_strategy()` and `set_action_grid()` after loading strategy. |
| `eval_harness/match_engine.py` | Added `confidence_nearest` mapping mode. Passes `mapped_action_id` to `compute_confidence()`. Threshold at alpha >= 0.05 before blending. |
| `run_eval_harness.py` | Added `run_gauntlet_multiseed()` function. Added `_USE_OPPONENT_MODEL` module flag. Added `--seeds`, `--no-opponent-model` CLI args. Updated gauntlet dispatch and summary to handle multi-seed format. |
| `run_phase0_validation.py` | New file. Subcommands: `validate` (multi-seed gauntlet + dual exploitability), `diagnose` (B0 vs v7 EV decomposition + bridge pain), `cliff` (checkpoint coverage analysis), `all`. |
| `CLAUDE.md` | Updated current state to v10. Corrected gauntlet table. Added v10 changes section. Updated known issues. Added architecture gotchas 8-10 (grid detection, OP disabled, Cython grid independence). Added v10 results and plan references. |
| `docs/plans/ACTION_PLAN_v10.md` | New file. Full v10 research plan (13 sections, 5 phases). |

---

## 9. Artifacts Produced

| Artifact | Path | Contents |
|----------|------|----------|
| V10 action plan | `docs/plans/ACTION_PLAN_v10.md` | Full research plan |
| Phase 0 report | `docs/results/v10_phase0_report.md` | Bugs, corrected baseline, cliff |
| Phase 0 validation data | `docs/results/v10_phase0_validate.json` | Gauntlet + dual exploitability JSON |
| Phase 0 diagnostics data | `docs/results/v10_phase0_diagnose.json` | B0 vs v7 EV decomposition + bridge pain |
| Phase 0 cliff data | `docs/results/v10_phase0_cliff.json` | Per-checkpoint coverage metrics |
| confidence_nearest results | `docs/results/v10_confidence_nearest_5k.json` | Per-bot per-seed gauntlet data |
| V10 complete report | `docs/results/v10_complete_report.md` | This document |

---

## 10. Recommendations for v11

1. **Make `confidence_nearest` the default mapping.** It's strictly better than `nearest` (+197 bb/100 avg) and requires no retraining. Change `GTOAgent.__init__` default from `"nearest"` to `"confidence_nearest"`.

2. **Add configurable action grid to Cython.** Pass `action_grid_size` to `train_fast()` and use it in `get_actions()`. Required for fair solver-dynamics comparisons and 9-bucket experiments.

3. **Redesign or remove OpponentProfile.** Current model is counterproductive — 0.05x multiplier is too extreme. Options: (a) remove entirely, (b) cap minimum multiplier at 0.5, (c) require 200+ samples before adjusting, (d) validate against gauntlet before deploying.

4. **Test polynomial weight schedule.** `(t-delay)^2` is the best candidate: grows gently enough at 100M scale, weights all iterations, and is already implemented in Cython. Validate on OpenSpiel first.

5. **Investigate local refinement (V10-B2).** `confidence_nearest` reduced WeirdSizing from -771 to -203, but -203 is still the worst matchup. A mini-CFR at the decision point with the actual concrete bet size (as in Libratus subgame solving) could close the remaining gap.

6. **Selective action expansion.** Use bridge pain data to add one size in one context: the 0.57x→2.0x mapping on flop/turn is the worst offender. Adding a ~0.5x pot action specifically for facing-bet postflop nodes might help without doubling the tree.

---

## 11. Key Lessons

1. **Evaluation discipline matters more than solver tricks.** The biggest v10 win was discovering that published results were inflated by ~650 bb/100. Without fixing measurement first, every subsequent experiment would have been wasted.

2. **Small protocol improvements compound.** Multi-seed gauntlets + CIs + grid auto-detection transformed the project from "we think it's +677" to "it's +28.5 ± known bounds." This is the difference between guessing and knowing.

3. **Bridge quality > action count.** `confidence_nearest` improved the average by +197 bb/100 without any retraining. The 16-action grid expansion cost 200M iterations of compute and gained nothing. The literature (Brown & Sandholm, Libratus) said this would happen — subgame refinement beats action translation.

4. **500-hand gauntlets are unreliable.** The v10 plan correctly predicted this. At 500 hands, the standard error of bb/100 is ~130+ bb/100 — large enough to flip the sign of most results. 5,000 hands with multiple seeds is the minimum for branch-selection decisions.

5. **Parameterization matters as much as algorithm choice.** Exponential weighting is a valid technique used in the literature. At base=1.001 and 100M iterations, it produces garbage. The same algorithm at base=1.0000001 might work. Always validate parameters on small games first.

6. **Auto-detection prevents silent failures.** The 16-action grid bug was silent — it didn't crash, it just produced bad results that looked plausible at 500 hands. Grid auto-detection and hit-rate monitoring catch these failures immediately.
