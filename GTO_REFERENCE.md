# GTO Poker Solver - Architecture & Results Reference

## Overview

A professional-grade Counterfactual Regret Minimization (CFR+) solver for heads-up No-Limit Texas Hold'em. The solver computes approximate Nash Equilibrium strategies offline using Cython-accelerated training, which are then served to AI opponents in a live game via a web server.

Designed to compete at the highest levels of online Texas Hold'em.

---

## System Architecture

```
Live Game                         Offline Training
---------                         ----------------
server/gto/engine.py              train_gto.py
  |                                 |
  +- Loads strategy.json            +- Creates CFRTrainer
  +- Buckets live hand              +- Runs CFR+ (Cython-accelerated)
  +- Maps betting history           +- tqdm progress bar (~200 updates)
  |  to abstract actions            +- Saves strategy.json
  +- Samples from strategy          +- Logs to experiments/matrix.json
     distribution                   |
                                  run_experiment.py
                                    |
                                    +- Named experiments with eval
                                    +- Exploitability (3-seed avg)
                                    +- Strategy audit
                                    +- Comparison tools
```

### Data Flow (Live Game)

1. **Player hand + community cards** arrive at `engine.py`
2. `equity.py` computes E[HS^2] and classifies hand type
3. Hand is mapped to a **2D bucket** (equity x hand_type)
4. Betting history is abstracted to the solver's action space
5. `cfr.py` looks up the average strategy for that info set
6. Engine **samples** an action from the strategy distribution
7. Abstract action is converted to a concrete bet amount

---

## Core Modules

### `server/gto/cfr.py` - CFR+ Trainer

The heart of the solver. Implements CFR+ (Tammelin 2014) with external sampling MCCFR (Lanctot et al. 2009).

**Key classes:**

- **`CFRNode`** - Stores regret sums and strategy sums for one information set. Implements regret-matching+ (regrets floored at 0) and linear-weighted strategy averaging.
- **`CFRTrainer`** - Orchestrates training across all four streets. Manages the node table (`dict[str, CFRNode]`), sampling, and traversal.

**Training loop (per iteration):**

1. For each street (preflop, flop, turn, river):
   - Sample correlated bucket pairs for both players
   - Run external sampling MCCFR for both traversers (alternating updates)
2. At street boundaries, continuation value uses next-street bucket comparison
3. CFR+ averaging: weight = max(T - delay, 0)

**Phase schedule:** Flop and turn get 2x training iterations per cycle. The Cython hot path (`cfr_fast.pyx`) has its own hardcoded 2x schedule — the Python `PHASE_SCHEDULE` is only used when Cython is unavailable.

**All-in regret dampening:** All-in regrets are scaled by 0.7x when `raise_count < 2`, discouraging premature shoves. Note: the next retrain will tighten this to `raise_count == 0` only (see `docs/ACTION_PLAN_v7.md`).

**Cython acceleration:** When `cfr_fast.pyx` is built, training automatically uses the Cython inner loop (56x speedup). Falls back to pure Python if not available.

**Correlated bucket sampling:**
- Hand type is **fixed** across all streets (structural property of hole cards)
- Equity bucket **drifts** via random walk (+/-1 or stay) between streets
- This models how community cards change hand strength while hand structure persists

**Parallel training:** `train_gto.py --workers N` spawns N parallel worker processes with shared-memory node tables. Includes tqdm progress bar for both warmup (sequential) and parallel phases.

**Persistence:** `save()` / `load()` serialize the full node table (regret_sum, strategy_sum) to JSON. Strategy version tracking (currently v5) ensures outdated formats trigger retraining.

### `server/gto/cfr_fast.pyx` - Cython-Accelerated CFR+

Pure C inner loop for the CFR+ traversal. All hot-path operations run without Python overhead.

**Key design:**
- **Flat `NodeData` struct array** with int64-encoded infoset keys for O(1) lookup
- **Key encoding:** `make_key()` packs phase(2 bits), position(1 bit), bucket(8 bits), history(4 bits each), hlen(4 bits) into int64
- **Dynamic pool growth:** starts at 500k nodes, doubles on overflow
- **Chunked training:** `train_fast()` accepts batch sizes for progress callback integration
- **Build:** `venv/bin/python setup_cython.py build_ext --inplace`
- **Compiler flags:** `-O3 -march=native -ffast-math`

**Performance:** 0.04 ms/iteration (vs ~2-3 ms Python). 5M iterations in ~3.5 min.

### `server/gto/abstraction.py` - Game Abstraction

Reduces the poker game to a tractable size through two abstractions:

**Card Abstraction - 2D Bucketing (v5):**

| Dimension | Buckets | Source |
|-----------|---------|--------|
| Equity | 8 | E[HS^2] quantized to 0-7 |
| Hand Type | 15 | Structural classification |
| **Total** | **120** | 8 x 15 |

**Hand Types (15 categories):**

| ID | Type | Examples |
|----|------|----------|
| 0 | PREMIUM_PAIR | AA, KK, QQ |
| 1 | HIGH_PAIR | JJ, TT |
| 2 | MID_PAIR | 99-66 |
| 3 | LOW_PAIR | 55-22 |
| 4 | STRONG_BROADWAY | AK, AQ (suited) |
| 5 | BROADWAY | KQ, QJ, KJ (suited) |
| 6 | SUITED_ACE | Axs (non-broadway) |
| 7 | HIGH_SUITED_CONNECTOR | T9s, 98s, 87s |
| 8 | LOW_SUITED_CONNECTOR | 65s, 54s, 43s |
| 9 | SUITED_GAPPER | T8s, 97s, 86s |
| 10 | STRONG_OFFSUIT_ACE | AK, AQ, AJ (offsuit) |
| 11 | WEAK_OFFSUIT_ACE | AT-A2 (offsuit) |
| 12 | OFFSUIT_BROADWAY | KJo, QTo, KQo |
| 13 | SUITED_TRASH | Low suited non-connectors |
| 14 | TRASH | Everything else |

**Action Abstraction - Phase-Aware Menus:**

*Preflop:*

| Situation | Available Actions |
|-----------|-------------------|
| SB facing BB (no raises) | Fold, Call (limp), Open Raise (~2.5bb), All-in |
| BB after limp (no raises, no bet) | Check, Open Raise (~2.5bb), All-in |
| Facing open | Fold, Call, 3-bet (~3x), All-in |
| Facing 3-bet | Fold, Call, 4-bet (~2.2x), All-in |
| Facing 4-bet+ | Fold, Call, All-in |

*Postflop:*

| Situation | Available Actions |
|-----------|-------------------|
| No bet | Check, 1/3 pot, Half-pot, 2/3 pot, Pot, Overbet, All-in |
| Facing bet | Fold, Call, 1/3 pot raise, Half-pot raise, 2/3 pot raise, Pot raise, Overbet raise, All-in |
| Donk bet (v6) | Donk small, Donk medium (OOP leading into preflop aggressor) |

Max 4 raises per street.

**Information Set Key Format (v5):** `{phase}:{position}:{bucket}:{action_history}`
- Position: `oop` (out of position, acts first postflop) or `ip` (in position)
- Example: `preflop:oop:45:` = preflop, OOP, bucket 45, empty history (first to act)
- Example: `preflop:ip:72:6` = preflop, IP, bucket 72, facing open raise (action 6)

### `server/gto/equity.py` - Hand Evaluation

**E[HS^2] (Expected Hand Strength Squared):**
- For non-river streets: samples future community cards, computes equity at each, squares and averages
- Captures both current strength and improvement potential
- Better than raw equity for card abstraction (Zinkevich et al. 2007)
- River: E[HS^2] = equity^2 (no more cards to come)

**`hand_strength_bucket()`** combines:
1. E[HS^2] -> quantized to equity bucket (0-7)
2. `classify_hand_type()` -> hand type (0-14)
3. `make_bucket()` -> single index (equity * 15 + hand_type)

### `server/gto/engine.py` - Live Game Bridge

Translates between the real game and the abstract solver:

1. **`get_trainer()`** - Lazy-loads the strategy (file or train on first use)
2. **`gto_decide()`** - Main entry point for AI decisions:
   - Computes 2D bucket from actual cards (200 MC sims)
   - Maps concrete betting history to abstract actions
   - Looks up strategy distribution
   - Samples an action (mixed strategy)
   - Converts abstract action to concrete bet amount
3. **`_to_concrete_action()`** - Maps abstract sizing to actual chip amounts

### `server/gto/exploitability.py` - Strategy Quality Measurement

**Kuhn Poker (exact):** Full game tree traversal with opponent card distribution tracking. Used for correctness validation.

**Multi-street (Monte Carlo):** Samples random bucket pairs and computes best response value. At BR player nodes, picks the max-value action; at opponent nodes, follows the average strategy.

- `exploitability_abstracted(trainer, phases, samples)` - Overall exploitability
- `exploitability_multiseed(trainer, seeds, samples)` - Multi-seed average with confidence intervals
- `exploitability_breakdown(trainer, phase, top_n)` - Per-infoset regret analysis
- `strategy_audit(trainer)` - Detects premium limps, all-in overuse, flat strategies, frequency anomalies
- `river_bluff_audit(trainer)` - Diagnoses river bluff-to-value ratio by sizing, position, and bucket. Classifies as HEALTHY (<40%), SLIGHTLY_HIGH (<50%), OVER_BLUFFING (<60%), or SEVERE_LEAK

**Metric:** Exploitability = (BR_value_p0 + BR_value_p1) / 2. At Nash, this equals 0.

### `server/gto/kuhn.py` - Correctness Benchmark

Minimal 3-card poker (J, Q, K) with known Nash equilibrium:
- P1: bet K always, bluff J at 1/3, check Q
- P2: call K always, call Q vs bet at 1/3, fold J always
- Game value: -1/18 for P1

All 16 Nash equilibrium tests pass, validating the core CFR+ engine.

---

## Training & Experiment Tracking

### `train_gto.py` - Production Training

```bash
venv/bin/python train_gto.py [iterations] [sampling]
venv/bin/python train_gto.py 5000000                    # 5M iters (Cython: ~3.5 min)
venv/bin/python train_gto.py 20000000                   # 20M iters (Cython: ~14 min)
venv/bin/python train_gto.py 20000000 --fresh            # Fresh start (discard existing)
venv/bin/python train_gto.py 20000000 --fresh --workers 6  # Parallel with 6 workers
```

Features:
- tqdm progress bar with nodes/ms-per-iter stats (both sequential and parallel modes)
- Parallel training with `--workers N` (shared-memory node tables, tqdm status polling)
- `--fresh` flag to start from scratch (ignores existing strategy)
- Loads existing strategy if available (continues training)
- Saves to `server/gto/strategy.json` (used by live game)
- Prints strategy tables (OOP root, IP vs raise/limp, flop sample)
- Prints exploitability breakdown (top 10 most exploitable infosets)
- **Automatically logs to experiment matrix** with exploitability eval, strategy audit, and strategy snapshot

### `run_experiment.py` - Experiment Management

```bash
venv/bin/python run_experiment.py --name test_5m --iterations 5000000
venv/bin/python run_experiment.py --list
venv/bin/python run_experiment.py --compare exp1 exp2
```

- Named experiments with full eval (3-seed exploitability, audit)
- Strategy snapshots saved to `experiments/`
- Results appended to `experiments/matrix.json`

### `run_eval_harness.py` - Comprehensive Evaluation Suite

Runs four independent test suites to validate strategy quality beyond exploitability:

```bash
venv/bin/python run_eval_harness.py                          # Full suite
venv/bin/python run_eval_harness.py --gauntlet               # Opponent gauntlet only
venv/bin/python run_eval_harness.py --offtree                # Off-tree tests only
venv/bin/python run_eval_harness.py --bridge                 # Bridge A/B only
venv/bin/python run_eval_harness.py --leakage                # Rare-node leakage only
venv/bin/python run_eval_harness.py --hands 2000             # More hands (slower)
venv/bin/python run_eval_harness.py --quick                  # Fast check (100 hands)
venv/bin/python run_eval_harness.py --save eval_results/v6.json  # Save to specific file
```

**Test Suites:**

| Suite | What it tests | Method |
|-------|---------------|--------|
| **Gauntlet** | bb/100 against 7 exploit bots | Parallel play (ProcessPoolExecutor) |
| **Off-tree stress** | EV loss from action translation | 11 forced-sizing variants |
| **Bridge A/B** | Mapping scheme comparison | 4 mappings × 3 opponents = 12 matches |
| **Leakage** | Regret hotspots, coverage, bluff ratios | MC exploitability + strategy audit |

**Adversary Bots (Gauntlet):**
NitBot, AggroBot, OverfoldBot, CallStationBot, DonkBot, WeirdSizingBot, PerturbBot

**Bridge Mappings (A/B):**
nearest (default), conservative (always smaller sizing), stochastic (interpolate between nearest), resolve (heuristic re-solve)

**Progress tracking:** tqdm bars at every level — overall stage progress, per-matchup gauntlet, per-sizing off-tree, per-match bridge, per-phase leakage hotspots.

Results are saved to `eval_results/` as JSON.

### `simulate.py` - Self-Play Simulation

Framework for running GTO agent vs heuristic AI opponents in simulated hands with detailed statistics.

---

## Results

### v7 (current — 2026-03-12)

| Iterations | Exploitability | Nodes | Train Time |
|-----------:|---------------:|------:|-----------:|
| **50,000,000** | **1.2822** | **1,047,504** | **~29 min** |

**v7 Eval Harness Results (`eval_results/eval_1773348554.json`):**

| Metric | Value |
|--------|-------|
| Exploitability | **1.2822 bb/100** |
| Preflop | 1.05 |
| Flop | 1.42 |
| Turn | 1.44 |
| River | 1.17 |
| Node coverage | 100% (1,047,504 nodes) |
| River bluff ratio | 35.1% (HEALTHY) |

**Gauntlet (nearest mapping):**

| Bot | bb/100 | Notes |
|-----|-------:|-------|
| AggroBot | +365.6 | |
| OverfoldBot | -291.4 | Active issue — not exploiting folds |
| DonkBot | +266.5 | |
| WeirdSizingBot | -852.7 | Off-tree translation gap |
| PerturbBot | -210.4 | |
| NitBot | +235.9 | Fixed from -19 at v6 |
| CallStationBot | +439.5 | Fixed from -463 at v6 |
| **Average** | **-6.7** | |

**Bridge A/B (nearest is best):**

| Mapping | Avg bb/100 |
|---------|----------:|
| nearest (default) | +276.9 |
| stochastic | +276.6 |
| resolve | +237.9 |
| conservative | -33.5 |

**Changes from v6:**
- Reverted eq_bucket >= 2 donk restriction (was causing 1.82 regression via action-set mismatch)
- Fixed exploitability.py RNG sharing bug (br0/br1 now use independent Random instances)
- GTOAgent default mapping reverted to `nearest` (conservative now -33.5 with healthy river bluffs)

### v6 (50M, deprecated reference)

| Iterations | Exploitability | Nodes |
|-----------:|---------------:|------:|
| 50,000,000 | 1.28 ± 0.04 | ~1.05M |
| 20,000,000 | 1.31 ± 0.03 | ~251k |

Known issues at v6: CallStationBot -463, NitBot -19, conservative mapping as default.
Both critical issues resolved in v7.

### v5.1 Convergence (with postflop pot fix + preflop fold fix)

| Iterations | Exploitability | Std | Nodes | Train Time |
|-----------:|---------------:|----:|------:|-----------:|
| 10,000 | 45.92 | 1.50 | 224,142 | 1.9s |
| **20,010,000** | **1.41** | **0.03** | **251,040** | **655s** |

**Key findings:**
- **96.9% reduction** in exploitability from 10k to 20M iterations
- Exploitability of **1.41** — a major improvement over v5's 0.72 (which had broken postflop)
- Preflop exploitability: **1.66** (nearly solved)
- Flop/turn/river strategies now fully converged (previously stuck at uniform)
- Training speed: **0.03 ms/iter** with Cython (20M in ~11 minutes)

### v5 (deprecated — broken postflop)

Previous v5 results (0.72 exploitability at 20M) were misleading: postflop strategies were completely uniform (20% each action) due to a zero-pot bug. The low exploitability score was an artifact of the evaluator also using the same broken investment model. See "Critical Bugs Fixed" below.

### Baselines (frozen references)

| Baseline | Version | Iterations | Exploitability | Nodes | File |
|----------|---------|----------:|---------------:|------:|------|
| **v6 (current)** | 6 | 50M | **1.30** | ~1.05M | `eval_results/eval_v6_20M.json` |
| v5.1 | 5.1 | 20M | 1.41 | 251,040 | `benchmarks/baseline_v51_20m.json` |
| v4 | 4 | 1M | 0.87* | 67,936 | `benchmarks/baseline_1m.json` |

*v4's 0.87 used 88 buckets (no position). v5 pre-fix showed 0.72 but postflop was broken (uniform).

### Variance Study (BR Estimator Stability)

| Sample Count | Mean | Std | CV |
|-------------:|-----:|----:|---:|
| 500 | 29.76 | 0.57 | 1.9% |
| 2,000 | 29.10 | 0.51 | 1.7% |

500 samples is sufficient — only ~2% noise. Differences >5% between measurements are real signal.

---

## Strategy Quality (50M, v6)

### Preflop OOP (acts first)

The solver has learned position-aware opening ranges with proper fold/limp/raise decisions:
- **Low equity (EQ 0-2):** Fold 80-95% — correctly dumping trash preflop
- **Mid equity (EQ 4):** Check/limp 98-99% — don't build pots OOP with marginal holdings
- **Above-average equity (EQ 6):** Mostly limp (72-94%), some raises (6-28%) — pot control
- **High equity (EQ 7):** Mix of limp (21-58%), open raise (23-58%), and all-in (12-50%) — hand-type dependent
- **Premium pairs at EQ 7:** 35% limp, 33% open raise, 31% all-in (balanced three-way mix)
- **High suited connectors at EQ 7:** 21% limp, 26% open raise, 50% all-in (aggressive)
- **Trash at EQ 0:** 91% fold — proper discipline with junk

### Preflop IP (responding)

Clean fold/call/3-bet ranges scaling with equity:
- **EQ 0 vs raise:** 99% fold across all hand types
- **EQ 3 vs raise:** 97-98% fold (very tight facing raises)
- **EQ 5 vs raise:** 98-99% call (correct flat-calling range)
- **EQ 7 vs raise:** Heavy 3-bet/all-in (high pairs: 40% 3-bet, 20% all-in; suited ace: 48% 3-bet)
- **Vs limp:** Aggressive raises at high equity (high suited connectors EQ 7: 76% open raise)

### Postflop (flop OOP sample — fully converged with donk betting)

- **Low equity (EQ 0):** Check 39-54%, donk small 26-44%, donk medium 6-10% — probing with small leads
- **Mid equity (EQ 3):** 99% check — classic pot control with marginal hands
- **Above-average equity (EQ 5):** Check 29-41%, donk small 45-62%, donk medium 8-13% — thin value leads
- **High equity (EQ 7):** Check 51-62% (trapping), donk small 12-21%, bet 2/3-pot 4-8%, pot 7-9%, all-in 6-9% — polarized sizing

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Training speed (Cython) | 0.03 ms/iter |
| Training speed (Python) | ~2-3 ms/iter |
| Cython speedup | ~70x |
| 1M iterations | ~30s (Cython) |
| 5M iterations | ~2.5 min (Cython) |
| 20M iterations | ~11 min (Cython) |
| 50M iterations | ~29 min (Cython, 6 workers) |
| Exploitability eval (500 samples, 3 seeds) | ~10s |
| Game tree nodes | ~1.05M (at 50M iters) |
| Bucket count | 120 (8 equity x 15 hand types) |
| Strategy file size | ~30-40 MB (JSON) |

---

## Critical Design Decisions & Bugs Fixed

### Continuation Values (v5)

**Problem:** v5 initially used recursive frozen-policy evaluation (`_evaluate_street`) at street boundaries. This caused **convergence regression**: exploitability *increased* with more training.

**Fix:** Bucket-comparison showdown utility at street boundaries. Each street trains independently with correlated bucket sampling.

**Lesson:** In abstracted multi-street CFR, simple continuation value approximations converge reliably, while recursive policy evaluation creates feedback loops.

### Postflop Zero-Pot Bug (v5 → v5.1)

**Problem:** Postflop `_player_investments()` started with `inv = [0.0, 0.0]`, `outstanding = 0.0`. Since all bet sizes are pot-fractions, every postflop bet computed as zero (half of 0 = 0). Every action had identical utility, regrets were always 0, and **postflop strategies stayed perfectly uniform forever** — even after 20M iterations.

**Fix:** Postflop now starts with `inv = [0.5, 0.5]` (normalized pot = 1.0). Pot-fraction bets now produce real stakes, enabling the solver to learn differentiated strategies. This is correct because per-street training is independent and poker decisions are relative to pot size.

**Impact:** Flop strategies went from 20%/20%/20%/20%/20% (broken) to fully converged with sensible patterns (check with weak hands, thin value bet with strong, trap/shove with nuts).

### Preflop Fold Missing (v5 → v5.1)

**Problem:** `_has_bet_to_call()` returned `False` for preflop empty history, so the SB facing the BB only had CHECK_CALL, OPEN_RAISE, ALL_IN — no FOLD option. The solver never learned to fold trash preflop.

**Fix:** `_has_bet_to_call()` now returns `True` for preflop empty history (BB is a forced bet). Added `raise_count == 0` case to preflop action menu giving FOLD, CALL, OPEN_RAISE, ALL_IN.

### Blind Investment Tracking (v5 → v5.1)

**Problem:** Preflop started with `inv = [0.5, 0.5]`, `outstanding = 0.0`. SB (0.5) facing BB (1.0) should have 0.5 outstanding. Limping was free.

**Fix:** Preflop now starts with `inv = [0.5, 1.0]`, `outstanding = 0.5`.

---

## Theoretical Background

### CFR+ (Tammelin 2014)

Standard CFR accumulates regrets that can go arbitrarily negative. CFR+ floors regrets at zero (regret-matching+) and uses linear-weighted strategy averaging:

```
R+(I,a) = max(R(I,a) + r(I,a), 0)           # Floor at zero
weight_T = max(T - delay, 0)                  # Linear averaging
```

This converges faster in practice and avoids numerical issues from large negative regrets.

### External Sampling MCCFR (Lanctot et al. 2009)

Instead of traversing the full game tree (vanilla CFR), external sampling:
- At the **traverser's** nodes: explores all actions (computes counterfactual regrets)
- At the **opponent's** nodes: samples one action according to current strategy

This reduces per-iteration cost from O(|tree|) to O(|traverser's tree|), enabling much larger games.

### E[HS^2] Card Abstraction (Zinkevich et al. 2007)

Raw hand equity doesn't capture hand potential (draws). E[HS^2] accounts for this by:
1. Sampling possible next community cards
2. Computing equity conditioned on each
3. Averaging the squared equities

Hands with the same equity but different draw potential get different E[HS^2] values.

---

## File Map

```
poker/
+-- train_gto.py                # Offline training with tqdm + parallel workers
+-- run_eval_harness.py         # Comprehensive eval suite (gauntlet/offtree/bridge/leakage)
+-- run_experiment.py           # Named experiments, comparison tools
+-- simulate.py                 # Self-play simulation (GTO vs heuristic AI)
+-- convergence_study.py        # Exploitability vs iterations diagnostic
+-- variance_study.py           # BR estimator noise measurement
+-- setup_cython.py             # Cython build script
+-- server/
|   +-- gto/
|       +-- cfr.py              # CFR+ trainer (weighted phases, all-in dampening)
|       +-- cfr_fast.pyx        # Cython-accelerated CFR+ inner loop
|       +-- abstraction.py      # 2D bucketing (15 hand types) + action menus (13 actions)
|       +-- equity.py           # E[HS^2] and hand classification
|       +-- engine.py           # Live game bridge
|       +-- exploitability.py   # Exploitability + strategy audit + river bluff audit
|       +-- kuhn.py             # Kuhn poker benchmark
|       +-- strategy.json       # Production strategy file
+-- eval_harness/
|   +-- match_engine.py         # HeadsUpMatch engine, GTOAgent
|   +-- adversaries.py          # 7 exploit bots (Nit, Aggro, Overfold, CallStation, etc.)
|   +-- offtree_stress.py       # 11 off-tree sizing stress tests
|   +-- translation_ab.py       # 4 mapping schemes × 3 opponents A/B test
|   +-- fast_equity.py          # Preflop cache + fast postflop equity
+-- eval_results/
|   +-- eval_v6_20M.json        # v6 eval harness results
+-- experiments/
|   +-- matrix.json             # Experiment results matrix
|   +-- *_strategy.json         # Strategy snapshots per experiment
+-- benchmarks/
    +-- baseline_v51_20m.json   # v5.1 baseline (20M, exploitability 1.41)
    +-- baseline_v51_20m_strategy.json  # v5.1 strategy snapshot
    +-- baseline_1m.json        # Legacy v4 baseline reference
```

---

## Next Steps to Improve

See `docs/ACTION_PLAN_v7.md` for the full prioritized plan. Summary:

### Short-Term (next retrain)
1. **Tighten all-in dampening** — Change `raise_count < 2` → `raise_count == 0`, factor `0.7x` → `0.5x`. Stops penalizing jam-over-open lines.
2. **3x Cython phase schedule** — Change `cfr_fast.pyx` `train_fast()` from 2x to 3x flop/turn. Reduces 1,334 flat strategy nodes and improves flop/turn exploitability (currently 1.42/1.44).
3. **Post-training strategy clamps** — Cap premium limp at 5% and EQ6 fold at 20% for 7 specific node types.

### Architectural (v8)
4. **Split EQ0 bucket** — Separate `EQ0_DRAW` (4+ outs) from `EQ0_AIR`. Requires 9-bucket rebuild but would eliminate the remaining river over-bluffing at specific nodes.
5. **Full multi-street best response** — Current exploitability (1.28) is per-street; true full-game BR would chain streets via correlated bucket sampling.
6. **Opponent-adaptive bluff frequencies** — Detect call-rate in real-time and reduce river bluff frequency vs stations.
7. **Finer equity abstraction** — Expanding from 8 to 12+ equity buckets (180+ total) likely more impactful than more iterations at current granularity.
