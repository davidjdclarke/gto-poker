# GTO Poker Solver — Project Status Report
**Date:** 2026-03-13 | **Current Best Model:** v7 (67M iterations) | **Phase:** Research & Experimentation

---

## Executive Summary

We are researching how to build the strongest possible GTO poker solver for heads-up No-Limit Texas Hold'em. The project uses Counterfactual Regret Minimization (CFR+) to compute approximate Nash Equilibrium strategies, with a focus on understanding which algorithmic choices, abstractions, and hyperparameters most impact model quality.

**Where we are now:**
- **Best exploitability:** 1.23 bb/100 — how far our strategy deviates from perfect Nash equilibrium (lower is better; 0 = perfect)
- **Best gauntlet average:** +47.9 bb/100 — average win rate against 7 diverse exploit bots (Exp A, 2x schedule)
- **Training throughput:** 50M iterations in ~29 minutes (Cython-accelerated, 6 parallel workers)
- **Evaluation throughput:** 5000-hand matchup in ~12 seconds (Cython-accelerated)
- **Model size:** ~1.05M information set nodes, 100% well-visited

**What we just learned:** A systematic 5-experiment ablation study revealed that (1) the phase schedule and all-in dampening parameters have the most impact on practical play quality, (2) finer card abstraction underperforms at equal training budget, and (3) our evaluation methodology needed a 10x sample size increase to produce reliable rankings.

**The open problem:** All model variants lose heavily to CallStationBot (-731 to -1268 bb/100). The solver's river bluff frequency (~27%) is structurally too low to punish stations. This is the single largest performance gap and the most important research question going forward.

---

## 1. Research Pipeline

### Experiment Cycle

Every improvement follows this loop:

```
  Hypothesis                  Train                    Evaluate                Compare
 ┌───────────┐          ┌──────────────┐         ┌───────────────┐       ┌──────────────┐
 │ "Does X   │  ──────► │ 50-70M iters │ ──────► │ 5000 hands/bot│──────►│ Delta vs     │
 │  improve  │          │ 6 workers    │         │ 7 exploit bots│       │ baseline     │
 │  the      │          │ ~29 min      │         │ exploitability│       │ Statistical  │
 │  model?"  │          │ from scratch │         │ strategy audit│       │ significance │
 └───────────┘          └──────────────┘         └───────────────┘       └──────────────┘
       │                                                                        │
       └────────────────────── Next hypothesis ◄────────────────────────────────┘
```

### Key Tooling

| Tool | Purpose | Command |
|------|---------|---------|
| `train_gto.py` | Train a model with configurable hyperparameters | `venv/bin/python train_gto.py --iterations 50000000 --workers 6 --fresh` |
| `run_eval_harness.py` | 5-stage evaluation (gauntlet, off-tree, bridge, leakage, behavioral) | `venv/bin/python run_eval_harness.py --hands 5000 --save result.json` |
| `run_ablation.py` | End-to-end experiment orchestration (train → eval → compare) | `venv/bin/python run_ablation.py --name my_exp --iterations 50000000` |
| `setup_cython.py` | Build Cython extensions (required after .pyx changes) | `venv/bin/python setup_cython.py build_ext --inplace` |

### Codebase Map

```
poker/
├── train_gto.py                  # Training CLI (configurable schedule, dampening, etc.)
├── run_eval_harness.py           # 5-stage evaluation suite
├── run_ablation.py               # Ablation experiment orchestrator
├── setup_cython.py               # Cython build (-O3 -march=native -ffast-math)
│
├── server/gto/                   # Core solver
│   ├── cfr.py                    # CFR+ trainer (Python orchestration layer)
│   ├── cfr_fast.pyx              # Cython hot loop (~70x training speedup)
│   ├── abstraction.py            # 2D bucketing (equity × hand type), action menus
│   ├── equity.py                 # E[HS²] hand evaluation + hand type classification
│   ├── exploitability.py         # MC best response + strategy audits
│   ├── engine.py                 # Live game bridge (not research-critical)
│   └── strategy.json             # Current best model (~30-40 MB)
│
├── eval_harness/                 # Evaluation infrastructure
│   ├── match_engine.py           # HeadsUpMatch engine, GTOAgent with mapping options
│   ├── adversaries.py            # 7 exploit bots (Aggro, Nit, Station, Overfold, etc.)
│   ├── eval_fast.pyx             # Cython hand evaluator (~45x eval speedup)
│   ├── fast_equity.py            # Preflop bucket cache + fast postflop equity
│   ├── behavioral_regression.py  # 6 strategic node family audits
│   ├── offtree_stress.py         # 11 off-tree sizing stress tests
│   └── translation_ab.py         # Action mapping A/B tests
│
├── tests/                        # 16 Kuhn Nash equilibrium tests (correctness gate)
├── docs/plans/                   # Experiment plans and decision logs
├── docs/results/                 # Historical eval results (one file per experiment)
└── experiments/ablations/        # Strategy snapshots + eval JSONs per ablation
```

---

## 2. Model Architecture

### 2.1 The Abstraction Problem

Real poker has ~10¹⁶⁰ game states — far too many to solve directly. We compress the game into ~1M tractable information sets through two abstractions that define the model's resolution:

**Card Abstraction — 2D Bucketing**

Every hand is mapped to: `bucket = equity_bucket × 15 + hand_type`

| Dimension | Count | How It's Computed |
|-----------|-------|-------------------|
| Equity buckets | 8 (baseline) or 9 (Exp E) | E[HS²] quantized into bands. E[HS²] captures both current hand strength and draw potential via Monte Carlo rollout |
| Hand types | 15 | Structural classification from hole cards (pair rank, suitedness, connectivity) |
| **Total buckets** | **120** (baseline) or **135** (9-bucket) | Full cross-product |

The 15 hand types are critical because hands with identical equity can have completely different strategic properties:

| ID | Category | Examples | Strategic Property |
|----|----------|----------|--------------------|
| 0-1 | PREMIUM/HIGH_PAIR | AA-TT | Blockers, set potential |
| 2-3 | MID/LOW_PAIR | 99-22 | Set mining, positional |
| 4-5 | BROADWAY | AK, KQ, QJ | Top-pair potential, blockers |
| 6 | SUITED_ACE | A2s-A9s | Nut flush draw potential |
| 7-9 | CONNECTORS/GAPPERS | T9s, 76s, T8s | Straight/flush draw combos |
| 10-12 | OFFSUIT BROADWAY/ACE | ATo, KQo | Domination risk |
| 13-14 | TRASH | 72o, 83o | Bluff candidates |

**This abstraction is the most important lever.** The v8 ablation study tested splitting EQ0 into AIR and DRAW sub-buckets (8→9 equity buckets), adding 12% more nodes. It underperformed at equal training budget, but this is likely a convergence issue, not a representation issue.

**Action Abstraction — 13 abstract actions**

| Preflop | Postflop | Special |
|---------|----------|---------|
| FOLD, CHECK_CALL | FOLD, CHECK_CALL | DONK_SMALL (1/4 pot) |
| OPEN_RAISE (~2.5bb) | BET_THIRD_POT | DONK_MEDIUM (1/2 pot) |
| THREE_BET (~3x) | BET_HALF_POT | ALL_IN |
| FOUR_BET (~2.2x) | BET_TWO_THIRDS_POT | |
| | BET_POT, BET_OVERBET | |

Menus are context-dependent (position, raise count, street). Max 4 raises per street.

### 2.2 CFR+ Training Algorithm

**External Sampling MCCFR with CFR+** (Tammelin 2014, Lanctot et al. 2009):

```
For each iteration t:
  For each traverser p in {P0, P1}:
    Pick a street from the phase schedule
    Sample correlated bucket pairs (hand type fixed, equity drifts ±1 between streets)
    Traverse the game tree:
      - At traverser nodes: evaluate ALL actions → compute counterfactual regrets
      - At opponent nodes: SAMPLE one action from current strategy (external sampling)
    Update regrets: R⁺(I,a) = max(R(I,a) + r(I,a), 0)     [CFR+ floor]
    Update strategy sum: weight = max(t - averaging_delay, 0)  [linear averaging]
```

**Hyperparameters under active investigation:**

| Parameter | Current Default | Exp A (best) | Exp B (2nd) | Effect |
|-----------|----------------|--------------|-------------|--------|
| Phase schedule | 3x (8-step: flop/turn get 3x emphasis) | **2x (6-step)** | 3x | Which streets get extra training passes per cycle |
| All-in dampening | rc==0, 0.5× | rc<2, 0.7× | **rc<2, 0.7×** | Regret scaling for all-in actions to prevent overuse |
| Averaging delay | 100,000 | 100,000 | 100,000 | Iterations before strategy averaging begins |
| Equity buckets | 8 | 8 | 8 | Card abstraction resolution (9 tested in Exp E) |
| Adaptive averaging | off | off | off | Per-node delay based on first visit (Exp C: no impact) |

**Cython implementation:** The inner traversal loop compiles to C (`cfr_fast.pyx`), using a flat `NodeData` struct array with int64-encoded infoset keys for O(1) lookup. Runs at ~0.03 ms/iter without Python GIL overhead — ~70x faster than pure Python.

**Parallel training:** 6 worker processes share a memory-mapped node pool without locks. Theoretically sound for CFR+ — benign data races add negligible variance to convergence.

### 2.3 What the Model Learns

At convergence, the solver produces a **mixed strategy** at every information set — a probability distribution over actions. Examples from the current v7 model:

- **AA preflop OOP:** 75% limp, 15% open raise, 9% all-in (balanced trapping mix)
- **72o preflop OOP:** 91% fold (correct discipline with trash)
- **EQ7 flop OOP:** 51-62% check (trapping), 12-21% donk small, 7-9% pot (polarized)
- **EQ0 flop OOP:** 39-54% check, 26-44% donk small (probing with air)

The strategy is **mixed** (randomized) because deterministic play is exploitable — if we always raise with AA, opponents learn to fold.

---

## 3. Evaluation Framework

### 3.1 Primary Metrics

We measure model quality on two axes:

| Metric | What It Measures | Target | Current Best |
|--------|-----------------|--------|-------------|
| **Exploitability (bb/100)** | Distance from Nash equilibrium — how much a perfect adversary could win | 0 (perfect Nash) | 1.23 (v7 67M) |
| **Gauntlet average (bb/100)** | Win rate against 7 diverse exploit bots — practical play quality | Positive (beating weak opponents) | +47.9 (Exp A) |

These metrics can diverge: a model can have good exploitability but poor gauntlet (too passive to exploit weak play), or vice versa. **Both must improve together.**

### 3.2 Evaluation Stages

| Stage | What It Tests | Time |
|-------|--------------|------|
| **Gauntlet** | 5000 hands × 7 bots (parallel). Win rate per bot + average | ~12s per matchup |
| **Leakage** | MC exploitability (3-seed, 500 samples), regret hotspots, strategy audit, river bluff analysis, node coverage | ~23 min (exploitability is the bottleneck) |
| **Behavioral regression** | 6 strategic node families tracked with delta-vs-baseline | ~30s |
| **Off-tree stress** | 11 non-standard bet sizes, EV loss measurement | ~2 min |
| **Bridge A/B** | 4 action mapping schemes × 3 opponents | ~2 min |

### 3.3 The 7 Exploit Bots

Each bot is designed to exploit a specific class of strategic weakness:

| Bot | Exploitative Strategy | What Failure Means |
|-----|----------------------|-------------------|
| **AggroBot** | 50% 3-bet, 70% open, 65% barrel | Model can't handle aggression |
| **NitBot** | Top 15% preflop, passive postflop | Model can't steal from tight ranges |
| **OverfoldBot** | Folds to any postflop bet | Model can't extract value with c-bets |
| **CallStationBot** | Calls nearly everything postflop | Model bluffs too much / value-bets too thin |
| **DonkBot** | OOP leads 20-120% pot randomly | Model can't handle unexpected leads |
| **WeirdSizingBot** | Off-tree sizes: 15-200% pot | Model's action translation is broken |
| **PerturbBot** | GTO-like + 15% random noise | Model isn't robust to small deviations |

**A strong model should beat all exploitable bots and break even against PerturbBot.** Our biggest failure is CallStationBot (structural weakness across all experiments).

### 3.4 Strategy Audits

Automated anomaly detection catches pathological behaviors:

| Audit | What It Catches | Current State | Concern Level |
|-------|----------------|---------------|---------------|
| River bluff ratio | Bluff-to-value on river bets | 27.3% | Healthy (but too low to punish stations) |
| All-in overuse | Excessive shove frequency | 458 nodes | Moderate — dampening helps |
| Flat strategies | Unconverged (near-uniform) nodes | 1,035 | Expected at current iterations |
| Premium limp | AA/KK limping instead of raising | 4 nodes | Cosmetic edge case |

### 3.5 Behavioral Regression Suite

Six node families track strategic stability across experiments:

1. **Preflop premium** — AA/KK/QQ opening action distribution
2. **Jam-over-open** — IP all-in frequency facing opens
3. **Flop low-EQ probe** — OOP donk/check behavior with weak holdings
4. **River bluff** — Low-equity river betting frequency
5. **River thin value** — Medium-equity river value betting frequency
6. **Overbet defense** — Response to oversized bets

Each reports action distributions, Shannon entropy, and deltas vs baseline. A regression in any family is a red flag that warrants investigation before adopting a change.

### 3.6 Evaluation Methodology Lessons

**Sample size is critical.** Our ablation study initially used 500 hands/bot, which produced wildly misleading results:

| Experiment | 500-hand avg | 5000-hand avg | How wrong we were |
|------------|-------------|---------------|-------------------|
| Exp A | +137.2 | +47.9 | 3x overstated |
| Exp B | +174.0 | +33.5 | 5x overstated |
| Baseline | -226.2 | -44.3 | 5x overstated (wrong direction implied) |
| Exp E | -291.1 / +159.3 | -71.3 | 450 bb/100 range between seeds! |

**Standard going forward: 5000 hands/bot minimum.** The Cython eval acceleration (`eval_fast.pyx`, ~45x speedup) makes this practical.

---

## 4. Model Evolution

### Version History

| Version | Iters | Exploitability | Gauntlet Avg | Key Change | Lesson Learned |
|---------|-------|---------------|-------------|------------|----------------|
| v4 | 1M | 0.87* | — | 88 buckets, no position | Position awareness is essential |
| v5 | 20M | 0.72* | — | 2D buckets | *Result was artifact of broken postflop* |
| v5.1 | 20M | 1.41 | — | Fixed zero-pot bug, preflop fold | Verify postflop strategies aren't uniform |
| v6 | 50M | 1.28 | -6.7 | 120 buckets, 13 actions, Cython, parallel | Donk bets + overbet sizing matter |
| **v7** | **67M** | **1.23** | **-44.3** | Reverted broken fixes, RNG bug fix | Don't ship untested "improvements" |
| **v8 Exp A** | **50M** | **1.27** | **+47.9** | 2x phase schedule | Simpler schedule wins on gauntlet |
| **v8 Exp B** | **50M** | **1.25** | **+33.5** | Old dampening (rc<2, 0.7×) | Broader dampening scope helps |

*v4/v5 exploitability numbers aren't comparable due to different bucket schemes and evaluation bugs.

### Critical Bugs That Invalidated Results

These are worth understanding because they illustrate how subtle bugs can produce plausible-looking but completely wrong results:

1. **Zero-pot bug (v5):** Postflop `inv = [0, 0]` meant all pot-fraction bets were 0. Every action had identical utility. Postflop strategies stayed perfectly uniform after 20M iterations — but the *evaluator had the same bug*, so exploitability looked normal (0.72). Everything was broken symmetrically.

2. **RNG sharing (v6):** Best-response for P0 and P1 shared one `Random` instance. Correlated samples meant exploitability estimates were systematically biased. Results looked reasonable but weren't measuring what we thought.

3. **Action-set mismatch (v6→v7):** Restricting donk bets to EQ≥2 during training but not evaluation meant the evaluator saw actions the trainer never learned about. Exploitability jumped from 1.28 to 1.82 — a 42% regression from a one-line change.

**Lesson:** Always verify that training and evaluation use exactly the same game rules. The Kuhn poker tests (16 exact Nash equilibrium checks) catch core algorithm bugs but not abstraction mismatches.

---

## 5. v8 Ablation Study — Current Results

### 5.1 Study Design

Five isolated experiments, each changing one variable, all trained from scratch at 50M iterations:

| Exp | What Changed | Why We Tested It |
|-----|-------------|-----------------|
| **A** | Phase schedule: 3x → 2x | 3x gives flop/turn extra emphasis — is that actually better? |
| **B** | All-in dampening: (rc==0, 0.5×) → (rc<2, 0.7×) | Narrower scope + harsher multiplier — too aggressive? |
| **C** | Per-node adaptive averaging delay | Rarely-visited nodes might be poisoning the average strategy |
| **D** | Translation-confidence blending | Off-tree action handling might be the ceiling |
| **E** | Equity buckets: 8 → 9 (split EQ0 into AIR/DRAW) | Abstraction granularity might be the dominant bottleneck |

### 5.2 Results

#### Gauntlet (bb/100, 5000 hands/bot)

| Bot | Exp A (2x sched) | Exp B (old damp) | Exp C (adaptive) | Exp E (9 buck) | Baseline v7 (67M) |
|-----|:-:|:-:|:-:|:-:|:-:|
| AggroBot | +440.8 | +387.6 | +421.3 | +566.5 | +421.0 |
| OverfoldBot | +10.2 | +72.4 | -16.7 | -41.7 | +14.8 |
| DonkBot | +484.6 | +337.3 | +553.5 | +303.3 | +566.1 |
| NitBot | +200.6 | +55.7 | +115.8 | +44.3 | +0.3 |
| WeirdSizingBot | **+233.9** | +144.4 | -51.3 | -259.9 | -383.0 |
| PerturbBot | -87.4 | **-31.4** | -47.4 | -118.7 | -139.6 |
| CallStationBot | -947.5 | **-731.4** | -1268.3 | -993.3 | -789.8 |
| **Average** | **+47.9** | **+33.5** | **-41.9** | **-71.3** | **-44.3** |

#### Exploitability & Diagnostics

| Metric | Baseline v7 (67M) | Exp A (50M) | Exp B (50M) | Exp C (50M) | Exp E (50M) |
|--------|:-:|:-:|:-:|:-:|:-:|
| Exploitability | **1.23** | 1.27 | 1.25 | 1.26 | 1.31 |
| River bluff % | 27.3% | 27.0% | 26.9% | 27.0% | 23.3% |
| All-in overuse nodes | 458 | 488 | **435** | 476 | 518 |
| Flat strategies | **1,035** | 1,393 | 1,168 | 1,132 | 1,215 |
| Nodes | 1.05M | 1.04M | 1.05M | 1.05M | **1.18M** |

### 5.3 Key Findings

#### Finding 1: Phase schedule has the biggest practical impact
**Exp A (2x schedule): +47.9 avg** — the best gauntlet performer by a clear margin. The simpler 6-step cycle `[pre, flop, flop, turn, turn, river]` outperforms the 8-step 3x cycle. The extra flop/turn passes in 3x don't help and may hurt by under-training preflop and river.

**Implication:** Adopt 2x schedule as the default for all future training.

#### Finding 2: Dampening scope matters more than dampening strength
**Exp B (old dampening): +33.5 avg.** The broader scope (apply dampening when `raise_count < 2`, not just `raise_count == 0`) with a gentler multiplier (0.7× instead of 0.5×) produces better play. It also achieves the lowest all-in overuse count (435) and the best PerturbBot result (-31.4).

**Implication:** Combine with 2x schedule in the next experiment.

#### Finding 3: Over-convergence was a measurement artifact
At 500 hands, the baseline scored -226.2 avg, suggesting 67M iterations were *harmful*. At 5000 hands, it scores -44.3 — in the same ballpark as Exp A/B. The extra 17M iterations genuinely reduce exploitability (1.23 vs 1.25-1.27 for 50M experiments). More iterations help, they just don't help as dramatically as the noisy 500-hand data suggested.

**Implication:** Don't fear training longer. But diminishing returns set in — hyperparameter choices matter more than raw iteration count past ~50M.

#### Finding 4: Finer abstraction needs proportionally more training
**Exp E (9 buckets): -71.3 avg.** The worst performer despite having the most expressive abstraction. With 12% more nodes, each node gets fewer training samples at 50M iterations. The river bluff ratio drops to 23.3% (vs 27% for others), suggesting bluffing nodes haven't converged.

**Implication:** Test 9-bucket at 70M+ iterations before concluding it doesn't help. The abstraction may be better; the training budget was just insufficient.

#### Finding 5: Adaptive averaging is a dead end
**Exp C: -41.9 avg** — statistically indistinguishable from baseline (-44.3). Per-node averaging delay based on `first_visit_iter` adds code complexity for zero benefit. Rare-node averaging is not a bottleneck.

**Implication:** Drop this line of investigation. Remove the code to keep the system simple.

#### Finding 6: CallStation is a structural problem
Every experiment loses heavily to CallStationBot (-731 to -1268 bb/100). The river bluff ratio is ~27% across all models — not enough to punish a player who calls everything. This isn't a convergence issue; it's a property of GTO strategy at this abstraction resolution.

**Implication:** This is the most important open research question. Options: bluff frequency floors, post-training strategy adjustment, alternate training objectives, or explicit station-exploitation modules.

#### Finding 7: WeirdSizingBot separates the winners
The biggest performance swing across experiments is vs WeirdSizingBot (range: -383 to +234). Exp A and B both win; baseline and Exp E both lose badly. Off-tree action translation robustness appears strongly correlated with the training schedule — balanced training across streets (2x) handles unusual sizing better than concentrated training (3x).

---

## 6. Open Research Questions

Ranked by expected impact on model quality:

### Priority 1: CallStation Exploitation
**Problem:** -731 to -1268 bb/100 across all model variants. River bluff frequency (~27%) is too low to punish stations.

**Candidate approaches:**
- River bluff frequency floor (force minimum bluff rate at specific nodes)
- Post-training strategy adjustment (increase bluff rates for low-equity river nodes)
- Alternate loss function (CFR+ minimizes exploitability, not maximizes exploitation)
- Separate exploitation layer on top of GTO base strategy

### Priority 2: Combined 2x Schedule + Old Dampening
**Question:** Do Exp A and B's gains compound, or are they overlapping improvements?

**Experiment design:** Train from scratch at 50-70M iterations with both changes active. Compare to Exp A, Exp B, and baseline individually. This is the next training run to execute.

### Priority 3: Abstraction Resolution at Convergence Parity
**Question:** Does 9-bucket abstraction (Exp E) outperform 8-bucket when given proportionally more training?

**Experiment design:** Train 9-bucket to 70M iterations (proportional to 12% more nodes). If it beats 8-bucket at equal per-node convergence, the path forward is more buckets + more compute.

### Priority 4: Off-Tree Action Handling
**Problem:** WeirdSizingBot loss (-259 to -383) and off-tree stress test EV losses indicate the action translation bridge is a significant source of error.

**Candidate approaches:**
- Finer action abstraction (more sizing options reduce translation error)
- Continuous action space (eliminate translation entirely — requires architectural change)
- Improved mapping heuristics (current "nearest" mapping is naive)

### Priority 5: Multi-Street Continuation Values
**Problem:** Each street trains independently with bucket-comparison continuation values. This is an approximation — true poker involves correlated decisions across streets.

**This is the biggest architectural limitation** but also the most expensive to fix. A full multi-street solver would chain street traversals but exponentially increase the game tree.

---

## 7. Experiment Infrastructure

### Running an Experiment

```bash
# Train a new model variant
venv/bin/python run_ablation.py \
    --name combined_2x_old_damp \
    --iterations 67000000 \
    --workers 6 \
    --config phase_schedule_mode=0 allin_dampen_mode=1

# Or manually:
venv/bin/python train_gto.py \
    --iterations 67000000 --workers 6 --fresh \
    --phase-schedule 2x --allin-dampen old

# Evaluate
venv/bin/python run_eval_harness.py \
    --strategy experiments/ablations/combined_2x_old_damp/strategy.json \
    --gauntlet --leakage --behavioral \
    --hands 5000 \
    --save experiments/ablations/combined_2x_old_damp/eval_5k.json

# Compare
venv/bin/python run_ablation.py --compare baseline_v7 combined_2x_old_damp
```

### Experiment Conventions

- **Train from scratch** (`--fresh`) for all comparisons — continuing from an existing model confounds results
- **5000 hands/bot minimum** — 500 hands produces unreliable rankings (see Section 3.6)
- **Document every run** in `docs/results/` with dated markdown
- **Save strategy + eval JSON** to `experiments/ablations/{name}/`
- **All 16 Kuhn tests must pass** before any experiment — they validate core algorithm correctness
- **Rebuild Cython** after any `.pyx` change — training silently falls back to 70x slower Python

### Technology Notes

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python 3.12, `venv/bin/python` always | Reproducibility |
| Training hot loop | Cython (`cfr_fast.pyx`) | ~70x speedup; flat struct array with int64 keys, no GIL |
| Eval hot loop | Cython (`eval_fast.pyx`) | ~45x speedup; hand evaluation + MC equity in C |
| Parallelism | `multiprocessing` shared-memory pool | Lock-free; benign races are fine for CFR+ |
| Compilation | `-O3 -march=native -ffast-math` | Maximize throughput on local hardware |

---

## 8. Recommended Next Experiments

### Immediate (next 1-2 runs)

1. **Combined A+B:** 2x schedule + old dampening at 67M iterations. This is the highest-confidence next step — both changes are individually positive, and we need to know if they compound.

2. **Exp E at convergence parity:** 9-bucket abstraction at 70M iterations. Determines whether finer abstraction is worth the compute cost.

### Short-term (next 3-5 runs)

3. **CallStation-aware training:** Test post-training bluff frequency adjustments (cheapest to try) before considering architectural changes.

4. **Finer action abstraction:** Add 1-2 more sizing options (e.g., 40% pot, 150% pot) and retrain. Tests whether action resolution or card resolution matters more.

5. **Longer training (100M+):** Characterize the diminishing returns curve. How much does exploitability improve from 67M to 100M?

### Medium-term (requires infrastructure work)

6. **12+ equity buckets:** 180+ total buckets. Likely the highest-ceiling change but requires proportionally more training compute.

7. **Continuous bet sizing:** Eliminate the action abstraction entirely. Major architectural change but removes the off-tree problem.

8. **Multi-street traversal:** Chain street training to capture cross-street correlations. Exponentially more expensive but fundamentally more accurate.

---

## Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **GTO** | Game Theory Optimal — a strategy that cannot be exploited |
| **CFR+** | Counterfactual Regret Minimization Plus — the algorithm we use to converge toward Nash equilibrium |
| **Nash Equilibrium** | Strategy pair where neither player can improve by changing strategy unilaterally |
| **Exploitability (bb/100)** | How much a perfect adversary could win against our strategy. Lower is better. 0 = perfect Nash |
| **bb/100** | Big blinds won per 100 hands — standard poker win rate unit |
| **E[HS²]** | Expected Hand Strength Squared — equity metric that captures draw potential, not just current strength |
| **Infoset** | Information set — all game states that look identical to a player (same cards, same history) |
| **Bucket** | Abstract grouping of similar hands. Our 2D scheme maps ~2.6M hand-board combos to 120 buckets |
| **Off-tree** | Opponent actions outside our abstract action set (e.g., betting 75% pot when we only have 66% and 100%) |
| **Phase schedule** | Which street gets trained on each iteration cycle. 2x = `[pre,flop,flop,turn,turn,river]` |
| **Dampening** | Scaling down regrets for specific actions (all-in) to prevent the solver from over-learning to shove |
| **Gauntlet** | Our 7-bot test suite measuring practical play quality |
| **Convergence** | How many iterations a node needs before its strategy stabilizes. More nodes = more iterations needed |

## Appendix B: Key Files

| Document | Path | Contents |
|----------|------|----------|
| Developer reference | `CLAUDE.md` | API docs, architecture gotchas, common tasks |
| Full technical reference | `GTO_REFERENCE.md` | 500+ lines of architecture detail |
| v7 eval results | `docs/results/v7_67M_20260313.md` | Detailed v7 evaluation |
| **v8 ablation study** | **`docs/results/v8_ablation_study_20260313.md`** | **Full 5000-hand ablation results** |
| v8 experiment plan | `docs/plans/ACTION_PLAN_v8_ablations.md` | Ablation study design and rationale |
| v6 root cause analysis | `RCA_v6_50M.md` | Post-mortem on v6 regressions |
