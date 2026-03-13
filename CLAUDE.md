# Poker GTO Solver — Claude Reference

> **For Claude:** Keep this file up to date. Any time you make changes to the codebase — new files, modified APIs, fixed bugs, changed parameters, updated results — edit the relevant section of this document before finishing the task.

## Quick Orientation

Professional-grade GTO poker solver for heads-up Texas Hold'em. Uses CFR+ (external sampling MCCFR) with a 2D abstraction (equity × hand type). A Cython-accelerated hot loop provides ~70x speedup. All 16 Kuhn poker Nash equilibrium tests pass.

**Python 3.12. Always use `venv/bin/python` (never system python).**

```bash
# Build Cython extension (required after any .pyx change)
venv/bin/python setup_cython.py build_ext --inplace

# Train
venv/bin/python train_gto.py --iterations 1000000 --workers 6

# Evaluate
venv/bin/python run_eval_harness.py

# Run tests
venv/bin/python -m pytest tests/
```

---

## Directory Layout

```
poker/
├── train_gto.py               # Training CLI (tqdm + parallel workers)
├── run_eval_harness.py        # Full eval suite (4 test suites)
├── run_experiment.py          # Experiment tracking & comparison
├── simulate.py                # Self-play simulation (GTO vs heuristic bots)
├── setup_cython.py            # Cython build (-O3 -march=native)
├── convergence_study.py       # Exploitability vs iteration analysis
│
├── server/gto/
│   ├── cfr.py                 # CFR+ trainer (Python orchestration)
│   ├── cfr_fast.pyx           # Cython inner loop (~70x speedup)
│   ├── abstraction.py         # 2D bucketing, action menus, infoset keys
│   ├── equity.py              # E[HS²] evaluation + hand type classifier
│   ├── engine.py              # Live game bridge (gto_decide entry point)
│   ├── exploitability.py      # MC best response + audit tools
│   ├── kuhn.py                # Kuhn poker benchmark (3-card game)
│   └── strategy.json          # Saved strategy (30–40 MB)
│
├── eval_harness/
│   ├── match_engine.py        # HeadsUpMatch, GTOAgent
│   ├── adversaries.py         # 7 exploit bots (Nit, Aggro, Station, etc.)
│   ├── offtree_stress.py      # 11 off-tree sizing variants
│   ├── translation_ab.py      # 4 mapping schemes A/B test
│   └── fast_equity.py         # Preflop cache + fast postflop equity
│
├── tests/
│   └── test_kuhn_benchmark.py # All 16 Nash eq. tests — must pass
│
├── experiments/               # matrix.json + per-experiment strategy snapshots
├── benchmarks/                # baseline_v51_20m.json, baseline_1m.json
├── eval_results/              # eval_v6_20M.json (gauntlet/off-tree/bridge/leakage)
│
├── GTO_REFERENCE.md           # Full architecture & results (500+ lines)
└── RCA_v6_50M.md              # Root cause analysis of v6 issues
```

---

## Core Abstractions

### 2D Bucketing (120 total buckets)

`bucket = equity_bucket * 15 + hand_type`  (range: 0–119)

**Equity buckets (8):** uniform 12.5% bands, 0 = weakest (0–12.5%), 7 = strongest (87.5–100%)

**Hand types (15):** structural category beyond raw equity

| Value | Name               | Examples            |
|-------|--------------------|---------------------|
| 0     | PREMIUM_PAIR       | AA, KK, QQ          |
| 1     | HIGH_PAIR          | JJ, TT              |
| 2     | MID_PAIR           | 99–66               |
| 3     | LOW_PAIR           | 55–22               |
| 4     | STRONG_BROADWAY    | AK, AQ              |
| 5     | BROADWAY           | KQ, QJ, KJ          |
| 6     | SUITED_ACE         | A2s–A9s             |
| 7     | HIGH_SUITED_CONN   | T9s, 98s            |
| 8     | LOW_SUITED_CONN    | 76s–32s             |
| 9     | SUITED_GAPPER      | T8s, 97s            |
| 10    | STRONG_OFFSUIT_ACE | ATo+                |
| 11    | WEAK_OFFSUIT_ACE   | A9o–A2o             |
| 12    | OFFSUIT_BROADWAY   | KQo, QJo            |
| 13    | SUITED_TRASH       | non-connected suits |
| 14    | TRASH              | 72o, 83o            |

**Why 2D?** AKo and 72o may share similar raw equity in some spots but have completely different strategic properties (blockers, drawing potential). The hand type separates them.

### Infoset Key Format

```
"{phase}:{position}:{bucket}:{history}"

# Examples:
"preflop:oop:15:()"           # OOP at bucket 15 (equity=1, type=BROADWAY), no history
"flop:ip:45:(1,2)"            # IP after CHECK_CALL then BET_THIRD_POT
"river:oop:112:(1,3,1)"       # River, OOP at bucket 112, complex history
```

- **phase**: `preflop` | `flop` | `turn` | `river`
- **position**: `ip` (in position) | `oop` (out of position)
- **bucket**: integer 0–119
- **history**: tuple of `Action` integers for the current street

### Action Enum (abstraction.py)

```python
class Action(IntEnum):
    FOLD               = 0
    CHECK_CALL         = 1
    BET_THIRD_POT      = 2   # ~1/3 pot
    BET_HALF_POT       = 3   # ~1/2 pot
    BET_POT            = 4   # ~pot-sized
    ALL_IN             = 5
    OPEN_RAISE         = 6   # preflop: ~2.5bb
    THREE_BET          = 7   # preflop: ~3x opener
    FOUR_BET           = 8   # preflop: ~2.5x 3bet
    BET_TWO_THIRDS_POT = 9   # ~2/3 pot
    BET_OVERBET        = 10  # ~1.25x pot
    DONK_SMALL         = 11  # OOP lead ~1/4 pot
    DONK_MEDIUM        = 12  # OOP lead ~1/2 pot
```

---

## Key Classes & Functions

### CFRTrainer (server/gto/cfr.py)

```python
class CFRTrainer:
    nodes: dict[str, CFRNode]   # infoset key → node
    iterations: int

    # Train (uses Cython if available, else pure Python)
    def train(num_iterations, averaging_delay=0, sampling='external',
              progress_callback=None, num_workers=1)

    # Strategy lookup (returns dict of action_int → probability)
    def get_strategy(phase, bucket, history, position) → dict[int, float]

    def save(filepath: str)
    def load(filepath: str) → bool
```

### CFRNode (server/gto/cfr.py)

```python
class CFRNode:
    def get_strategy() → np.ndarray           # regret-matching+ current strategy
    def get_average_strategy() → np.ndarray   # converged strategy (use this)
    def update_regrets(regrets: np.ndarray)    # CFR+ update (floor at 0)
    def accumulate_strategy(strategy, weight)  # linear weighting
```

### equity.py

```python
# Main entry point for hand evaluation
hand_strength_bucket(hole_cards, community, num_opponents=1,
                     num_buckets=None, simulations=300,
                     use_ehs2=True) → int   # returns bucket 0–119

# Raw E[HS²] value
hand_strength_squared(hole_cards, community, num_opponents=1,
                      simulations=300) → float

# Classify hand type from hole cards only (used preflop)
classify_hand_type(rank1: int, rank2: int, suited: bool) → HandType
```

### engine.py — Live Game Bridge

```python
# Main entry point from the poker server
gto_decide(player, community_cards, pot, current_bet, min_raise,
           big_blind, num_opponents=1, betting_history=None,
           is_in_position=True) → GTODecision

# GTODecision fields:
#   action: str    ("fold" | "check" | "call" | "raise")
#   amount: int
#   strategy_info: dict   # includes "position" and "position_caller" for mismatch debugging

# _abstract_history() now delegates to abstraction.concrete_to_abstract_history()
# — same mapping as match_engine.py (C1 fix)
# is_in_position mismatch vs history-derived position now logs a WARNING
```

### exploitability.py — Evaluation

```python
# Primary eval: exploitability in bb/100 (lower = better GTO)
exploitability_abstracted(trainer, phases=None, samples=500, seed=None) → float

# Multi-seed with confidence intervals (use for reporting)
exploitability_multiseed(trainer, phases=None, samples=500,
                         seeds=None) → dict   # keys: mean, ci, per_phase

# Audit tools — detect strategic anomalies
strategy_audit(trainer, phases=None) → dict
    # Returns: premium_limp, allin_overuse, flat_strategies, frequency_anomalies

river_bluff_audit(trainer, value_threshold=4) → dict
    # Detects river over-bluffing (the CallStation leak)

allin_audit(trainer, phases=None) → dict
```

### match_engine.py — Concrete Evaluation

```python
@dataclass
class HandContext:
    hole_cards, community_cards, pot, current_bet, my_bet, my_chips,
    opp_chips, min_raise, big_blind, phase, is_ip, betting_history,
    hand_number, street_pot_start

class GTOAgent(Agent):
    def __init__(trainer, name="GTO", mapping="nearest", simulations=80)
    # mapping options: "nearest" (default) | "conservative" | "stochastic" | "resolve"
    # v7: nearest = +276.9 avg (best); conservative = -33.5 (too passive now that river bluffs fixed)

class HeadsUpMatch:
    def __init__(p0: Agent, p1: Agent, big_blind=2, seed=None)
    def play(num_hands: int) → MatchResult
```

### adversaries.py — The 7 Exploit Bots

| Bot             | Style                             | GTO Result (v6, 50M) |
|-----------------|-----------------------------------|----------------------|
| NitBot          | Tight preflop, passive postflop   | -19 bb/100 (regressed from +715 at 20M) |
| AggroBot        | Hyper-aggressive 3-better         | varies               |
| OverfoldBot     | Folds to all postflop bets        | exploitable          |
| **CallStationBot** | Calls everything postflop      | **-463 bb/100 (critical leak)** |
| DonkBot         | Unusual lead sizings              | moderate             |
| WeirdSizingBot  | Off-tree bets (75%, min-click)    | moderate             |
| PerturbBot      | ~5% random off-tree actions       | near-GTO             |

---

## CFR+ Algorithm (How It Works)

**External Sampling MCCFR:**
- For each traverser (P0, P1 alternately):
  - Opponent actions: sample 1 (external sampling)
  - Traverser actions: evaluate all (get regrets)
- Regret update: `R⁺ = max(R + r, 0)` (CFR+ floor)
- Strategy: regret-matching over positive regrets
- Averaging: linear weighting `w = max(T - delay, 0)`

**Phase schedule:** `[preflop, flop, flop, turn, turn, river]` (indices into PHASES)
Flop and Turn get 2x iterations per training cycle. Cython (`cfr_fast.pyx`) has its own hardcoded 2x schedule — Python `PHASE_SCHEDULE` is only used when Cython is unavailable.

**All-in dampening:** multiply all-in regrets by 0.7× when `raise_count < 2`.

**Donk path:** OOP flop/turn first action (no prior bet, `hlen == 0`) uses 6-action donk menu for ALL equity buckets.

**Postflop normalization:** `inv = [0.5, 0.5]` at each street start (pot = 1.0 per player)

---

## Training Parameters

```bash
venv/bin/python train_gto.py \
    --iterations 5000000 \    # 5M per run, cumulative from previous save
    --workers 6 \             # parallel workers (shared-memory node pool)
    --averaging-delay 100000  # linear weighting starts after 100k iters
```

**Performance:** ~0.03–0.04 ms/iter with Cython (6 workers), ~29 min for 50M iterations total.

---

## Current State (v7, 50M iterations — 2026-03-12)

**Exploitability: 1.2822 bb/100** (per eval harness 3-seed), **1.2708 ± 0.0019** (training-time estimate)
**Per-phase:** Preflop 1.05, Flop 1.42, Turn 1.44, River 1.17
**Nodes:** 1,047,504 (100% well-visited) | **Strategy:** `server/gto/strategy.json`
**Experiment:** `experiments/v6_50.0M_20260312_162803_strategy.json`
**Eval results:** `eval_results/eval_1773348554.json`

### Gauntlet Results (v7)

| Bot | v7 bb/100 | v6 50M bb/100 | Change |
|-----|-----------|---------------|--------|
| AggroBot | +365.6 | — | — |
| OverfoldBot | **-291.4** | — | new issue |
| DonkBot | +266.5 | — | — |
| WeirdSizingBot | **-852.7** | — | known |
| PerturbBot | **-210.4** | — | — |
| NitBot | **+235.9** | -19 | ✓ FIXED |
| CallStationBot | **+439.5** | -463 | ✓ FIXED |
| **Average** | **-6.7** | — | |

### Bridge Mapping Results (v7)

| Mapping | AggroBot | CallStation | WeirdSizing | **Avg** |
|---------|----------|-------------|-------------|---------|
| **nearest** | +365.6 | **+912.9** | -447.7 | **+276.9** ← default |
| conservative | +257.4 | -72.5 | -285.3 | -33.5 |
| stochastic | +528.4 | -70.3 | +371.6 | +276.6 |
| resolve | +670.7 | +531.3 | -488.3 | +237.9 |

`nearest` is now the best and is the default (reverted from `conservative`). The `conservative` 15% bleed was helpful when CallStation was -463; now that river bluffing is healthy (+439.5), it hurts value betting.

### What changed v6 → v7
- Reverted broken RCA "fixes" that caused exploitability regression to 1.82–1.87:
  - `cfr_fast.pyx`: removed `eq_bucket >= 2` donk restriction (was changing EQ0/EV1 from 6→7 actions)
  - `cfr_fast.pyx` + `cfr.py`: all-in dampening back to `raise_count < 2`, 0.7×
  - `abstraction.py`: removed `eq_bucket >= 2` from `get_available_actions()` to match training
- Fixed `exploitability.py` RNG sharing bug: br0/br1 now use isolated `Random(seed)` instances
- Changed GTOAgent default mapping `conservative` → `nearest` (conservative now -33.5 avg)

### Known Issues

| Issue | Severity | Description | Fix |
|-------|----------|-------------|-----|
| OverfoldBot -291.4 | High | GTO not exploiting passive folders enough | Investigate postflop bet frequency vs folding opponents |
| WeirdSizingBot -852.7 | High | Large loss to off-tree sizings (esp. 120% overbet → -1305 raw) | Translation improvements |
| PerturbBot -210.4 | Medium | Losing slightly to near-random play | Expected for GTO vs exploitative variance |
| EQ0 river 100% bet | Medium | Specific EQ0 nodes still 100% bet on river (overall bluff ratio 35.1% HEALTHY) | Structural abstraction limit |
| Premium pair limp | Medium | 4 nodes: PREMIUM/HIGH_PAIR at EQ5/EQ6 limp 21-100% | Rare unconverged path |
| Strong hand fold | Medium | EQ6 folds 100% — bucket boundary artifact | Widen equity bucket ranges |
| All-in overuse | Medium | 520 nodes flagged | Dampening too broad (`raise_count < 2`) |
| Flat strategies | Medium | 1334 nodes with flat (uniform) strategies | 2x phase schedule undertrained |
| Frequency anomalies | Low | 65814 nodes flagged | Abstraction granularity limit |
| #2 Bridge default | ~~Critical~~ **FIXED** | Now `nearest` (reverted from conservative) | match_engine.py |
| #3 engine.py history | ~~High~~ **FIXED** | BET_HALF_POT + aliases added | engine.py |
| #11 NitBot regression | ~~Medium~~ **FIXED** | +235.9 (was -19) | — |
| #1 CallStation -463 | ~~Critical~~ **FIXED** | +439.5 (river bluff ratio now 35% healthy) | — |

Full details: `RCA_v6_50M.md`

---

## Common Tasks

### Add a New Bot to Eval Harness

1. Add class to `eval_harness/adversaries.py` inheriting `Agent`
2. Implement `decide(ctx: HandContext) → AgentDecision`
3. Add to `get_all_adversaries()` list

### Change the Action Abstraction

1. Edit `Action` enum in `abstraction.py`
2. Update `get_available_actions()` in `abstraction.py`
3. Update `_PREFLOP_CONCRETE_MAP` / `_POSTFLOP_CONCRETE_MAP` in `abstraction.py` (shared by engine + eval)
4. Update `_to_concrete_action()` in `engine.py`
5. Update `get_actions()` in `cfr_fast.pyx` if needed
6. Rebuild Cython: `venv/bin/python setup_cython.py build_ext --inplace`
7. Retrain from scratch (abstraction changes invalidate strategy.json)

### Run Exploitability Check

```python
from server.gto.cfr import CFRTrainer
from server.gto.exploitability import exploitability_multiseed

trainer = CFRTrainer()
trainer.load("server/gto/strategy.json")
result = exploitability_multiseed(trainer, samples=500, seeds=[42, 123, 456])
print(result)  # {'mean': 1.28, 'ci': 0.04, 'per_phase': {...}}
```

### Query Strategy for a Hand

```python
from server.gto.cfr import CFRTrainer
from server.gto.abstraction import Action

trainer = CFRTrainer()
trainer.load("server/gto/strategy.json")

# Get preflop OOP strategy for PREMIUM_PAIR (bucket 0)
strategy = trainer.get_strategy('preflop', 0, (), 'oop')
# {0: 0.0, 6: 0.95, 5: 0.05}  → 95% open raise, 5% all-in

# IP strategies at empty history show uniform (33/33/33) — this is CORRECT
# IP only acts after OOP; query with non-empty history
strategy = trainer.get_strategy('preflop', 0, (Action.OPEN_RAISE,), 'ip')
```

---

## Architecture Gotchas

1. **Single history reconstruction**: `abstraction.concrete_to_abstract_history()` is the canonical implementation. Both `engine.py` and `match_engine.py` delegate to it — they are guaranteed identical. `game.py` and `simulate.py` have a separate *multiplayer-compression* function (`_get_abstract_history`) which is a different operation and intentionally not unified.

2. **Cython required for performance**: Pure Python fallback runs ~70x slower. If `cfr_fast.pyx` changes, rebuild with `setup_cython.py` or training will silently use slow path.

3. **IP empty history is intentionally uniform**: Don't mistake IP empty-history uniform strategy for a bug. IP only acts after OOP acts first.

4. **strategy.json is the source of truth**: If you change abstraction (bucket scheme, action set), you must retrain. Old strategy files are invalid.

5. **Parallel training has benign races**: Workers share a memory-mapped node pool without locks. This is theoretically sound for CFR+ (near-zero impact on convergence) but adds small variance.

6. **Postflop pot = 1.0 normalized**: All bet sizings in the solver are fractions of the normalized pot. The engine translates to real chips. `inv = [0.5, 0.5]` means each player has contributed 0.5 to a pot of 1.0.

7. **`nearest` is the default mapping**: GTOAgent default is `"nearest"`. The `conservative` mapping (15% bleed from bet actions → CHECK_CALL) was default in v6 to compensate for river over-bluffing. With the river bluff ratio now healthy (35.1%), conservative hurts value betting — nearest is +276.9 vs conservative -33.5 avg.
