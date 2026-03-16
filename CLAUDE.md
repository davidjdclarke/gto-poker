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
├── run_eval_harness.py        # Full eval suite (4 test suites + multi-seed gauntlet)
├── run_h2h.py                 # v12: GTO vs GTO head-to-head match runner with detailed analysis
├── run_phase0_validation.py   # V10 Phase 0: validation, diagnostics, cliff analysis
├── run_experiment.py          # Experiment tracking & comparison
├── simulate.py                # Self-play simulation (GTO vs heuristic bots)
├── setup_cython.py            # Cython build (-O3 -march=native)
├── convergence_study.py       # Exploitability vs iteration analysis
├── run_toy_validation.py      # v11: Kuhn poker validation for solver dynamics
│
├── server/gto/
│   ├── cfr.py                 # CFR+ trainer (Python orchestration)
│   ├── cfr_fast.pyx           # Cython inner loop (~70x speedup, configurable grid)
│   ├── abstraction.py         # 2D bucketing, action menus, infoset keys, selective actions
│   ├── equity.py              # E[HS²] evaluation + hand type classifier
│   ├── engine.py              # Live game bridge (gto_decide entry point)
│   ├── exploitability.py      # MC best response + audit tools
│   ├── kuhn.py                # Kuhn poker benchmark (3-card game)
│   ├── local_refine.py        # v12: Refine 2.0 — blueprint CFVs, adaptive threshold, texture blend
│   ├── board_texture.py       # v11: board texture classification (dry/mono/draw/paired)
│   └── strategy.json          # Saved strategy (30–40 MB)
│
├── eval_harness/
│   ├── match_engine.py        # HeadsUpMatch, GTOAgent (+ compute_strategy(), DecisionRecord extensions)
│   ├── adversaries.py         # 7 exploit bots (Nit, Aggro, Station, etc.)
│   ├── advanced_adversaries.py # v12: 12 policy-distorted bots (4 styles × 3 intensities)
│   ├── aivat.py               # v12: Half-AIVAT variance reduction
│   ├── offtree_stress.py      # 11 off-tree sizing variants
│   ├── translation_ab.py      # 4 mapping schemes A/B test
│   ├── fast_equity.py         # Preflop cache + fast postflop equity
│   └── external/              # v12: external benchmarks
│       ├── slumbot_client.py  # Slumbot API client
│       ├── slumbot_match.py   # Slumbot match orchestrator
│       └── openspiel_adapter.py # OpenSpiel stub (v13)
│
├── tests/
│   └── test_kuhn_benchmark.py # All 16 Nash eq. tests — must pass
│
├── docs/plans/                # Action plans for each iteration/improvement cycle
├── docs/results/              # Per-iteration eval result docs (historical tracking)
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
    # v9 additions (fills WeirdSizingBot translation gaps)
    BET_QUARTER_POT         = 13  # ~1/4 pot (check ↔ bet_third gap)
    BET_THREE_QUARTER_POT   = 14  # ~3/4 pot (2/3 ↔ pot gap)
    BET_DOUBLE_POT          = 15  # ~2x pot (overbet ↔ all-in gap)
    BET_TRIPLE_POT          = 16  # ~3x pot (river only, via selective action — v12)
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
    --iterations 5000000 \        # 5M per run, cumulative from previous save
    --workers 6 \                 # parallel workers (shared-memory node pool)
    --averaging-delay 100000 \    # linear weighting starts after 100k iters
    --phase-schedule 2x \         # 2x or 3x flop/turn emphasis
    --allin-dampen old \          # old (rc<2,0.7x) or new (rc==0,0.5x)
    --regret-discount 0.995 \     # DCFR: discount factor <1.0 (1.0=CFR+)
    --weight-schedule exponential \ # linear/exponential/polynomial
    --weight-param 1.001 \        # base for exp, power for polynomial
    --checkpoint-interval 10000000 \ # save checkpoint every N iters
    --checkpoint-eval \           # compute exploitability at checkpoints
    --checkpoint-gauntlet \       # run quick gauntlet at checkpoints
    --action-grid 13              # v11: explicit grid size (13 or 16)
```

**Performance:** ~0.03–0.04 ms/iter with Cython (6 workers), ~29 min for 50M iterations total.

### Solver Dynamics Options (v9)

- **CFR+ (default):** `--regret-discount 1.0` — standard regret flooring, no decay
- **DCFR:** `--regret-discount 0.995` — discount old regrets by gamma each iteration
- **Weight schedules:** `--weight-schedule linear` (default), `exponential`, `polynomial`, `scheduled`
  - Exponential: weight = param^(t-delay). Try `--weight-param 1.001`
  - Polynomial: weight = (t-delay)^param. Try `--weight-param 2.0`
  - Scheduled (v11): DCFR with time-varying discount. weight = (t/(t+1))^gamma, regret_discount = t^1.5/(t^1.5+1). `--weight-param` sets gamma (default 2.0)

### Evaluation Diagnostics (v9)

```python
# EV decomposition (requires detailed_tracking=True)
from eval_harness.ev_decomposition import decompose_by_street, callstation_dashboard
match = HeadsUpMatch(gto, opponent, detailed_tracking=True)
result = match.play(5000)
street_ev = decompose_by_street(result.hands, big_blind=20)
dashboard = callstation_dashboard(result.hands, big_blind=20)

# Bridge pain map
from eval_harness.bridge_pain import analyze_bridge_pain, format_pain_map
analysis = analyze_bridge_pain(gto.bridge_log, result.hands, big_blind=20)
print(format_pain_map(analysis))
```

---

## Current State (v12, 2026-03-16)

**BLUEPRINT DECISION: poly(2.0) + refine** — `experiments/v11_poly2_100M.json` with `mapping="refine"`
**Previous best: B0 + refine** — `experiments/best/v9_B0_100M_allbots_positive.json` with `mapping="refine"`
**Best mapping: `refine`** (v12: Refine 2.0 — blueprint CFVs, adaptive threshold, board-texture blend)
**B0 exploitability:** 1.2211 bb/100 | **poly(2.0) exploitability:** 1.2496 bb/100
**Nodes:** B0: 1,055,003 | poly(2.0): 1,059,640
**Gauntlet average (poly2 + refine):** +215.9 bb/100 classic, +102.8 bb/100 advanced (10k hands × 3 seeds)
**Gauntlet average (B0 + refine):** +202.6 bb/100 classic, +124.7 bb/100 advanced (10k hands × 3 seeds)
**WS3 training status:** COMPLETE — REJECTED (two runs). See below.
**⚠️ strategy.json WARNING:** Currently contains WS3-v2 (24.28 bb/100, underconverged). For production, load `experiments/v11_poly2_100M.json` explicitly.

### Gauntlet Results (v12, 10k hands × 3 seeds — Blueprint Decision)

#### Classic Gauntlet

| Bot | B0 + conf_nearest | B0 + refine | poly2 + refine | poly2 + conf_nearest |
|-----|------------------:|------------:|---------------:|---------------------:|
| NitBot | +216.1 | +253.1 | +158.1 | +151.3 |
| AggroBot | +479.6 | +528.0 | +549.0 | +536.8 |
| OverfoldBot | +3.3 | -14.0 | -31.4 | -43.2 |
| CallStationBot | +54.5 | +57.9 | +89.6 | +84.6 |
| DonkBot | +592.0 | +625.4 | +735.5 | +739.0 |
| WeirdSizingBot | -159.7 | -86.4 | **+32.7** | +6.4 |
| PerturbBot | +41.2 | +54.4 | -22.1 | +17.1 |
| **Average** | **+175.3** | **+202.6** | **+215.9** | **+213.2** |

#### Advanced Gauntlet (12 policy-distorted bots)

| Style | B0 + refine | poly2 + refine | poly2 + conf_nearest |
|-------|------------:|---------------:|---------------------:|
| Aggressive (avg) | +340.5 | +258.7 | — |
| Nit (avg) | -87.2 | +2.6 | — |
| Station (avg) | +319.9 | +181.4 | — |
| Overfolder (avg) | -74.3 | -31.5 | — |
| **Average** | **+124.7** | **+102.8** | **+105.7** |
| Robustness score | +11.5 | **+31.2** | — |
| Worst case | -144.6 | **-81.0** | — |

> **Decision rationale:** poly2+refine wins the weighted scoring matrix (88.5 vs 66.8) primarily due to WeirdSizingBot (+32.7 vs -86.4 = +119.1 bb/100 difference × 20% weight = +23.8 composite points). Classic avg also leads (+13.3 bb/100 × 30% = +4.0). B0+refine leads on advanced avg (+21.9 × 20% = +4.4), OverfoldBot (+17.4 × 10% = +1.7), and exploitability (0.029 × 20% = negligible). WeirdSizingBot gap is decisive. See `docs/results/v12_blueprint_decision.md` for full analysis.

### AIVAT Limitation Note

AIVAT (WS2) requires per-opponent calibration to be valid for gauntlet evaluation. The bucket-EV table built from GTO self-play has near-zero expected values, but continuation values against weak opponents are +100s of bb. The correction `cont_value - baseline ≈ cont_value` produces massive variance inflation, not reduction. **AIVAT is only valid for near-GTO matchups** (e.g., Slumbot or GTO exploitability measurement). The `--aivat` flag will produce unreliable numbers in gauntlet mode until per-opponent calibration is added.

### v10 Experiment Results Summary

**V10 Phase 0:** Corrected B0 baseline to +28.5 bb/100 (nearest). Found and fixed 16-action grid bug and OpponentProfile regression.

**V10 Phase 3:** `confidence_nearest` mapping improved average to +225.1 bb/100 without retraining. WeirdSizingBot improved +568 bb/100.

**V9-B0 (13-action, 100M):** Exploitability 1.22, corrected gauntlet +28.5 (nearest) / +225.1 (confidence_nearest).

**16-action grid (200M):** Exploitability **41.4** (did NOT converge). Gauntlet +318 — weaker than B0 despite 2x training. The 3 new bet sizes created ~2x more nodes (2.0M vs 1.05M) and exploitability plateaued at ~40 instead of converging. NitBot -1289, WeirdSizingBot +218 (no improvement over B0's +220). **Verdict: not viable at current iteration budgets.**

**DCFR sweep (16-action, 20M):** Tested gamma = {0.999, 0.995}. Both showed exploitability *increasing* from 10M to 20M. DCFR destabilizes convergence on wider action trees. **Verdict: counterproductive for this grid.**

### What changed v11 → v12
- **Advanced gauntlet (WS0):** `eval_harness/advanced_adversaries.py` — 12 `PolicyDistortedBot` instances (4 styles: aggressive/nit/station/overfolder × 3 intensities). Distorts GTO base strategy via multiplicative family reweighting with `min_support=0.01` floor. `robustness_score = mean - 0.5×std`. `--gauntlet-mode {classic,advanced,full}` CLI flag.
- **AIVAT variance reduction (WS2):** `eval_harness/aivat.py` — bucket-EV approximation control variate. `AivatResult` dataclass. `build_bucket_ev_table()`, `aivat_adjusted_result()`. `--aivat` and `--build-bucket-ev` flags. `DecisionRecord` extended with `strategy`, `available_actions`, `infoset_key`, `abstract_action`, `pot_odds`.
- **`compute_strategy()` refactor (prereq):** `GTOAgent.decide()` split into `compute_strategy(ctx) → dict` + sample step. Allows `PolicyDistortedBot` to intercept strategy before sampling.
- **Refine 2.0 (WS1):** `local_refine.py` rewritten — blueprint CFV 1-ply backup (`_blueprint_cfv_for_action()`) replacing heuristic payoffs, adaptive threshold (`compute_adaptive_threshold()` using visit count + entropy + board texture), board-texture-aware blend alpha (`_compute_blend_alpha()`). Budget cap `_MAX_REFINE_BUDGET=100`. Diagnostic counters `blueprint_cfv_count` / `heuristic_fallback_count`.
- **Selective river overbet (WS3):** `BET_TRIPLE_POT = 16` added to `Action` enum. NOT in base grid — registered only via `add_selective_action('river', ...)`. `--selective-river-overbet` flag in `train_gto.py`. Cython `ACT_BET_TRIPLE_POT = 16` with bet size `3.0×pot`. Sizing in both `match_engine.py` and `engine.py`.
- **External calibration scaffold (WS5):** `eval_harness/external/` — `SlumbotClient`, `SlumbotMatch`, `openspiel_adapter.py` (stub). `--slumbot` and `--tier2` CLI flags.

### What changed v10 → v11
- **Cython grid configurability (A2):** `cfr_fast.pyx` now accepts `action_grid_size` parameter (13 or 16). `set_action_grid_size()` / `get_action_grid_size()` in Cython. `--action-grid` CLI flag in `train_gto.py`. Python/Cython mismatch is now impossible.
- **OpponentProfile retired (A3):** Default disabled in `run_eval_harness.py`. Use `--opponent-model` flag to re-enable.
- **confidence_nearest default (C1):** GTOAgent now defaults to `confidence_nearest` mapping instead of `nearest`.
- **Scheduled DCFR (B2):** New `weight_schedule_mode=3` / `--weight-schedule scheduled` implements time-varying DCFR: regret discount `t^α/(t^α+1)` with α=1.5, strategy weight `(t/(t+1))^γ`.
- **Local refinement prototype (C2):** `server/gto/local_refine.py` — mini-CFR at off-tree decision points. New `"refine"` mapping mode in GTOAgent. Trigger policy with bridge-pain integration.
- **Toy-game validation (B3):** `run_toy_validation.py` — validates all solver dynamics variants on Kuhn Poker before HUNL.
- **Board texture (D3):** `server/gto/board_texture.py` — classifies boards as dry/monotone/draw-heavy/paired/connected. Auxiliary signal for confidence blending.
- **Selective action additions (D1):** `add_selective_action()` / `clear_selective_actions()` in `abstraction.py` — adds actions to specific postflop contexts without expanding the entire grid.
- **Configurable equity buckets (D2):** `set_equity_buckets()` in `abstraction.py` — supports 9-bucket experiments.

### What changed v9 → v10
- **Bugs fixed:** 16-action grid compatibility bug, OpponentProfile regression
- **New mapping:** `confidence_nearest` (+225.1 avg vs +28.5 for `nearest`)
- **Tooling:** Multi-seed gauntlet with CIs, Phase 0 validation script, grid auto-detection
- **Evaluation corrected:** Published +677 avg corrected to +28.5 (nearest) / +225.1 (confidence_nearest)

### What changed v7 → v9-B0
- **Training config:** 2x phase schedule (was 3x), old all-in dampening, 100M iterations (was 67M)
- **v9 infrastructure added:**
  - Phase 0: EV decomposition, bridge pain map, checkpoint gauntlet
  - Phase 2: DCFR regret discounting, pluggable weight schedules
  - Phase 3A: 16-action grid (code in place, but 13-action strategy is production)

### Known Issues

| Issue | Severity | Description | Status |
|-------|----------|-------------|--------|
| WeirdSizingBot (B0+refine) | ~~Medium~~ **RESOLVED** | Was -86.4 with B0+refine; poly2+refine achieves +32.7 | Blueprint switched to poly2+refine (v12 decision) |
| AIVAT gauntlet calibration | Medium | GTO self-play table produces biased corrections vs weak bots | Known limitation — only valid for near-GTO matches |
| 16-action grid 15% coverage | High | Only 15% of nodes well-visited at 200M | Selective expansion available (v11 D1); WS3 rejected — node count doubled to 2M, exploitability 88.5 |
| BET_TRIPLE_POT / WS3 | ~~High~~ **REJECTED** | Selective river overbet doubled node count; NitBot -1,114 bb/100 | Re-attempt only with narrow selectivity predicate (equity bucket ≥ 6 filter) |
| strategy.json overwritten | **High** | WS3-v2 training wrote 24.28 bb/100 strategy to strategy.json | Load `experiments/v11_poly2_100M.json` explicitly for all production use |
| street_ev attribution in run_h2h.py | Low | `street_ev` tracks chip investment, not EV; by-street numbers are not meaningful EVs | Replace with proper attribution (hand net → last street reached) in v13 |
| OpponentProfile counterproductive | ~~Medium~~ **RETIRED** | Hurts performance, especially vs CallStation | Disabled by default (v11 A3) |
| EQ0 river 100% bet | Low | Specific EQ0 nodes still 100% bet on river | Structural abstraction limit |
| `--averaging-delay` not parsed | ~~Medium~~ **FIXED** | Value leaked as positional arg → "unknown sampling" error | `averaging_delay_override` variable added to `train_gto.py` |
| Cython hardcodes 16-action | ~~Medium~~ **FIXED** | Can't train 13-action grid with Cython | Configurable via `set_action_grid_size()` (v11 A2) |
| 16-action grid bug | ~~Critical~~ **FIXED** | `_postflop_actions()` broke 13-action B0 eval | Grid auto-detection added (v10) |
| Published v9 results inflated | ~~Critical~~ **FIXED** | +677 avg was from 500-hand noise + grid bug | Corrected to +28.5 nearest / +225 conf_nearest |

Full details: `docs/results/v12_complete_report_20260316.md`, `docs/results/v12_blueprint_decision.md`, `docs/results/v11_results_20260315.md`, `docs/results/v10_complete_report.md`

---

## Results Tracking

**Every eval run must be documented in `docs/results/`** with a file named `{version}_{iterations}_{date}.md`.

Each results doc should include: exploitability, gauntlet table, bridge mapping A/B, off-tree stress test, strategy audit, and a comparison to the previous iteration. This creates a historical record of how the strategy evolves over time.

Existing results:
- `docs/results/v12_complete_report_20260316.md` — **v12 final report**: all workstreams, 50k H2H analysis, tree saturation finding, v13 recommendations
- `docs/results/v12_blueprint_decision.md` — **v12 blueprint decision**: poly2+refine chosen; full gauntlet matrix scoring + WS3 rejection
- `docs/results/v11_results_20260315.md` — **v11 results**: local refinement, solver dynamics, selective abstraction
- `docs/results/v11_baseline_report.md` — v11 baseline template and configuration
- `docs/results/v10_complete_report.md` — v10 definitive report: corrected baseline, confidence mapping, full analysis
- `docs/results/v10_phase0_report.md` — v10 Phase 0: bugs found, corrected gauntlet, cliff analysis
- `docs/results/v9_complete_report.md` — v9 definitive report: B0 baseline, DCFR sweep, 16-action grid
- `docs/results/v9_B0_100M_20260314.md` — v9-B0 baseline detail (results now known to be inflated)
- `docs/results/v9_16action_200M_20260315.md` — 16-action grid experiment detail
- `docs/results/v7_67M_20260313.md` — v7 at 67M iterations

## Action Plans

**Before starting a new improvement cycle, create an action plan in `docs/plans/`** describing the goals, hypotheses, and planned changes. Update the plan as work progresses. This provides a decision log for why changes were made.

Existing plans:
- `docs/plans/ACTION_PLAN_v13.md` — **v13 research plan (current)**: pseudo-harmonic mapping, safe subgame solving, VR-MCCFR, suit isomorphism, EMD bucketing, board texture key dimension
- `docs/plans/ACTION_PLAN_v12.md` — v12 research plan (complete)
- `docs/plans/ACTION_PLAN_v11.md` — v11 research plan (complete)
- `docs/plans/ACTION_PLAN_v10.md` — v10 research plan
- `docs/plans/ACTION_PLAN_v9.md` — v9 improvement plan
- `docs/plans/ACTION_PLAN_v8_ablations.md` — v8 ablation study plan
- `docs/plans/ACTION_PLAN_v7.md` — v7 improvement plan

---

## Common Tasks

### Run V10 Phase 0 Validation

```bash
# Full validation (5k hands, 3 seeds — ~15 min)
venv/bin/python run_phase0_validation.py all --hands 5000

# Quick smoke test (200 hands, 2 seeds)
venv/bin/python run_phase0_validation.py validate --quick

# Causal diagnostics only (B0 vs v7)
venv/bin/python run_phase0_validation.py diagnose --bots CallStationBot

# Checkpoint cliff analysis
venv/bin/python run_phase0_validation.py cliff
```

### Run Multi-Seed Gauntlet

```bash
# Multi-seed gauntlet with confidence intervals
venv/bin/python run_eval_harness.py --gauntlet --hands 5000 --seeds 42,123,456
```

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

### Run Head-to-Head Analysis (GTO vs GTO)

```bash
# Default: poly2+refine (P0) vs B0+refine (P1), 50k hands, 3 seeds
venv/bin/python run_h2h.py

# Custom configs
venv/bin/python run_h2h.py \
    --p0-strategy experiments/v11_poly2_100M.json --p0-mapping refine \
    --p1-strategy experiments/best/v9_B0_100M_allbots_positive.json --p1-mapping refine \
    --hands 50000 --seeds 42,123,456
```

Reports: overall EV (bb/100 per seed), by position (IP/OOP), by equity bucket, action frequency comparison, showdown stats, top-15 strategy divergence infosets (frequency-weighted TV distance).

Note: by-street numbers use chip-investment attribution (see Known Issues) and should not be interpreted as per-street EV.

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

7. **`refine` is the recommended mapping (v12)**: Layers mini-CFR solve on top of `confidence_nearest` for extreme off-tree bets. The v12 blueprint is **poly2+refine**: load `experiments/v11_poly2_100M.json` and pass `mapping="refine"`. This achieves +215.9 classic avg and +32.7 WeirdSizingBot (vs -86.4 for B0+refine). GTOAgent default is `"confidence_nearest"` — pass `mapping="refine"` for best results.

8. **Action grid auto-detection is required**: When loading a strategy, call `detect_action_grid_from_strategy()` and `set_action_grid()`. The B0 strategy uses a 13-action grid; the Cython trainer creates 16-action nodes. Mismatched grids cause ~45% lookup failures. `get_trainer()` in engine.py handles this automatically.

9. **OpponentProfile is disabled for evaluation**: The `OpponentProfile` model was found counterproductive (v10 Phase 0). Use `--no-opponent-model` flag or pass `opponent_profile=None` to GTOAgent for accurate results. The OP model needs redesign before it's useful.

10. **Cython action grid is now configurable (v11)**: Call `cfr_fast.set_action_grid_size(13)` or `cfr_fast.set_action_grid_size(16)` before training. The `train_fast()` function also accepts `action_grid_size` parameter. `train_gto.py --action-grid 13` does this automatically. Python's `set_action_grid()` and Cython's `set_action_grid_size()` must agree.

11. **Local refinement is heuristic, not safe**: `local_refine.py` approximates subgame solving with a simplified payoff model. It's a prototype for evaluation, not a theoretically safe implementation. Use `"refine"` mapping in GTOAgent to enable it.

12. **Selective actions don't affect existing strategies**: `add_selective_action()` adds actions to the Python action menu only. Existing strategy files won't have nodes for the new actions, so lookups will miss and use uniform. Must retrain to populate the new nodes.

13. **OpponentProfile is disabled by default since v11**: The `--opponent-model` flag re-enables it. `--no-opponent-model` is still accepted but is now a no-op.

14. **`train_gto.py` uses positional arg parsing, not argparse**: Pass iterations as `train_gto.py 100000000` (first positional), sampling as second positional (`external` or `vanilla`). All `--flag value` pairs are parsed separately and their values are excluded from the positional list. If a flag value is mistakenly left as positional, it appears as `sampling` and triggers "Unknown sampling method" error.

15. **AIVAT is only valid for near-GTO matchups**: `--aivat` with a GTO self-play calibration table produces biased corrections when GTO plays against weak opponents (gauntlet bots). The control variate is zero-mean only when `E[cont_value | bucket] ≈ baseline`, which holds for GTO vs GTO but not GTO vs NitBot/AggroBot/etc. For gauntlet evaluation, ignore the AIVAT column or use per-opponent calibration.
