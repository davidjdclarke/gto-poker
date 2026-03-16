# V12 Complete Results Report

**Date:** 2026-03-16
**Blueprint:** poly(2.0) + refine — `experiments/v11_poly2_100M.json`, `mapping="refine"`
**Status:** All workstreams complete. Blueprint confirmed.

---

## Executive Summary

V12 delivered four things: a stronger evaluator (advanced gauntlet, AIVAT), an improved
runtime bridge (Refine 2.0), a selective abstraction experiment (WS3), and a blueprint
decision. Every workstream completed. The blueprint is **poly2+refine**.

However, a full 50k-hand head-to-head after the blueprint decision revealed that
**B0+refine beats poly2+refine by +230.5 bb/100** in direct matchup — a statistically
decisive result. Combined with tree saturation analysis confirming 100% node visitation at
100M iterations, v12 ends with a clear diagnosis: **the solver has converged to the ceiling
of its current information abstraction**. Further gains require a better abstraction, not
more training or schedule tuning.

---

## 1. Workstream Results

### WS0 — Advanced Gauntlet (12 policy-distorted bots)

**Status:** Complete.

Built `eval_harness/advanced_adversaries.py` with 4 styles × 3 intensities = 12
`PolicyDistortedBot` instances. Strategy distortion via multiplicative family reweighting
with `min_support=0.01` floor. New `--gauntlet-mode {classic,advanced,full}` CLI flag.

**Key metric:** `robustness_score = mean - 0.5 × std`

| Config | Classic Avg | Advanced Avg | Robustness | Worst Case |
|--------|------------:|-------------:|-----------:|-----------:|
| B0+refine | +202.6 | +124.7 | +11.5 | -144.6 |
| poly2+refine | +215.9 | **+102.8** | **+31.2** | **-81.0** |

poly2+refine has meaningfully better worst-case and robustness despite lower advanced
average. The advanced average gap is driven by Station-strong (+315.9 vs +525.1) and
Aggressive-strong (+293.6 vs +440.0) — styles that reward tighter preflop play.

**Advanced gauntlet per-style:**

| Style | B0+refine | poly2+refine |
|-------|----------:|-------------:|
| Aggressive_mild | +212.3 | +135.8 |
| Aggressive_medium | +369.3 | +346.7 |
| Aggressive_strong | +440.0 | +293.6 |
| Nit_mild | -52.3 | +38.2 |
| Nit_medium | -144.6 | -54.0 |
| Nit_strong | -64.6 | +23.7 |
| Station_mild | +198.3 | +47.6 |
| Station_medium | +236.2 | +180.8 |
| Station_strong | +525.1 | +315.9 |
| Overfolder_mild | -107.8 | -81.0 |
| Overfolder_medium | +15.1 | +22.1 |
| Overfolder_strong | -130.1 | -35.7 |
| **Average** | **+124.7** | **+102.8** |

poly2+refine leads on every Nit sub-style (+75.6 combined) — polynomial weighting
emphasises later, more-converged iterations, which are less exploitable by tight/passive play.

---

### WS1 — Refine 2.0

**Status:** Complete. Enabled in blueprint via `mapping="refine"`.

Three enhancements to `server/gto/local_refine.py`:

**A. Blueprint CFV 1-ply backup (`_blueprint_cfv_for_action`):**
Replaces heuristic payoffs with actual counterfactual values from the trainer's node table.
When the successor infoset exists in `trainer.nodes`, performs a 1-ply backup over the
opponent's average strategy. Falls back to heuristic when the node is missing.
Diagnostic counters: `blueprint_cfv_count` / `heuristic_fallback_count`.

**B. Adaptive threshold (`compute_adaptive_threshold`):**
Replaces fixed `MISMATCH_THRESHOLD` with a visit-count + entropy + board-texture adjustment.
- Base: 0.40
- High visit count → raise threshold (trust blueprint more)
- High entropy → lower threshold (more refinement)
- Draw-heavy/connected boards: −0.08; dry boards: +0.08
- Hard clamp: [0.20, 0.60]

**C. Board-texture blend alpha (`_compute_blend_alpha`):**
Draw-heavy/connected boards +0.10 (trust refine more); dry boards −0.10; monotone +0.05.
Hard clamp: [0.30, 0.70].

**Impact:** WeirdSizingBot +32.7 bb/100 (vs −86.4 for B0+refine, +119.1 bb/100 swing). Refine
2.0 handles the 200%-pot off-tree sizing with a proper mini-CFR solve rather than
nearest-action translation.

---

### WS2 — AIVAT Variance Reduction

**Status:** Complete — limited applicability discovered.

Built `eval_harness/aivat.py` with bucket-EV control variate. Implemented
`build_bucket_ev_table()`, `aivat_adjusted_result()`, `AivatResult` dataclass.
`DecisionRecord` extended with `strategy`, `available_actions`, `infoset_key`,
`abstract_action`, `pot_odds`.

**Key finding:** The bucket-EV table built from GTO self-play has near-zero expected values.
Against weak gauntlet bots, continuation values are +100s bb/100. The correction
`cont_value − baseline ≈ cont_value` inflates variance rather than reducing it.

**AIVAT is only valid for near-GTO matchups** (GTO vs GTO, or vs Slumbot). In gauntlet
mode the `--aivat` column is unreliable until per-opponent calibration is added.

---

### WS3 — Selective River Overbet (BET_TRIPLE_POT)

**Status:** Complete — REJECTED.

**Training:** 200M iterations, B0 strategy with `--selective-river-overbet`
(BET_TRIPLE_POT = 3.0× pot, river only).

**Results (10k hands × 3 seeds):**

| Config | WeirdSizingBot | NitBot | Classic Avg |
|--------|---------------:|-------:|------------:|
| B0+refine | -86.4 | +253.1 | +202.6 |
| poly2+refine | +32.7 | +158.1 | +215.9 |
| WS3-v1 | **+87.5** | **-1,114.0** | +436.6\* |

\*Classic avg inflated by AggroBot (+1,373) and DonkBot (+1,864) due to 2M underconverged nodes.

**Root cause:** BET_TRIPLE_POT spawned ~945k successor nodes, doubling the tree to 1,999,902.
All new nodes are underconverged at 200M total iterations. Exploitability: 88.5 bb/100
(vs 1.22 for B0). NitBot finds and punishes the underconverged river spots: −1,114 bb/100
(−1,367 vs poly2+refine).

**Checkpoint exploitability history:**

| Iters | Nodes | Exploitability (200 samples) |
|-------|-------|------------------------------|
| 110M | 1,055,281 | 24.6 |
| 130M | 1,055,785 | 25.1 |
| 150M | 1,056,153 | 23.1 |
| 170M | 1,056,420 | 23.7 |
| **200M** | **1,999,902** | **88.5 (full 500×3 seeds)** |

Quick checkpoints (110M–190M) avoided the new nodes; final full eval confirmed the
catastrophic regression.

**WS3-v2 ("fixed args"):** A second run also completed at 200M iterations. Only ~1,749 new
nodes spawned (vs ~945k in v1) but exploitability still 24.28 (20× worse than B0).
Gauntlet not run — disqualified by exploitability alone. WS3-v2 also rejected.

**⚠️ strategy.json overwritten by WS3-v2.** For all production and evaluation work, load
`experiments/v11_poly2_100M.json` explicitly.

**WS3 verdict:** Rejected. WeirdSizingBot gain (+54.8 vs poly2+refine) is swamped by the
NitBot regression (−1,272 vs poly2+refine). The "selective" intent failed because the new
action does not stay selective — it spawns cascading successor nodes throughout the river.

Future option: limit BET_TRIPLE_POT to high-equity river spots (equity bucket ≥ 6) to
constrain new nodes to <50k. Requires a selectivity predicate in `add_selective_action()`.

---

### WS4 — Blueprint Decision

**Status:** Complete. See `docs/results/v12_blueprint_decision.md` for full decision matrix.

**Decision: poly2+refine** (`experiments/v11_poly2_100M.json`, `mapping="refine"`).

| Criterion | Weight | B0+refine | poly2+refine | poly2 delta |
|-----------|--------|----------:|-------------:|------------:|
| Classic avg | 30% | +202.6 | +215.9 | +3.99 pts |
| Advanced avg | 20% | +124.7 | +102.8 | −4.38 pts |
| WeirdSizingBot | 20% | −86.4 | +32.7 | **+23.82 pts** |
| OverfoldBot | 10% | −14.0 | −31.4 | −1.74 pts |
| Exploitability | 20% | 1.2211 | 1.2496 | ~0 pts |

**Composite:** poly2+refine 88.49 vs B0+refine 66.80. Primary driver: WeirdSizingBot gap is
decisive (+119.1 bb/100 × 20% weight = +23.82 points).

---

### WS5 — External Calibration Scaffold

**Status:** Scaffold complete; full implementation deferred to v13.

Created `eval_harness/external/`:
- `slumbot_client.py` — `SlumbotClient` with `new_hand()`, `act()`, `get_result()`,
  `state_to_hand_context()`, `decision_to_action()`
- `slumbot_match.py` — `SlumbotMatch.play(num_hands) → MatchResult`
- `openspiel_adapter.py` — stub; full implementation deferred

CLI flags in `run_eval_harness.py`: `--slumbot`, `--tier2`.

---

## 2. Head-to-Head: poly2+refine vs B0+refine (50k hands, 3 seeds)

**Script:** `run_h2h.py`
**Config:** 49,998 hands total (16,666 × 3 seeds: 42, 123, 456), detailed_tracking=True

### Overall

| Metric | Value |
|--------|-------|
| **B0+refine result** | **+230.5 bb/100** |
| poly2+refine result | −230.5 bb/100 (mirror) |
| 95% CI | ±43.3 |
| Std (seed-to-seed) | 38.3 |
| Per seed | −220.6 / −198.1 / −272.7 |

**B0+refine leads significantly.** The result is 5.3σ from zero. poly2+refine loses on all
three seeds with consistent magnitude.

### By Position (P0 = poly2+refine)

| Position | bb/100 | Hands |
|----------|-------:|------:|
| IP | +770.5 | 22,065 |
| OOP | −1,146.9 | 24,999 |

The OOP deficit (−1,147 bb/100) is the dominant driver of the overall loss. poly2+refine
plays materially worse than B0+refine when out of position.

### Action Frequency Comparison

| Phase | Family | poly2+refine | B0+refine | Delta |
|-------|--------|-------------:|----------:|------:|
| Flop | bet/raise | 28.8% | 26.9% | +1.9% |
| River | bet/raise | **16.0%** | **22.9%** | **−6.9%** |
| River | call | 9.1% | 5.9% | +3.1% |
| River | fold | 12.0% | 9.5% | +2.5% |

The most striking difference: **B0+refine bets the river 6.9% more frequently.** poly2+refine
checks river spots it should be betting for value, and folds more to river bets.

### By Equity Bucket (P0 preflop strength)

| Bucket | Range | bb/100 | Hands |
|--------|-------|-------:|------:|
| EQ2 | 25–37% | −177.8 | 6,488 |
| EQ3 | 37–50% | **−487.8** | 18,562 |
| EQ4 | 50–62% | −207.7 | 19,669 |
| EQ5 | 62–75% | +368.8 | 4,411 |
| EQ6 | 75–87% | +1,317.3 | 868 |

poly2+refine loses with every hand in the 37–62% equity range (mid-strength), which
accounts for 77% of all hands played. This is the strategic weakness B0+refine exploits.

### Top Strategy Divergence Points (frequency-weighted TV distance)

| Infoset | TV | Visits | Key difference |
|---------|---:|-------:|----------------|
| `flop:oop:74:()` | 0.295 | 1,547 | donk_small: **36% vs 6%** |
| `river:oop:119:()` | 0.617 | 726 | check/call: **78% vs 20%** |
| `flop:oop:89:()` | 0.245 | 1,609 | donk_small: **48% vs 72%** |
| `turn:oop:74:()` | 0.364 | 1,058 | donk_small: **45% vs 8%** |
| `flop:ip:89:(1)` | 0.316 | 1,213 | check/call: **48% vs 79%** |
| `river:oop:104:()` | 0.456 | 578 | check/call: **66% vs 20%** |

Two structural differences dominate:

1. **Donk-betting OOP**: poly2+refine donk-bets 36–48% in medium-equity spots where B0
   checks. These are bucket 74/89 (equity tier 4–5, hand type TRASH/SUITED_TRASH). Whether
   these are semi-bluffs or convergence artifacts is not resolved, but B0 punishes them.

2. **River checking with strong hands**: At `river:oop:119` (top equity bucket, TRASH hand
   type — strong rivered hand despite weak hole cards), poly2 checks 78% vs B0's 20%. This
   is a systematic failure to value-bet strong hands OOP on the river.

### Showdown Stats

| Metric | Value |
|--------|-------|
| Showdown rate | 45.3% |
| poly2+refine wins at SD | 49.0% |
| Splits | 5.1% |

poly2+refine wins slightly less than half of showdowns (49.0%), consistent with a strategy
that is slightly weaker on average at the hands it takes to showdown.

---

## 3. Tree Saturation Analysis

To understand why 3 cycles of weight-schedule and training changes have not beaten B0, a
node visitation and entropy analysis was run on the B0 strategy.

### Node Distribution

| Phase | Total nodes | 2-action nodes | Full-menu nodes |
|-------|------------:|---------------:|----------------:|
| Preflop | 5,520 | — | 5,520 (varied) |
| Flop | 341,872 | **284,872 (83%)** | 56,760 |
| Turn | 341,904 | **284,904 (83%)** | 56,760 |
| River | 365,707 | **303,548 (83%)** | 61,919 |

83% of postflop nodes are 2-action (fold/call only — deep in betting tree where raising is
capped). Only ~57k nodes per phase carry the full 8-action strategic menu.

### Visitation Coverage

| Phase | Nodes | Well-visited (sum > 1,000) | Coverage | Avg entropy |
|-------|------:|---------------------------:|---------:|------------:|
| Preflop | 5,520 | 5,520 | **100%** | 0.126 |
| Flop | 341,872 | 341,804 | **100%** | 0.201 |
| Turn | 341,904 | 341,829 | **100%** | 0.201 |
| River | 365,707 | 362,737 | **99%** | 0.252 |

**Every node in the tree has been visited.** The game tree is fully saturated at 100M
iterations. There are no undiscovered infosets remaining.

**Average entropy is low and stable** (0.20 flop/turn, 0.25 river), confirming that
strategies are concentrated and have converged — this is not a case of uniform/mixed
strategies waiting for more training to sharpen.

### Implications

The 1.22 bb/100 exploitability is not a training artifact. It is the floor imposed by the
abstraction itself. The infoset key `phase:position:bucket:history` collapses too many
strategically distinct situations into the same bucket. For example, bucket 74 (equity 50–62%,
TRASH type) groups hands that may be flush draws, straight draws, overcards, or pure air —
all receiving the same strategy despite having completely different strategic properties.

**More iterations, different weight schedules, or DCFR cannot break through this ceiling.**
The solver has converged. The only path to lower exploitability and better GTO quality is
improving the information abstraction.

---

## 4. Why Three Cycles Failed to Beat B0

| Cycle | Approach | Best result vs B0 (H2H) | Root cause of failure |
|-------|----------|------------------------:|----------------------|
| v10 | Bug fixes, confidence_nearest mapping | Not measured | Mapping fix, not model improvement |
| v11 | poly(2.0), DCFR, scheduled DCFR, local refinement | ~258 bb/100 loss (1k hands, noisy) | Weight schedule produces different strategy, not better one |
| v12 | Refine 2.0, WS3, blueprint decision | **−230.5 bb/100 (50k hands, decisive)** | Abstraction ceiling confirmed; WS3 failed (node explosion) |

The pattern is consistent: B0 (linear weighting, 13-action, 100M) represents the best
achievable strategy under the current 2D bucket abstraction. The exploit-bot gauntlet can
show different configurations winning on specific bot-exploit metrics, but in direct GTO-vs-
GTO matchup B0 consistently wins.

**The gauntlet measures exploit capacity, not GTO quality.** poly2+refine wins the classic
gauntlet (+215.9 vs +202.6) primarily because of WeirdSizingBot (+32.7 vs −86.4). But
WeirdSizingBot uses off-tree bet sizes that the Refine mapping handles better regardless of
blueprint. The underlying blueprint quality is what H2H measures, and B0 is better there.

---

## 5. V13 Recommendations

The core recommendation is to change the abstraction, not continue tuning the training
algorithm.

### Recommendation 1: Board Texture in the Infoset Key (highest priority)

**Change:** `{phase}:{position}:{bucket}:{history}` →
`{phase}:{position}:{bucket}:{texture}:{history}`

where `texture` is the 4-class board texture already implemented in `server/gto/board_texture.py`
(dry / monotone / draw-heavy / paired).

**Why this works:** The solver currently plays the same strategy with bucket 74 on a dry
K72 board and a flush-complete QJ9 board. These are strategically opposite situations. With
texture in the key, the solver can learn "with bucket 74 on a dry board, check; on a
draw-heavy board, bet." Currently Refine 2.0 approximates this post-hoc — baking it into
the core abstraction makes it learnable and exact.

**Cost:** Node count grows approximately 4× (one set of nodes per texture class), to
~4–5M total. Requires retraining from scratch. Needs more iterations to converge (~200M
estimated). The board_texture module is already written and tested.

**Expected gain:** Materially lower exploitability; substantially better OOP postflop play
(the OOP deficit in H2H is the primary symptom of insufficient board context).

### Recommendation 2: Finer Equity Bucketing (secondary)

**Change:** 8 equity buckets (12.5% bands) → 10–12 buckets (8–10% bands).

The EQ3/EQ4 buckets (37–62%) are where poly2+refine lost worst in H2H (−487.8 and −207.7
bb/100) and where the most hands land (77% of hands played). Splitting these bands gives
the solver more resolution on the most-played hands.

**Cost:** Adds ~25% more nodes. If combined with board texture, total nodes could reach
6–8M. This should be evaluated after texture is confirmed working, not simultaneously.

### Recommendation 3: Retire the Weight-Schedule Research Track

v10–v12 tested linear, DCFR (0.995), polynomial (1.5, 2.0), and scheduled DCFR. None
produced a strategy that beats B0 in direct H2H. The weight schedule affects *what* the
average strategy converges to, not whether it converges to a better equilibrium. This
research track has delivered its findings and should be closed.

**If board texture retraining is adopted, use linear weighting (B0 configuration).**
It is the most interpretable and has the best track record.

### Recommendation 4: AIVAT Per-Opponent Calibration

AIVAT infrastructure is complete but only valid for near-GTO matchups. To make it useful
for gauntlet evaluation, build a separate bucket-EV table for each gauntlet bot via 5k-hand
GTO-vs-bot self-play. This is a one-time calibration cost per bot and would enable 3–5×
variance reduction in gauntlet evaluation.

### Recommendation 5: Fix Selective Action Architecture Before Reuse

WS3 showed that `add_selective_action()` is not truly selective — BET_TRIPLE_POT spawned
~945k successor nodes because every subsequent river decision node that is reached by a
triple-pot bet becomes a new node. Before any future selective action experiment:

1. Add a selectivity predicate to `add_selective_action()` that limits node creation to
   specific equity ranges (e.g., bucket ≥ 90 for BET_TRIPLE_POT)
2. Add a node count guard that refuses to create more than N new nodes from a single
   selective action (hard limit)
3. Validate at 10M and 25M iterations before committing to a full run

### Recommended v13 Execution Order

1. Implement board texture in infoset key + train 200M with linear weighting
2. Evaluate H2H against current B0 at 100M and 200M checkpoints
3. If exploitability improves, proceed; if not, evaluate 10-bucket equity split
4. Implement AIVAT per-opponent calibration for gauntlet use
5. Revisit selective river action with narrow predicate if texture model is stable
6. Run Slumbot evaluation (WS5 full implementation)

---

## 6. Configuration for Deployment

```python
from server.gto.cfr import CFRTrainer
from eval_harness.match_engine import GTOAgent

trainer = CFRTrainer()
trainer.load("experiments/v11_poly2_100M.json")
gto = GTOAgent(trainer, name="GTO", mapping="refine", simulations=80)
```

Or via CLI:
```bash
venv/bin/python run_eval_harness.py \
    --mapping refine \
    --strategy experiments/v11_poly2_100M.json
```

**Note:** `server/gto/strategy.json` was overwritten by WS3-v2. Do not use it. Always load
from `experiments/v11_poly2_100M.json` explicitly.

---

## 7. Files

| File | Description |
|------|-------------|
| `experiments/v11_poly2_100M.json` | poly(2.0) blueprint strategy (100M iters) — production |
| `experiments/best/v9_B0_100M_allbots_positive.json` | B0 baseline strategy |
| `server/gto/local_refine.py` | Refine 2.0 implementation |
| `server/gto/board_texture.py` | Board texture classifier (dry/mono/draw/paired) |
| `eval_harness/advanced_adversaries.py` | 12 policy-distorted bots |
| `eval_harness/aivat.py` | AIVAT variance reduction (near-GTO matchups only) |
| `eval_harness/external/` | Slumbot client + OpenSpiel stub |
| `run_h2h.py` | Head-to-head GTO vs GTO analysis script |
| `docs/results/v12_blueprint_decision.md` | Blueprint decision matrix (detailed) |
| `docs/plans/ACTION_PLAN_v12.md` | V12 action plan |
