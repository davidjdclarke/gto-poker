# V12 Complete Results Report

**Date:** 2026-03-16
**Blueprint:** poly(2.0) + refine — `experiments/v11_poly2_100M.json`, `mapping="refine"`
**Status:** All workstreams complete. Blueprint confirmed.

---

## Summary

V12 improved the evaluator (advanced gauntlet, AIVAT), the runtime bridge (Refine 2.0), and evaluated a selective abstraction expansion (WS3). The blueprint decision is **poly2+refine**, confirmed after WS3 was rejected due to catastrophic NitBot regression despite passing the WeirdSizingBot threshold.

---

## Workstream Results

### WS0 — Advanced Gauntlet (12 policy-distorted bots)

**Status:** Complete.

Built `eval_harness/advanced_adversaries.py` with 4 styles × 3 intensities = 12 `PolicyDistortedBot` instances. Strategy distortion via multiplicative family reweighting with `min_support=0.01` floor.

**Key metric:** robustness score = `mean - 0.5 × std`

| Config | Classic Avg | Advanced Avg | Robustness | Worst Case |
|--------|------------:|-------------:|-----------:|-----------:|
| B0+refine | +202.6 | +124.7 | +11.5 | -144.6 |
| poly2+refine | +215.9 | +102.8 | **+31.2** | **-81.0** |

poly2+refine has meaningfully better worst-case and robustness despite lower advanced average. The advanced average gap is driven by Station-strong (+315.9 vs +525.1) and Aggressive-strong (+293.6 vs +440.0) — styles that reward tighter preflop play (B0 advantage).

**Advanced gauntlet per-style (B0+refine vs poly2+refine):**

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

poly2+refine leads on every Nit style (total +75.6 advantage over Nit subtypes) — this is structurally expected since poly(2.0) weighting emphasises later, more-converged iterations which are less exploitable by tight/passive play.

---

### WS2 — AIVAT Variance Reduction

**Status:** Complete — limited applicability.

Built `eval_harness/aivat.py` with bucket-EV control variate. Implemented `build_bucket_ev_table()`, `aivat_adjusted_result()`, `AivatResult` dataclass.

**Key limitation:** The bucket-EV calibration table is built from GTO self-play, where continuation values average near zero. When GTO plays against weak gauntlet bots (NitBot, AggroBot, etc.), continuation values are large and positive (+100s bb/100). The correction `cont_value - baseline ≈ cont_value` inflates variance rather than reducing it.

**AIVAT is only valid for near-GTO matchups** (GTO vs GTO, or vs Slumbot). In gauntlet mode, ignore the `--aivat` column until per-opponent calibration is added.

---

### WS1 — Refine 2.0

**Status:** Complete. Enabled in blueprint via `mapping="refine"`.

Three enhancements to `server/gto/local_refine.py`:

**A. Blueprint CFV 1-ply backup (`_blueprint_cfv_for_action`):**
Replaces heuristic payoffs with actual counterfactual values from the trainer's node table. When the successor infoset exists in `trainer.nodes`, performs a 1-ply backup over opponent's average strategy. Falls back to heuristic when node is missing.
- Diagnostic counters: `blueprint_cfv_count` / `heuristic_fallback_count`
- Target: `blueprint_cfv_count / total >= 0.60`

**B. Adaptive threshold (`compute_adaptive_threshold`):**
Replaces fixed `MISMATCH_THRESHOLD` with a visit-count + entropy + board-texture adjustment:
- Base: 0.40
- High visit count → raise threshold (trust blueprint)
- High entropy → lower threshold (more refinement)
- Draw-heavy/connected boards → lower threshold (-0.08)
- Dry boards → raise threshold (+0.08)
- Hard clamp: [0.20, 0.60]

**C. Board-texture blend alpha (`_compute_blend_alpha`):**
Draw-heavy/connected boards +0.10 (trust refine more); dry boards -0.10; monotone +0.05.
Hard clamp: [0.30, 0.70].

**Impact on WeirdSizingBot:** +32.7 bb/100 (vs -86.4 for B0+refine, +119.1 swing). Refine 2.0 handles the 200%-pot off-tree sizing by doing a proper mini-CFR solve for that decision point rather than pattern-matching to the nearest abstract action.

---

### WS3 — Selective River Overbet (BET_TRIPLE_POT)

**Status:** Complete — REJECTED.

**Training:** 200M iterations, B0 strategy with `--selective-river-overbet` (BET_TRIPLE_POT = 3.0× pot on river only).

**Results (10k hands × 3 seeds):**

| Config | WeirdSizingBot | NitBot | Classic Avg |
|--------|---------------:|-------:|------------:|
| B0+refine | -86.4 | +253.1 | +202.6 |
| poly2+refine | +32.7 | +158.1 | +215.9 |
| **WS3 (B0+selective-river-overbet)** | **+87.5** | **-1,114.0** | **+436.6\*** |

\*Classic avg inflated by AggroBot (+1373) and DonkBot (+1864) — structurally different from B0/poly2 results due to 2M underconverged nodes.

**Root cause of NitBot regression:**
- BET_TRIPLE_POT spawned ~945k successor nodes, doubling the tree (1,055,003 → 1,999,902)
- All new nodes were underconverged at 200M total iterations
- Exploitability at 200M: **88.5 bb/100** (vs 1.22 for B0, 1.25 for poly2)
- NitBot, with its tight/passive style, finds and exploits the underconverged river spots
- NitBot collapse: -1,114.0 bb/100 (was +253.1 with B0+refine → -1,367 swing)

**Checkpoint exploitability history (WS3 training):**

| Iters | Nodes | Exploitability (quick, 200 samples) |
|-------|-------|--------------------------------------|
| 110M | 1,055,281 | 24.6 bb/100 |
| 120M | 1,055,554 | 24.8 bb/100 |
| 130M | 1,055,785 | 25.1 bb/100 |
| 140M | 1,055,974 | 26.8 bb/100 |
| 150M | 1,056,153 | 23.1 bb/100 |
| 160M | 1,056,273 | 23.7 bb/100 |
| 170M | 1,056,420 | 23.7 bb/100 |
| 180M | 1,056,530 | 26.0 bb/100 |
| 190M | 1,056,672 | 24.6 bb/100 |
| **200M** | **1,999,902** | **88.5 bb/100 (full eval, 500 × 3 seeds)** |

Note: checkpoints 110M–190M had 1.05M nodes (BET_TRIPLE_POT active in training but not yet spawning many nodes). The full node explosion occurred in the final training segment. The checkpoint quick-evals (24–27) used only 200 samples and likely avoided the new nodes; the final full eval confirmed 88.5.

**Decision:** WS3-v1 rejected. The WeirdSizingBot gain (+54.8 vs poly2+refine) is swamped by the NitBot loss (-1,272 vs poly2+refine). Blueprint remains poly2+refine.

#### WS3-v2 ("fixed args") — Additional Run

A second WS3 run (with corrected argument parsing) also completed at 200M total iterations:

| Metric | WS3-v1 | WS3-v2 | B0 (ref) |
|--------|---------|---------|----------|
| Total nodes | 1,999,902 | 1,056,752 | 1,055,003 |
| New nodes (BET_TRIPLE_POT) | ~944,899 | ~1,749 | — |
| Exploitability (full eval) | 88.5 bb/100 | 24.28 bb/100 | 1.22 bb/100 |

WS3-v2 barely spawned any BET_TRIPLE_POT nodes (~1,749 vs ~945k in v1). Despite minimal node expansion, exploitability remained 20× worse than B0 (24.28 vs 1.22). The new nodes are underconverged regardless of count. Gauntlet not run on WS3-v2 (exploitability alone is disqualifying).

**WS3-v2 also rejected.** WS3 rejection stands.

**⚠️ Note:** WS3-v2 overwrote `server/gto/strategy.json`. For production use, load `experiments/v11_poly2_100M.json` explicitly — do NOT use `strategy.json` until it is restored.

**Future option:** If BET_TRIPLE_POT is revisited with a much narrower selector (high-equity river spots only, limiting to <50k new nodes), node explosion may be avoidable. Requires a custom selectivity predicate in `add_selective_action()`.

---

### WS5 — External Calibration Scaffold

**Status:** Complete (scaffold only; full impl deferred to v13).

Created `eval_harness/external/`:
- `slumbot_client.py` — `SlumbotClient` with `new_hand()`, `act()`, `get_result()`, `state_to_hand_context()`, `decision_to_action()`
- `slumbot_match.py` — `SlumbotMatch.play(num_hands) -> MatchResult`
- `openspiel_adapter.py` — stub; full impl deferred

CLI flags added to `run_eval_harness.py`:
- `--slumbot`: runs Slumbot match (requires network access)
- `--tier2`: stub (NotImplementedError)

---

### WS4 — Blueprint Decision

**Status:** Complete.

Decision matrix (10k hands × 3 seeds, B0+refine vs poly2+refine):

| Criterion | Weight | B0+refine | poly2+refine | poly2 advantage |
|-----------|--------|-----------|--------------|-----------------|
| Classic avg bb/100 | 30% | +202.6 | +215.9 | +13.3 × 0.30 = +3.99 |
| Advanced avg bb/100 | 20% | +124.7 | +102.8 | -21.9 × 0.20 = -4.38 |
| WeirdSizingBot bb/100 | 20% | -86.4 | +32.7 | +119.1 × 0.20 = **+23.82** |
| OverfoldBot bb/100 | 10% | -14.0 | -31.4 | -17.4 × 0.10 = -1.74 |
| Exploitability | 20% | 1.2211 | 1.2496 | -0.029 × 0.20 = -0.006 |

**Composite scores:** poly2+refine 88.49 vs B0+refine 66.80. **Winner: poly2+refine (+21.69 points).**

---

## Head-to-Head: poly2+refine vs B0+refine

**Script:** `run_h2h.py` — 1,000 hands, seed=42 (smoke test)

```
Overall: poly2+refine -258.4 bb/100 vs B0+refine (1k hands, 1 seed — noisy)
```

**Action frequency differences (>0.5%):**

| Phase | Family | poly2+refine | B0+refine | Delta |
|-------|--------|-------------:|----------:|------:|
| Preflop | call | 43.4% | 47.9% | -4.4% |
| Preflop | check | 45.6% | 43.1% | +2.5% |
| Preflop | fold | 8.8% | 7.1% | +1.8% |
| Flop | bet/raise | 30.5% | 28.3% | +2.2% |
| Flop | check | 52.4% | 53.9% | -1.5% |
| River | bet/raise | 19.8% | 25.8% | **-6.0%** |
| River | check | 60.5% | 56.6% | +3.9% |
| River | fold | 10.6% | 8.2% | +2.4% |

Key difference: **B0+refine bets river 6% more frequently.** poly2+refine checks back more on the river and folds more to river bets, suggesting a tighter river value-betting range but similar bluff frequency.

**Top strategy divergence infosets** (frequency-weighted TV distance):

| Infoset | TV | Visits | Key difference |
|---------|-----|--------|----------------|
| flop:oop:74: | 0.295 | 28 | donk_small 36% vs 6% |
| flop:ip:89:1 | 0.316 | 26 | check/call 48% vs 79% |
| river:oop:119: | 0.617 | 12 | check/call 78% vs 20% |
| flop:oop:89: | 0.245 | 30 | donk_small 48% vs 72% |
| turn:oop:74: | 0.364 | 18 | donk_small 45% vs 8% |

Notable: the two strategies diverge most on **donk betting** (OOP leading into IP) and on high-equity river spots. poly2+refine donks more frequently with medium-equity hands (bucket 74 = equity tier 4, hand type 14 = TRASH), possibly using them as semi-bluffs; B0+refine prefers a check-based strategy in these spots.

**Showdown stats (1k hand smoke test):**
- Showdown rate: 45.6%
- poly2+refine win rate at showdown: 48.2% (splits: 4.6%)

**Note on by-street/by-position numbers:** The `street_ev` field in `HandRecord` tracks chip investments per street (always negative when betting), not EV. Pot recovery happens in `_finish_hand` outside any street attribution. The by-street and by-position breakdown numbers in `run_h2h.py` output are therefore NOT per-street EV — they represent chip investment, which does not sum to `p0_net`. These columns should be interpreted with caution or replaced in a future iteration with proper per-street EV attribution (e.g., attribute `p0_net` to the last street reached, or use AIVAT continuation values).

---

## Configuration for Deployment

```python
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

---

## Open Items for V13

1. **AIVAT per-opponent calibration**: Build a separate bucket-EV table for each gauntlet bot (5k-hand GTO-vs-bot self-play) to make AIVAT valid for gauntlet evaluation.
2. **Slumbot evaluation**: Run `--slumbot` flag to test vs external benchmark (WS5 full impl). Validates strategy against non-synthetic opponent.
3. **WS3 revisit (narrow selector)**: If BET_TRIPLE_POT is revisited, add a selectivity predicate limiting it to high-equity river spots (e.g., equity bucket >= 6) to constrain new nodes to <50k.
4. **`street_ev` fix in `run_h2h.py`**: Replace investment-based street attribution with proper per-street EV attribution (attribute hand net to last street reached, or use decision-weighted attribution).
5. **Larger h2h sample**: Run full 50k-hand poly2+refine vs B0+refine head-to-head to get statistically reliable action-frequency and divergence data.
