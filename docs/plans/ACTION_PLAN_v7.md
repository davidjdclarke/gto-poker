# GTO Solver — Action Plan (v7, 2026-03-12)

## Current Status

**v7 67M trained and evaluated (2026-03-13).** Exploitability improved to 1.2311 bb/100
(from 1.2822 at 50M), 1,052,718 nodes, 100% coverage.

However, **gauntlet regressed significantly** — average went from -6.7 → **-226.2 bb/100**.
NitBot and CallStationBot both flipped from large wins to large losses. River bluff ratio
dropped from 35.1% → 27.3%. The additional 17M iterations improved theoretical convergence
but made the strategy too passive in practice.

### 67M vs 50M Comparison

| Metric | 50M | 67M | Delta |
|--------|-----|-----|-------|
| Exploitability | 1.2822 | 1.2311 | -0.05 (better) |
| Gauntlet avg | -6.7 | -226.2 | -219.5 (worse) |
| NitBot | +235.9 | -409.3 | -645.2 (regressed) |
| CallStationBot | +439.5 | -437.0 | -876.5 (regressed) |
| OverfoldBot | -291.4 | +107.8 | +399.2 (fixed) |
| River bluff ratio | 35.1% | 27.3% | -7.8pp (too low) |
| Flat strategies | 1,334 | 1,035 | -299 (better) |
| Frequency anomalies | 65,814 | 17,669 | -48,145 (better) |

**Key finding:** More iterations does NOT equal better practical play. The solver is
over-converging toward a tight, passive strategy that fails to exploit weak opponents.
The 50M checkpoint was likely closer to a practical optimum for the current abstraction.

---

## Priority 1 — Immediate (no retrain, do now)

### 1A. ~~OverfoldBot -291.4 bb/100~~ ✓ FIXED at 67M
**Was -291.4 at 50M, now +107.8 at 67M.** The additional training iterations resolved the
under-betting against passive folders without any code changes. No further action needed.

### 1B. ~~Position Bug in GTO Suggestions (Multiplayer)~~ ✓ FIXED
**Was:** `_compute_gto_suggestions_sync` in `server/game.py` derived `position` from
`len(history) % 2`, ignoring the `is_in_position` parameter. At a 6-player table the
compressed abstract history has arbitrary length — a dealer/button player with even-length
history got OOP strategy, causing fold recommendations on strong hands like AQs.

**Fix applied (2026-03-12):** Line ~535 changed to `position = 'ip' if is_in_position else 'oop'`.

### 1C. Opponent-Adaptive Bluff Frequency ✓ IMPLEMENTED
**Impact: High.** Originally addressed OverfoldBot -291.4 and CallStationBot stability.

**Approach implemented:**
- New `server/gto/opponent_model.py` — `OpponentProfile(window=50, min_samples=15)`
  tracks per-phase fold/call rates over a rolling 50-action window
- Post-lookup probability adjustment: bet multiplier up to 2.0× vs folders (fold_rate > 0.5)
  and down to 0.05× vs callers (call_rate > 0.9). Linear ramp, not threshold step.
- Only applies to EQ0-3 (bluff/thin-value buckets). EQ4-7 (strong hands) always unchanged.
- Only adjusts bet actions (2,3,4,9,10,11,12). FOLD, CHECK_CALL, ALL_IN untouched.
- Kicks in only after 15+ decisions per phase (no wild early adjustments).

**67M update:** OverfoldBot is now fixed (+107.8) even without opponent modeling active in
the eval harness gauntlet — the training itself resolved it. However, opponent modeling
is now **even more critical** given the 67M regressions against NitBot (-409.3) and
CallStationBot (-437.0). The base strategy is too passive at 67M; the adaptive layer
should compensate by boosting bluffs vs stations and aggression vs nits.

**Files modified:**
- `server/gto/opponent_model.py` — new
- `eval_harness/match_engine.py` — GTOAgent accepts `opponent_profile`, HeadsUpMatch records actions
- `run_eval_harness.py` — fresh OpponentProfile per gauntlet matchup
- `server/game.py` — session-level profile on Game object, AI actions recorded, suggestions adjusted

### 1D. WeirdSizingBot Translation (-952.4 at 67M, was -852.7 at 50M)
**Impact: High — getting worse with more training.** The 120% overbet scenario alone loses
-1305 bb/100 raw. The GTO agent responds to a 120% pot bet using the `bet_overbet` (1.25x)
node, which is undertrained. This regressed further at 67M (-952.4 vs -852.7).

- When facing an overbet, consider bypassing the undertrained node and applying a
  heuristic: fold EQ0-2, call EQ3-5, raise EQ6+ — then blend with trained strategy at
  a weight proportional to node visit count
- This is a translation fix, not a training fix, so it can be deployed now

**Files:** `eval_harness/translation_ab.py`, `eval_harness/match_engine.py`

---

## Priority 2 — Next Retrain Changes

These should all be bundled into a single retrain. Each change alone has minimal impact;
combined they address the root causes of the remaining issues.

### 2A. Harden All-In Dampening (Issue #6)
**Change:** `raise_count < 2` → `raise_count == 0`, factor `0.7x` → `0.5x`

```python
# cfr_fast.pyx + cfr.py — current (too broad):
if actions[i] == ACT_ALL_IN and rc < 2:
    regrets[i] *= 0.7

# Fix:
if actions[i] == ACT_ALL_IN and rc == 0:
    regrets[i] *= 0.5
```

This stops dampening jam-over-open lines (raise_count==1) while hitting cold jams harder.
Expected to reduce 520 `allin_overuse` nodes and improve preflop exploitability (currently 1.05).

**Files:** `server/gto/cfr_fast.pyx`, `server/gto/cfr.py`
**Rebuild required:** Yes (`setup_cython.py build_ext --inplace`)

### 2B. Increase Cython Phase Schedule to 3x (Issues #7, #8, #9)
**Change:** Hardcode 3x flop/turn in `cfr_fast.pyx` `train_fast()` (lines ~797-805)

```cython
# cfr_fast.pyx train_fast() — current:
phase_schedule[0] = PHASE_PREFLOP
phase_schedule[1] = PHASE_FLOP
phase_schedule[2] = PHASE_FLOP
phase_schedule[3] = PHASE_TURN
phase_schedule[4] = PHASE_TURN
phase_schedule[5] = PHASE_RIVER

# Fix (3x schedule):
phase_schedule[0] = PHASE_PREFLOP
phase_schedule[1] = PHASE_FLOP
phase_schedule[2] = PHASE_FLOP
phase_schedule[3] = PHASE_FLOP
phase_schedule[4] = PHASE_TURN
phase_schedule[5] = PHASE_TURN
phase_schedule[6] = PHASE_TURN
phase_schedule[7] = PHASE_RIVER
# Also update schedule_len = 8
```

Also update `cfr.py` `PHASE_SCHEDULE = [0, 1, 1, 1, 2, 2, 2, 3]` to document the intent.

Expected: reduce flat strategies (1334 → <500), reduce frequency anomalies, improve
flop/turn exploitability (currently 1.42/1.44 — the worst streets).

**Files:** `server/gto/cfr_fast.pyx`, `server/gto/cfr.py`
**Rebuild required:** Yes

### 2C. Post-Training Strategy Overrides
Apply after training, before saving `strategy.json`. Prevents catastrophic leaks
without changing training:

1. **Premium limp clamp**: Set `PREMIUM_PAIR/HIGH_PAIR` `CHECK_CALL` prob ≤ 5% when
   `phase=preflop, not has_bet, eq_bucket >= 5`. Renormalize.
2. **Strong hand fold clamp**: Set `FOLD` prob ≤ 20% when `eq_bucket >= 6, has_bet=True`.
   Renormalize.

These are heuristic patches for 4+3 = 7 specific node types. Very low risk.

**Files:** `train_gto.py` (post-training hook) or `server/gto/cfr.py` (`save()` override)

---

## Priority 3 — Architectural (v8 planning)

These require more significant design work and should be scoped before committing.

### 3A. Split EQ0 Into EQ0_DRAW / EQ0_AIR
**Impact: Critical** for river accuracy. EQ0 currently merges:
- Hands with draws (flush draw, straight draw — semi-bluff equity)
- Pure air (missed draws, wrong side of board)

On the river there are no draws — EQ0 is always air. The solver can't distinguish these.

**Approach:**
- Increase `NUM_EQUITY_BUCKETS` from 8 to 9-10
- Split the 0-12.5% equity band: EQ0 = 0-6%, EQ1 = 6-12.5%
- OR: add a boolean "has_draw" flag to the bucket (9th equity dimension)
- Requires full retrain (invalidates all nodes)

**Estimated gain:** Likely -0.2 to -0.3 exploitability, major improvement vs CallStation
(currently +439 but driven by other hands picking up slack)

### 3B. Full Multi-Street Best Response Evaluator
**Impact: Medium** for measurement accuracy. Current per-street exploitability (1.28)
is not directly comparable to PioSOLVER-style full-game Nash distance. True exploitability
chains streets using correlated bucket sampling — the reported number is likely optimistic.

**Approach:** Add `--full-br` flag to `run_eval_harness.py` that runs a proper 4-street
best response with equity drift between streets.

### 3C. Per-Node Adaptive Averaging Delay
**Impact: Medium** for rare-node convergence. The global `averaging_delay = iterations // 4`
discards early signal equally for all nodes. Nodes first visited late in training lose
fewer useful iterations but nodes that are rare throughout stay near-uniform.

**Approach:** Track per-node `first_visit_iter`. Set node averaging delay to
`max(0, first_visit_iter - warmup_buffer)` instead of global threshold.

### 3D. Opponent-Adaptive Bluff Frequency — ✓ IMPLEMENTED as Priority 1C
See **1C** above. Implemented with linear ramp (not threshold steps), all streets (not just
river), EQ0-3 (not just EQ0-1), and min_samples guard. Full details in
`server/gto/opponent_model.py`.

---

## What NOT to Do

- **Do not retrain with the eq_bucket >= 2 donk restriction** — this caused the 1.82 regression
  by changing the action set size for EQ0/EV1 nodes, creating a training/eval mismatch.
- **Do not change the 3x phase schedule without also updating cfr_fast.pyx** — Python
  `PHASE_SCHEDULE` is dead code when Cython is available. Must change both.
- **Do not change the bridge default back to `conservative`** — with healthy river bluff
  ratios, conservative's 15% bet→check bleed now hurts value betting. Only appropriate if
  river over-bluffing returns.
- **Do not blindly train more iterations** — 67M showed that more iterations can degrade
  gauntlet performance despite improving exploitability. The solver over-converges toward
  passive play. Either fix the abstraction or use early stopping based on gauntlet metrics,
  not just exploitability.

---

## Retrain Command (when ready)

```bash
# After making 2A + 2B changes and rebuilding Cython:
venv/bin/python setup_cython.py build_ext --inplace
mv server/gto/strategy.json server/gto/strategy_v7_archived.json
venv/bin/python train_gto.py --iterations 50000000 --workers 6
venv/bin/python run_eval_harness.py
```

Expected training time: ~29 min (6 workers, 0.03ms/iter).
Target exploitability: < 1.20 (improvement from 3x schedule + tighter dampening).

---

## Metrics Baseline (v7, for comparison)

| Metric | 50M | 67M | Trend |
|--------|-----|-----|-------|
| Exploitability | 1.2822 | 1.2311 | better |
| Preflop | 1.05 | 1.02 | better |
| Flop | 1.42 | 1.30 | better |
| Turn | 1.44 | 1.34 | better |
| River | 1.17 | 1.13 | better |
| AggroBot | +365.6 | +179.4 | worse |
| OverfoldBot | -291.4 | +107.8 | **fixed** |
| DonkBot | +266.5 | +293.9 | better |
| WeirdSizingBot | -852.7 | -952.4 | worse |
| PerturbBot | -210.4 | -366.0 | worse |
| NitBot | +235.9 | -409.3 | **regressed** |
| CallStationBot | +439.5 | -437.0 | **regressed** |
| Gauntlet avg | -6.7 | -226.2 | **regressed** |
| Bridge best (nearest) | +276.9 | +122.5 | worse |
| Flat strategies | 1,334 | 1,035 | better |
| All-in overuse | 520 | 458 | better |
| Frequency anomalies | 65,814 | 17,669 | better |
| River bluff ratio | 35.1% | 27.3% | lower |

Full eval details: `docs/results/v7_67M_20260313.md`
