# GTO Solver — V9 Action Plan (2026-03-13)

## Diagnosis Corrections (from expert review)

- **CallStation losses are NOT a bluff-frequency problem.** Against a never-fold opponent, the correct exploit is: bluff less, value bet thinner/more often, reduce slowplay. Evidence: AA limps 75% OOP, EQ7 mostly checks flop = under-value-betting + over-slowplaying.
- **v7 67M is NOT over-converged.** At 5k hands, baseline scores -44.3 (not -226). More iterations help equilibrium; hyperparameters matter more than iteration count after ~50M.
- **Exp E did NOT prove finer abstraction is bad.** It proved undertrained (12% more nodes, same budget).

---

## Phase 0 — Measurement Hardening

### 0A: EV Decomposition

**New file:** `eval_harness/ev_decomposition.py`
- `decompose_by_street(hands: list[HandRecord]) -> dict[str, float]` — per-street bb/100
- `decompose_by_action_family(hands, phase) -> dict` — fold/check/bet/raise EV split
- `decompose_by_bucket(hands, phase) -> dict` — per-equity-bucket EV
- `callstation_dashboard(hands) -> dict` — flags: checked-strong (slowplay leak), bet-weak-called (bluff leak), checked-medium-positive-EV (thin-value leak)

**Modify:** `eval_harness/match_engine.py`
- Extend `HandRecord` with: `street_ev: dict[str, float]`, `street_actions: dict[str, list]`, `p0_bucket_per_street: dict[str, int]`
- In `_play_hand` / `_betting_round`, track P0 investment delta per street
- Gate behind `detailed_tracking=True` on `HeadsUpMatch` to avoid overhead

### 0B: Bridge Pain Map

**New file:** `eval_harness/bridge_pain.py`
- `analyze_bridge_pain(bridge_log, hands) -> dict` — per off-tree ratio: mapped action, distance, EV
- `summarize_pain_zones() -> list` — ranked worst translation gaps

**Modify:** `eval_harness/match_engine.py` `GTOAgent.decide()` — log `(concrete_ratio, mapped_action, phase, bucket)` to `self.bridge_log` when facing off-tree bet

### 0C: Checkpoint Gauntlet

**Modify:** `train_gto.py`
- Add `--checkpoint-gauntlet` flag
- At each checkpoint interval, run quick gauntlet (500 hands/bot) and record to `checkpoints/convergence_curve.json`
- Reuse existing `--checkpoint-interval` and `--checkpoint-eval` infrastructure

**Verification:** `venv/bin/python train_gto.py 100000 --checkpoint-interval 50000 --checkpoint-eval --checkpoint-gauntlet`

---

## Phase 1 — V9-B0 Baseline

**Combine Exp A (2x schedule) + Exp B (old dampening).** Both were positive in v8 ablation; test if they compound.

**Config:**
```bash
venv/bin/python train_gto.py 100000000 --workers 6 --fresh \
  --phase-schedule 2x --allin-dampen old \
  --checkpoint-interval 10000000 --checkpoint-eval --checkpoint-gauntlet
```

**Files:** No code changes — CLI args already exposed. Just a training run.

**Pass/fail:**
- Gauntlet avg > 0 bb/100 (v7 = -44.3)
- Exploitability <= 1.25
- Best checkpoint identified from convergence curve

---

## Phase 2 — Solver Dynamics

### 2A: DCFR (regret discounting)

Update: `R_{t+1} = gamma * R_t + r_{t+1}`, then floor at 0.

**Modify:** `server/gto/cfr_fast.pyx` `node_update_regrets()` ~line 333
- Add `double discount` param, multiply `regret_sum[i] *= discount` before accumulation
- New global: `cdef double _regret_discount = 1.0`

**Modify:** `server/gto/cfr_fast.pyx` `train_fast()` ~line 824
- Accept `double regret_discount=1.0`, set global

**Modify:** `server/gto/cfr.py` `CFRNode.update_regrets()` ~line 91
- Accept `discount=1.0`, apply `self.regret_sum *= discount` before `+= regrets`

**Modify:** `server/gto/cfr.py` `train()` ~line 153
- Accept `regret_discount=1.0`, thread through to `_cfr_single_street` and `_parallel_worker`

**Modify:** `train_gto.py` — add `--regret-discount` CLI flag (default 1.0)

**Rebuild:** `venv/bin/python setup_cython.py build_ext --inplace`

**Experiments:** Test gamma = {0.999, 0.995, 0.99} at 20M first, then best at 100M.

### 2B: Pluggable Weight Schedules

Replace `weight = max(t - delay, 0)` with configurable schedule.

**Modify:** `server/gto/cfr_fast.pyx` ~line 884
- New globals: `cdef int _weight_schedule_mode = 0`, `cdef double _weight_schedule_param = 1.0`
- Mode 0 = linear (current), 1 = exponential (`param ** (t-delay)`), 2 = polynomial (`(t-delay) ** param`)

**Modify:** `server/gto/cfr.py` `train()` — accept `weight_schedule_mode=0, weight_schedule_param=1.0`

**Modify:** `train_gto.py` — add `--weight-schedule {linear,exponential,polynomial}` and `--weight-param`

**Experiments:** Test exponential(1.001), polynomial(2.0) vs linear at 50M on B0 backbone.

**Pass/fail (Phase 2):**
- At least one variant beats V9-B0 on exploitability AND gauntlet avg
- Winner becomes backbone for subsequent phases

---

## Phase 3 — Action Abstraction & Off-Tree

### 3A: Expand Action Grid (3 new sizes)

Chosen based on WeirdSizingBot gaps (0.15→bet_third, 0.75→bet_pot, 2.0→all_in are all lossy mappings):

| New Action | Value | Size | Fills Gap Between |
|-----------|-------|------|-------------------|
| BET_QUARTER_POT | 13 | 0.25x | check ↔ bet_third (0.33) |
| BET_THREE_QUARTER_POT | 14 | 0.75x | bet_two_thirds (0.67) ↔ bet_pot (1.0) |
| BET_DOUBLE_POT | 15 | 2.0x | bet_overbet (1.25) ↔ all_in |

**Modify:** `server/gto/abstraction.py` — extend `Action` enum, update `get_available_actions()` postflop menus, update `_POSTFLOP_CONCRETE_MAP`, update `ACTION_NAMES`
**Modify:** `server/gto/cfr_fast.pyx` — new action constants, update `get_postflop_actions()`, `player_investments()`, `is_raise_action()`. Expand `NodeData.regret_sum[16]` and `strategy_sum[16]`
**Modify:** `server/gto/cfr.py` — update `_player_investments()`, `_has_bet_to_call()`, `_is_street_complete()`
**Modify:** `server/gto/engine.py` — update `_to_concrete_action()` with new sizes
**Modify:** `eval_harness/match_engine.py` — update `_abstract_to_concrete()`, `classify_raise()` ratio ranges
**Rebuild Cython.** **Retrain from scratch** (abstraction change invalidates all strategies).

### 3B: Confidence-Aware Nearest Mapping

**Modify:** `eval_harness/match_engine.py` `GTOAgent.decide()` — for `nearest` mapping, when mapped action is >0.2 pot-fractions from concrete size, apply small blend (alpha capped at 0.15) with equity heuristic. Log these events.

### 3C: Local Refinement Prototype (river/turn overbet-facing only)

**New file:** `server/gto/local_refine.py`
- `resolve_subtree(trainer, phase, bucket, history, concrete_ratio) -> dict[int, float]`
- Mini-CFR (1000 iters) on the immediate subtree with the actual bet size
- Only invoked for `mapping="resolve"`, only on river/turn facing overbet

**Pass/fail (Phase 3):**
- WeirdSizingBot improves from worst-case to > -400 bb/100
- No exploitability regression > 0.1 from action grid expansion
- Resolve beats nearest on WeirdSizingBot by > 100 bb/100

---

## Phase 4 — Abstraction at Convergence Parity

### 4A: Parameterize Bucket Count

**Modify:** `server/gto/abstraction.py` — `NUM_EQUITY_BUCKETS = int(os.environ.get('POKER_EQUITY_BUCKETS', 8))`
**Modify:** `server/gto/cfr_fast.pyx` — add `set_num_equity_buckets(int n)` to override hardcoded `NUM_EQUITY_BUCKETS_C = 8`

### 4B: 9-Bucket Parity Run

- Use best dynamics from Phase 2
- `POKER_EQUITY_BUCKETS=9` — 135 buckets (12.5% more than 120)
- Train to 115M (100M * 135/120) for per-node parity
- Full eval + EV decomposition

**Pass/fail:** 9-bucket achieves exploitability within 10% of 8-bucket at parity AND gauntlet avg improves

---

## Execution Order

| Run | Branch | Iterations | Depends On | Status |
|-----|--------|-----------|------------|--------|
| 0 | Phase 0 (measurement code) | n/a | none | **DONE** |
| 1 | V9-B0 (2x + old damp) | 100M + ckpts | Phase 0C | **DONE** — +677 avg, expl 1.22 |
| 2 | V9-S1 DCFR sweep | 2x 20M | Phase 1 | **DONE** — NEGATIVE (diverges) |
| 3 | V9-S1/S2 best at 100M | 100M | Run 2 | **SKIPPED** — DCFR not viable |
| 4 | V9-A1 action grid expand | 200M | Phase 0B | **DONE** — NEGATIVE (expl 41, gauntlet +318) |
| 5 | V9-A2/A3 translation | eval only | Run 4 | **SKIPPED** — grid didn't converge |
| 6 | V9-C1 9-bucket parity | 115M | Run 3 | not started |

---

## Progress Log

### 2026-03-14: Phase 0 + Phase 2 infrastructure complete
All measurement tools (EV decomposition, bridge pain map, checkpoint gauntlet) and solver dynamics options (DCFR, weight schedules) implemented and tested.

### 2026-03-14: V9-B0 baseline trained (Run 1)
100M iterations, 2x schedule + old dampening. Exploitability 1.2211, gauntlet avg +677.2 bb/100 (all 7 bots positive). Massive improvement from v7's -226.2. See `docs/results/v9_B0_100M_20260314.md`.

### 2026-03-14: Phase 3A action grid expansion (Run 4 code)
Added 3 new postflop bet sizes: BET_QUARTER_POT (0.25x), BET_THREE_QUARTER_POT (0.75x), BET_DOUBLE_POT (2.0x). Updated all 12 files. NodeData expanded from [13] to [16]. Cython rebuilt, all 16 Kuhn tests pass.

### 2026-03-15: DCFR sweep (Run 2) — NEGATIVE
Tested gamma = {0.999, 0.995} at 20M on 16-action grid. Both showed exploitability *increasing* from 10M to 20M. DCFR destabilizes convergence on wider action trees. Abandoned gamma=0.99 and skipped Run 3.

### 2026-03-15: 16-action grid 200M (Run 4) — NEGATIVE
Trained 200M iterations (100M fresh + 100M continued). Exploitability plateaued at ~40 bb/100 (vs B0's 1.22). Gauntlet +318 (vs B0's +677). NitBot -1289 (vs B0's +719), WeirdSizingBot +218 (no improvement over B0's +220). The 2x node explosion (2.0M vs 1.05M) causes a convergence problem that can't be solved with proportionally more iterations. See `docs/results/v9_16action_200M_20260315.md`.

---

## Success Criteria

v9 is successful if ALL of:
1. **Gauntlet avg > +50 bb/100** (from -44.3) — **PASS: +677.2**
2. **CallStation loss decomposed** and thin-value metric improved (not just bluff ratio) — **PASS: +1071.4** (was -437)
3. **WeirdSizing > -400 bb/100** (from -952) — **PASS: +220.4**
4. **One solver-dynamics variant beats fixed CFR+** (DCFR or schedule) — **FAIL: DCFR counterproductive**
5. **One abstraction variant tested at fair convergence parity** — **FAIL: 16-action grid did not converge at 200M**

---

## Out of Scope (deferred to v10)

- Structured/board-texture-aware abstraction features (draw density, nuttedness proxy, SPR)
- Embedding CFR / learned abstraction
- Preference-CFR / bounded exploit layer
- Full subgame solving architecture (beyond the Phase 3C prototype)
- Adaptive averaging (Exp C confirmed no impact)
