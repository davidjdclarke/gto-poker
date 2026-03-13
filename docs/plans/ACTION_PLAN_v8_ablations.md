# Research Program: Breaking the v7 Solver Performance Wall

## Goal
Build infrastructure for isolated ablation experiments, then implement and run 5 experiments at 50M iterations each to identify the dominant bottleneck (abstraction, averaging, translation, schedule, or update rule).

---

## Part 1: Experiment Infrastructure

### 1A. Config flags for existing changes
Expose 3x schedule and all-in dampening as CLI parameters so they can be reverted for ablation.

- `train_gto.py`: Add `--phase-schedule {2x,3x}` and `--allin-dampen {old,new}` CLI args, pass to `trainer.train()`
- `server/gto/cfr.py`: Add `phase_schedule_mode` and `allin_dampen_mode` params to `train()`, define `PHASE_SCHEDULE_2X = [0,1,1,2,2,3]` alongside existing 3x
- `server/gto/cfr_fast.pyx`: Add `phase_schedule_mode` and `allin_dampen_mode` params to `train_fast()`, select schedule/dampening based on mode
- Rebuild Cython after

### 1B. Mid-training checkpoints
Save strategy snapshots at intervals during training for convergence trajectory analysis.

- `train_gto.py`: Add `--checkpoint-interval N` and `--checkpoint-dir DIR` CLI args
- Split parallel training into checkpoint-sized segments: each segment does fork-join-import-save-export
- Checkpoint file: `{dir}/ckpt_{total_iters}.json` + `checkpoint_log.json` index
- Optional `--checkpoint-eval` flag runs exploitability at each checkpoint
- Overhead: ~10s per checkpoint for 1M+ nodes (import/export cycle)

### 1C. Behavioral regression suite
Fixed audit of strategically meaningful node families, comparable across experiments.

- **New file**: `eval_harness/behavioral_regression.py`
- 6 node families: preflop premium, jam-over-open, flop low-EQ probe, river bluff, river thin-value, overbet-defense
- Each family: action distribution, entropy, fold/call/raise split, delta vs baseline
- Reuse key-parsing patterns from `strategy_audit()` in `server/gto/exploitability.py`
- Add `--behavioral` flag to `run_eval_harness.py` as 5th eval stage

### 1D. Ablation experiment runner
Orchestrate: pin baseline -> apply change -> train from scratch -> eval -> compare.

- **New file**: `run_ablation.py`
- `AblationConfig` dataclass: name, iterations, workers, checkpoint_interval, config_overrides
- Creates `experiments/ablations/{name}/` with checkpoints, eval.json, behavioral regression, comparison report
- Comparison: delta per metric with 3-seed replication, flags statistical significance

---

## Part 2: Five Ablation Experiments (all at 50M iterations)

### Exp A: Revert 3x schedule to 2x (isolation test)
- `--phase-schedule 2x --fresh --iterations 50000000 --workers 6`
- Full eval + behavioral regression vs baseline
- **Tests H4**: is the 3x schedule actually helping?

### Exp B: Revert all-in dampening to old (isolation test)
- `--allin-dampen old --fresh --iterations 50000000 --workers 6`
- Full eval + behavioral regression vs baseline
- **Tests**: is rc==0/0.5x better than rc<2/0.7x?

### Exp C: Per-node adaptive averaging delay
- `server/gto/cfr.py`: Add `first_visit_iter` field to `CFRNode`, compute per-node weight as `max(t - max(global_delay, node.first_visit_iter + 1000), 0)`
- `server/gto/cfr_fast.pyx`: Add `first_visit_iter` to `NodeData` struct, set on first visit in `cfr_single_street()`, pass `current_iter` and `averaging_delay` through recursion for per-node weight calc. During warmup (single-threaded), all nodes get correct values; parallel phase creates no new nodes (`_no_create_mode`)
- Serialization: include `first_visit_iter` in `to_dict()`/`from_dict()` and Cython export/import
- Train 50M fresh, full eval + behavioral regression
- **Tests H2**: is rare-node averaging suppressing aggression?

### Exp D: Translation-confidence blending (no retrain needed)
- **New file**: `eval_harness/confidence.py`
- `compute_confidence(strategy, visit_count, action_mismatch, entropy) -> alpha` where alpha in [0, 0.5]
  - `mismatch_penalty`: 0 if exact match, scales with bet-size distance
  - `visit_confidence`: sigmoid of log(visit_count)
  - `entropy_factor`: high entropy = uncertain = blend more toward heuristic
- `equity_heuristic(eq_bucket, has_bet, phase) -> strategy_dict` — fold trash, check/call medium, bet strong
- `eval_harness/match_engine.py`: Add `"blend"` mapping to GTOAgent that blends `(1-alpha)*trained + alpha*heuristic`
- Add to bridge A/B test as 5th mapping in `eval_harness/translation_ab.py`
- Run eval against existing v7 67M strategy (no retrain)
- **Tests H3**: is the ceiling caused by off-tree action handling?

### Exp E: EQ0 split (draw-aware low equity)
- `server/gto/abstraction.py`: `NUM_EQUITY_BUCKETS = 9`, `NUM_BUCKETS = 135`. Non-uniform boundaries: EQ0_AIR=[0, 0.06), EQ0_DRAW=[0.06, 0.125), EQ1=[0.125, 0.25), ..., EQ7=[0.875, 1.0)
- `server/gto/equity.py`: Non-uniform bucket boundaries in `hand_strength_bucket()`. On river (5 community cards), force EQ0_DRAW -> EQ0_AIR since no draws remain
- `server/gto/cfr_fast.pyx`: Update `NUM_EQUITY_BUCKETS_C = 9`, `NUM_BUCKETS_C = 135`
- `server/gto/cfr.py`: Bump `STRATEGY_VERSION` to 7 (invalidates old strategies)
- `eval_harness/fast_equity.py`: Update bucket cache for 9 equity buckets
- Train 50M fresh from scratch, full eval + behavioral regression
- **Tests H1**: is state abstraction the dominant bottleneck?

---

## Execution Order

1. **Part 1A**: Config flags — parameterize schedule + dampening in cfr.py, cfr_fast.pyx, train_gto.py
2. **Part 1B**: Checkpoint infra — segment parallel training, save/eval at intervals
3. **Part 1C**: Behavioral regression suite — 6 node families, delta-vs-baseline reporting
4. **Part 1D**: Ablation runner — orchestration script with comparison logic
5. **Pin baseline**: copy v7 67M strategy to `experiments/ablations/baseline_v7/`, run behavioral regression to snapshot
6. **Exp D**: Translation-confidence blending (no retrain, fastest signal)
7. **Exp A + B**: Schedule/dampening reversions via config flags (50M each, ~29 min)
8. **Exp C**: Adaptive averaging delay (Cython struct change + 50M retrain)
9. **Exp E**: EQ0 split (most invasive, 50M retrain from scratch)

---

## Files Modified/Created Summary

| File | Parts |
|------|-------|
| `train_gto.py` | 1A (flags), 1B (checkpoints) |
| `server/gto/cfr.py` | 1A (schedule/dampen params), C (adaptive delay), E (version bump) |
| `server/gto/cfr_fast.pyx` | 1A (schedule/dampen params), C (NodeData struct), E (bucket constants) |
| `server/gto/abstraction.py` | E (bucket constants) |
| `server/gto/equity.py` | E (non-uniform boundaries) |
| `eval_harness/match_engine.py` | D (blend mapping) |
| `eval_harness/translation_ab.py` | D (5th mapping) |
| `eval_harness/fast_equity.py` | E (bucket cache) |
| `run_eval_harness.py` | 1C (behavioral flag) |
| **NEW** `eval_harness/behavioral_regression.py` | 1C |
| **NEW** `eval_harness/confidence.py` | D |
| **NEW** `run_ablation.py` | 1D |

---

## Verification

After each experiment: `venv/bin/python run_eval_harness.py --strategy <exp_strategy> --save experiments/ablations/{name}/eval.json`

Pass/fail per experiment:
- Improvement in >= 1 primary metric (exploitability, gauntlet avg, bluff ratio)
- No catastrophic regression in behavioral audit families
- Reproducible across 3 eval seeds
- All 16 Kuhn tests pass: `venv/bin/python -m pytest tests/`

Baseline reproduction gate (before any experiments): v7 67M exploitability within +/-3%, gauntlet EVs within confidence bands, bluff ratio within +/-2pp.
