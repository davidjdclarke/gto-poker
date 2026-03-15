# V9 Complete Results Report (2026-03-15)

## Executive Summary

V9 achieved its primary goal: a strategy that beats all 7 adversary bots (+677 bb/100 average). This was accomplished through training configuration changes (2x phase schedule, old all-in dampening) rather than algorithmic changes. Two further experiments — expanding the action grid from 13 to 16 sizes, and DCFR regret discounting — both produced negative results and were abandoned.

**Production strategy:** `experiments/best/v9_B0_100M_allbots_positive.json`
**Exploitability:** 1.2211 bb/100 | **Gauntlet:** +677.2 bb/100 | **Nodes:** 1,055,003

---

## Experiment 1: V9-B0 Baseline (13-action grid, 100M)

### Hypothesis
Combining the two best ablation results from v8 (2x phase schedule + old all-in dampening) would produce a stronger strategy than v7.

### Configuration
```bash
venv/bin/python train_gto.py 100000000 --workers 6 --fresh \
  --phase-schedule 2x --allin-dampen old \
  --checkpoint-interval 10000000 --checkpoint-eval --checkpoint-gauntlet
```
- **Action grid:** 13 actions (original v6 abstraction)
- **Phase schedule:** 2x (flop/turn get 2x iterations; was 3x in v7)
- **All-in dampening:** old (rc<2, 0.7x; v7 used rc==0, 0.5x)
- **Dynamics:** Standard CFR+ with linear averaging
- **Performance:** ~0.15 ms/iter, ~50 min total

### Results

| Metric | V9-B0 | v7 (67M) | Delta |
|--------|-------|----------|-------|
| Exploitability | **1.2211** | 1.2311 | -0.01 |
| Gauntlet avg | **+677.2** | -226.2 | +903.4 |
| Nodes | 1,055,003 | 1,052,718 | +2,285 |

### Per-Bot Gauntlet (500 hands/bot, seed=42)

| Bot | V9-B0 | v7 | Delta | Analysis |
|-----|-------|-----|-------|----------|
| NitBot | **+718.5** | -409.3 | +1128 | Aggression properly calibrated by old dampening |
| AggroBot | **+893.4** | +179.4 | +714 | Better preflop/river convergence from 2x schedule |
| OverfoldBot | **+115.9** | +107.8 | +8 | Stable — already positive |
| CallStationBot | **+1071.4** | -437.0 | +1508 | Thin-value leak fixed by 2x schedule (more river training) |
| DonkBot | **+1024.0** | +293.9 | +730 | Better flop/turn donk-bet handling |
| WeirdSizingBot | **+220.4** | -952.4 | +1173 | Positive but weakest matchup |
| PerturbBot | **+696.7** | -366.0 | +1063 | Robustness improved from balanced convergence |
| **Average** | **+677.2** | **-226.2** | **+903.4** | |

### Convergence Curve

| Iters | Expl | Nit | Aggro | OFold | CStat | Donk | Weird | Pert | AVG |
|-------|------|-----|-------|-------|-------|------|-------|------|-----|
| 10M | 130.1 | -857 | +1919 | -3084 | +188 | -1101 | -4048 | -3219 | -1457 |
| 20M | 123.2 | -1285 | -456 | -2064 | -236 | -1101 | -4048 | -3219 | -1773 |
| 30M | **1.38** | +607 | +443 | +220 | +1066 | -218 | +698 | +484 | +471 |
| 40M | 1.31 | +634 | +658 | +122 | +1157 | +37 | +394 | -174 | +404 |
| 50M | 1.32 | +685 | +750 | +114 | +683 | -66 | +464 | -282 | +336 |
| 60M | 1.35 | +586 | +479 | +111 | +783 | +28 | +204 | -260 | +276 |
| 70M | 1.27 | +627 | +644 | +114 | +884 | +353 | +87 | +1111 | +546 |
| 80M | 1.28 | +755 | +400 | +115 | +889 | +355 | -28 | -552 | +276 |
| 90M | **1.25** | +863 | +524 | +115 | +887 | +650 | +601 | +368 | +572 |
| 100M | 1.33 | +719 | +893 | +116 | +1071 | +1024 | +220 | +697 | **+677** |

**Key finding:** Sharp phase transition at 30M — exploitability drops from 123 to 1.38 in a single checkpoint. Gauntlet performance continues improving through 100M even as exploitability plateaus. This suggests the strategy is refining its play at well-visited nodes while exploitability is dominated by rare edge cases.

### Why B0 beat v7

The v7 strategy had a specific failure mode: the 3x flop/turn schedule **over-converged flop/turn at the expense of preflop/river.** This caused:

1. **Under-trained river value betting** — the solver hadn't learned to bet thin for value on the river, so it checked strong hands too often (CallStation leak: -437).
2. **Miscalibrated preflop aggression** — under-trained preflop nodes produced erratic open/3bet frequencies (NitBot regression: -409).
3. **Brittle noise handling** — under-trained preflop/river nodes responded poorly to random perturbations (PerturbBot: -366).

The 2x schedule rebalanced iteration budget across all streets. The old all-in dampening (rc<2, 0.7x) provided broader control of jam frequency, preventing the all-in overuse that the narrower new dampening (rc==0, 0.5x) allowed.

### Verdict: POSITIVE — Production strategy

---

## Experiment 2: DCFR Regret Discounting

### Hypothesis
Discounting old regrets (DCFR) would speed up convergence on the wider 16-action grid by forgetting stale early-game mistakes faster.

### Configuration
Tested on the 16-action grid at 20M iterations each:
- gamma=0.999 (mild — 0.1% decay per iteration)
- gamma=0.995 (moderate — 0.5% decay per iteration)
- gamma=0.99 (aggressive — abandoned after first two results)

### Results

| Gamma | 10M expl | 20M expl | Trend | 20M gauntlet |
|-------|---------|---------|-------|-------------|
| 0.999 | 4.37 | 8.60 | diverging | -62.9 |
| 0.995 | 6.90 | 12.99 | diverging | +411.5 |
| 1.0 (CFR+) | 133.2 | 132.7 | converging slowly | -1316.7 |

### Analysis

DCFR produced *lower* exploitability at 10M than CFR+ (4.37 vs 133.2) because regret discounting aggressively prunes early noise. But by 20M, exploitability was *increasing* — the discount was erasing useful learned regrets faster than new information replaced them.

This is a known failure mode of DCFR on large action spaces: with 10-11 actions per node, the signal-to-noise ratio per action is low, and discounting amplifies this by reducing the effective sample size. Standard CFR+ accumulates regret information permanently, which converges more reliably even if it starts slower.

### Verdict: NEGATIVE — Counterproductive on wider action trees

---

## Experiment 3: 16-Action Grid Expansion (200M)

### Hypothesis
Adding 3 new postflop bet sizes would improve WeirdSizingBot performance and reduce off-tree translation errors.

**New actions:**
- BET_QUARTER_POT (0.25x) — fills check-to-bet_third gap
- BET_THREE_QUARTER_POT (0.75x) — fills bet_two_thirds-to-bet_pot gap
- BET_DOUBLE_POT (2.0x) — fills bet_overbet-to-all_in gap

### Configuration
```bash
# Phase 1: Fresh 100M
venv/bin/python train_gto.py 100000000 --workers 6 --fresh \
  --phase-schedule 2x --allin-dampen old \
  --checkpoint-interval 10000000 --checkpoint-eval --checkpoint-gauntlet

# Phase 2: Continue to 200M
venv/bin/python train_gto.py 100000000 --workers 6 \
  --phase-schedule 2x --allin-dampen old \
  --checkpoint-interval 10000000 --checkpoint-eval \
  --checkpoint-gauntlet --gauntlet-interval 50000000
```
- **Performance:** ~0.55 ms/iter (~3.5x slower than 13-action due to wider tree)

### Results

| Metric | 16-act (200M) | B0 13-act (100M) | Delta |
|--------|--------------|------------------|-------|
| Exploitability | **41.40** | 1.22 | +40.18 (33x worse) |
| Gauntlet avg | +318.0 | +677.2 | -359 |
| Nodes | 1,999,902 | 1,055,003 | +945K (1.89x) |
| Iters/node | 100 | 95 | ~parity |

### Per-Bot Comparison at Best Checkpoints

| Bot | 16-act 200M | B0 100M | Delta | Notes |
|-----|------------|---------|-------|-------|
| NitBot | **-1289.0** | +718.5 | -2008 | Worst regression — random sizing exploited |
| AggroBot | **+1527.3** | +893.4 | +634 | Only bot that improved |
| OverfoldBot | +31.6 | +115.9 | -84 | Can't efficiently exploit folds |
| CallStationBot | +705.2 | +1071.4 | -366 | Value betting diluted across too many sizes |
| DonkBot | +1046.5 | +1024.0 | +23 | Roughly equal |
| WeirdSizingBot | +217.6 | +220.4 | **-3** | Zero improvement (the whole point) |
| PerturbBot | -13.4 | +696.7 | -710 | Noisy strategy can't absorb perturbation |
| **Average** | **+318.0** | **+677.2** | **-359** | |

### Convergence Curve

| Iters | Expl | AVG | Notes |
|-------|------|-----|-------|
| 10M | 133.2 | -1840 | Unconverged |
| 20M | 132.7 | -1317 | Still unconverged |
| 30M | 51.0 | -210 | Partial convergence begins |
| 40M | 44.2 | +238 | Gauntlet turns positive |
| 50M | 47.6 | +315 | |
| 60M | 40.2 | +182 | |
| 70M | 39.1 | +424 | Best gauntlet in first 100M |
| 80M | 38.2 | +64 | |
| 90M | 44.1 | +105 | |
| 100M | 38.9 | +184 | |
| 110M | 43.0 | — | |
| 120M | 41.3 | — | |
| 130M | 42.2 | — | |
| 140M | 42.6 | — | |
| 150M | 40.0 | +227 | |
| 160M | 39.8 | — | |
| 170M | 44.2 | — | |
| 180M | 40.1 | — | |
| 190M | 42.9 | — | |
| 200M | 41.4 | +318 | **Exploitability plateaued** |

### Why the 16-action grid failed to converge

The per-node iteration budget (100 iters/node) is at parity with B0 (95 iters/node), yet exploitability is 33x worse. This means the problem is **not simply undertrained nodes** — it's structural:

1. **Combinatorial path explosion.** Each postflop node has 10-11 actions instead of 7-8. A 3-action betting sequence has 10^3 = 1,000 paths vs 7^3 = 343 paths — 2.9x more sequences to explore per street, compounding across streets.

2. **Adjacent size confusion.** Quarter (0.25) vs third (0.33) and two-thirds (0.67) vs three-quarter (0.75) have very similar EV. Regrets between them hover near zero, keeping strategies flat/uniform at these nodes instead of polarizing to one choice. This is wasted capacity.

3. **Deeper effective trees.** More actions per node means more raise-reraise sequences before the depth cap (8 actions). These deep nodes are rarely visited, contributing to exploitability but never converging.

4. **WeirdSizingBot unchanged.** The hypothesis was wrong: WeirdSizingBot's losses come from the *solver's* unconverged sizing choices, not from translation gaps. Adding sizes the solver can't converge on makes it worse, not better.

### Verdict: NEGATIVE — Not viable at current iteration budgets

---

## Infrastructure Delivered

V9 added significant tooling that will benefit future iterations:

### Measurement (Phase 0)
- **EV decomposition** (`eval_harness/ev_decomposition.py`): Per-street, per-bucket, per-action-family EV breakdown. CallStation dashboard for detecting slowplay/bluff/thin-value leaks.
- **Bridge pain map** (`eval_harness/bridge_pain.py`): Identifies worst off-tree translation gaps by concrete bet ratio.
- **Checkpoint gauntlet** (`train_gto.py`): Per-bot tracking during training with configurable interval (`--gauntlet-interval`).

### Solver dynamics (Phase 2)
- **DCFR** (`--regret-discount`): Implemented and tested. Not beneficial for current grid sizes.
- **Weight schedules** (`--weight-schedule`, `--weight-param`): Linear, exponential, polynomial averaging. Not yet tested independently.

### Action grid (Phase 3A)
- **16-action grid**: Code complete across all 12 files. BET_QUARTER_POT (13), BET_THREE_QUARTER_POT (14), BET_DOUBLE_POT (15). NodeData expanded to [16]. Ready if longer training budgets become available.

---

## Success Criteria Assessment

| # | Criterion | Target | Result | Status |
|---|-----------|--------|--------|--------|
| 1 | Gauntlet avg | > +50 bb/100 | +677.2 | **PASS** |
| 2 | CallStation fix | decomposed + improved | +1071.4 (was -437) | **PASS** |
| 3 | WeirdSizing | > -400 bb/100 | +220.4 | **PASS** |
| 4 | Solver dynamics | beat CFR+ | DCFR diverges | **FAIL** |
| 5 | Abstraction variant | tested at parity | 16-act plateaus at expl 41 | **FAIL** |

**Overall: 3/5 criteria met.** The primary goals (positive gauntlet, fix bot regressions) are achieved. The research goals (better solver dynamics, better abstraction) produced valuable negative results — we now know the boundaries of what works at current scale.

---

## Recommendations for v10

1. **Ship B0 as production.** Exploitability 1.22, all bots positive, robust.
2. **For WeirdSizingBot improvement:** Pursue translation-layer improvements (Phase 3B confidence-aware mapping, Phase 3C local refinement) rather than expanding the action grid.
3. **For abstraction improvement:** Test 9 equity buckets at convergence parity (Phase 4) — this adds ~12% more nodes vs the 90% more from action expansion.
4. **If revisiting action expansion:** Try adding only BET_THREE_QUARTER_POT (the single largest translation gap) on river-only, keeping node growth under 20%.

## Files

| File | Description |
|------|-------------|
| `experiments/best/v9_B0_100M_allbots_positive.json` | Production strategy (13-action, 100M) |
| `experiments/best/v7_67M_reference.json` | Previous best (for comparison) |
| `checkpoints/checkpoint_log.json` | Full checkpoint data for latest run |
| `dcfr_sweep/` | DCFR experiment logs and strategies |
| `experiments/archive/` | Historical v5/v6 strategies |
