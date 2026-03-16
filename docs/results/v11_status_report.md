# V11 Iteration — Status Report

**Prepared:** 2026-03-16
**Audience:** Chief Data Scientist (v12 roadmap planning)
**Full technical detail:** `docs/results/v11_results_20260315.md`

---

## Executive Summary

V11 achieved two of its three success gates. The cycle produced one runtime improvement that requires no retraining (local refinement, +102 bb/100 average) and one new model that improves practical play against exploit-style opponents (poly(2.0), +41 bb/100 average). Both are deployed and available. The third gate — selective abstraction — produced working infrastructure but the experiment needs more training compute to converge.

The strongest single finding: **B0 + refine mapping achieves +288 bb/100 average across the 7-bot gauntlet with zero regressions**, up from +170 baseline. This is a code-only change on the existing blueprint with no retraining cost.

---

## What We Shipped

### 1. Local Refinement Mapping (code-only, no retrain)

When the opponent bets a size far outside the solver's abstract action set, the agent now runs a 200-iteration mini-CFR solve at the decision point, blended 50/50 with the blueprint strategy. Fires on turn and river only, when mismatch exceeds 40% of pot.

| Metric | Before (confidence_nearest) | After (refine) | Delta |
|--------|----------------------------:|---------------:|------:|
| Average bb/100 | +186.0 | +288.3 | **+102.4** |
| WeirdSizingBot | -202.8 | -57.4 | **+145.4** |
| DonkBot | +717.6 | +1159.3 | **+441.7** |
| Regressions | — | 0 of 7 bots | |

Latency: 4.7 ms/hand (10x overhead vs confidence_nearest). Triggers on ~11% of hands against off-tree opponents. Acceptable for evaluation; needs optimization for real-time play.

### 2. poly(2.0) Model (new trained strategy)

Trained from scratch with polynomial strategy weighting (weight = (t-delay)^2 instead of linear t-delay). Same grid, same iterations, same config otherwise.

| Metric | B0 (linear) | poly(2.0) |
|--------|------------:|----------:|
| Exploitability | 1.2211 | 1.2496 |
| Gauntlet average | +186.0 | +227.0 |
| WeirdSizingBot | -202.8 | -84.6 |
| AggroBot | +444.0 | +546.8 |
| CallStationBot | +73.0 | +163.7 |

Slightly higher exploitability (+0.03) but substantially better practical play (+41 avg). The EV decomposition shows poly(2.0) bluffs less (-8% frequency) and loses less per bluff (-14% EV). Thin-value decisions flip from -0.29 to +0.18 avg EV against CallStation.

### 3. Infrastructure

| Capability | Description | Status |
|-----------|-------------|--------|
| Cython grid configurability | `--action-grid 13/16` flag; Python/Cython mismatch impossible | Production |
| Cython selective actions | Bitmask-based per-phase/context action additions | Production |
| Scheduled DCFR | `--weight-schedule scheduled` with time-varying discount | Available, marginal results |
| Board texture classifier | Dry/monotone/draw-heavy/paired/connected classification | Infrastructure only (not integrated) |
| Configurable equity buckets | `set_equity_buckets(9)` for bucket experiments | Infrastructure only |
| Toy-game validation | Kuhn poker preflight for solver dynamics | Production |
| OpponentProfile retired | Disabled by default; `--opponent-model` to re-enable | Production |
| confidence_nearest default | GTOAgent default mapping changed from nearest | Production |
| AIVAT feasibility study | Design memo for variance-reduced evaluation | Document |
| External benchmark plan | Slumbot/OpenSpiel integration design | Document |
| Branch report template | Standardized evaluation format | Document |

---

## What We Learned

### Positive Results

1. **Runtime bridge refinement is the highest-leverage path.** +102 bb/100 with no retraining. The literature predicted this (subgame solving > action translation) and the data confirmed it.

2. **Polynomial weighting trades exploitability for robustness.** Slightly more exploitable in theory, substantially better in practice. The mechanism is measurable: fewer bluffs, better thin-value extraction.

3. **The gains partially stack.** poly(2.0) + refine achieves +278 avg and is the only configuration where WeirdSizingBot goes positive (+60.4). But a PerturbBot regression (-215) prevents it from beating B0 + refine overall.

### Negative Results

4. **DCFR diverges on HUNL.** Dominates on Kuhn poker but exploitability explodes on HUNL (3.97 vs 1.53 baseline at 10M). Toy-game validation is necessary but not sufficient — results do not transfer to full-scale games.

5. **Global selective action expansion doesn't converge at current compute.** Adding BET_THREE_QUARTER_POT to turn/river grew nodes by 51% and worsened exploitability by +0.19 at 10M. Same pattern as the v10 16-action grid failure.

6. **Scheduled DCFR is marginal.** Time-varying discount (t^1.5/(t^1.5+1)) produced only -0.005 exploit improvement. Not enough to justify the added complexity.

### Methodology Findings

7. **Seed-based evaluation has a fundamental limitation.** When two strategies make different decisions, the shared RNG diverges and all subsequent hands see different cards. Cross-strategy EV decomposition by street becomes unreliable. This is exactly the problem AIVAT was designed to solve.

8. **bb/100 calculation gotcha.** `MatchResult.p0_total_won` is already in BB units after `play()`. Dividing by big_blind again produces numbers that are 20x too low. Found and fixed mid-cycle.

9. **Default strategy file was wrong.** `strategy.json` contained the rejected 16-action model. Corrected: B0 is now the default. Warning added to eval harness when `--strategy` is not specified.

---

## Current Best Configurations

| Rank | Configuration | Average bb/100 | WeirdSizing | Regressions | Notes |
|------|--------------|---------------:|------------:|-------------|-------|
| 1 | B0 + refine | +288.3 | -57.4 | 0/7 | Best risk-adjusted. No retrain. |
| 2 | poly2 + refine | +278.1 | **+60.4** | 3/7 | Only positive WeirdSizing. PerturbBot regression. |
| 3 | poly2 + conf_nearest | +227.0 | -84.6 | 2/7 | Good if refine latency is a concern. |
| 4 | B0 + conf_nearest | +170.5 | -291.7 | 0/7 | Safe baseline. |

---

## Unsolved Problems

| Problem | Severity | Evidence | Suggested Direction |
|---------|----------|---------|-------------------|
| WeirdSizingBot still negative on B0+refine | Medium | -57.4 bb/100 | Improve refine payoff model with blueprint counterfactuals |
| 2x pot bridge pain (dist 1.34) | Medium | 327 events/match, largest translation gap | Add ~3x pot abstract action (targeted, not global) |
| Refine parameters don't generalize across models | Medium | poly2+refine regresses on PerturbBot | Scale thresholds relative to strategy sum magnitudes |
| Evaluation variance | High | 5k hands still produces wide CIs (e.g. PerturbBot CI spans 551 bb/100) | AIVAT pilot for 3-5x variance reduction |
| No external calibration | Medium | All results are against internal heuristic bots | Slumbot API integration |
| Board texture signal unused | Low | Classifier built but not integrated | Wire into confidence alpha and refine trigger |

---

## Recommendations for V12

Ordered by expected return on investment:

### Tier 1 — High confidence, clear path

1. **Improve refine payoff model.** Replace the heuristic with blueprint counterfactual values. This fixes the PerturbBot regression, unlocks full poly2+refine stacking (projected +300+ avg), and makes refine parameters model-agnostic. Estimated effort: 2-3 days.

2. **AIVAT pilot.** Implement half-AIVAT (GTO baseline only) for 3-5x variance reduction. Makes 1,000-hand evaluations as reliable as current 5,000-hand protocol. Estimated effort: 3-5 days. Design memo ready at `docs/plans/v11_aivat_feasibility.md`.

3. **Promote refine as default mapping.** B0 + refine is strictly better than B0 + confidence_nearest with zero regressions. Flip the default in GTOAgent after payoff model improvement.

### Tier 2 — Worth testing, needs validation

4. **Targeted 3x pot abstract action.** Bridge pain data shows concrete bets at ~3.3x pot mapping to 2.0x abstract (distance 1.34). Add a single ~3x pot facing-bet action on river only, train at 100M. Node growth should be ~5-10% (vs 51% for the global addition).

5. **Adaptive refine thresholds.** Scale visit_count threshold and blend alpha by the model's strategy sum magnitude. Enables refine to work across different training schedules (linear, polynomial, etc.).

6. **Board texture integration.** Feed the texture signal into confidence blending alpha. Monotone and draw-heavy boards should have higher alpha (trust heuristic more) since trained strategy is less reliable on dynamic boards.

### Tier 3 — Strategic investment

7. **External benchmarks.** Integrate Slumbot API for real-world calibration. Design ready at `docs/plans/v11_external_benchmarks.md`. Estimated effort: 3 days.

8. **poly(2.0) as new default blueprint.** If refine parameters are made model-agnostic (item 1+5), retrain poly(2.0) with the improved refine as the evaluation mapping. This could produce the strongest overall configuration.

---

## Files Delivered

| File | Description |
|------|-------------|
| `server/gto/strategy.json` | B0 baseline (now the default) |
| `experiments/best/v11_poly2_100M.json` | poly(2.0) trained strategy |
| `server/gto/local_refine.py` | Local refinement prototype |
| `server/gto/board_texture.py` | Board texture classifier (not yet integrated) |
| `run_toy_validation.py` | Kuhn poker solver dynamics validation |
| `docs/results/v11_results_20260315.md` | Full technical results with all four configs |
| `docs/plans/ACTION_PLAN_v11.md` | Action plan (status: complete) |
| `docs/plans/v11_aivat_feasibility.md` | AIVAT design memo |
| `docs/plans/v11_external_benchmarks.md` | External benchmark design |
| `docs/plans/v11_branch_report_template.md` | Standardized branch report template |

---

## Pass/Fail

| Gate | Outcome | Evidence |
|------|---------|---------|
| Local refinement improves WeirdSizing | **PASS** | +145 on WeirdSizing, +102 avg, zero regressions |
| Solver dynamics beats baseline | **PASS** | poly(2.0) +41 avg, better bluff discipline |
| Selective abstraction yields gain | **FAIL** | +51% nodes, needs more compute |

**V11 passes.** Two of three gates met with strong causal clarity on all three.
