# V12 Action Plan — GTO Solver

Date: March 16, 2026
Status: In Progress
Audience: Data Science / PokerAI team
Scope: Research and implementation plan for the next iteration

## 1. Objective

The objective of V12 is to convert the V11 findings into one clear system improvement and one clear evaluation improvement.

V12 should do five things:

- upgrade the gauntlet so model ranking is based on stronger opponents,
- improve runtime refinement so it becomes the true default response layer,
- reduce evaluation variance so branch comparisons become cheaper and more trustworthy,
- run one selective abstraction experiment tied directly to measured bridge pain,
- choose the best blueprint + runtime stack for the next phase.

The guiding principle for V12 is:

> Improve the evaluator first, then improve the runtime bridge, then change the model only where evidence says it matters.

## 2. Baseline for V12

All V12 experiments should compare against this baseline:

- Blueprint: B0, linear weighting, 13-action grid, 100M iterations
- Runtime mapping: current refine if stable, otherwise confidence_nearest
- Evaluation protocol: 5,000 hands per bot, 3 seeds minimum
- Opponent model: disabled by default

Primary metrics:
- gauntlet average
- exploitability
- WeirdSizing EV
- worst-case bot EV
- bridge pain summary

This baseline remains the control because V11 showed that B0 + refine is currently the best overall system-level configuration.

## 3. Core problems V12 is solving

**Problem A — The current gauntlet is not rich enough**

The existing handcrafted bots are still useful, but they are too simple to be the only practical evaluation set. They test a few specific failure modes, but they do not give enough coverage over the space of "competent but biased" opponents.

**Problem B — Runtime response quality is now a major driver of strength**

The biggest gains are no longer coming from retraining alone. They are coming from how the model reacts when the opponent takes actions that the blueprint handles poorly.

**Problem C — Evaluation is still expensive and noisy**

Even with the corrected V10/V11 protocol, final ranking still costs a lot of hands and the confidence intervals are still wider than ideal.

**Problem D — Abstraction changes are still risky**

Broad action expansion remains too expensive and too unstable. Any further abstraction work must be selective and justified by measured bridge pain.

## 4. V12 success criteria

V12 should be considered successful if it achieves at least three of the following:

- Advanced gauntlet is operational and useful: stronger than the classic gauntlet, separates candidate models more clearly
- Refine 2.0 improves over current refine: better WeirdSizing, no broad regressions
- Evaluation variance is reduced: AIVAT or half-AIVAT makes ranking cheaper and more stable
- Selective abstraction yields a measurable gain: one targeted action change improves practical performance, no convergence collapse
- Blueprint decision becomes clear: strong evidence for either B0 or poly2 as the better long-term base

## 5. Workstream 0 — Advanced gauntlet (first step)

This should be the first implementation step in V12.

### 5.1 Goal

Build a stronger second-layer gauntlet before running the main V12 model experiments.

The gauntlet should become a two-layer system:
- Classic gauntlet = current handcrafted exploit bots
- Advanced gauntlet = policy-distorted versions of the base model

### 5.2 Why this matters

The current handcrafted bots are easy to interpret and useful for catching obvious regressions, but strategically narrow. A policy-distorted pool gives harder opponents, smoother difficulty control, and a better test of robustness against competent but biased play.

### 5.3 Design of the advanced gauntlet

Each advanced bot starts from the current base strategy and applies controlled distortions to action families.

At each infoset:
1. load the base strategy vector
2. group actions into families
3. apply family-specific multipliers
4. renormalize
5. sample from the distorted strategy

### 5.4 Action families

- fold
- passive = check / call
- small bet
- medium bet
- large bet / overbet
- raise / re-raise
- all-in

### 5.5 Bot families to implement

4 styles, each at 3 strengths (mild / medium / strong):

1. **Aggressive** — more raises, more large bets, more continuation aggression, less folding
2. **Nit** — more folds, fewer marginal continues, fewer bluffs, less thin value aggression
3. **Call station** — more calling, less folding, somewhat less raising; still value bets strong hands
4. **Overfolder** — folds too much versus bets and raises, especially on turn and river

### 5.6 Intensity levels

mild / medium / strong — gives an initial pool of 12 bots.

### 5.7 Distortion rules

Street-aware, action-family-aware, moderate at first, fully logged.

Multiplicative reweighting:
```
π'(a) ∝ π(a) · w(a)
```
then renormalize. Add a minimum support floor for legal actions and cap distortion strength.

### 5.8 What to log for every advanced bot

- family name, intensity
- per-street multipliers
- per-action-family multipliers
- effective aggregate frequencies after distortion

### 5.9 Metrics for the advanced gauntlet

- average EV by family
- average EV by intensity
- worst-case EV across the pool
- fraction of advanced bots beaten
- robustness score: mean EV minus 0.5 × std across the pool
- top-3 hardest bots for the candidate model

### 5.10 CLI support

```
--gauntlet classic
--gauntlet advanced
--gauntlet full
```

### 5.11 Pass criteria

- bots behave in line with their intended style
- pool creates stronger separation between candidate models than classic alone
- results are stable enough to use in branch ranking

## 6. Workstream 1 — Refine 2.0

### Goal

Turn refine from a strong heuristic improvement into a model-aware runtime solver.

### Main changes

1. **Replace heuristic payoff logic with blueprint counterfactual values** — the runtime refine logic should trust the blueprint more intelligently instead of relying on generic payoff heuristics.

2. **Use adaptive thresholds** — scale refine triggers based on mismatch magnitude, visit count / strategy mass, strategy entropy, local confidence.

3. **Integrate board texture** — use board texture to modulate whether refine should trigger, how strongly to blend, and how much to trust the fallback logic.

### Pass criteria

- B0 + refine2 beats B0 + refine on full gauntlet
- poly2 + refine2 materially improves over poly2 + refine
- WeirdSizing improves further
- no broad regressions

## 7. Workstream 2 — Evaluation variance reduction

### Goal

Make branch ranking cheaper and more reliable via half-AIVAT.

### Plan

Implement a half-AIVAT pilot. Use bucket-EV approximation as the control variate.

Test on B0 + confidence_nearest, B0 + refine, B0 + refine2, poly2 + refine2.

Compare raw 1k hands vs AIVAT-adjusted 1k hands vs raw 5k hands.

### Pass criteria

- at least 3x variance reduction
- 1k adjusted ranking broadly agrees with 5k raw ranking

## 8. Workstream 3 — Selective abstraction

### Goal

Run one targeted abstraction experiment tied directly to bridge pain.

### Plan

Add one selective ~3x pot action on river only. Do not expand globally, add multiple sizes, or broaden to all streets.

### Metrics

node growth, coverage, exploitability, WeirdSizing, advanced gauntlet robustness, bridge pain reduction

### Pass criteria

- controlled node growth
- meaningful bridge pain reduction
- WeirdSizing improvement
- no major regression elsewhere

## 9. Workstream 4 — Blueprint decision

### Goal

Choose the blueprint backbone for the next phase.

### Candidates

B0 linear vs poly(2.0)

### Rule

Do not decide until refine2 is complete and tested on both. Choose the blueprint that gives the best overall: exploitability, classic gauntlet, advanced gauntlet, full gauntlet, runtime compatibility.

## 10. Workstream 5 — External calibration

### Goal

Begin moving beyond fully internal benchmarking.

### Scope for V12

Secondary task. Focus on: setting up the harness, defining benchmark protocol, running one pilot if time allows.

## 11. Recommended execution order

1. Advanced gauntlet
2. Refine 2.0
3. Half-AIVAT pilot
4. Selective river 3x action
5. Blueprint decision
6. External calibration

## 12. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Advanced gauntlet too similar to base model | Use multiple styles and intensities; keep classic gauntlet |
| Refine 2.0 becomes too expensive | Narrow trigger policy, log trigger frequency, cap runtime |
| Selective abstraction destabilizes training | Only one action, one street, one experiment |
| Team falls back to noisy decisions | Final branch ranking always uses full protocol |

## 13. Deliverables

Each V12 branch should produce: hypothesis, config, checkpoint metrics, classic gauntlet report, advanced gauntlet report, full gauntlet report, exploitability report, EV decomposition, bridge pain summary, keep/discard/merge recommendation.

## 14. Final recommendation

1. Strengthen the evaluator with an advanced distorted-policy gauntlet
2. Improve refine so it becomes model-aware
3. Reduce evaluation variance
4. Run one selective abstraction experiment
5. Choose the best blueprint + runtime stack
