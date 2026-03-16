# V11 Action Plan — GTO Solver

**Date:** March 15, 2026
**Status:** Complete — see `docs/results/v11_results_20260315.md`
**Scope:** Research and experimentation

---

## 1. Objective

The objective of v11 is to convert the v10 findings into one real model-strength gain and one real system-capability gain.

In practical terms, v11 should do four things:

1. Preserve the corrected scientific discipline established in v10
2. Improve practical strength beyond the current B0 + confidence_nearest baseline
3. Reduce the remaining off-tree weakness, especially versus WeirdSizing-like play
4. Test at least one solver-dynamics improvement fairly on the correct 13-action training setup

This cycle should not try to solve everything at once. The evidence from v10 says the highest-return path is:

1. Bridge refinement first
2. Solver-dynamics second
3. Selective abstraction third

That ordering is consistent with the data and with the literature showing that subgame solving/refinement is a stronger answer to off-tree actions than static action translation, and that newer CFR weighting schedules can materially improve convergence versus older fixed schemes.

---

## 2. Current baseline and interpretation

### 2.1 Baseline to carry into v11

The v11 control baseline should be:

- **Blueprint:** v9-B0, 13-action grid, 100M iterations
- **Runtime mapping:** confidence_nearest
- **Evaluation mode:** no OpponentProfile
- **Selection protocol:** 5,000 hands per bot, 3 seeds minimum, dual exploitability settings

### 2.2 What v10 established

v10 established three important facts:

**First**, the old published gauntlet results were inflated by protocol problems and implementation mismatches. The corrected baseline is much weaker than previously believed, but also much more trustworthy.

**Second**, the remaining dominant weakness is bridge quality, not obviously blueprint collapse. The strongest evidence is that confidence_nearest improved average gauntlet performance by roughly 197 bb/100 and improved WeirdSizing by roughly 568 bb/100 without retraining.

**Third**, global action-space expansion is not viable at current compute and current solver dynamics. The 16-action branch plateaued with very low effective coverage and poor exploitability, so repeating that class of experiment would be wasteful.

### 2.3 Strategic implication

The current system should now be treated as:

- A stable blueprint baseline
- With a still-imperfect but useful bridge patch
- And a measurement stack that is finally good enough to support branch decisions

That means v11 should target causal gains, not cosmetic experimentation.

---

## 3. v11 hypotheses

### H1 — Local bridge refinement will beat confidence-only mapping

The current confidence_nearest result strongly suggests that the main remaining EV loss in WeirdSizing-like spots comes from answering the wrong abstract question. The literature supports this interpretation: subgame solving adapted to off-tree actions significantly outperformed state-of-the-art action translation, and repeated solving down the tree lowered exploitability further.

**Prediction:** a narrow local-refinement prototype on turn/river overbet-facing spots will outperform confidence_nearest on WeirdSizing without broad regressions elsewhere.

### H2 — Solver dynamics still have unclaimed headroom

The failed exponential schedule did not invalidate dynamic CFR weighting. It invalidated one extreme parameterization. Recent work shows that hyperparameter schedules can outperform earlier practical CFR baselines and that dynamic discounted CFR can outperform fixed discounting approaches.

**Prediction:** a fair polynomial or schedule-based test on the 13-action grid will either improve convergence speed or match current quality with less training.

### H3 — Selective abstraction changes will outperform global expansion

The 16-action result and recent action-abstraction literature point the same way: the issue is not "too few actions everywhere," it is "the wrong action support in the wrong places." RL-CFR reports HUNL gains by improving action abstraction more intelligently than fixed baselines.

**Prediction:** one selective action addition in a high-pain context will outperform any broad menu expansion at the same compute budget.

### H4 — Evaluation variance is still high enough that protocol shortcuts will mislead

The poker evaluation literature already showed that standard match results in HUNL can require very long play to reach strong conclusions, and AIVAT was introduced to reduce the number of hands needed by more than 10x in no-limit poker settings.

**Prediction:** if v11 does not harden ranking rules further, the team will again over-read noisy branch deltas.

---

## 4. Success criteria

A v11 cycle should be considered successful only if it achieves at least three of the following:

| Criterion | Condition |
|-----------|-----------|
| **Bridge improvement** | WeirdSizing improves materially beyond current confidence_nearest; no broad regression against standard-size opponents |
| **Solver-dynamics improvement** | A schedule-based branch matches or beats baseline quality at lower compute, or beats baseline quality at matched compute |
| **Selective abstraction improvement** | One targeted abstraction change improves performance without exploding node count or collapsing convergence |
| **Evaluation improvement** | Branch ranking is fully based on corrected protocol; no final decision relies on 500-hand evidence |
| **Causal clarity** | For every adopted change, the team can state why it worked in terms of specific streets, node families, or bridge-pain reductions |

---

## 5. Non-negotiable protocol rules

These rules apply to all v11 work.

### 5.1 Final evaluation

Final ranking requires:
- 5,000 hands per bot minimum
- 3 seeds minimum
- No OpponentProfile
- Confidence intervals reported for all final gauntlet numbers
- High-fidelity exploitability for final contenders

### 5.2 Checkpoint use

Checkpoint gauntlets at low hand counts may be used for triage only. They may not be used for final ranking.

### 5.3 Hit-rate and grid sanity checks

Every serious evaluation run must record:
- Strategy hit rate
- Detected action-grid size
- Mapping mode
- OpponentProfile status

### 5.4 Causal diagnostics

Every final branch report must include:
- EV decomposition by street
- EV decomposition by action family
- Bridge pain map summary
- Delta versus baseline in the relevant node families

---

## 6. Workstream A — Baseline lock and engineering prerequisites

**Goal:** Eliminate infrastructure ambiguity before expensive experimentation.

### Tasks

#### A1. Lock the v11 baseline

Adopt:
- B0 blueprint
- confidence_nearest runtime mapping
- Corrected evaluation protocol

**Deliverable:** One baseline report with multi-seed gauntlet, dual exploitability, EV decomposition, and bridge pain summary.

#### A2. Make the action grid configurable in Cython

The current Python/Cython split is unacceptable for fair experimentation. The training stack must use one shared source of truth for:
- Action grid size
- Action menus
- Optional selective action additions

**Deliverable:**
- Cython training accepts explicit action-grid config
- Training logs record the effective grid
- Mismatch between Python and Cython menus becomes impossible or loudly fails

#### A3. Retire OpponentProfile from the default path

Remove it from all baseline and branch defaults. Archive the current implementation with a failure note.

**Deliverable:**
- Explicit deprecation note
- No hidden adaptive layer in evaluation

### Exit criteria

No branch starts until:
- Grid consistency is solved
- Baseline is reproducible
- Evaluation metadata is complete

---

## 7. Workstream B — Solver dynamics

**Goal:** Test solver-dynamics improvements fairly on the correct 13-action baseline.

**Why this matters:** Recent CFR research reports that dynamic schedules outperform older practical baselines and are easy to layer onto existing solvers. The exponential failure does not invalidate this; it only invalidated one unusable base value.

### Experiments

#### B1. Polynomial weighting

Test polynomial schedules already implemented in Cython.

**Recommended candidates:**
- power = 1.25
- power = 1.5
- power = 2.0

Run all on:
- 13-action grid
- 2x phase schedule
- Old dampening
- Same checkpoint cadence as baseline

**Primary readout:**
- Speed to reach baseline-quality exploitability
- Speed to reach baseline-quality gauntlet
- Final 100M result

#### B2. Hyperparameter Schedules approximation

Implement the closest practical version of the Hyperparameter Schedules idea that fits the codebase. The paper reports state-of-the-art practical performance over prior CFR variants in two-player zero-sum games and emphasizes ease of implementation.

**Primary readout:**
- Matched-compute quality versus baseline
- Quality-at-checkpoint curves

#### B3. Toy-game validation

Before burning HUNL compute, validate schedule behavior on smaller games in a benchmark environment. OpenSpiel is appropriate for this kind of preflight testing.

**Deliverables:**
- Toy-game convergence plots
- HUNL checkpoint comparison plot
- One solver-dynamics recommendation

### Decision rule

Adopt a solver-dynamics change only if it:
- Matches or beats baseline on final protocol, **or**
- Reaches baseline quality materially earlier

---

## 8. Workstream C — Bridge refinement

**Goal:** Reduce the remaining off-tree loss, especially against WeirdSizing.

**Why this matters:** This is the clearest practical weakness in the current system. The literature supports moving beyond action translation toward subgame solving/refinement for off-tree actions.

### Experiments

#### C1. Make confidence_nearest the default

This is not experimental anymore. Promote it to baseline.

**Deliverable:**
- Default runtime mapping switched
- Updated baseline report

#### C2. Narrow local refinement prototype

Build `local_refine.py` with very tight scope:
- Turn and river only
- Facing overbet or high-pain mapped actions only
- Limited action support
- Small number of CFR iterations
- Blueprint used as prior

**Primary comparison:** nearest vs confidence_nearest vs resolve

**Primary metric:** WeirdSizing; secondary: DonkBot, PerturbBot, aggregate bridge pain

#### C3. Trigger policy from bridge pain map

Do not invoke local refinement everywhere. Invoke only when:
- Mismatch exceeds threshold
- Mapped node confidence is low
- Action falls into the known top pain families

**Deliverables:**
- Trigger policy specification
- Latency measurement
- EV improvement report

### Decision rule

Adopt local refinement if it:
- Materially improves WeirdSizing beyond confidence_nearest
- Without creating broad regressions or unacceptable runtime

---

## 9. Workstream D — Selective abstraction

**Goal:** Improve representation where it matters without recreating the 16-action failure.

**Why this matters:** Recent action- and hand-abstraction work suggests that the path forward is smarter structure, not uniform expansion.

### Experiments

#### D1. Single selective action addition

Use the bridge pain map to choose one size in one context.

**Candidate:** A ~0.5–0.6 pot facing-bet action in postflop spots where current mapping jumps to 2.0x.

This is much more defensible than reintroducing three new actions globally.

**Primary metric:**
- Reduction in worst bridge pain family
- Effect on WeirdSizing
- Node growth and coverage

#### D2. 9-bucket parity run

Only after Cython grid configurability is fixed. Train the 9-bucket branch to real parity, not token parity.

**Primary metric:**
- Exploitability and gauntlet at matched compute
- Bridge-sensitive and middle-street behavior
- Convergence/coverage

#### D3. Structured abstraction mini-upgrade

Add one extra state signal instead of broadening everything.

**Candidates:**
- Draw/no-draw flag
- Thin-value/showdownability class
- Simple board-texture class

This is the higher-upside abstraction experiment, but it should come after bridge/local-refine work unless capacity allows parallelization.

### Decision rule

Adopt an abstraction change only if the gain survives final protocol and the complexity increase is justified.

---

## 10. Workstream E — Evaluation modernization

**Goal:** Prepare the project for stronger claims and lower-variance decisions.

**Why this matters:** v10 already proved that weak protocols can mislead the whole program. Poker-specific variance reduction remains highly relevant. AIVAT was designed for exactly this problem and reduced required hand counts by more than 10x in no-limit poker settings.

### Tasks

#### E1. AIVAT feasibility study

Document:
- What additional logs are needed
- How current simulations would need to change
- What a pilot implementation would evaluate

#### E2. External benchmark ladder design

Define which public or external agents will eventually matter. The internal gauntlet is useful, but not sufficient for long-run capability claims.

#### E3. Branch-ranking template

Formalize one final report template that every v11 branch must use.

**Deliverables:**
- AIVAT design memo
- External benchmark plan
- Standardized branch report template

---

## 11. Recommended sequence

| Step | Task | Dependencies |
|------|------|-------------|
| 1 | Lock baseline with confidence_nearest and corrected protocol | None |
| 2 | Fix Cython grid configurability and remove Python/Cython action ambiguity | None |
| 3 | Run polynomial schedule experiments on 13-action only | Steps 1–2 |
| 4 | Build and test narrow local refinement | Steps 1–2 |
| 5 | Run one selective action-addition experiment, informed by bridge pain | Steps 1–2, 4 |
| 6 | Run one larger abstraction experiment only after above are complete | Steps 3–5 |
| 7 | Draft AIVAT feasibility and external benchmark plan | None |

---

## 12. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| **Solver-dynamics experiments are confounded again** | Require grid parity; validate on toy games first; log effective schedule weights |
| **Local refinement is too slow** | Narrow trigger policy; street restriction; cap CFR iterations; benchmark latency from the first prototype |
| **Selective abstraction still causes convergence collapse** | Only one change at a time; require checkpoint coverage reports; reject any branch that loses coverage dramatically without strong EV gain |
| **Team drifts back to noisy branch ranking** | Formal protocol gate; no final conclusions from 500-hand results; CI reporting mandatory |

---

## 13. Deliverables

Every v11 branch must produce:

1. Branch hypothesis
2. Exact config
3. Checkpoint metrics
4. Final 5k/3-seed gauntlet
5. Exploitability report
6. EV decomposition
7. Bridge pain summary
8. Recommendation: keep, discard, or merge

---

## 14. v11 pass/fail gates

### Pass

v11 is a success if at least one of these happens under final protocol:

- Local refinement materially improves WeirdSizing beyond confidence_nearest
- A solver-dynamics branch beats or matches baseline more efficiently
- A selective abstraction change yields a real practical gain without convergence collapse

### Fail

v11 is a failure if:

- No branch beats the current baseline under corrected protocol
- Bridge weakness remains largely unchanged
- No causal clarity is gained about why

---

## 15. Recommendation summary

The v11 strategy should be:

1. Stabilize the corrected baseline
2. Promote confidence_nearest
3. Fix training-stack configurability
4. Test polynomial / schedule-based solver dynamics on the correct grid
5. Prioritize narrow local refinement over broad abstraction expansion
6. Treat selective abstraction as a targeted follow-on, not the main event

That is the shortest path to a meaningful next gain. The data from v10 supports it, and the strongest relevant research supports it.

---

## Reference papers

- *Faster Game Solving via Hyperparameter Schedules*
- *Dynamic Discounted Counterfactual Regret Minimization*
- *Safe and Nested Subgame Solving for Imperfect-Information Games*
- *AIVAT: A New Variance Reduction Technique for Agent Evaluation in Imperfect Information Games*
