# V10 Research Plan — GTO Solver

**Date:** March 15, 2026
**Status:** Proposed
**Scope:** Pure model development and evaluation, not deployment

---

## 1. Executive summary

V9 established a new best baseline with the 13-action grid + 2x phase schedule + old all-in dampening. On your internal harness, that branch materially outperformed the prior v7 recipe and achieved positive results against all seven adversary bots, while keeping exploitability roughly flat to slightly better. At the same time, two major research directions produced negative results: naive action-grid expansion to 16 actions and the tested DCFR discounting setup on that widened tree. Those results are useful, but they do not justify closing the books on action abstraction or solver-dynamics research. The stronger conclusion is that your system is still sensitive to training allocation, bridge quality, and abstraction design, and that the next cycle should prioritize measurement hardening, modern solver schedules, off-tree refinement, and structured abstraction, not another broad action explosion.

The central thesis for v10 is:

> **The current ceiling is not raw CFR correctness. It is the interaction of evaluation discipline, solver weighting dynamics, action translation/off-tree handling, and abstraction quality.**

---

## 2. What v9 proved, and what it did not prove

### 2.1 What v9 proved

Your best current branch is the right baseline for the next cycle:

- 13-action grid
- 2x phase schedule
- old all-in dampening
- standard CFR+ style averaging

That result is strong enough to carry forward as the default comparison branch for v10.

### 2.2 What v9 did not prove

The following conclusions would be too strong and should be avoided:

**"Action abstraction is solved."**
The 16-action branch only showed that adding three more sizes everywhere was a bad tradeoff for your current solver and budget. Recent work on RL-CFR argues that dynamic or selective action abstraction can outperform fixed abstractions in HUNL without increasing solve time.

**"Discounting methods do not help."**
Your tested DCFR configuration failed on the widened tree, but newer methods such as Dynamic Discounted CFR and Hyperparameter Schedules specifically target the weakness of fixed discounting schemes and report better practical convergence than older baselines.

**"WeirdSizing is no longer a bridge problem."**
The poker literature remains clear that subgame refinement outperforms pure action translation when opponents take off-tree actions. That remains the most relevant external guidance for your WeirdSizing class of failure.

**"Card abstraction is fine."**
Recent hand-abstraction research argues that current approaches lose important information when they omit relevant history or observation structure, and proposes more principled abstraction evaluation.

---

## 3. Prior-cycle review: what was right, what was wrong

The v9 plan was directionally right in three ways:

1. It correctly elevated measurement and bridge quality as first-class issues.
2. It correctly predicted that 2x schedule + old dampening was the most likely immediate gain.
3. It correctly identified confidence-aware mapping and local refinement as higher-value follow-ons than blindly widening the tree.

Where the prior cycle still fell short:

- **500-hand gauntlets remained overused.** Your own earlier work already showed that short matches can badly mis-rank branches. In v10, 500-hand results should be treated as checkpoint triage only, not branch-selection evidence.
- **The report still over-inferred causes from aggregates.** "Thin-value leak fixed by 2x schedule" is plausible, but it needs to be demonstrated with the EV decomposition tooling you built, not asserted.
- **The 30M exploitability cliff was not investigated hard enough.** A drop from triple digits to ~1.4 is abnormal enough to deserve instrumentation before it becomes part of the team's mental model.

---

## 4. Research objectives for v10

### Primary objective

Identify the highest-ROI path to the next material strength gain over v9-B0 without sacrificing exploitability.

### Secondary objectives

- Separate core strategy improvement from bridge/translation improvement.
- Test at least one modern CFR weighting/schedule method against the current CFR+ baseline.
- Run one structured abstraction experiment that is not confounded by massive action-space growth.
- Build evidence for or against local refinement as the long-term path for off-tree handling.

---

## 5. Hypotheses

### H1 — Solver dynamics still have unclaimed headroom

The current baseline uses a fixed CFR+/linear averaging scheme, but more recent methods such as Hyperparameter Schedules and DDCFR argue that dynamic weighting can significantly speed convergence or improve practical performance.

**Prediction:** At least one schedule-based variant will match or beat B0 on exploitability and/or reach the same quality earlier in training.

### H2 — WeirdSizing is primarily a bridge problem, not a global-action-count problem

Libratus-style work and safe nested subgame solving show that responding to off-tree actions via subgame refinement outperforms static action translation.

**Prediction:** Confidence-aware mapping or local refinement will improve WeirdSizing more efficiently than adding actions globally.

### H3 — Better abstraction will come from better dimensions, not just more buckets

Recent hand-abstraction work emphasizes the cost of losing historical or observation structure, while RL-CFR suggests that action abstraction should be chosen adaptively rather than frozen.

**Prediction:** A small structured abstraction change will outperform a blunt "more sizes everywhere" strategy at similar compute.

### H4 — The current evaluation protocol still hides too much causal information

The project now has EV decomposition and bridge-pain tooling. Those should become mandatory for interpreting results, otherwise the team will keep mistaking symptoms for causes.

**Prediction:** The main gains of B0 over v7 will be traceable to a small number of specific node families and streets.

---

## 6. Hard protocol changes for v10

These are not optional.

### 6.1 Final ranking protocol

- **500 hands/bot:** checkpoint triage only
- **5,000 hands/bot minimum:** all branch selection
- Matched seeds where possible
- Mean and confidence interval reported for every final gauntlet result

### 6.2 Exploitability protocol

- Report exploitability with more than one seed/sample setting
- Add a "high-fidelity" exploitability run for final candidates
- Track exploitability by checkpoint, not just endpoint

### 6.3 Mandatory diagnostics on every serious branch

- EV decomposition by street
- EV decomposition by node family
- bridge pain map
- action-frequency deltas vs baseline
- strong-hand check rate and thin-value rate
- off-tree sizing loss breakdown

---

## 7. v10 experiment program

### Phase 0 — Measurement hardening and baseline validation

**Goal:** Make the project scientifically trustworthy before starting expensive new retrains.

**Deliverables:**

**B0 validation suite**
- rerun the current best branch with multiple gauntlet seeds
- confirm exploitability under at least two evaluation settings

**Causal diagnostics**
- B0 vs v7 EV decomposition
- street-level EV changes
- thin-value / slowplay / bluff / jam action-family deltas
- bridge pain map by exact pot-size mismatch

**30M cliff instrumentation**
- log node coverage, averaging weights, exploitability sample counts, and regret mass across checkpoints
- determine whether the 20M→30M cliff is real or a metric artifact

**Success criteria:**
- You can explain why B0 beat v7 with concrete evidence
- You can identify the worst remaining off-tree size gaps
- You can trust the protocol enough to compare future branches

**Kill criteria:**
- If B0 does not remain clearly superior under 5k-hand/multi-seed confirmation, pause architecture work and debug evaluation first.

### Phase 1 — Locked baseline branch

**Branch:** V10-T0-B0-validated

This is the control branch for all v10 work.

**Config:**
- 13-action grid
- 2x phase schedule
- old all-in dampening
- standard CFR+ / current linear averaging
- 100M training target, with checkpoints every 10M

**Questions answered:**
- Is the current "best" branch stable under stronger evaluation?
- Does 100M remain the best checkpoint under proper ranking?

**Required outputs:**
- checkpoint curve
- gauntlet mean/CI
- exploitability mean/CI
- EV decomposition summary

### Phase 2 — Solver-dynamics branch

This is the highest-priority algorithmic branch.

**Branch: V10-S1-HS**

Implement Hyperparameter Schedules as faithfully as practical. The paper's core claim is that simple dynamic schedules can outperform earlier fixed discounting schemes and generalize to realistic poker domains.

**Experiment design:**
- Use the 13-action grid, not the failed 16-action branch
- Keep 2x schedule and old dampening fixed
- Train to 100M with checkpoints every 10M
- Compare against T0 on both speed-to-quality and final quality

**Branch: V10-S2-WeightSchedules**

You already implemented alternative weight schedules. Test them independently on the good grid.

**Candidate schedules:**
- linear
- exponential
- polynomial

**Branch: V10-S3-DDCFR (optional)**

Test only if S1/S2 are inconclusive. DDCFR remains promising in the literature, but it is a more complex branch and your prior DCFR result was negative under a bad context for fair judging.

**Success criteria:**
A solver-dynamics branch succeeds if it does one of the following without harming the rest:
- reaches T0-quality materially earlier
- beats T0 on exploitability at matched compute
- beats T0 on 5k-hand gauntlet at matched compute

**Kill criteria:**
If none of S1/S2/S3 outperform T0 after a fair test, solver dynamics move off the critical path for the next cycle.

### Phase 3 — Off-tree and bridge branch

This is the highest-priority practical-strength branch after solver dynamics.

**Branch: V10-B1-ConfidenceMap**

Implement the deferred confidence-aware nearest mapping with no retrain.

**Design:**
- trained strategy remains dominant when mapped size is close
- blend toward an equity/heuristic fallback only when:
  - size mismatch is large
  - mapped branch has low visit confidence
  - public state is one of the known pain families

**Primary target:**
- WeirdSizingBot
- overbet-defense node families
- bridge pain map hotspots

**Branch: V10-B2-LocalRefine**

Implement the deferred local refinement prototype:
- turn/river only
- facing overbet-sized off-tree bets
- limited local action set
- limited CFR iterations for the resolve
- compare directly to nearest mapping

This branch is strongly aligned with the literature. Brown and Sandholm explicitly show that subgame solving adapted to off-tree actions significantly outperforms action translation, and that repeated refinement as the game progresses lowers exploitability further.

**Success criteria:**
- WeirdSizing improvement over T0/B0 by a meaningful margin
- no broad regression on standard-size opponents
- reduced worst-case bridge pain

**Kill criteria:**
If confidence mapping and local refinement both fail to improve WeirdSizing materially, then your bridge problem is less about response mechanics and more about core blueprint strategy.

### Phase 4 — Smarter abstraction branch

Not "bigger tree everywhere." Smarter representation.

**Branch: V10-C1-9BucketParity**

Run the already-planned 9-bucket experiment at real convergence parity, not just a token increase. This remains worth doing because the prior 9-bucket result was confounded by lower effective budget per model component.

**Branch: V10-C2-StructuredAbstraction**

Add one structured axis rather than just more raw buckets.

**Recommended candidates:**
- draw/no-draw flag
- board texture class
- showdownability / thin-value class
- SPR band

This branch is better aligned with recent hand-abstraction work, which argues that richer observation structure matters and that omitting it causes measurable information loss.

**Branch: V10-C3-SelectiveActionAbstraction**

Do not re-run a global 16-action grid. Instead, use the bridge pain map to add one strategically important size in one context or street family.

This is the practical version of the RL-CFR lesson: action abstraction should be selected where it matters, not expanded everywhere.

**Success criteria:**
- equal or better exploitability at matched compute
- improved robustness on bridge-sensitive spots
- no explosion in flatness or node count relative to benefit

**Kill criteria:**
If structured abstraction adds cost without measurable gain, hold abstraction constant and focus on bridge/refinement instead.

### Phase 5 — Tooling and external benchmarking

**OpenSpiel integration**

Use OpenSpiel as a benchmarking and validation harness for smaller imperfect-information games before spending HUNL compute on every solver-dynamics idea. OpenSpiel 1.6.11 is currently available on PyPI.

**Use cases:**
- validate schedule implementations
- compare regret-weighting variants on toy games
- catch implementation mistakes before HUNL retrains

**PokerKit integration**

Use PokerKit 0.7.3 for simulation, scenario generation, and regression support. Its docs emphasize simulation, hand evaluation, and statistical analysis support.

**Use cases:**
- generate reproducible edge-case scenarios
- independent rules sanity checks
- hand-history-driven regression testing

---

## 8. Execution order

Recommended order if compute is limited:

1. **Phase 0** — measurement hardening and B0 validation
2. **Phase 2 / S1-S2** — solver-dynamics tests on the 13-action baseline
3. **Phase 3 / B1** — confidence-aware mapping
4. **Phase 3 / B2** — local refinement prototype
5. **Phase 4 / C1** — 9-bucket parity run
6. **Phase 4 / C2** — structured abstraction
7. **Phase 4 / C3** — selective action addition

This order is deliberate:
- cheap scientific cleanup first
- low-cost schedule upside next
- low-cost bridge improvements next
- abstraction expansion only after the system is being measured correctly

---

## 9. Success criteria for v10

A successful v10 cycle should meet at least three of the following five criteria:

1. **Validated baseline** — B0 remains superior under 5k-hand multi-seed evaluation
2. **Solver-dynamics gain** — one schedule-based method beats or matches baseline more efficiently
3. **Bridge gain** — WeirdSizing and bridge pain improve materially via confidence mapping or local refinement
4. **Abstraction gain** — one structured abstraction branch shows a measurable quality improvement at fair compute
5. **Causal clarity** — the team can explain in detail why B0 beat v7, and why any v10 winner beat B0

---

## 10. Risks

| Risk | Mitigation |
|------|------------|
| Evaluation remains noisy | Enforce 5k-hand minimum for final decisions and report confidence intervals |
| Solver-dynamics gains fail to transfer from papers to your setting | Use OpenSpiel first, then HUNL |
| Local refinement is engineering-heavy | Constrain the prototype to turn/river overbet-response only |
| Abstraction changes confound too many variables | Run one-issue branches and hold the rest fixed |

---

## 11. Concrete deliverables

Each serious branch should produce:

- Markdown experiment note
- Reproducible command/config
- Checkpoint trajectory plot
- Final 5k-hand gauntlet report
- Exploitability report
- EV decomposition summary
- Bridge pain summary
- Keep / discard / merge recommendation

---

## 12. Recommended paper list

### Solver dynamics / CFR variants

- **Faster Game Solving via Hyperparameter Schedules** — schedule-based CFR weighting; directly relevant to your next solver-dynamics branch.
- **Dynamic Discounted Counterfactual Regret Minimization (ICLR 2024)** — dynamic learned discounting; relevant if simple schedules are inconclusive.

### Action abstraction / dynamic abstraction

- **RL-CFR: Improving Action Abstraction for Imperfect Information Extensive-Form Games with Reinforcement Learning** — strong argument against naive fixed action abstraction; relevant to your selective-action branch.

### Hand abstraction / state abstraction

- **Signal Observation Models and Historical Information Integration in Poker Hand Abstraction** — argues current abstraction methods lose important information when history/observation structure is omitted.

### Off-tree handling / subgame refinement

- **Safe and Nested Subgame Solving for Imperfect-Information Games** — key paper showing subgame solving beats action translation for off-tree responses.
- **Superhuman AI for heads-up no-limit poker: Libratus beats top professionals** — blueprint + nested refinement remains the most relevant architecture reference for your long-term path.
- **Libratus: The Superhuman AI for No-Limit Poker (IJCAI 2017)** — more implementation-oriented overview of the Libratus architecture.

### Tooling

- **OpenSpiel on PyPI** — useful for validating solver variants on smaller extensive-form games before running HUNL-scale retrains.
- **PokerKit documentation** — useful for simulation, analysis, and regression support.

---

## 13. Final recommendation

The next cycle should not chase another broad action-grid expansion. It should:

1. validate the current best branch properly,
2. test one modern solver-schedule family,
3. improve off-tree handling through bridge logic and local refinement,
4. then run one structured abstraction experiment.

That is the shortest path to learning something real and moving the ceiling.
