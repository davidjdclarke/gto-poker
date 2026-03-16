# V13 Action Plan — GTO Solver

**Date:** 2026-03-16
**Status:** Final
**Scope:** Breaking through the 1.22 bb/100 abstraction ceiling

---

## 1. Context and Diagnosis

V12 confirmed what had been suspected since v10: the solver has converged to the best
possible strategy under its current 2D information abstraction. Tree saturation analysis
(100% node visitation at 100M iterations) and three consecutive cycles without H2H
improvement against B0 are definitive evidence.

The primary culprit is the **infoset representation**:

```
{phase}:{position}:{bucket}:{history}
```

where `bucket = equity_bucket × 15 + hand_type` collapses 120 distinct hand classes. This
loses board texture, draw type, equity trajectory across streets, and suit isomorphism at
the postflop level. The 1.22 bb/100 exploitability floor is the theoretical ceiling of this
abstraction, not a training shortfall.

The v12 head-to-head (50k hands, 3 seeds) confirmed the strategic symptoms:
- poly2+refine loses −1,147 bb/100 OOP vs B0 — OOP postflop strategy is systematically
  wrong in spots where board texture differentiates hands in the same equity bucket
- Top divergence: `river:oop:119:()` — 78% check vs B0's 20% on top-bucket hands; TRASH
  hand type spans wildly different rivered-hand strengths on different board runouts
- River value betting frequency: poly2+refine bets 16% vs B0's 22.9% — consistent with
  a strategy that cannot distinguish "strong made hand" from "missed draw" within a bucket

V13 must focus primarily on the abstraction, but the paper research also surfaced two
near-drop-in training algorithm improvements (PCFR+ and hyperparameter schedules) that
could push exploitability lower before any abstraction work is done.

---

## 2. Objectives

V13 has four goals in priority order:

1. **Improve the runtime bridge without retraining** — pseudo-harmonic action translation,
   safe subgame solving (Refine 3.0), per-opponent AIVAT calibration
2. **Improve the B0 blueprint via targeted retraining** — PCFR+ optimistic updates,
   VR-MCCFR variance reduction, hyperparameter schedules, and suit isomorphism
3. **Redesign the card abstraction** — potential-aware EMD bucketing to replace E[HS²]
   single-number bucketing with distribution-aware clustering; board texture as key
   dimension; Embedding CFR as an alternative to discrete buckets entirely
4. **Research track: dynamic action abstraction** — prototype RL-CFR to learn which
   bet sizes belong at each node, rather than using a fixed grid

---

## 3. Literature and Prior Art

These papers inform specific workstreams below. All have been confirmed and reviewed.

| Paper | Year | Venue | Relevance |
|-------|------|-------|-----------|
**Card / Information Abstraction**

| Paper | Year | Venue | Relevance |
|-------|------|-------|-----------|
| Ganzfried & Sandholm — "Potential-Aware Imperfect-Recall Abstraction with EMD" | 2014 | AAAI | EMD-based equity trajectory clustering; directly attacks abstraction ceiling |
| Fu et al. — "Signal Observation Models and Historical Information Integration" (KrwEmd) | 2024 | arXiv | k-recall winrate features with EMD; encodes board runout history in bucket; proves information loss in imperfect-recall schemes |
| Fu et al. — "Beyond Outcome-Based Imperfect-Recall" | 2025 | arXiv | Proves standard outcome-based bucketing "suffers substantial losses by discarding history"; constructive framework for higher-resolution abstractions |
| Fu et al. — "No-Regret Strategy Solving via Pre-Trained Embedding" (Embedding CFR) | 2026 | AAAI 2026 | Replaces discrete 120-bucket scheme with continuous learned embeddings; regret generalizes across similar infosets without hard clustering |
| Waugh — "A Fast and Optimal Hand Isomorphism Algorithm" | 2013 | CMU | Lossless suit canonicalization, postflop board-relative |

**Action Translation**

| Paper | Year | Venue | Relevance |
|-------|------|-------|-----------|
| Ganzfried & Sandholm — "Action Translation: Pseudo-Harmonic Mapping" | 2013 | IJCAI | Provably less exploitable than nearest/confidence; axiomatically derived formula |
| Li, Fang, Huang — "RL-CFR: Dynamic Action Abstraction" | 2024 | ICML | Learns context-specific action menus via RL; beats Slumbot by 84 mbb/h |

**Subgame Solving**

| Paper | Year | Venue | Relevance |
|-------|------|-------|-----------|
| Brown & Sandholm — "Safe and Nested Subgame Solving" | 2017 | NeurIPS (Best Paper) | Gift-action construction makes local_refine.py theoretically safe |
| Brown, Sandholm, Amos — "Depth-Limited Solving / Modicum" | 2018 | NeurIPS | K=2 leaf strategies; blueprint as value function |
| Zhou et al. — "DecisionHoldem: Safe Depth-Limited Solving with Diverse Opponents" | 2022 | arXiv | Open-source blueprint+subgame; beats Slumbot by 730 mbb/h; direct reference implementation |
| Kubíček, Lisý, Sandholm — "Equilibrium Refinements Improve Subgame Solving" | 2026 | arXiv | Sequential equilibria gadgets reduce exploitability >50% vs standard subgame solving |
| Ge et al. — "Safe and Robust Subgame Exploitation" | 2024 | ICML | Safety guarantees + opponent exploitation simultaneously |

**CFR Algorithm Improvements**

| Paper | Year | Venue | Relevance |
|-------|------|-------|-----------|
| Tammelin — "Solving Large Imperfect Information Games Using CFR+" | 2014 | arXiv | Foundation algorithm currently in use |
| Schmid et al. — "VR-MCCFR" | 2019 | AAAI | Baseline variance reduction in external sampling (~1000× lower variance) |
| Brown & Sandholm — "DCFR: Discounted Regret Minimization" | 2019 | AAAI | Asymmetric α/β discounting (not yet tested); different from symmetric version in v10 sweep |
| Farina, Kroer, Sandholm — "Optimistic CFR / PCFR+" | 2020 | NeurIPS | T⁻¹ convergence (vs CFR+'s T⁻¹/²); near-drop-in Cython replacement for the regret update |
| Zhang, McAleer, Sandholm — "Faster Game Solving via Hyperparameter Schedules" | 2026 | AAAI 2026 | Automated CFR discounting schedule; 10–30% faster convergence validated on large poker |
| McAleer et al. — "ESCHER" | 2022 | NeurIPS workshop | History value function eliminates importance sampling; orders-of-magnitude variance reduction |

**Deep / Neural Approaches (research track)**

| Paper | Year | Venue | Relevance |
|-------|------|-------|-----------|
| Brown & Sandholm — "ReBeL" | 2020 | NeurIPS | PBS value network as learned terminal evaluator; open source |
| Brown et al. — "Deep CFR" | 2019 | ICML | Replaces discrete abstraction with neural CFR; removes ceiling by definition |

**Open-source reference implementations:**
- `ericgjackson/slumbot2019` — production-scale HUNL, no postflop card abstraction (the target to beat)
- `lbn187/RL-CFR` — ICML 2024 dynamic action abstraction, Python
- `AI-Decision/DecisionHoldem` — blueprint + safe depth-limited subgame solving, Python (2022 SOTA open-source HUNL)
- `facebookresearch/rebel` — full ReBeL implementation (C++/Python)
- `kdub0/hand-isomorphism` — C library for lossless suit canonicalization

---

## 4. Workstreams

### WS0 — Pseudo-Harmonic Action Translation (no retrain, high value)

**Goal:** Replace `confidence_nearest` as the primary mapping with a theoretically grounded
pseudo-harmonic mapping that is provably less exploitable than nearest-action approaches.

**Background:** Ganzfried & Sandholm (IJCAI 2013) proved that nearest-action translation
violates monotonicity and is systematically exploitable near bucket boundaries. Their
pseudo-harmonic mapping satisfies all desiderata and was the ACPC 2012 state of the art.
WeirdSizingBot's residual issues are exactly the near-boundary translation pathology the
paper describes.

**Design:**

The pseudo-harmonic mapping assigns probability `p` to the lower abstract action and
`(1-p)` to the higher action, where:

```
p(x, a, b) = [b(x - a)] / [x(b - a)]
```

`x` is the concrete bet, `a` is the lower abstract action, `b` is the higher. This is
the unique mapping satisfying: boundary conditions, monotonicity, shift-invariance, and
the scale-invariance axiom. Unlike nearest-action (which picks whichever abstract action
is closer), pseudo-harmonic interpolates between the two bracketing actions, weighting
the strategy accordingly.

**Implementation:**

1. Add `mapping="pseudo_harmonic"` option to `GTOAgent` in `match_engine.py`
2. In `compute_strategy()`: when the concrete bet falls between two abstract actions,
   compute `p` and `(1-p)` using the formula above and blend the two strategy vectors
3. The blended strategy should use the abstract-action pot-fraction sizing (not the
   concrete one) for the output action

**Success criteria:**
- WeirdSizingBot > +50 bb/100 (current best: +32.7 with poly2+refine)
- No regression vs. confidence_nearest on standard-bet bots
- Per-seed variance lower than refine (no mini-CFR latency)

**Estimated effort:** 8–12 hours

---

### WS1 — Safe Subgame Solving in Refine (Refine 3.0)

**Goal:** Make `local_refine.py` theoretically safe by implementing the "gift action"
construction from Brown & Sandholm (NeurIPS 2017), and upgrade from a 1-ply backup to
a K=2 depth-limited solve per Modicum (NeurIPS 2018).

**Why the current implementation is unsafe:**

Refine 2.0 solves a subgame from the current decision point using a simplified payoff
model. Brown & Sandholm (2017) prove that naively solving a subgame in isolation can
increase the exploitability of the overall strategy — the opponent can exploit the fact
that the agent has deviated from its blueprint in a predictable direction. The fix is to
augment every subgame with a "gift action": an auxiliary action valued at exactly what
the blueprint gives the opponent (the counterfactual value from `trainer.nodes`). This
guarantees exploitability cannot increase relative to the blueprint.

**K=2 depth-limited leaves (Modicum):**

Current Refine 2.0 uses a 1-ply backup (`_blueprint_cfv_for_action()`) with a single
leaf evaluation. Brown, Sandholm & Amos (NeurIPS 2018) show K≥2 opponent strategies at
the depth limit are required for the safety guarantee in depth-limited solving. The
practical K=2 implementation uses:
- Strategy 1: opponent folds everything (lower bound on leaf value)
- Strategy 2: opponent calls everything (upper bound)
Then max over these two leaf values when choosing our action.

**Implementation changes to `local_refine.py`:**

1. **Gift action construction:** Before running mini-CFR, add a new action to the subgame
   tree with value = `blueprint_cfv` (the 1-ply backup already computed). The opponent is
   always allowed to "take the gift" — choose the blueprint counterfactual value instead
   of responding to the refine strategy.

2. **K=2 leaves:** Replace the single heuristic payoff with two evaluations (fold-all,
   call-all). Return `max(v_fold_leaf, v_call_leaf)` as the terminal value.

3. **Budget cap increase:** With safe subgame solving, increase `_MAX_REFINE_BUDGET` from
   100 to 200 — theoretically safe refinements can fire more often without degrading the
   overall strategy.

**Additional reference: Kubíček et al. (2026)** proved that using sequential equilibria
(rather than Nash) as the subgame solution concept reduces the exploitability of the full
strategy by >50% compared to standard subgame solving. If WS1 is implemented, testing
sequential equilibria as the subgame objective is a natural follow-on experiment.

**Reference implementation: DecisionHoldem** (`github.com/AI-Decision/DecisionHoldem`) —
open-source Python, beat Slumbot by 730 mbb/h using this exact architecture. Study its
`subgame_solver.py` before implementing WS1.

**Success criteria:**
- Refine 3.0 beats Refine 2.0 on WeirdSizingBot (target: +50)
- PerturbBot regression eliminated (was a known Refine 2.0 weakness)
- Formal guarantee: refine can only improve exploitability relative to the blueprint

**Estimated effort:** 16–20 hours

---

### WS2a — PCFR+: Optimistic Regret Updates (near-drop-in Cython change)

**Goal:** Replace the CFR+ regret update in `cfr_fast.pyx` with an optimistic / predictive
variant (PCFR+) that achieves T⁻¹ convergence instead of CFR+'s T⁻¹/².

**Background:** Farina, Kroer & Sandholm (NeurIPS 2020) showed that adding a predictive
(optimistic) correction term to regret minimization changes convergence from T⁻¹/² to T⁻¹
— the same improvement rate as going from vanilla gradient descent to Nesterov acceleration.
The update is:

```
# CFR+:
R[t+1](a) = max(R[t](a) + r[t](a), 0)

# PCFR+ (optimistic):
r̂[t](a) = r[t-1](a)                        # prediction: last iteration's regret
R[t+1](a) = max(R[t](a) + r[t](a) - r̂[t](a), 0)
```

The optimistic term `r̂[t](a)` is the previous iteration's counterfactual regret, which
requires storing one extra `double[16]` array per node (the "prediction buffer").

**Why this matters for your plateau:** The theoretical improvement is most pronounced at
lower iteration counts. If your 100M-iteration convergence is limited by CFR+'s T⁻¹/²
rate, PCFR+ could reach the same exploitability in ~50M iterations, or push below
1.22 bb/100 by 100M. Combined with WS2b (VR-MCCFR baselines), this is a multiplicative
improvement in effective convergence speed.

**Cython changes:** Add `double prev_regret[16]` field to `NodeData` struct; update
inner loop to compute the prediction correction; add `--solver pcfr+` flag to
`train_gto.py`.

**Success criteria:**
- Exploitability at 50M iterations < 1.22 bb/100 (current 100M baseline)
- No regression in gauntlet or H2H

**Estimated effort:** 12–16 hours

---

### WS2b — VR-MCCFR: Variance Reduction in Training

**Goal:** Add a baseline function to the external sampling MCCFR traversal to reduce
regret estimate variance by ~1000×, enabling faster convergence per iteration.

**Background:** Schmid et al. (AAAI 2019) show that the Monte Carlo regret estimator in
external sampling MCCFR has high variance because a single opponent sample may be
unrepresentative of the opponent's average. Adding a baseline `b(I)` — any function of
the infoset — to the regret estimate and correcting analytically gives an unbiased
estimator with dramatically lower variance:

```
r̂(a) = (r(a) - b(I)) + E[b(I)]   (unbiased, baseline cancels in expectation)
```

The simplest effective baseline is the current average strategy value at the node
(`node.get_average_strategy()` × expected payoffs). This requires no additional data
structure — the strategy sum is already tracked.

**Implementation in `cfr_fast.pyx`:**

```cython
# Current external sampling traversal:
regrets[a] = cfv_a - cfv_chance

# With VR-MCCFR baseline:
baseline = sum(strategy[a] * cfv_a for a in actions)  # E[V] under current policy
regrets[a] = (cfv_a - baseline) + baseline_correction  # same expectation, lower variance
# baseline_correction = 0 for the traverser (full tree evaluation, no sampling)
# baseline_correction = baseline for sampled opponent actions
```

This change is confined to the inner loop of `cfr_fast.pyx` and does not affect the
node structure, save format, or any other code.

**Expected impact:** Faster convergence at same wall-clock time; lower variance in
regret estimates means less training noise at the convergence frontier. This may push
exploitability below the current 1.22 bb/100 plateau if any residual variance is
masking additional strategy refinement.

**Success criteria:**
- Exploitability at 50M iterations better than current 100M baseline (i.e., 2× efficiency)
- No regression in gauntlet (same strategy quality faster)

**Estimated effort:** 10–14 hours

---

### WS2c — Hyperparameter Schedules for CFR Discounting

**Goal:** Replace the hand-tuned `--weight-schedule` and `--regret-discount` flags with
an automatically derived schedule validated on large-scale poker.

**Background:** Zhang, McAleer & Sandholm (AAAI 2026) showed that a single time-varying
CFR discounting schedule derived from three small benchmark games generalises to 17 diverse
games including large-scale HUNL. Their schedule aggressively discounts early iterations
and gradually stabilises. Empirically 10–30% faster convergence than CFR+ or DCFR with
fixed schedules.

The schedule is parameterised as:
```
discount[t] = t^α / (t^α + c)     # α and c derived from small-game sweep
weight[t]   = t^β                  # strategy weight
```

where α, β, c are found by grid search on Kuhn/Leduc/Goofspiel — your `run_toy_validation.py`
already provides the Kuhn poker test bed.

**Implementation:**

1. Run the schedule sweep on Kuhn poker (10k iterations) to find α, β, c
2. Add `--weight-schedule zhang2026 --weight-param α,β,c` to `train_gto.py`
3. Train B0-schedule (100M iterations) and compare exploitability to B0

**Why this is a quick win:** It requires zero architectural changes. If the Zhang schedule
outperforms your current linear weighting, retrain B0 with it at no additional cost.
Your `--weight-schedule scheduled` mode in v11 was an approximation; the Zhang 2026
calibration is the proper derivation.

**Success criteria:**
- Exploitability with Zhang schedule < 1.15 bb/100 at 100M iterations
- Validated first on Kuhn poker (reduces risk before full training run)

**Estimated effort:** 6–8 hours

---

### WS3 — Suit Isomorphism (Lossless Free Accuracy)

**Goal:** Implement board-relative suit isomorphism postflop to collapse strategically
identical infosets, concentrating training signal without adding any approximation error.

**Background:** Waugh (2013) defines the isomorphism: two hands on a board are
isomorphic if one can be obtained from the other by a permutation of suits that leaves
the board invariant. For example, K♠T♠ on a 7♠8♣2♥ board is isomorphic to K♥T♥ on
7♥8♣2♠. These nodes should have identical strategies; currently they are separate nodes
receiving separate training.

**Why this matters at your scale:** With ~340k flop nodes, a naive estimate is that
~30–40% are suit-isomorphic duplicates — nodes that are receiving identical (or should be
identical) strategy signals but counted separately. Collapsing them concentrates training
signal: each remaining canonical node is visited 1.4–1.7× as often for the same
iteration budget. This is a free convergence improvement.

**Implementation:**

1. Add `canonicalize_hand_board(hole_cards, community)` to `equity.py` or `abstraction.py`
   - Uses Waugh's algorithm: enumerate suit permutations that fix the board, apply the
     lexicographically smallest to the hole cards
   - Output: canonical (rank, suit_class) representation

2. Modify `InfoSet.key` to use canonical hole-card representation when computing `bucket`
   - The bucket assignment already happens in `hand_strength_bucket()` — canonical
     representation just changes which cards are passed in

3. No change needed to CFR training, node storage, or Cython — the canonicalization
   happens at the infoset key level before any CFR logic

**Important:** This is a **lossless** operation. There is no abstraction error introduced.
Exploitability cannot increase; it can only stay the same or decrease (due to better
training signal concentration).

**Success criteria:**
- Node count reduction of ≥20% at same game tree depth (if less, canonicalization may
  already be partially happening via hand_type)
- Exploitability at 50M iterations ≤ current 100M baseline
- Zero regression in H2H vs. B0

**Estimated effort:** 12–16 hours

---

### WS4 — Potential-Aware EMD Card Abstraction (Major)

**Goal:** Replace the E[HS²] single-equity-number bucketing with potential-aware
clustering using Earth Mover's Distance — the AAAI 2014 state of the art for card
abstraction. This directly attacks the abstraction ceiling.

**Why E[HS²] has a ceiling:**

E[HS²] compresses the full distribution of future equity outcomes into a single number.
A hand with 45% equity can be:
- A nut flush draw (high variance: 70% equity if flush completes, 15% if not)
- A trapped top pair (low variance: 48% equity on all runouts)

These have the same E[HS²] but completely different strategic properties. The flush draw
should bluff more, slowplay less, and size larger. The trapped top pair should check-call
and let opponents bluff. Currently both receive the same strategy from bucket 45.

**Potential-aware EMD clustering:**

Instead of bucketing by E[HS²] value, compute for each hand a *histogram* of equity
outcomes across all possible runout cards. The histogram has N bins representing equity
ranges (e.g., N=10, covering 0–10%, 10–20%, ... 90–100%). Cluster hands by EMD between
their histograms using k-means.

Two hands are in the same bucket if their equity *distributions* are similar — not just
their means. The flush draw and trapped pair get different buckets despite identical means.

**Implementation plan:**

**Phase 4a: Precompute equity histograms** (~40 hours, one-time)

```python
# For each canonical hand × board combination (sampled):
def compute_equity_histogram(hole_cards, board, n_bins=10, n_rollouts=1000):
    # Sample possible runout cards
    # Compute final equity on each runout
    # Bin into histogram
    # Return: np.array of shape (n_bins,)
```

Cache results: with 169 preflop hands × ~500 board samples = 84,500 histograms.
Store in `equity_histograms.npy` (~20 MB).

**Phase 4b: K-means clustering with EMD** (~8 hours)

```python
from scipy.stats import wasserstein_distance

def emd_distance(hist_a, hist_b):
    return wasserstein_distance(range(len(hist_a)), range(len(hist_b)), hist_a, hist_b)

# K-means variant with EMD as the distance metric
# K = 8 (matching current 8 equity buckets) to maintain node count
# Or K = 12 for finer resolution
cluster_assignments = emd_kmeans(histograms, K=8, distance=emd_distance)
```

**Phase 4c: Replace bucketing in `equity.py`**

```python
# Current:
def hand_strength_bucket(hole_cards, community, ...) -> int:
    ehs2 = hand_strength_squared(...)
    equity_bucket = int(ehs2 * N_EQUITY_BUCKETS)
    bucket = equity_bucket * NUM_HAND_TYPES + hand_type
    return bucket

# New:
def hand_strength_bucket(hole_cards, community, ...) -> int:
    canonical = canonicalize_hand_board(hole_cards, community)  # WS3
    equity_cluster = EMD_CLUSTER_TABLE[canonical]               # lookup
    bucket = equity_cluster * NUM_HAND_TYPES + hand_type
    return bucket
```

**Phase 4d: Retrain from scratch**

This changes the card abstraction, invalidating any existing strategy. Train B0-EMD using
the same hyperparameters as B0:
- 100M iterations, 2x phase schedule, linear weighting, 13-action grid
- Expected node count: similar to current 1.05M (same k, same action grid)
- Expected exploitability: lower than 1.22 (reduced abstraction error per EMD paper)

**Success criteria:**
- Exploitability < 1.0 bb/100 (vs current 1.22)
- H2H vs. B0: B0-EMD wins or draws within CI
- Gauntlet average ≥ current best (+215.9 bb/100)

**Estimated effort:** 60–80 hours (including precomputation, clustering, retrain, eval)

---

### WS5 — Board Texture as Infoset Key Dimension (Major)

**Goal:** Add the 4-class board texture as a fifth field in the infoset key, allowing the
solver to learn genuinely different strategies for the same hand on dry vs. draw-heavy boards.

**This should be implemented AFTER WS4** — the combination of EMD bucketing + board texture
will give better node utilisation than either alone.

**Design:**

```
New key format:
{phase}:{position}:{bucket}:{texture}:{history}

where texture ∈ {0=dry, 1=monotone, 2=draw_heavy, 3=paired}
```

`board_texture.py` already implements this classification. The change is in `InfoSet.key`.

**Node count impact:**
- Preflop: no change (no board yet)
- Flop/turn/river: ×4 (one node set per texture class)
- Total: ~1.05M × 3 (postflop phases) × 4 / 3 ≈ 4.2M nodes
- Training to convergence: estimated 300–400M iterations (3× current budget)
- Wall time: ~8–10 hours per training run

**Cython key encoding:**

The current 64-bit key will overflow with texture added. Two options:

Option A — Compact key (recommended):
```cython
# Use 2 bits for texture (4 classes fits in 2 bits)
# Reduce history depth from 12 to 11 actions (saves 4 bits)
# Key: phase(2) | pos(1) | bucket(8) | texture(2) | history(44) | hlen(4) = 61 bits ✓
```

Option B — Python dict with 128-bit key (fallback):
```cython
# Use Python dict keyed by (int64_base_key, texture_byte)
# Slower (~5× penalty on key lookup) but no bit packing complexity
```

**Success criteria:**
- H2H vs B0 (no texture): B0-texture wins significantly
- Exploitability < 0.8 bb/100
- OOP deficit in H2H narrows below −500 bb/100 (current: −1,147)

**Estimated effort:** 40–60 hours (implementation) + training time

---

### WS5b — Embedding CFR: Continuous Infoset Representations (Research Track)

**Goal:** Evaluate replacing the discrete 120-bucket scheme with pre-trained continuous
embeddings, per Fu et al. (AAAI 2026), as an alternative to the WS4 + WS5 retrain.

**Background:** Fu et al. (AAAI 2026) showed that replacing the hard cluster assignment
`bucket = equity_bucket × 15 + hand_type` with a continuous embedding where regret
accumulation generalises across nearby infosets achieves better exploitability per node
than discrete bucketing. The embedding is trained from game play data (infoset features →
low-dimensional vector) and used to define "soft" bucket membership.

This is the most architecturally different approach in the plan. It eliminates the
bucketing ceiling by definition — there are no hard bucket boundaries to exploit. However
it requires:
- A training pipeline for the embedding network (offline, using existing strategy data)
- Modification of CFR to accumulate regrets in embedding space rather than per-bucket
- Significant departure from the current tabular architecture

**Prototype scope for v13:**
1. Train a simple MLP embedding on (hole_cards, board, phase) → 16-dimensional vector
   using the existing `hand_strength_bucket` outputs as supervision
2. Implement a "soft bucket" version of `get_strategy()` that interpolates between the
   K nearest cluster centers rather than doing a hard lookup
3. Measure exploitability and gauntlet vs. hard-bucket baseline

This is explicitly a research track — if the prototype shows improvement at 10M
iterations, it becomes a Phase D workstream. If not, discard.

**Estimated effort:** 20–30 hours

---

### WS6 — RL-CFR Dynamic Action Abstraction (Research Track)

**Goal:** Prototype RL-CFR (Li et al., ICML 2024) using the existing `add_selective_action()`
infrastructure to learn which bet sizes to include per decision node, rather than using a
fixed 13-action grid.

**Background:** RL-CFR beat Slumbot by 84±17 mbb/h (=840 bb/100) over 600k+ hands by
dynamically choosing the action set at each decision point. The key insight: a 75%-pot bet
matters enormously in some spots and is irrelevant in others. A fixed grid wastes capacity
on useless actions in most spots while missing crucial actions in key spots.

Your `add_selective_action()` API in `abstraction.py` is already the right abstraction —
it's a manual version of what RL-CFR automates. The RL policy would decide per-infoset
whether to add BET_QUARTER, BET_TRIPLE, etc.

**Prototype scope:**

Rather than full RL-CFR (which requires a separate RL training loop), prototype as:

1. **Sensitivity analysis:** For each infoset, compute the counterfactual regret for each
   non-grid action (e.g., BET_QUARTER_POT at spots where it's not in the grid). Infosets
   where the regret of the missing action exceeds a threshold are "underserved."

2. **Greedy expansion:** Add the missing action to the top-K underserved infosets (e.g.,
   K=50k). Retrain for 50M additional iterations on the expanded grid.

3. **Measure:** Compare exploitability and gauntlet vs. baseline.

This is a tractable first step toward RL-CFR without the full RL training infrastructure.

**Success criteria:**
- Identify ≥10 infosets where a non-grid action would have positive regret > 5 bb/100
- Targeted expansion of those infosets improves WeirdSizingBot without node explosion
- Node growth stays below 20% (lessons from WS3)

**Estimated effort:** 20–30 hours

---

### WS7 — AIVAT Per-Opponent Calibration

**Goal:** Make AIVAT variance reduction valid for gauntlet evaluation by building a
separate bucket-EV calibration table for each gauntlet bot.

**Current limitation:** The bucket-EV table from GTO self-play has near-zero expected
values. Against weak opponents (NitBot, AggroBot), continuation values are +100s bb/100.
The control variate `cont_value − baseline ≈ cont_value` inflates variance.

**Fix:** For each gauntlet bot, run 5k-hand GTO vs. bot calibration match (seed=0) and
build `bucket_ev_table_{botname}.json`. Load the opponent-specific table at match time.

**Implementation:** Modify `aivat.py` to accept an optional `opponent_name` parameter.
Add `--build-opponent-aivat {botname}` flag to `run_eval_harness.py`.

**Success criteria:**
- AIVAT variance ≤ raw variance / 3 for all 7 classic gauntlet bots
- 1k AIVAT-adjusted ranking agrees with 5k raw ranking ≥ 90% of the time

**Estimated effort:** 8–10 hours

---

### WS8 — Slumbot External Benchmark

**Goal:** Run the poly2+refine blueprint against Slumbot via the existing
`eval_harness/external/slumbot_client.py` scaffold to get an external validation point.

**Why this matters:** All internal evaluation is circular — the gauntlet bots and exploit
bots were designed around the solver's known weaknesses. Slumbot is an independent
external benchmark trained by Eric Jackson (author of `slumbot2019`). A positive result
vs. Slumbot would validate the blueprint's real-world quality. A loss would indicate
which strategic patterns an independent solver exploits.

**Protocol:**
- 10k hands minimum, 3 seeds if API allows
- Report: bb/100 vs Slumbot, strategy divergence analysis

**Success criteria:**
- Positive result (≥ 0 bb/100) validates blueprint quality
- If negative: identify top-5 divergent infosets and add to WS0/WS1 targets

**Estimated effort:** 4–6 hours (depends on API stability)

---

## 5. Improvements to the B0 Model Specifically

Beyond the new workstreams, several targeted improvements to the existing B0 blueprint
are worth pursuing before any major retrain:

### B0 Post-Hoc Improvements (no retraining)

**1. Pseudo-harmonic mapping (WS0)** — the most impactful zero-retrain improvement.
Action translation is provably wrong with nearest/confidence approaches near bucket
boundaries. Implementing pseudo-harmonic would close the WeirdSizingBot gap further and
improve robustness to any off-tree sizing.

**2. Board texture integration in confidence_nearest** — `texture_adjustments()` in
`board_texture.py` exists but is never called. Wiring it into `confidence.py`'s alpha
calculation would give board-aware blending on the current blueprint at zero training cost.
Monotone boards → lower confidence in the blueprint (more blending); dry boards → higher.

**3. River-specific visit threshold calibration** — The Refine 2.0 trigger uses
`visit_count < 200` calibrated on B0. The river divergence analysis shows the largest
strategic differences are on the river. Setting a river-specific threshold of 150 (more
aggressive refinement on river spots) could improve the river value-betting deficit.

### B0 Retrain Improvements

**4. VR-MCCFR in Cython (WS2)** — Adding a variance reduction baseline to
`cfr_fast.pyx` allows the same node count to converge faster. A B0 retrain with
VR-MCCFR would reach the current 1.22 bb/100 exploitability in ~50M iterations and
potentially reach ~0.8 bb/100 by 100M. This is the single highest ROI training change.

**5. Suit isomorphism (WS3)** — Collapse strategically identical postflop nodes.
A B0 retrain with canonicalized infosets would have fewer total nodes but each better
trained. Combined with VR-MCCFR, a new "B0-v2" retrain could outperform current B0
at half the iteration budget.

**6. DCFR asymmetric discounting** — Brown & Sandholm (DCFR, AAAI 2019) introduced
asymmetric `α/β` discounting for positive vs. negative regrets separately. Your v10
DCFR sweep used symmetric discounting (`γ < 1` applied to both). The asymmetric variant
(e.g., `α=3/4, β=1/2` from the paper) was not tested and may avoid the divergence seen
with symmetric DCFR on wider action trees.

---

## 6. Execution Order

V13 should proceed in three phases:

### Phase A — Quick wins (no retraining, 2–4 weeks)

1. **WS0:** Pseudo-harmonic mapping — measurable gain, no training cost
2. **WS7:** AIVAT per-opponent calibration — makes all future eval more reliable
3. **WS8:** Slumbot evaluation — external validation before major investment
4. Board texture wired into `confidence_nearest` — low effort, free improvement

### Phase B — Improved blueprint via smarter retraining (4–8 weeks)

5. **WS2c:** Zhang 2026 hyperparameter schedule sweep on Kuhn poker (1–2 days)
6. **WS2a:** PCFR+ optimistic regret update in `cfr_fast.pyx`
7. **WS2b:** VR-MCCFR baseline variance reduction in `cfr_fast.pyx`
8. **WS3:** Suit isomorphism canonicalization
9. **B0-v2 retrain:** B0 with PCFR+ + VR-MCCFR + suit isomorphism + Zhang schedule
10. **WS1:** Refine 3.0 (safe subgame solving + K=2 leaves) against B0-v2

**Gate:** B0-v2 must beat current B0 in H2H before proceeding to Phase C. The PCFR+ and
Zhang schedule changes together should be sufficient to move exploitability — if neither
moves the needle, the ceiling is confirmed as abstraction-only and we skip to Phase C.

### Phase C — Abstraction redesign (8–12 weeks)

11. **WS4:** Potential-aware EMD bucketing (precompute histograms, cluster, retrain)
12. **WS5:** Board texture as key dimension (implement after WS4 converges)
13. **WS5b:** Embedding CFR prototype (in parallel with WS5, research track)
14. **WS6:** RL-CFR dynamic action abstraction (in parallel, research track)

---

## 7. Success Criteria for V13

V13 should be considered successful if it achieves **at least two** of the following:

| Criterion | Target | Current | Primary WS |
|-----------|--------|---------|-----------|
| Exploitability (best model) | < 0.80 bb/100 | 1.22 (B0) | WS2a/b/c, WS4 |
| H2H B0-v2 vs B0 | B0-v2 wins significantly | — | WS2a/b/c + WS3 |
| WeirdSizingBot (best mapping) | > +80 bb/100 | +32.7 (poly2+refine) | WS0, WS1 |
| OOP deficit in H2H | < −500 bb/100 | −1,147 (poly2 vs B0) | WS4, WS5 |
| Slumbot | > 0 bb/100 | unknown | WS8 |
| AIVAT variance reduction | ≥ 3× vs raw | currently inflates | WS7 |
| Refine 3.0 vs Refine 2.0 on PerturbBot | no regression | −215 bb/100 | WS1 |

---

## 8. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| EMD precomputation too slow for 169×500 boards | Medium | Parallelize with `multiprocessing`; use GPU if needed; limit to 200 board samples per hand |
| VR-MCCFR baseline requires storing additional values in cfr_fast.pyx | Low | Baseline = node average value, already computable from `strategy_sum`; no new storage needed |
| Suit isomorphism implementation bugs (incorrect canonicalization) | Medium | Validate using `kdub0/hand-isomorphism` C library as reference; check all 169 preflop hands |
| Board texture dimension causes Cython key overflow | High | Use 2-bit encoding with reduced history depth (WS5 design section above) |
| RL-CFR training loop destabilizes the solver | Medium | Run on isolated copy of B0; do not write to main strategy file |
| Slumbot API changes or rate limits | Medium | Run in batches; cache all hands for offline analysis |

---

## 9. Non-Goals for V13

The following were considered and explicitly excluded:

- **Deep CFR / neural architecture** — high implementation cost, uncertain gain, requires
  GPU infrastructure. Only relevant if you need exploitability < 0.1 bb/100.
- **Multiplayer (3+ player)** — requires fundamentally different CFR variant; out of scope
- **Opponent exploitation (non-GTO)** — OpponentProfile was retired in v10; rebuilding it
  is a separate research track from GTO improvement
- **Broader action grid (16 actions globally)** — confirmed not viable at 200M iterations
  (16-action grid exploitability 41.4, v10). Only selective expansion is viable.
- **Further weight-schedule exploration** — linear, poly, DCFR, scheduled all tested;
  none beat B0 in H2H; this research track is closed.

---

## 10. Files to Create / Modify

| File | Action | WS |
|------|--------|-----|
| `eval_harness/match_engine.py` | Add `pseudo_harmonic` mapping to `GTOAgent` | WS0 |
| `server/gto/abstraction.py` | Add `pseudo_harmonic_translate()` function | WS0 |
| `server/gto/local_refine.py` | Gift action + K=2 leaves (Refine 3.0) | WS1 |
| `server/gto/cfr_fast.pyx` | PCFR+ prev_regret buffer + optimistic update | WS2a |
| `server/gto/cfr_fast.pyx` | VR-MCCFR baseline in inner loop | WS2b |
| `train_gto.py` | `--solver pcfr+` flag, Zhang 2026 schedule mode | WS2a, WS2c |
| `server/gto/equity.py` | `canonicalize_hand_board()` + EMD lookup table | WS3, WS4 |
| `server/gto/abstraction.py` | Infoset key + texture dimension | WS5 |
| `server/gto/cfr_fast.pyx` | Updated key encoding for texture | WS5 |
| `server/gto/emd_clustering.py` | NEW: equity histogram precomputation + EMD k-means | WS4 |
| `eval_harness/aivat.py` | Per-opponent calibration flag | WS7 |
| `run_eval_harness.py` | `--build-opponent-aivat`, `--mapping pseudo_harmonic` | WS0, WS7 |
| `docs/plans/ACTION_PLAN_v13.md` | This document | — |

---

## 11. Recommended Reading Order

For anyone starting fresh on v13 implementation, read in this order:

1. **Ganzfried & Sandholm 2013 (Pseudo-Harmonic Mapping)** — understand action translation
   theory before implementing WS0; 30-minute read, immediate payoff
2. **Brown & Sandholm 2017 (Safe and Nested Subgame Solving)** — understand the gift-action
   construction before touching `local_refine.py`; read `AI-Decision/DecisionHoldem` source
   alongside
3. **Farina, Kroer & Sandholm 2020 (PCFR+/Optimistic CFR)** — understand the prediction
   buffer before touching `cfr_fast.pyx` for WS2a
4. **Zhang, McAleer & Sandholm 2026 (Hyperparameter Schedules)** — read before running
   WS2c; the small-game sweep methodology translates directly to `run_toy_validation.py`
5. **Schmid et al. 2019 (VR-MCCFR)** — understand the baseline construction for WS2b
6. **Ganzfried & Sandholm 2014 (Potential-Aware EMD)** — understand equity trajectory
   clustering before starting WS4
7. **Fu et al. 2024 (KrwEmd / Signal Observation Models)** — read alongside WS4 to
   understand the theoretical ceiling on imperfect-recall bucketing
8. **Waugh 2013 (Hand Isomorphism)** — study `kdub0/hand-isomorphism` C code before WS3
9. **Li et al. 2024 (RL-CFR)** — read the GitHub (`lbn187/RL-CFR`) before WS6
10. **Fu et al. 2026 (Embedding CFR)** — read before WS5b prototype
