# V11 Workstream E1: AIVAT Feasibility Study

**Date:** March 15, 2026
**Status:** Feasibility assessment
**Scope:** Variance reduction for agent evaluation via AIVAT control variates

---

## 1. Background

AIVAT (A New Variance Reduction Technique for Agent Evaluation) reduces variance in poker agent evaluation by subtracting a zero-mean control variate computed from the agent's own strategy. The core idea: if we know what an agent *would have done* at every decision point and every chance node, we can subtract the expected contribution of luck, leaving a lower-variance estimate of true skill difference.

At 5,000 hands with 3 seeds, the current gauntlet produces 95% confidence intervals of roughly +/- 100-150 bb/100 for most bots (see `docs/results/v10_complete_report.md` Section 5.5). This is wide enough that a +50 bb/100 improvement from a new strategy branch might be statistically indistinguishable from noise. AIVAT could tighten these CIs by 3-10x, making it possible to detect smaller improvements with fewer hands.

---

## 2. What AIVAT Requires (Theory)

AIVAT needs three things per hand:

### 2.1 Full Decision Tree Logging

Every decision point must be recorded with:
- Who acted (player index)
- Game state at decision time (pot, bets, community cards, phase)
- All available actions and their probabilities under the agent's strategy
- The action actually taken
- The counterfactual value of each unchosen action

Currently, `HeadsUpMatch._betting_round()` in `eval_harness/match_engine.py` records `DecisionRecord` when `detailed_tracking=True`, but only for the action taken. It does not log:
- The full strategy distribution at the decision point
- The available action set
- The pot/bet state needed to compute counterfactual values

### 2.2 Expected Values Under the Agent's Strategy

At each decision node for the controlled agent, we need:
```
EV_agent(node) = sum over actions a: pi(a | node) * V(node, a)
```

where `V(node, a)` is the expected value of taking action `a` at that node. For the GTO agent, `pi(a | node)` is available from `trainer.get_strategy()` or directly from the node's average strategy. But `V(node, a)` requires either:
- (a) Rolling out each action to terminal (expensive: O(actions * depth) per node)
- (b) Precomputed node values from the CFR solution (available as regret sums, but not directly as EV)

### 2.3 Chance Node Baselines

At each chance node (card dealt), AIVAT subtracts the expected value over all possible cards weighted by their probability. This requires:
- Knowing which card was dealt at each chance event (deck deal for hole cards, flop, turn, river)
- Computing the expected value over all possible cards at that point
- Access to the full deck state (which cards remain)

Currently, `HeadsUpMatch._play_hand()` deals cards via `Deck()` but does not expose the deck state or enumerate card probabilities.

---

## 3. What the Current Codebase Has

### 3.1 Available Infrastructure

| Component | Location | What it provides | Gap for AIVAT |
|-----------|----------|-----------------|---------------|
| `HeadsUpMatch` | `eval_harness/match_engine.py` | Plays hands, records `HandRecord` with p0_net, street_ev | No full decision tree, no strategy at each node |
| `GTOAgent.decide()` | `eval_harness/match_engine.py:91` | Computes strategy, samples action, returns concrete decision | Strategy computed but not recorded; only chosen action logged |
| `trainer.get_strategy()` | `server/gto/cfr.py` | Full strategy lookup by (phase, bucket, history, position) | Available, but need to call at every decision point and log result |
| `DecisionRecord` | `eval_harness/match_engine.py:322` | Records player, phase, action, bucket, eq_bucket, pot_before, amount | Missing: full strategy distribution, available actions, counterfactual values |
| `HandRecord` | `eval_harness/match_engine.py:335` | Per-hand result with decisions list, street_ev | Missing: chance node events, full decision trees |
| `bridge_log` | `eval_harness/match_engine.py:89` | GTOAgent records (concrete_ratio, mapped_action, phase, bucket) | Only captures off-tree translation events, not all decisions |
| `fast_bucket()` | `eval_harness/fast_equity.py` | Quick bucket computation | Already used in decide(); available for node value computation |

### 3.2 Critical Gaps

**Gap 1: Adversary bots are black boxes.** `NitBot`, `AggroBot`, `CallStationBot`, etc. in `eval_harness/adversaries.py` implement `decide()` as heuristic functions. They do not expose:
- A strategy distribution over actions (they sample internally and return one action)
- Expected values at decision points
- Any policy representation that AIVAT could use

This means full AIVAT (which needs both players' strategies) is not possible without either:
- (a) Refactoring all bots to expose strategy distributions (high effort, fundamentally changes bot architecture)
- (b) Using "half-AIVAT" that only uses the GTO agent's baseline (practical, reduced but still significant variance reduction)

**Gap 2: No chance node enumeration.** `Deck.deal()` returns cards but the match engine does not record what the deck state was before each deal, nor enumerate over alternative outcomes.

**Gap 3: No counterfactual rollout infrastructure.** Computing `V(node, a)` for unchosen actions requires either rollout or precomputed values. The CFR nodes store cumulative regrets (`node.regret_sum`) and strategy sums (`node.strategy_sum`), but not expected values. Regrets approximate the *difference* in EV between actions and the policy EV, not absolute values.

---

## 4. Implementation: Half-AIVAT (GTO Baseline Only)

The practical path is "half-AIVAT": use the GTO agent's strategy as a control variate while treating the opponent as a black box. This is a standard simplification described in the AIVAT literature and still provides significant variance reduction.

### 4.1 Core Algorithm

For each hand, compute the AIVAT-adjusted payoff:

```python
adjusted_payoff = actual_payoff - sum(baseline_corrections)
```

where each baseline correction corresponds to one of the GTO agent's decision points:

```python
# At each GTO decision node:
strategy = trainer.get_strategy(phase, bucket, history, position)
chosen_action = the_action_gto_actually_took
baseline_correction = actual_continuation_value - sum(
    strategy[a] * continuation_value[a] for a in available_actions
)
```

For the chosen action, `continuation_value[chosen_action]` is the actual hand payoff from that point forward. For unchosen actions, we use an approximation (see 4.3).

### 4.2 Decision Tree Logging (Required Changes)

**File: `eval_harness/match_engine.py`**

Extend `DecisionRecord` to include AIVAT-relevant fields:

```python
@dataclass
class DecisionRecord:
    player: int
    phase: str
    action: str              # concrete action taken
    bucket: int
    eq_bucket: int
    pot_before: int
    amount: int
    # --- AIVAT additions ---
    strategy: dict = None    # {action_int: probability} at this node
    available_actions: list = None  # list of Action ints
    infoset_key: str = ""    # for node lookup
    abstract_action: int = -1  # abstract action chosen
```

In `GTOAgent.decide()`, before sampling the action, store the computed strategy and infoset key on the agent instance so `_betting_round()` can capture them:

```python
# In GTOAgent.decide():
self._last_strategy = strategy  # full distribution
self._last_actions = action_ids
self._last_infoset_key = key
self._last_abstract_action = chosen  # before concrete conversion
```

In `_betting_round()`, when constructing `DecisionRecord`:

```python
if hasattr(agent, '_last_strategy'):
    decisions.append(DecisionRecord(
        ...,
        strategy=agent._last_strategy,
        available_actions=agent._last_actions,
        infoset_key=agent._last_infoset_key,
        abstract_action=agent._last_abstract_action,
    ))
```

### 4.3 Continuation Value Approximation

The expensive part of AIVAT is computing continuation values for unchosen actions. Three options, ordered by cost:

**Option A: Bucket-EV table (cheapest).** Precompute average EV per equity bucket from historical hand data. Use `EV[eq_bucket]` as the continuation value for unchosen actions. This is a crude approximation but:
- Requires no rollout at evaluation time
- Can be built from existing gauntlet results
- Provides ~3-5x variance reduction based on literature estimates

**Option B: Single-rollout sampling.** For each unchosen action, simulate one random continuation from the decision point. This is unbiased but noisy per-hand; across many hands, the noise averages out. Cost: ~(A-1) additional rollouts per GTO decision, where A is the number of available actions (typically 3-8). Rough estimate: 3-7x slowdown per hand.

**Option C: Multi-rollout sampling.** Like Option B but with 5-10 rollouts per unchosen action. Better per-hand estimates, 15-70x slowdown.

**Recommendation:** Start with Option A for the pilot. It requires no simulation infrastructure changes and can be implemented in ~100 lines. If variance reduction is insufficient, upgrade to Option B.

### 4.4 Chance Node Corrections (Optional Enhancement)

For full AIVAT, chance node corrections further reduce variance. At each card deal:

```python
# After dealing community card(s):
actual_board = dealt_cards
# Compute E[payoff] over all possible boards (weighted by probability)
expected_payoff = mean(payoff(alternate_board) for alternate_board in all_possible_boards)
chance_correction = actual_payoff_from_here - expected_payoff
```

This is expensive for flop (C(remaining, 3) = ~17,000 combinations) but cheap for turn/river (C(remaining, 1) = ~45 combinations). Implementation:
- Turn/river chance corrections: feasible, ~45 evaluations per deal
- Flop chance corrections: expensive, consider sampling 200 random flops instead of enumerating all

**Recommendation:** Defer chance node corrections to a second iteration. Decision-point corrections alone should provide most of the variance reduction.

---

## 5. Effort Estimate

### 5.1 Half-AIVAT with Bucket-EV Approximation (Pilot)

| Task | Lines | Hours | Files |
|------|-------|-------|-------|
| Extend `DecisionRecord` with strategy/actions fields | ~30 | 1 | `match_engine.py` |
| Modify `GTOAgent.decide()` to expose last strategy | ~15 | 0.5 | `match_engine.py` |
| Capture extended records in `_betting_round()` | ~20 | 0.5 | `match_engine.py` |
| Build bucket-EV table from historical data | ~60 | 2 | new: `eval_harness/aivat.py` |
| AIVAT correction calculator | ~120 | 3 | `eval_harness/aivat.py` |
| Integration with `MatchResult` reporting | ~40 | 1 | `match_engine.py`, `run_eval_harness.py` |
| Tests | ~80 | 2 | `tests/test_aivat.py` |
| **Total** | **~365** | **~10** | |

### 5.2 Half-AIVAT with Single-Rollout Sampling (Phase 2)

| Task | Lines | Hours | Files |
|------|-------|-------|-------|
| Rollout simulator from mid-hand state | ~150 | 4 | `eval_harness/aivat.py` |
| Action continuation value computation | ~80 | 2 | `eval_harness/aivat.py` |
| Integration with pilot infrastructure | ~40 | 1 | |
| **Total (incremental)** | **~270** | **~7** | |

### 5.3 Full AIVAT (Phase 3, Requires Bot Strategy Exposure)

| Task | Lines | Hours | Files |
|------|-------|-------|-------|
| Define `StrategyAgent` protocol (strategy distribution API) | ~30 | 1 | `match_engine.py` |
| Retrofit all 7 adversary bots with strategy distributions | ~300 | 8 | `adversaries.py` |
| Dual-player AIVAT corrections | ~100 | 3 | `eval_harness/aivat.py` |
| Chance node corrections (turn/river) | ~120 | 3 | `eval_harness/aivat.py` |
| **Total (incremental)** | **~550** | **~15** | |

---

## 6. Expected Variance Reduction

Based on AIVAT literature and the structure of this codebase:

| Approach | Expected Variance Reduction | Effective Sample Multiplier | Practical Impact |
|----------|---------------------------|---------------------------|-----------------|
| Current (raw payoff) | Baseline | 1x | 5k hands needed for +/- 100 bb/100 CIs |
| Half-AIVAT (bucket-EV) | ~60-80% | 3-5x | 1k hands equivalent to current 5k |
| Half-AIVAT (rollout) | ~80-90% | 5-8x | 1k hands equivalent to current 5-8k |
| Full AIVAT | ~90-95% | 10-20x | 500 hands equivalent to current 5-10k |

The practical value: **with half-AIVAT, a 1,000-hand evaluation would produce CIs comparable to the current 5,000-hand protocol.** This means:
- Branch comparison runs 5x faster (minutes instead of hours)
- Or: keep 5,000 hands and get CIs of +/- 30-50 bb/100 instead of +/- 100-150
- Enables detecting +20 bb/100 improvements that are currently invisible in noise

---

## 7. Pilot Plan

### 7.1 Phase 1: Validate Concept (1-2 days)

1. **Build bucket-EV table.** Run 10,000 hands of GTO vs CallStationBot with `detailed_tracking=True`. Compute average P0 net bb by equity bucket at each phase. Save as `eval_harness/bucket_ev_table.json`.

2. **Implement half-AIVAT calculator.** New file `eval_harness/aivat.py`:
   - `compute_aivat_correction(hand: HandRecord, bucket_ev: dict) -> float`
   - `aivat_adjusted_result(result: MatchResult, bucket_ev: dict) -> AivatResult`
   - `AivatResult` dataclass with `raw_bb100`, `aivat_bb100`, `raw_ci`, `aivat_ci`, `variance_reduction_pct`

3. **Compare CI widths.** Run GTO vs each bot at 1,000 hands (3 seeds) with and without AIVAT correction. Report:
   - Raw CI width vs AIVAT CI width per bot
   - Variance reduction percentage
   - Sanity check: AIVAT mean should be close to raw mean (unbiased)

### 7.2 Phase 2: Integrate into Eval Harness (1 day)

4. **Add `--aivat` flag to `run_eval_harness.py`.** When enabled:
   - Force `detailed_tracking=True` on all matches
   - Compute AIVAT-adjusted results alongside raw results
   - Report both in the gauntlet table

5. **Add to multi-seed gauntlet.** `run_gauntlet_multiseed()` reports both raw and AIVAT-adjusted per-bot CIs.

### 7.3 Phase 3: Validate Reduction (1 day)

6. **1k vs 5k comparison test.** Run GTO vs all 7 bots at:
   - 1,000 hands with AIVAT (3 seeds)
   - 5,000 hands without AIVAT (3 seeds)

   Compare: are the CIs similar? Is the mean stable? This validates that AIVAT at 1k is as reliable as raw at 5k.

7. **Document results** in `docs/results/v11_aivat_pilot.md`.

---

## 8. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Bucket-EV approximation too crude | Medium | Fall back to single-rollout sampling (Phase 2 effort) |
| AIVAT bias from abstraction errors | Low | The control variate is zero-mean by construction; abstraction errors affect variance reduction magnitude, not bias |
| Performance overhead from extended logging | Low | `DecisionRecord` additions are small dataclass fields; measured overhead should be <5% per hand |
| Bot strategy exposure too expensive | Medium | Stay with half-AIVAT; the 3-5x reduction from GTO-only baseline is sufficient for branch selection |
| Bucket-EV table opponent-dependent | Medium | Build separate tables per opponent class, or use a universal table from self-play |

---

## 9. Decision

**Recommendation: Proceed with half-AIVAT pilot (bucket-EV approximation).**

The effort is modest (~10 hours, ~365 lines), the expected payoff is high (3-5x variance reduction), and it directly addresses the evaluation bottleneck identified in v10 (wide CIs requiring 5k hands per measurement). The pilot is self-contained and carries no risk to existing infrastructure.

If the pilot achieves >= 3x variance reduction, proceed to integration with the eval harness. If < 2x, upgrade to single-rollout sampling before integrating.
