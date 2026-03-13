# Root Cause Analysis — GTO Solver v6→v7
*Originally generated: 2026-03-12 against eval_1773329751*
*Updated: 2026-03-12 — v7 deployed (eval_1773348554). See `docs/ACTION_PLAN_v7.md` for next steps.*

---

## Executive Summary

**v7 status as of 2026-03-12:** Issues 2, 3, and 11 are fully resolved. Issue 1 (CallStation)
is largely resolved (+439.5 bb/100 from -463). Issues 4–9 remain and are planned for the
next retrain. See `docs/ACTION_PLAN_v7.md` for the prioritized plan.

| # | Issue | Severity | Status (v7) |
|---|-------|----------|-------------|
| 1 | River over-bluffing with EQ0 hands | ~~Critical~~ Medium | Mostly fixed — bluff ratio 35.1% HEALTHY; isolated EQ0 nodes still 100% |
| 2 | Default bridge mapping `nearest` is negative EV | ~~Critical~~ **FIXED** | `nearest` now +276.9 avg (was -200.8). `conservative` is now -33.5 |
| 3 | `engine.py` history reconstruction incomplete | ~~High~~ **FIXED** | BET_HALF_POT added, unified with match_engine via abstraction.py |
| 4 | Premium limp (BB not raising over limps) | High | Still present (4 nodes) — planned post-training clamp |
| 5 | Strong hand folding at EQ6 | High | Still present — planned post-training clamp |
| 6 | All-in dampening too broad (raise_count < 2) | Medium | Still present — next retrain fix: raise_count == 0 |
| 7 | `bet_overbet` bucket undertrained | Medium | Still present — next retrain fix: 3x phase schedule |
| 8 | Donk bets over-fired with zero-equity hands | Medium | Reverted restriction (caused regression). Monitoring only |
| 9 | Flat strategy nodes (1,360 unconverged) | Medium | Still present (1,334) — next retrain fix: 3x phase schedule |
| 10 | Exploitability metric is per-street not full-game | Low | Open design issue |
| 11 | OverfoldBot/NitBot — structural GTO limitation | ~~Low~~ **FIXED** | NitBot +235.9 (was -19). OverfoldBot -291.4 (new active issue) |

---

## Issue 1 — River Over-Bluffing with EQ0 Hands

### Symptom
- **CallStationBot: -463.1 bb/100** (50M), -285.3 bb/100 (20M) — worsening with training
- River bluff audit: `river:ip:3:1`, `river:ip:1:1`, `river:ip:8:1`, `river:ip:10:1`, `river:ip:13:1`
  all show **bet=100%, EQ0** (pure air betting 100% of the time)
- Overall bluff ratio 35% is healthy, but individual EQ0 nodes are pure-bluffing

### Root Cause
The CFR solver has no opponent model. GTO bluff frequencies are calibrated against a
**balanced range** that sometimes folds. Against CallStationBot (never folds postflop),
bluffing with EQ0 is pure chip donation.

More specifically: the pattern `river:ip:X:1` (history = single CHECK_CALL action) represents
IP on the river after OOP checks. The solver learned to bet 100% here with EQ0 because in
the training population, OOP sometimes folds to a bet. The regret for "bet 100% with EQ0"
is positive against a balanced opponent but catastrophically negative against a station.

The key structural problem: **the bucket abstraction collapses `EQ0 with outs` and `EQ0 pure
air` into the same bucket**. On the flop or turn, EQ0 might have draws and semi-bluff equity;
on the river, EQ0 is pure air with zero showdown value. The solver treats them identically.

### Evidence
```
river:ip:3:1:   bet=100%  EQ0 LOW_PAIR         ← pure bluff; LOW_PAIR missed
river:ip:1:1:   bet=100%  EQ0 HIGH_PAIR        ← missed board; should check/fold
river:ip:10:1:  bet=100%  EQ0 STRONG_OFFSUIT_ACE
river:ip:13:1:  bet=100%  EQ0 SUITED_TRASH
```
All are single-action history (OOP checked), IP position. These are 100% bluff frequencies
against an opponent who just checked—the exact spot CallStationBot traps in.

### Fixes
1. **Training fix**: Add a `bet_check` terminal regret penalty when EQ0 bet gets called at
   showdown. Weight the terminal utility by a "bluff-penalty" term for low-equity bets.
2. **Abstraction fix**: Split EQ0 into two sub-buckets: `EQ0_DRAW` (4+ outs) vs `EQ0_AIR`.
   This requires expanding `NUM_EQUITY_BUCKETS` from 8 to 9-10.
3. **Bluff cap (heuristic)**: Post-training, cap river bluff frequency at EQ0 to 25-30% max
   using the average strategy normalize step. This is lossy but prevents the worst outliers.
4. **Opponent-adaptive mode**: Allow the engine to detect "station" behavior patterns
   (call-rate > 85%) and shift to `conservative` bridge + reduced bluff frequency.

---

## Issue 2 — Default Bridge Mapping `nearest` Is Negative EV

### Symptom
```
nearest      -200.8 bb/100 avg  ← current default
conservative +274.5 bb/100 avg  ← best
stochastic   +246.3 bb/100 avg
resolve      -403.0 bb/100 avg
```
Gap between worst and best: **475.3 bb/100**. The default loses to all three alternatives.

### Root Cause
`nearest` maps off-tree actions to the closest abstract sizing by magnitude. A 120% overbet
gets mapped to `bet_overbet` (1.25x). But `bet_overbet` was added in v6 and has the most
frequency anomalies — it's a poorly-trained action node. `nearest` routes live decisions into
these undertrained nodes.

`conservative` instead takes 15% probability from every bet-sizing action and moves it to
`CHECK_CALL` (well-trained across all phases). This is a deliberate hedge that avoids
committing to poor nodes.

```python
# conservative (translation_ab.py:72-82)
for a in [BET_THIRD_POT, BET_HALF_POT, BET_TWO_THIRDS_POT, BET_POT, BET_OVERBET, ...]:
    steal = strategy[a] * 0.15
    strategy[a] -= steal
    boost += steal
strategy[CHECK_CALL] += boost
```

`resolve` performs worst (-403.0) because its pot-odds adjustment overcompensates — it shifts
fold/call probabilities up to 30% based on pot-odds ratio, amplifying small errors into
large strategy divergences.

### Fix
**Immediate**: Change the default mapping in `engine.py` and `match_engine.py` from `nearest`
to `conservative`. Zero training cost, immediate +475 bb/100 average improvement.

```python
# engine.py — change default bridge agent used at runtime
# Current (implicitly nearest):
#   strategy = trainer.get_strategy(phase, bucket, abstract_history, position=position)
#
# Fix: apply conservative adjustment before sampling
```

This is the single highest-ROI change available.

---

## Issue 3 — `engine.py` History Reconstruction Incomplete

### Symptom
`engine.py:213-248` — `_abstract_history()` maps live action strings to abstract action IDs.

### Root Cause

**Bug A — `BET_HALF_POT` has no entry in the postflop mapping:**
```python
# engine.py:229-240 postflop mapping
mapping = {
    "fold":         int(Action.FOLD),
    "check":        int(Action.CHECK_CALL),
    "call":         int(Action.CHECK_CALL),
    "raise_small":  int(Action.BET_THIRD_POT),
    "raise":        int(Action.BET_TWO_THIRDS_POT),  ← skips BET_HALF_POT entirely
    "raise_big":    int(Action.BET_POT),
    "raise_overbet":int(Action.BET_OVERBET),
    ...
}
# Fallback for unknown: int(Action.CHECK_CALL)
```
If a live game records a 50% pot bet as `"raise_half"` or any unlisted string, it silently maps
to `CHECK_CALL` (a check). This corrupts the abstract history, causing the wrong infoset node
to be queried on subsequent streets.

**Bug B — `is_in_position` parameter is ignored:**
```python
def gto_decide(player, community_cards, pot, current_bet, min_raise, big_blind,
               num_opponents=1, betting_history=None,
               is_in_position=True) -> GTODecision:  # ← parameter accepted
    ...
    # position is computed from history length, not is_in_position
    position = 'oop' if len(abstract_history) % 2 == 0 else 'ip'  # L140
```
The caller's knowledge of position is discarded. If `betting_history` is truncated (e.g., only
shows the current street, not prior streets), `len(abstract_history)` will be wrong and
`position` will be flipped. This would send the engine to the wrong infoset bucket with ~50%
hands in any real deployment.

### Fix
1. Add `BET_HALF_POT` to the postflop mapping: `"raise_half": int(Action.BET_HALF_POT)`
2. Optionally use `is_in_position` as a sanity check against the computed position, logging
   a warning on mismatch (do not blindly override — the history-derived position is more
   accurate when history is complete)
3. Document the expected `betting_history` format (current-street only, or full-game)

---

## Issue 4 — Premium Limp: BB Not Raising Over Limps

### Symptom
```
[high] premium_limp: PREMIUM_PAIR at EQ5 limps 100%
[high] premium_limp: HIGH_PAIR at EQ5 limps 99%
```
Strategy audit filters: `phase == 'preflop'`, `not has_bet`, `hand_type in (PREMIUM_PAIR,
HIGH_PAIR)`, `eq_bucket >= 5`. These nodes have `visit_weight >= 100` (not just unvisited).

### Root Cause
The specific infoset is: **BB with no bet to call (someone limped), holding a premium pair**.
Available actions: `CHECK_CALL` (check/trap), `OPEN_RAISE`, `ALL_IN`.

The issue is a **convergence failure on a rare path**, not a code bug. In training:
- OOP (SB) limping is not the dominant preflop strategy — SB open-raises most of the time
- Consequently, `preflop:ip:X:1` (BB facing limp, history = one CHECK_CALL) is visited rarely
- The `averaging_delay = iterations // 4 = 12.5M` means the first 12.5M iterations have
  weight 0. If this node accrues most visits in the early phase, its useful signal is discarded
- Later visits may be too sparse to overcome the uniform initialization

The solver "knows" to raise with premiums facing opens (✓) but hasn't converged on the
limp case because limping SB is an off-equilibrium path it rarely encounters.

### Fix
1. **Targeted extra training**: Run 5M additional iterations where P0 (SB) is forced to limp
   with 20% probability (overriding the policy). This creates more visits to the BB facing
   limp scenario.
2. **Post-processing clamp**: After training, clamp premium limp probability to 5% max for
   PREMIUM_PAIR/HIGH_PAIR at EQ5+ when `not has_bet` and `phase == preflop`. Override via
   renormalization.
3. **Reduce averaging_delay** for rare nodes (low `strategy_sum`): instead of `iter // 4`
   uniformly, apply lower delay for nodes with low visit weight.

---

## Issue 5 — Strong Hand Folding at EQ6

### Symptom
```
[high] strong_hand_folding: Folding 100% at EQ6  (× 3+ occurrences)
```
Audit condition: `fold_prob > 0.20 and eq_bucket >= 6 and has_bet`. EQ6 = top ~25% hands.

### Root Cause
Two possible causes, both likely contributing:

**Cause A — Bucket boundary artifacts**: Hands near the EQ6/EQ7 boundary may have equity
that drifts across the boundary when new streets are dealt. A hand that started as EQ7
preflop can end up in EQ6 on the turn after a bad run-out. The strategy learned for EQ7 is
aggressive; the strategy learned for EQ6 is more cautious. Boundary cases see conflicting
signal and may not converge to the correct fold frequency.

**Cause B — Rare history paths**: The `has_bet=True, eq_bucket=6` combination with a
specific history (e.g., after multiple raises) may be visited infrequently. At those nodes,
the solver defaults toward a uniform strategy, and if the regret for fold is slightly higher
than other actions early in training, it converges to 100% fold.

The equity bucket is assigned at the **start of each street** using `hand_strength_squared`,
but the showdown utility only uses `eq_bucket // NUM_HAND_TYPES` (the equity dimension).
A hand with EQ6 but a hand_type that creates a blocker (e.g., SUITED_ACE on a suited board)
might be treated as a weak hand by the strategy even though it has implied odds.

### Fix
1. Inspect the specific keys: query `trainer.nodes` for the folding keys and check their
   `regret_sum` — if the fold regret is huge and other actions are near zero, it's a
   convergence issue requiring more targeted training.
2. **Post-training override**: Cap fold probability at 30% for EQ6+ nodes with `has_bet=True`
   in the final strategy export. This is a heuristic patch but prevents catastrophic leaks.
3. **Equity drift smoothing**: When crossing a street boundary, if a hand's equity drops from
   EQ7 to EQ6, blend the two bucket strategies (70/30) instead of hard-switching. This
   requires changes to the continuation value computation in `cfr.py`.

---

## Issue 6 — All-In Dampening Too Broad

### Symptom
- `allin_overuse`: 488 instances despite 0.7x dampening
- Despite dampening, all-in is still over-fired; dampening may be slowing convergence of
  legitimate jam lines

### Root Cause
The dampening condition is `raise_count < 2` (cfr.py:519):
```python
if action == Action.ALL_IN and raise_count < 2:
    regrets[i] *= 0.7
```

This captures both:
- `raise_count == 0`: First street action, jam out of nowhere → correct to dampen
- `raise_count == 1`: Facing one raise (e.g., open raise), re-jam → **should not be dampened**

At `raise_count == 1`, jamming over an open is a valid 3-bet jam with premiums (QQ+, AK).
Dampening this 30% slows convergence of the "jam over open" line. The solver under-converges
on premium 3-bet jams while still over-using all-in in some raw spots.

Additionally, 488 nodes with `allin_overuse` at `raise_count < 2` suggests the dampening
threshold is tuned too permissively — nodes with raise_count=1 keep learning to jam.

### Fix
Change the dampening condition to `raise_count == 0` only:
```python
if action == Action.ALL_IN and raise_count == 0:
    regrets[i] *= 0.5  # Stronger dampening for cold jams
```
This preserves jam-over-open learning while suppressing cold all-ins (the actual problem).

---

## Issue 7 — `bet_overbet` Bucket Undertrained

### Symptom
- Off-tree test: 120% pot overbet → `bet_overbet` → **-1669.5 bb/100** (worst result)
- Strategy anomaly: `frequency_anomalies`: 65,865 instances (very high count)

### Root Cause
`BET_OVERBET` (1.25x pot) was added in v6. The PHASE_SCHEDULE gives it no additional weight
vs other actions:
```python
PHASE_SCHEDULE = [0, 1, 1, 2, 2, 3]  # flop/turn get 2x; overbet gets same as check
```

When a 120% pot overbet is faced by the GTO agent:
1. It maps to `bet_overbet` (nearest concrete = 1.25x abstract)
2. The response strategy is drawn from an undertrained node
3. Undertrained nodes default toward near-uniform strategy
4. Near-uniform strategy = calling and raising equally → poor pot odds response

The 65,865 frequency anomalies are disproportionately in `bet_overbet` response nodes where
entropy ratio > 0.95 — confirming insufficient training signal.

### Fix
1. **Phase schedule modifier**: Give overbet-related histories extra weighting in the
   PHASE_SCHEDULE, e.g., by adding a second pass that specifically samples overbet histories.
2. **Forced overbet exploration**: During training, force-sample overbet actions at 5% of
   iterations to ensure response nodes receive sufficient visits.
3. **Immediate heuristic**: When responding to an overbet, apply a conservative override:
   call with EQ5+ (50%+ equity), fold below. This is used only when the node entropy > 0.8.

---

## Issue 8 — Donk Bets Over-Fired with Zero-Equity Hands

### Symptom
- Flop OOP donk betting 37-62% at EQ0 and EQ5
- This contributes to the river over-bluffing pattern

### Root Cause
Donk bets (`DONK_SMALL`, `DONK_MEDIUM`) are only available at `history_len == 0` on
flop/turn for OOP. The solver learned to use them as probes. At EQ0, donk-betting can be
semi-profitable in training because:

1. EQ0 on the flop may include hands with draws (flush draws, straight draws have EQ0 in
   the E[HS^2] bucket because they have low current equity even with high potential)
2. The solver learns "donk EQ0 on flop" is profitable (fold equity from opponent)
3. This creates a **path dependency**: after donk-betting flop with air, turn and river nodes
   inherit a "bluffing" trajectory where further bets seem justified

The bucket abstraction does not distinguish `EQ0 + 9 outs flush draw` from `EQ0 pure air`.
Both learn the same donk frequency, but only the first is correct.

### Fix
1. **River EQ0 donk clamp**: The river leak is the worst outcome; clamp river EQ0 bet
   frequency to 20% max post-training.
2. **Equity re-check at action time**: When computing donk eligibility, additionally check
   if `num_outs >= 5` (draw exists). Use this to split EQ0 into `EQ0_DRAW` / `EQ0_NODRAW`.
   This is an abstraction change requiring retraining.
3. **Reduce donk availability threshold**: Restrict donk bets to EQ >= 2 (cut out pure air
   donks) by changing `_postflop_actions` to add `is_donk_eligible and eq_bucket >= 2`.

---

## Issue 9 — 1,360 Flat Strategy Nodes (Unconverged)

### Symptom
`flat_strategies: 1360` — nodes with entropy_ratio > 0.95 and 3+ actions, meaning the solver
is still returning near-uniform (random) strategy on 1,360 infosets after 50M iterations.

### Root Cause
The averaging delay of `iterations // 4 = 12.5M` means only the last 37.5M iterations
contribute to strategy averaging. Nodes first visited after 12.5M iterations have full weight,
but nodes that are **visited rarely throughout** never accumulate enough weighted signal to
deviate from uniform.

With 1,046,267 total nodes and 50M weighted iterations: average visits per node ≈ 48 weighted
iterations. For rare nodes (specific bucket × phase × history combinations), actual visits
may be < 10, leaving them near-uniform.

The flop/turn nodes are worst (exploitability 1.38/1.41 vs 1.08/1.16 for preflop/river),
and the flat nodes are disproportionately in flop/turn action subtrees.

### Fix
1. **Continue training**: An additional 50M iterations should resolve the worst flat nodes,
   though diminishing returns suggest < 5% improvement overall.
2. **Adaptive averaging delay**: Use per-node delay based on visit count rather than a global
   threshold. Nodes with < 1000 visits use no delay; heavily visited nodes use the standard
   `iter // 4` delay.
3. **Re-examine PHASE_SCHEDULE**: Increase flop/turn weight from 2x to 3x (change
   `PHASE_SCHEDULE = [0, 1, 1, 1, 2, 2, 2, 3]`) to drive more visits into the highest-
   exploitability streets.

---

## Issue 10 — Exploitability Metric Is Per-Street, Not Full-Game

### Symptom
Reported exploitability: 1.28 ± 0.04. This metric is not directly comparable to industry
standard (e.g., PioSOLVER uses full-game Nash distance).

### Root Cause
`exploitability_abstracted()` in `exploitability.py:182-208`:
```python
for phase in phases:
    br0 = best_response_abstracted(trainer, br_player=0, phase=phase, samples=500)
    br1 = best_response_abstracted(trainer, br_player=1, phase=phase, samples=500)
    total += (br0 + br1) / 2.0
return total / len(phases)   # ← averages 4 independent single-street evaluations
```

Each phase samples **random, independent bucket pairs** — not conditioned on prior street
actions. This ignores cross-street coupling. A hand that plays well preflop may face different
bucket distributions on the flop depending on the preflop action; the metric doesn't capture
this dependency.

The 1.28 value is likely **lower than the true full-game exploitability** because it misses
information-revelation effects across streets.

### Fix
Implement a proper multi-street best response that chains streets using the correlated
bucket structure:
1. Sample bucket pair at preflop (using correlated sampling as in training)
2. Run best response for preflop → get continuation action
3. Apply equity drift to get flop buckets conditioned on preflop action
4. Continue to flop/turn/river

This would give a true exploitability estimate. Computationally expensive but necessary for
accurate benchmarking. Could run as an optional `--full-br` flag.

---

## Issue 11 — Structural GTO Limitation vs OverfoldBot/NitBot

### Symptom
- **OverfoldBot: +18.3 bb/100** (should be ~+500 with pure c-bet strategy)
- **NitBot: -19.0 bb/100** (slightly losing to a tight-passive opponent)

### Root Cause
This is expected but worth documenting. GTO strategies maximize worst-case performance,
not performance against specific exploitable opponents:

**OverfoldBot** folds to any postflop bet. An exploitative strategy (c-bet every hand) would
yield +500 bb/100. GTO strategy checks back ~40% of hands for balance → leaves value on table.

**NitBot** has a top-15% preflop range. When NitBot bets (EQ > 70%), GTO calls with standard
ranges that assume a balanced betting range — not a pure value range. GTO over-calls NitBot's
value bets because it doesn't narrow the range. NitBot's ATC fold + strong-only-bet creates
an exploitable pattern GTO can't adapt to.

The 20M → 50M regression on NitBot (+715.9 → -19.0 bb/100) is significant and unexpected.
This suggests the solver may have learned a preflop opening strategy at 50M that over-calls
NitBot's 3bb opens (paying the "Nit tax"), whereas at 20M it was folding more appropriately.

### Fix
1. **Accept the OverfoldBot gap** — this is correct GTO behavior. Document as known.
2. **Investigate NitBot regression**: Query preflop strategy for `SB vs 3bb open` at 20M
   vs 50M to see if call frequency changed. If the solver is calling too wide vs NitBot's
   3bb open (because GTO assumes balanced range), add NitBot's actual equity distribution
   to the benchmark discussion.
3. **Future work**: opponent modeling layer on top of GTO for exploitative play.

---

## Cross-Cutting Issues

### C1 — `_concrete_to_abstract_history` in match_engine.py vs engine.py Are Different
`match_engine.py` and `engine.py` have separate implementations of history reconstruction
with slightly different mappings. If these diverge, the eval harness measures a different
system than the live engine. Recommend extracting to a single shared function in
`abstraction.py`.

### C2 — Parallel Training Race Conditions on Shared Memory
Parallel workers update `regret_sum` and `strategy_sum` without locks (documented as "benign
races"). For 6 workers × 8.3M iters each, concurrent writes to the same float64 array can
cause silent corruption on certain CPU architectures. While empirically this converges, it
is not provably correct and could explain some of the 65,865 frequency anomalies (strategy
sums accumulating garbage from torn writes).

### C3 — Bucket Decoder Consistency
`decode_bucket` is `(bucket // NUM_HAND_TYPES, bucket % NUM_HAND_TYPES)` but `make_bucket`
is `equity_bucket * NUM_HAND_TYPES + hand_type`. These are consistent, but the naming is
counter-intuitive: `bucket // 15 = equity`, `bucket % 15 = hand_type`. Confirmed correct
but fragile if NUM_HAND_TYPES ever changes (there are hardcoded `15` references in some
audit paths and in cfr_fast.pyx).

---

## Recommended Action Plan

*See `docs/ACTION_PLAN_v7.md` for full detail. Summary:*

### Completed (v7)
1. ~~**Switch bridge default to `conservative`**~~ — Done, then reverted to `nearest` (now best)
2. ~~**Add `BET_HALF_POT` to engine.py history mapping**~~ — Done (unified via abstraction.py)
3. ~~**Log/assert when `is_in_position` disagrees with computed position**~~ — Done

### Next Retrain (v8)
4. **Change all-in dampening** to `raise_count == 0` only, strengthen to `0.5x`
   — Affects: `server/gto/cfr_fast.pyx` + `server/gto/cfr.py`
5. **Increase Cython phase schedule** to 3x flop/turn (edit `cfr_fast.pyx` `train_fast()`)
   — Python `PHASE_SCHEDULE` is dead code; must change the Cython hardcoded schedule
6. **Post-training clamps**: premium limp ≤ 5%, EQ6 fold ≤ 20%
7. **Investigate OverfoldBot -291.4** (new issue — postflop c-bet frequency too low)

### Architectural (v9+)
8. **Split EQ0 bucket** into `EQ0_DRAW` / `EQ0_NODRAW` — requires 9-bucket rebuild
9. **Implement full multi-street best response** for accurate exploitability measurement
10. **Opponent modeling layer** for exploitative play vs stations/nits
11. **Per-node adaptive averaging delay** based on visit count

---

*Full codebase refs: `server/gto/cfr.py`, `server/gto/abstraction.py`, `server/gto/exploitability.py`,
`server/gto/engine.py`, `eval_harness/adversaries.py`, `eval_harness/translation_ab.py`*
