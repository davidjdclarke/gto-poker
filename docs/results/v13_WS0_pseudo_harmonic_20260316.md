# V13 WS0 — Pseudo-Harmonic Action Translation
**Date:** 2026-03-16
**Status:** Complete
**Blueprint:** `experiments/best/v9_B0_100M_allbots_positive.json` (`mapping="pseudo_harmonic"`)

---

## Summary

Implemented pseudo-harmonic action translation (Ganzfried & Sandholm, IJCAI 2013) as a
new `mapping="pseudo_harmonic"` mode in `GTOAgent`. No retraining required — pure
inference-time change. The new mapping is the v13 recommended mapping, replacing `refine`.

**Key result:** WeirdSizingBot improved from −86.4 (B0+refine) to **+79.5 bb/100**
(CI [+11.2, +147.8]). Overall classic gauntlet average: **+205.4 bb/100** — best B0
performance to date, narrowly ahead of B0+refine (+202.6) and B0+conf_nearest (+175.3).

---

## Algorithm

The pseudo-harmonic formula assigns blending weights when the opponent's concrete bet `x`
falls between two abstract actions `a` (lower) and `b` (upper) measured as pre-bet pot
fractions:

```
p_lower = a * (b - x) / [x * (b - a)]
p_upper = 1 - p_lower
```

Blended strategy = `p_lower * strategy(history + lower_action) + p_upper * strategy(history + upper_action)`

Boundary properties: `p_lower(x=a)=1`, `p_lower(x=b)=0`, `p_lower(harmonic_mean(a,b))=0.5`.

This is layered on top of `confidence_nearest` equity-heuristic blending for low-visit
nodes (same as the existing confidence blending code path).

**⚠️ Formula note:** The paper writes `p(x,a,b) = b*(x-a)/[x*(b-a)]` — this is the weight
for the **upper** action. The formula above is the correct `p_lower` derivation.

---

## Implementation Details

### New code (`server/gto/abstraction.py`)

```python
ABSTRACT_BET_FRACTIONS: dict[int, float]
# Canonical pre-bet pot fractions for all 13 postflop abstract bet actions.
# BET_THIRD=1/3, BET_HALF=0.5, BET_TWO_THIRDS=2/3, BET_THREE_QUARTER=0.75,
# BET_POT=1.0, BET_OVERBET=1.25, BET_DOUBLE=2.0, BET_TRIPLE=3.0,
# ALL_IN=3.0 (sentinel), DONK_SMALL=0.25, DONK_MEDIUM=0.5

_BRACKET_LADDER_13: list[tuple[float, int]]
# Sorted (fraction, action_id) for standard 13-action postflop bet sizes.

pseudo_harmonic_translate(concrete_ratio, bracket_actions=None) -> (lower_id, upper_id, p_lower)
# concrete_ratio = to_call / pre_bet_pot (NOT ctx.pot which is post-bet).
```

### Changes to `eval_harness/match_engine.py`

- Import `pseudo_harmonic_translate` from `server.gto.abstraction`
- When `mapping == "pseudo_harmonic"` and postflop and `to_call > 0` and history exists:
  - Compute `pre_bet_pot = ctx.pot - to_call`
  - Compute `concrete_ratio = to_call / pre_bet_pot`
  - Get `(lower_id, upper_id, p_lower)` from `pseudo_harmonic_translate()`
  - Look up strategy at `history[:-1] + (lower_id,)` and `history[:-1] + (upper_id,)`
  - Blend: `strategy[a] = p_lower * s_lower[a] + p_upper * s_upper[a]`
- Added `"pseudo_harmonic"` to the `confidence_nearest` blending condition (applies
  equity-heuristic adjustment for low-visit nodes on top of pseudo-harmonic blend).

### Files modified
- `server/gto/abstraction.py` — new `ABSTRACT_BET_FRACTIONS`, `_BRACKET_LADDER_13`, `pseudo_harmonic_translate()`
- `eval_harness/match_engine.py` — `mapping="pseudo_harmonic"` in GTOAgent
- `run_eval_harness.py` — added `"pseudo_harmonic"` to `--mapping` choices
- `run_h2h.py` — added `"pseudo_harmonic"` to `--p0-mapping` / `--p1-mapping` choices

---

## Bugs Encountered and Fixed

### Bug 1: Post-bet pot convention (gauntlet avg: −61.1 → fixed)

`ctx.pot` in `match_engine.py` includes the opponent's bet already. Using it directly
as the denominator means a half-pot bet (e.g., opponent bets 5 into pot of 10, so `ctx.pot
= 15`) gives `concrete_ratio = 5/15 = 0.333` which maps to BET_THIRD — wrong.

**Fix:** `pre_bet_pot = ctx.pot - to_call; concrete_ratio = to_call / pre_bet_pot`
Now `5 / (15-5) = 5/10 = 0.5` → correctly maps to BET_HALF.

### Bug 2: Formula inversion (gauntlet avg: +226.6, NitBot −254.9 → fixed)

The action plan reproduced the paper's formula as `p(x,a,b) = b*(x-a)/[x*(b-a)]` but
described it as the weight for the lower action. It is actually the weight for the upper
action. A bet of 0.489 (bracketed by BET_THIRD=0.333 and BET_HALF=0.5) gave p_lower=0.955
→ 95% weight on BET_THIRD response. Should be 4.5% (bet is nearly equal to BET_HALF).

**Fix:** `p_lower = a*(b-x)/[x*(b-a)]`. Verification: p_lower(x=a)=1 ✓, p_lower(x=b)=0 ✓,
p_lower at harmonic mean = 0.5 ✓.

### Bug 3: Missing confidence blending (gauntlet avg: +52.2, NitBot −363.9 → fixed)

Without the `confidence_nearest` equity-heuristic blending layer, low-visit nodes return
near-uniform strategies. Bots that bet in-grid amounts (NitBot, PerturbBot) expose this
because pseudo-harmonic only fires when the bet is off-tree — in-tree bets use raw node
lookups which may have low visit counts.

**Fix:** Added `"pseudo_harmonic"` to the existing `if self.mapping in ("blend",
"confidence_nearest", "refine"):` condition that applies equity-heuristic blending.

---

## Gauntlet Results

### Classic Gauntlet — 10k hands × 3 seeds

| Bot | B0+nearest | B0+conf_nearest | B0+refine | poly2+refine | **B0+pseudo_h** |
|-----|----------:|----------------:|----------:|-------------:|----------------:|
| NitBot | +28.5 | +216.1 | +253.1 | +158.1 | +200.0 |
| AggroBot | — | +479.6 | +528.0 | +549.0 | +380.3 |
| OverfoldBot | — | +3.3 | -14.0 | -31.4 | -5.2 |
| CallStationBot | — | +54.5 | +57.9 | +89.6 | +54.5 |
| DonkBot | — | +592.0 | +625.4 | +735.5 | **+694.9** |
| WeirdSizingBot | — | -159.7 | -86.4 | +32.7 | **+79.5** |
| PerturbBot | — | +41.2 | +54.4 | -22.1 | +33.9 |
| **Average** | **+28.5** | **+175.3** | **+202.6** | **+215.9** | **+205.4** |

WeirdSizingBot CI [+11.2, +147.8] (3 seeds: ~+60, ~+90, ~+90).

### Head-to-Head: B0+pseudo_harmonic vs B0+refine (50k hands, 3 seeds)

| Seed | P0 (pseudo_h) bb/100 |
|------|---------------------:|
| 42 | +31.2 |
| 123 | +67.4 |
| 456 | +45.4 |
| **Overall** | **+48.0** |

By position: IP +83.5 bb/100, OOP +11.7 bb/100. All 3 seeds positive.

Key divergence finding from H2H analysis:
- `refine` mapping has pathological "never fold" behavior in certain infosets
- Example: `flop:oop:44:19` — refine fold 0% vs pseudo_harmonic fold 94% for weak
  hands facing 2/3-pot overbets. The mini-CFR solve in refine overrides the blueprint's
  correct fold with a locally computed strategy that lacks global context.
- `pseudo_harmonic` relies on the blueprint for action frequencies, which is more stable.

---

## Comparison to Previous Benchmarks

| Mapping | Classic Avg | WeirdSizingBot | Notes |
|---------|------------:|---------------:|-------|
| nearest | +28.5 | ~-160 | v9 corrected baseline |
| confidence_nearest | +175.3 | -159.7 | v10 best (same blueprint) |
| refine (B0) | +202.6 | -86.4 | v12 (no training change) |
| refine (poly2) | +215.9 | +32.7 | v12 blueprint decision |
| **pseudo_harmonic (B0)** | **+205.4** | **+79.5** | **v13 WS0 — current best for WSBot** |

pseudo_harmonic on B0 surpasses B0+refine on classic avg (+2.8 bb/100) with a dramatic
WeirdSizingBot improvement (+165.9 bb/100). The tradeoff vs poly2+refine: +10.5 bb/100
better WeirdSizingBot at cost of −10.5 bb/100 classic avg (poly2+refine leads on classic
mainly via DonkBot +735.5 and AggroBot +549.0).

**For production use:** `experiments/best/v9_B0_100M_allbots_positive.json` with
`mapping="pseudo_harmonic"`.

---

## V13 Roadmap Impact

WS0 success confirms that inference-time mapping improvements are still viable. The
remaining abstraction ceiling (1.22 bb/100 exploitability) requires:
- WS1: Safe subgame solving (Refine 3.0) — gift action construction
- WS2: PCFR+ / hyperparameter schedules — training algorithm improvements
- WS3: EMD bucketing / board texture key — abstraction redesign (hardest, highest upside)

See `docs/plans/ACTION_PLAN_v13.md` for full workstream detail.
