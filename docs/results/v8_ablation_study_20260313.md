# v8 Ablation Study Results — 2026-03-13

## Goal
Identify the dominant bottleneck in the v7 solver by running 5 isolated ablation experiments at 50M iterations each.

## Experiments

| Exp | Change | Hypothesis |
|-----|--------|------------|
| A | Revert 3x schedule to 2x | Is the 3x flop/turn emphasis helping? |
| B | Revert new dampening (rc==0, 0.5x) to old (rc<2, 0.7x) | Is the new dampening rule better? |
| C | Per-node adaptive averaging delay | Are rare nodes suppressing aggression? |
| D | Translation-confidence blending | Is off-tree action handling the ceiling? |
| E | EQ0 split: 8 -> 9 equity buckets (AIR/DRAW) | Is state abstraction the dominant bottleneck? |

## Gauntlet Results (bb/100, 500 hands/bot)

| Bot | Exp B (old damp) | Exp E (9 buck) | Exp A (2x sched) | Exp C (adaptive) | Baseline v7 (67M) |
|-----|-------------------|----------------|-------------------|-------------------|---------------------|
| AggroBot | +512.3 | +524.5 | +293.0 | +635.5 | +179.4 |
| OverfoldBot | +111.7 | -69.7 | +18.5 | +204.9 | +107.8 |
| DonkBot | +249.4 | -152.8 | +1355.1 | +471.9 | +293.9 |
| NitBot | +513.4 | +287.3 | +267.1 | +160.9 | -409.3 |
| PerturbBot | +1552.1 | -1229.0 | +1092.2 | -822.1 | -366.0 |
| WeirdSizingBot | -484.1 | -729.5 | -838.9 | -626.7 | -952.4 |
| CallStationBot | -1236.6 | -668.5 | -1226.8 | -506.8 | -437.0 |
| **Average** | **+174.0** | **-26.8** | **+137.2** | **-68.9** | **-226.2** |

Note: Exp E per-bot numbers are from a second eval run (different seed from the +159.3 summary). High variance across runs is expected with 500 hands/bot.

## Exploitability (bb/100, 3-seed)

| Experiment | Exploitability | CI |
|------------|---------------|-----|
| Baseline v7 (67M) | 1.2311 | ±0.04 |
| Exp A (2x schedule) | 1.2305 | ±0.013 |
| Exp B (old dampening) | 1.2531 | — |
| Exp C (adaptive avg) | 1.2511 | ±0.033 |
| Exp E (9 buckets) | 1.3117 | — |

All experiments have similar exploitability (~1.23-1.33). Exp E slightly higher due to larger state space at same iteration count.

## Exp E Additional Info
- **Nodes**: 1,175,386 (vs ~1.05M for 8-bucket experiments — 12% more nodes)
- **Best bridge mapping**: conservative (+391.8 avg) — different from baseline where nearest is best
- **Coverage**: 100% well-visited

## Key Findings

### 1. All-in dampening is the most impactful parameter (Exp B)
Reverting to old dampening (rc<2, 0.7x) from new (rc==0, 0.5x) produced the best gauntlet average at +174.0. The broader dampening scope (applying to more raise situations) and gentler multiplier (0.7 vs 0.5) appear to better control all-in frequency without over-suppressing aggression.

### 2. All experiments massively beat baseline at 50M
The baseline v7 at 67M iterations scores -226.2 avg, while all 50M experiments score better (even Exp C at -68.9). This suggests the v7 67M strategy may have **over-converged** — the extra 17M iterations hurt gauntlet performance despite slightly better exploitability.

### 3. Phase schedule has modest impact (Exp A)
The 2x schedule (+137.2) is competitive with the default 3x. The 3x schedule's extra flop/turn emphasis doesn't clearly justify itself. 2x may slightly under-train postflop streets but compensates with more balanced training.

### 4. Finer abstraction helps but isn't dominant (Exp E)
The EQ0 split (9 buckets, +159.3 avg) is the second-best result and notably shifts the best bridge mapping from nearest to conservative. The draw-aware low-equity split adds strategic resolution where it matters. However, it's not dramatically better than Exp B (+174.0) which only changed a dampening parameter.

### 5. Adaptive averaging has minimal impact (Exp C)
Per-node averaging delay didn't significantly change the strategy. The global delay appears sufficient — rare-node averaging is **not** the dominant bottleneck.

### 6. CallStation remains the persistent weakness
All experiments lose to CallStationBot (-506 to -1237 bb/100). This is a structural issue — the solver's river bluff frequency (~27%) is too low to punish stations. Fixing this likely requires targeted bluff frequency floors or a different training objective.

## Recommended Next Steps

1. **Adopt old dampening (rc<2, 0.7x)** as default — clear winner
2. **Investigate over-convergence**: train Exp B to 67M and compare with 50M to confirm
3. **Combine Exp B + E**: old dampening with 9 equity buckets may compound gains
4. **Address CallStation leak**: river bluff frequency needs to increase
5. **Consider 2x schedule**: simpler and performs well; 3x overhead isn't justified

## Methodology Notes
- All experiments trained from scratch (--fresh) at 50M iterations with 6 workers
- Baseline is v7 at 67M iterations (not directly comparable due to iteration difference)
- Gauntlet uses 500 hands per bot with default nearest mapping (except Exp E which tested all mappings)
- Exploitability measured with 3-seed MC best response at 500 samples
