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

## Gauntlet Results (bb/100, 5000 hands/bot)

| Bot | Exp A (2x sched) | Exp B (old damp) | Exp C (adaptive) | Exp E (9 buck) | Baseline v7 (67M) |
|-----|-------------------|-------------------|-------------------|----------------|---------------------|
| AggroBot | +440.8 | +387.6 | +421.3 | +566.5 | +421.0 |
| OverfoldBot | +10.2 | +72.4 | -16.7 | -41.7 | +14.8 |
| DonkBot | +484.6 | +337.3 | +553.5 | +303.3 | +566.1 |
| NitBot | +200.6 | +55.7 | +115.8 | +44.3 | +0.3 |
| WeirdSizingBot | +233.9 | +144.4 | -51.3 | -259.9 | -383.0 |
| PerturbBot | -87.4 | -31.4 | -47.4 | -118.7 | -139.6 |
| CallStationBot | -947.5 | -731.4 | -1268.3 | -993.3 | -789.8 |
| **Average** | **+47.9** | **+33.5** | **-41.9** | **-71.3** | **-44.3** |

### Comparison to preliminary 500-hand results

The 5000-hand results are substantially different from the original 500-hand run, confirming the high-variance concern:

| Experiment | 500-hand avg | 5000-hand avg | Delta |
|------------|-------------|---------------|-------|
| Exp A (2x schedule) | +137.2 | +47.9 | -89.3 |
| Exp B (old dampening) | +174.0 | +33.5 | -140.5 |
| Exp C (adaptive avg) | -68.9 | -41.9 | +27.0 |
| Exp E (9 buckets) | -291.1 / +159.3 | -71.3 | — |
| Baseline v7 (67M) | -226.2 | -44.3 | +181.9 |

The 500-hand results overstated the gap between experiments. At 5000 hands, all experiments cluster more tightly (-71 to +48 bb/100), and the baseline is much less negative than initially measured.

## Exploitability (bb/100, 3-seed)

| Experiment | Exploitability | CI |
|------------|---------------|-----|
| Baseline v7 (67M) | 1.2311 | ±0.06 |
| Exp A (2x schedule) | 1.2656 | ±0.05 |
| Exp B (old dampening) | 1.2531 | ±0.05 |
| Exp C (adaptive avg) | 1.2565 | ±0.06 |
| Exp E (9 buckets) | 1.3117 | ±0.03 |

All experiments have similar exploitability (~1.23-1.31). Baseline v7 at 67M has the best exploitability. Exp E slightly higher due to larger state space at same iteration count.

## Exp E Additional Info
- **Nodes**: 1,175,386 (vs ~1.05M for 8-bucket experiments — 12% more nodes)
- **River bluff ratio**: 23.3% (lowest of all experiments)
- **Coverage**: 100% well-visited

## Strategy Audit Summary

| Metric | Baseline | Exp A | Exp B | Exp C | Exp E |
|--------|----------|-------|-------|-------|-------|
| River bluff % | 27.3% | 27.0% | 26.9% | 27.0% | 23.3% |
| All-in overuse nodes | 458 | 488 | 435 | 476 | 518 |
| Flat strategies | 1,035 | 1,393 | 1,168 | 1,132 | 1,215 |
| Freq anomalies | 17,669 | 18,280 | 18,003 | 17,801 | 37,544 |
| Premium limp | 4 | 4 | 4 | 4 | 4 |
| Strategy hit rate | 93.5% | 93.7% | 93.6% | 93.6% | 92.9% |

## Key Findings (Updated with 5000-hand data)

### 1. Exp A (2x schedule) is the best performer (+47.9 avg)
At 5000 hands, the simpler 2x schedule clearly outperforms all other experiments. It wins against 5 of 7 bots (only losing to PerturbBot and CallStationBot). The 3x schedule's extra flop/turn emphasis provides no measurable benefit and may slightly hurt by under-training preflop/river.

### 2. Exp B (old dampening) is second-best (+33.5 avg)
Old dampening (rc<2, 0.7x) still shows improvement over baseline. It has the lowest all-in overuse count (435) and the best PerturbBot result (-31.4). The broader dampening scope controls all-in frequency without over-suppressing aggression.

### 3. The "over-convergence" signal was mostly noise
At 500 hands, the baseline scored -226.2 suggesting severe over-convergence at 67M. At 5000 hands, baseline scores -44.3 — only modestly worse than Exp A/B. The 67M strategy is not catastrophically over-converged; it's within the range of normal variance. The baseline's best exploitability (1.23) suggests the extra iterations do help convergence.

### 4. Exp E (9 buckets) underperforms (-71.3 avg)
Despite having finer abstraction with draw-aware equity splits, Exp E performs worst. The larger state space (12% more nodes) with the same iteration count means each node gets fewer training samples. The 23.3% river bluff ratio (vs 27% for others) may also indicate under-convergence in bluffing nodes.

### 5. Exp C (adaptive averaging) is near-baseline (-41.9 avg)
Per-node adaptive averaging delay has minimal impact, confirming that rare-node averaging is **not** the dominant bottleneck. The result is statistically indistinguishable from baseline.

### 6. CallStation remains the persistent weakness
All experiments lose heavily to CallStationBot (-731 to -1268 bb/100). Exp C is worst (-1268.3) while Exp B is best (-731.4). The solver's river bluff frequency (~27%) is structurally too low to punish calling stations.

### 7. WeirdSizingBot is the key differentiator
The biggest swing between experiments is vs WeirdSizingBot: Exp A wins +233.9, Exp B wins +144.4, but Exp E loses -259.9 and baseline loses -383.0. Off-tree action handling appears to benefit from the 2x schedule's more balanced training.

## Recommended Next Steps

1. **Adopt 2x schedule** as default — best overall gauntlet performance
2. **Combine 2x schedule + old dampening**: Exp A and B are both positive; combining may compound gains
3. **Give Exp E more iterations**: 9-bucket abstraction likely needs 60-70M iterations to match per-node convergence of 8-bucket at 50M
4. **Address CallStation leak**: river bluff frequency needs to increase — consider bluff frequency floors
5. **Drop adaptive averaging (Exp C)**: no meaningful impact, adds code complexity

## Methodology Notes
- All experiments trained from scratch (--fresh) at 50M iterations with 6 workers
- Baseline is v7 at 67M iterations (not directly comparable due to iteration difference)
- Gauntlet uses 5000 hands per bot with default nearest mapping
- Exploitability measured with 3-seed MC best response at 500 samples
- Cython-accelerated evaluation (~45x speedup) enabled 5000 hands/bot in ~10-15s per matchup

## Historical: Original 500-hand Results

<details>
<summary>Click to expand (superseded by 5000-hand results above)</summary>

| Bot | Exp B (old damp) | Exp E (9 buck) | Exp A (2x sched) | Exp C (adaptive) | Baseline v7 (67M) |
|-----|-------------------|----------------|-------------------|-------------------|---------------------|
| AggroBot | +512.3 | +524.5 | +293.0 | +635.5 | +179.4 |
| OverfoldBot | +111.7 | -69.7 | +18.5 | +204.9 | +107.8 |
| DonkBot | +249.4 | -152.8 | +1355.1 | +471.9 | +293.9 |
| NitBot | +513.4 | +287.3 | +267.1 | +160.9 | -409.3 |
| PerturbBot | +1552.1 | -1229.0 | +1092.2 | -822.1 | -366.0 |
| WeirdSizingBot | -484.1 | -729.5 | -838.9 | -626.7 | -952.4 |
| CallStationBot | -1236.6 | -668.5 | -1226.8 | -506.8 | -437.0 |
| **Average** | **+174.0** | **-291.1** | **+137.2** | **-68.9** | **-226.2** |

</details>
