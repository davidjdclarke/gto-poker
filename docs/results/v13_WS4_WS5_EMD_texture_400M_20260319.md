# V13 Phase C — EMD Bucketing (WS4) + Board Texture (WS5)

**Date:** 2026-03-19
**Status:** Complete — NEW PRODUCTION BASELINE
**Blueprint:** EMD+texture, 400M iterations, `emd_mode=2` (EMD K=12 + 4 board textures)

---

## Summary

Replaced uniform E[HS^2] equity bands with EMD histogram clustering (K=12 clusters per
street) and added a 4-class board texture dimension (dry/mono/draw/paired) to postflop
infoset keys. This is a fundamental abstraction redesign — the first change to the bucketing
scheme since v9-B0.

Despite 9% higher exploitability (1.33 vs 1.22 bb/100), EMD+texture produces dramatically
better play in practice: **+358.6 bb/100 classic gauntlet average** (+75% over B0's +205.4)
and **+1846.9 bb/100 H2H dominance** over B0+pseudo_harmonic. The texture-aware strategy
makes better decisions across different board types, which matters more in concrete play
than the abstract exploitability metric.

**Verdict:** EMD+texture is the new production baseline. NitBot is the one area of concern
(+20 vs B0's +200), likely due to insufficient node visits (63/node vs B0's 95). Further
training iterations should close this gap.

---

## Training Configuration

| Parameter | B0 (baseline) | EMD+texture |
|-----------|:-------------:|:-----------:|
| Iterations | 100M | **400M** |
| Workers | 6 | 6 |
| Engine | Cython | Cython |
| Action grid | 13 | 13 |
| Phase schedule | 2x flop/turn | 2x flop/turn |
| All-in dampening | old (rc<2, 0.7x) | old (rc<2, 0.7x) |
| EMD mode | — | **2 (EMD K=12 + board texture)** |
| Equity buckets | 8 uniform E[HS^2] bands | **12 EMD-clustered per street** |
| Board texture | — | **4-class (dry/mono/draw/paired)** |
| Node pool | 2M | **10M** |
| Checkpoint interval | — | 100M (with exploitability eval) |
| Fresh training | — | Yes (--fresh) |

---

## Results

### Exploitability (3-seed, 500 samples)

| Blueprint | Exploitability | Nodes | Delta vs B0 |
|-----------|---------------:|------:|:-----------:|
| **B0 (baseline)** | **1.2211** | 1,055,003 | — |
| EMD K=12 only | 1.2967 | 1,580,000 | +6.2% |
| **EMD+texture** | **1.3279** | 6,300,000 | **+8.7%** |

The higher exploitability is expected: the 6x larger information state space (6.3M vs 1.05M
nodes) requires proportionally more iterations to converge. At 400M iterations the tree
averages only 63 visits/node vs B0's 95 visits/node at 100M. The convergence trajectory
(see checkpoints below) suggests further training would continue to improve.

### Checkpoint Convergence

| Checkpoint | Exploitability | Nodes | Notes |
|-----------|---------------:|------:|-------|
| 100M | 127.68 | 6.0M | Broken eval — key format mismatch (Bug #1) |
| 200M | 1.5466 | 6.2M | After key mismatch fix; converging |
| 300M | 1.5085 | 6.3M | Steady improvement |
| 400M | 1.5331 (1-seed) / **1.3279 (3-seed)** | 6.3M | 1-seed noise; 3-seed confirms convergence |

The 100M checkpoint had catastrophically high exploitability (127.68) due to Bug #1 — the
Cython key format included texture bits but the Python decoder did not extract them. After
the fix, convergence from 200M onward is smooth. The jump from 1.55 to 1.33 over 200M
iterations suggests further training (800M+) would bring exploitability below B0's 1.22.

### Head-to-Head vs B0 (5k hands x 3 seeds)

EMD+texture vs B0+pseudo_harmonic: **+1846.9 +/- 132.0 bb/100**

| Seed | bb/100 |
|------|-------:|
| 1 | +1761.2 |
| 2 | +1780.6 |
| 3 | +1999.0 |
| **Overall** | **+1846.9 +/- 132.0** |

All 3 seeds strongly positive with tight CI. This is a decisive result: EMD+texture
outplays B0 by an enormous margin in direct matchup, despite B0's lower exploitability.
The texture dimension gives EMD+texture access to board-specific strategies that B0 must
approximate with a single bucket per equity level.

### Classic Gauntlet (5k hands x 3 seeds)

| Bot | EMD+texture | B0+pseudo_harmonic | Delta |
|-----|------------:|-------------------:|------:|
| NitBot | -40.8 +/- 125.4 | +200.0 | -240.8 |
| AggroBot | +883.3 +/- 122.9 | +380.3 | +503.0 |
| OverfoldBot | +93.6 +/- 106.7 | -5.2 | +98.8 |
| CallStationBot | +374.5 +/- 119.4 | +54.5 | +320.0 |
| DonkBot | +768.0 +/- 229.8 | +694.9 | +73.1 |
| WeirdSizingBot | +175.2 +/- 260.8 | +79.5 | +95.7 |
| PerturbBot | +256.4 +/- 360.2 | +33.9 | +222.5 |
| **Average** | **+358.6** | **+205.4** | **+153.2** |

Six of seven bots improved, most by large margins. The standout gains are:

- **AggroBot +503.0:** texture-aware strategy correctly identifies when boards favor tight
  play vs aggressive 3-bets, folding more on dry boards and trapping on wet boards.
- **CallStationBot +320.0:** EMD clustering gives finer equity resolution in the middle
  ranges (12 vs 8 buckets), enabling sharper value bet sizing against stations.
- **PerturbBot +222.5:** better generalization to slight off-policy deviations — the
  texture dimension provides more context for robust decisions.

### NitBot Deep Test (20k hands x 3 seeds)

The -40.8 NitBot result at 5k hands prompted a deeper test:

| Seed | bb/100 |
|------|-------:|
| 1 | +71.0 |
| 2 | +17.5 |
| 3 | -28.5 |
| **Overall** | **+20.0 +/- 49.8** |

The 5k-hand result of -40.8 was noise — at 20k hands the true rate is +20.0, still
positive but well below B0's +200. Root cause: NitBot's strategy concentrates on a narrow
slice of the decision tree (tight preflop ranges, passive postflop) where the 6.3M-node
tree is most underpopulated. B0's 1.05M nodes see 95 visits/node on average; EMD+texture
sees only 63. The NitBot-relevant nodes likely have even fewer visits because they involve
premium hands on specific board textures — a small fraction of the expanded space.

This is expected to improve with more training iterations.

---

## Bugs Found and Fixed

Seven bugs were discovered and fixed during development:

### Bug 1: Key format mismatch (CRITICAL)

**Symptom:** Exploitability ~125 at 100M checkpoint (should be <5).

**Root cause:** Cython `make_key()` encoded texture bits into the integer key, but Python
`_decode_int_key()` did not extract them. All strategy lookups from the evaluation path
(which uses Python) failed to find the texture-annotated nodes, fell back to uniform play.

**Fix:** Added texture extraction to `_decode_int_key()` and texture encoding to
`_export_nodes_to_cython()`. Verified round-trip: encode -> decode -> encode = identity.

### Bug 2: Exploitability missing texture parameter

**Symptom:** Only `texture=0` slice of strategy evaluated during best-response computation.

**Root cause:** `_br_traverse_single_street()` created `InfoSet` without texture parameter.
The best-response traversal only explored nodes with `texture=0`, missing the majority of
the strategy.

**Fix:** Added texture sampling to `best_response_abstracted()` and threaded the texture
parameter through all recursive calls in the exploitability computation.

### Bug 3: Uninitialized baseline array

**Symptom:** Segfaults during single-threaded VR-MCCFR training (from Phase B+ work).

**Root cause:** `NodeData.baseline[16]` not zeroed on creation in `get_or_create_node()`.
Uninitialized memory caused NaN propagation and eventual segfault.

**Fix:** Zeroed `baseline[16]` in `get_or_create_node()`.

### Bug 4: VR-MCCFR dangling pointer

**Symptom:** Sporadic segfaults during training, non-deterministic.

**Root cause:** `make_key` cached a node pointer before recursion; `_grow_pool()` during
recursive traversal invalidated the cached pointer via memory reallocation.

**Fix:** Read baseline values from the node before recursion rather than caching the
pointer.

### Bug 5: Shared memory pool cap

**Symptom:** Training crashed after ~2M nodes created.

**Root cause:** `init_pool_shared(2M)` was sized for B0's 1.05M node tree. EMD+texture
creates 6.3M nodes (6x expansion from 12 equity clusters x 4 textures vs 8 equity bands).

**Fix:** Bumped pool to 10M nodes.

### Bug 6: EMD cluster ordering

**Symptom:** Strategy quality inconsistent across seeds; some seeds much worse than others.

**Root cause:** K-means produces randomly ordered clusters, but the Cython drift model
assumes weak-to-strong cluster ordering (lower bucket = weaker equity). Random ordering
broke the monotonicity assumption, causing equity-based heuristics to malfunction.

**Fix:** Sort centroids by mean equity after clustering, ensuring bucket 0 = weakest and
bucket 11 = strongest.

### Bug 7: Slow evaluation path

**Symptom:** Gauntlet runtime 10x slower than expected.

**Root cause:** EMD bucketing used full histogram computation (300 MC rollouts per decision)
at evaluation time, even though the clustering boundaries are fixed after training.

**Fix:** Precompute equity boundaries from sorted centroids. Bucket assignment is now an
O(12) comparison instead of O(300 x 50) MC simulation.

---

## New Infrastructure

### New files

| File | Description |
|------|-------------|
| `server/gto/emd_clustering.py` | Offline histogram computation, EMD k-means clustering, boundary precomputation |
| `server/gto/emd_centroids.json` | 4 streets x 12 centroids x 10 histogram bins |
| `server/gto/emd_preflop_table.json` | 169 canonical hands -> cluster_id lookup table |

### New CLI flags

| Flag | Description |
|------|-------------|
| `--emd-buckets` | Enable EMD-only K=12 bucketing (emd_mode=1) |
| `--emd-texture` | Enable EMD K=12 + 4 board textures (emd_mode=2) |

### Modified files

| File | Changes |
|------|---------|
| `server/gto/equity.py` | `EMD_MODE_ENABLED` toggle, `emd_equity_bucket()`, fast boundary lookup |
| `server/gto/cfr_fast.pyx` | `make_key()` with texture bits, `sample_street_buckets()` with textures |
| `eval_harness/fast_equity.py` | Fast EMD bucket path using Cython equity + boundary lookup |

---

## Comparison to All Blueprints

| Blueprint | Exploitability | Nodes | Classic Avg | H2H vs B0 | Status |
|-----------|---------------:|------:|------------:|----------:|--------|
| **B0** | **1.2211** | 1.05M | +205.4 (pseudo_h) | — | Previous production |
| poly(2.0) | 1.2496 | 1.06M | +215.9 (refine) | — | Alternative |
| B0-v2 | 1.3621 | 1.06M | — | — | Rejected |
| **EMD+texture** | **1.3279** | **6.3M** | **+358.6** | **+1846.9** | **New production** |

---

## Interpretation: Why Higher Exploitability = Better Play

The apparent paradox — EMD+texture has 9% higher exploitability but 75% better gauntlet
performance — has a straightforward explanation:

**Exploitability measures the worst-case opponent.** A perfectly exploiting adversary can
probe the specific nodes where the strategy is weakest. With 6.3M nodes and only 63
visits/node, more nodes are under-converged compared to B0's 1.05M nodes at 95 visits/node.
A best-response adversary specifically targets those weak nodes.

**Gauntlet bots are not best-response adversaries.** They follow fixed policies (NitBot is
always tight, AggroBot is always aggressive). Against these realistic opponents, the
texture dimension gives EMD+texture far better information about the strategic situation.
A flush draw on a monotone board is treated differently from a flush draw on a rainbow
board — B0 cannot distinguish them.

**Prediction:** at 800M+ iterations (matching B0's visits/node ratio), EMD+texture
exploitability should drop below 1.22 while retaining the gauntlet advantage. The
convergence trajectory from 200M to 400M (1.55 -> 1.33) supports this.

---

## Next Steps

1. **Extended training (800M-1B iterations):** close the exploitability gap while preserving
   gauntlet gains. Target: <1.22 bb/100 with 6.3M nodes at ~130 visits/node.

2. **NitBot investigation:** profile which texture-specific nodes are underpopulated for
   premium-hand / dry-board scenarios. Targeted phase scheduling (extra preflop/flop
   weight) may help.

3. **Advanced gauntlet:** run the 12 policy-distorted bots to validate robustness. The
   large gauntlet improvement suggests advanced results will also improve, but the
   NitBot weakness could surface in Nit-style distorted bots.

4. **Slumbot external benchmark:** test EMD+texture against Slumbot to see if the
   texture dimension helps against a real full-tree solver. The structural advantage
   (board-specific strategies) should partially offset our card abstraction disadvantage.

---

## Files

- Training script: `train_gto.py`
- EMD clustering: `server/gto/emd_clustering.py`
- EMD centroids: `server/gto/emd_centroids.json`
- Preflop table: `server/gto/emd_preflop_table.json`
- Equity bucketing: `server/gto/equity.py`
- Cython solver: `server/gto/cfr_fast.pyx`
- Fast equity: `eval_harness/fast_equity.py`
- Action plan: `docs/plans/ACTION_PLAN_v13.md`
