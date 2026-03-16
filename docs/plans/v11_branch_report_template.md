# V11 Branch Report Template

Every v11 experiment branch must produce a report following this template before merge or discard decisions. Copy this file to `docs/results/v11_{branch_name}_{date}.md` and fill in all sections.

---

## 1. Branch Metadata

| Field | Value |
|-------|-------|
| **Branch name** | `v11-{workstream}-{id}` (e.g., `v11-C2-poly-weight`) |
| **Hypothesis** | State the specific claim being tested (e.g., "Polynomial weight schedule (t^2) improves exploitability by >= 10% over linear CFR+ at 100M iterations") |
| **Workstream** | A (Convergence) / B (Bridge) / C (Solver Dynamics) / D (Abstraction) / E (Evaluation) |
| **Baseline** | Strategy file compared against (e.g., `experiments/best/v9_B0_100M_allbots_positive.json`) |
| **Training config** | Full `train_gto.py` command with all flags |
| **Date started** | YYYY-MM-DD |
| **Date completed** | YYYY-MM-DD |
| **Total compute** | Wall-clock time, iteration count, number of workers |
| **Strategy file** | Path to saved strategy (e.g., `experiments/v11_C2_poly/strategy_100M.json`) |
| **Nodes** | Total node count in trained strategy |

### Training Command

```bash
# Paste the exact command used
venv/bin/python train_gto.py \
    --iterations XXXXXXXX \
    --workers 6 \
    --averaging-delay XXXXXX \
    --phase-schedule 2x \
    --allin-dampen old \
    --regret-discount 1.0 \
    --weight-schedule linear \
    --weight-param 1.0 \
    --checkpoint-interval 10000000 \
    --checkpoint-eval
```

### Key Configuration Differences from Baseline

| Parameter | Baseline (B0) | This Branch | Rationale |
|-----------|---------------|-------------|-----------|
| (list each changed parameter) | | | |

---

## 2. Checkpoint Metrics

Record exploitability at each checkpoint to track convergence trajectory. Use `--checkpoint-eval` during training, or compute post-hoc with `exploitability_multiseed()`.

### 2.1 Exploitability by Checkpoint

| Checkpoint (iters) | Nodes | Coverage (%) | Exploitability (3-seed, 500 samples) | Delta vs B0 at same iters |
|--------------------|-------|-------------|--------------------------------------|--------------------------|
| 10M | | | | |
| 20M | | | | |
| 30M | | | | |
| 50M | | | | |
| 100M | | | | |

**Convergence rate:** Does exploitability decrease monotonically? At what rate compared to B0?

**Coverage note:** Report `well_visited / total_nodes` percentage. B0 baseline is 99.7% at 100M with 1,055,003 nodes. Any branch with < 90% coverage at final checkpoint has a structural problem.

### 2.2 Checkpoint Quick Gauntlet (Optional)

If `--checkpoint-gauntlet` was used, record 500-hand gauntlet averages at key checkpoints. These are directional only (high variance at 500 hands -- see v10 Phase 0 report).

| Checkpoint | Quick Gauntlet Avg (bb/100) | Worst Bot |
|------------|---------------------------|-----------|
| 50M | | |
| 100M | | |

---

## 3. Final Gauntlet

**Protocol:** 5,000 hands/bot, 3 seeds (42, 123, 456), `confidence_nearest` mapping, no OpponentProfile, bb=20, 10,000 starting chips.

**Command:**
```bash
venv/bin/python run_eval_harness.py --gauntlet --hands 5000 --seeds 42,123,456
```

### 3.1 Per-Bot Results

| Bot | Mean (bb/100) | Std | 95% CI Low | 95% CI High | B0 Mean | Delta | Significant? |
|-----|-------------|-----|-----------|-------------|---------|-------|-------------|
| NitBot | | | | | +200.3 | | |
| AggroBot | | | | | +444.0 | | |
| OverfoldBot | | | | | -3.9 | | |
| CallStationBot | | | | | +73.0 | | |
| DonkBot | | | | | +717.6 | | |
| WeirdSizingBot | | | | | -202.8 | | |
| PerturbBot | | | | | +347.2 | | |
| **Average** | | | | | **+225.1** | | |

**Significance criterion:** A delta is "significant" if the 95% CIs of the two branches do not overlap, or equivalently if the delta exceeds 2x the pooled standard error.

### 3.2 Summary Statistics

| Metric | This Branch | B0 Baseline | Delta |
|--------|-------------|-------------|-------|
| Average bb/100 | | +225.1 | |
| Worst bot | | WeirdSizingBot (-202.8) | |
| Best bot | | DonkBot (+717.6) | |
| Bots with positive mean | /7 | 5/7 | |
| Strategy hit rate | | 92%+ | |

---

## 4. Exploitability Report

### 4.1 Standard Evaluation

**Protocol:** 3 seeds (42, 123, 456), 500 MC samples per seed.

```bash
# Code to reproduce:
from server.gto.cfr import CFRTrainer
from server.gto.exploitability import exploitability_multiseed
trainer = CFRTrainer()
trainer.load("path/to/strategy.json")
result = exploitability_multiseed(trainer, samples=500, seeds=[42, 123, 456])
```

| Metric | Value |
|--------|-------|
| Mean exploitability (bb/100) | |
| Standard deviation | |
| 95% CI | |
| B0 baseline | 1.2211 |
| Delta | |

### 4.2 High-Fidelity Evaluation

**Protocol:** 5 seeds (42, 123, 456, 789, 1000), 1000 MC samples per seed.

| Metric | Value |
|--------|-------|
| Mean exploitability (bb/100) | |
| 95% CI | |

### 4.3 Per-Phase Exploitability

| Phase | Mean | B0 Baseline | Delta |
|-------|------|-------------|-------|
| Preflop | | 1.0204 | |
| Flop | | 1.3865 | |
| Turn | | 1.3840 | |
| River | | 1.1908 | |

**Interpretation:** Which phases improved? Which regressed? Does the per-phase pattern make sense given the hypothesis?

---

## 5. EV Decomposition

Run with `detailed_tracking=True` against at least CallStationBot and WeirdSizingBot (the two most diagnostic matchups).

**Protocol:** 5,000 hands, seed=42, `detailed_tracking=True`, `confidence_nearest` mapping.

```python
from eval_harness.match_engine import GTOAgent, HeadsUpMatch
from eval_harness.ev_decomposition import decompose_by_street, decompose_by_action_family
from eval_harness.adversaries import CallStationBot

match = HeadsUpMatch(gto, CallStationBot(), big_blind=20, seed=42, detailed_tracking=True)
result = match.play(5000)
street_ev = decompose_by_street(result.hands, big_blind=20)
action_ev = decompose_by_action_family(result.hands, big_blind=20)
```

### 5.1 Street EV (bb/100) vs CallStationBot

| Phase | This Branch | B0 Baseline | Delta |
|-------|-------------|-------------|-------|
| Preflop | | | |
| Flop | | | |
| Turn | | | |
| River | | | |

### 5.2 Street EV (bb/100) vs WeirdSizingBot

| Phase | This Branch | B0 Baseline | Delta |
|-------|-------------|-------------|-------|
| Preflop | | | |
| Flop | | | |
| Turn | | | |
| River | | | |

### 5.3 Action Family EV (vs CallStationBot)

| Family | Total EV (bb) | Count | Avg EV (bb) | B0 Avg EV | Delta |
|--------|-------------|-------|------------|----------|-------|
| Fold | | | | | |
| Check | | | | | |
| Call | | | | | |
| Bet/Raise | | | | | |

### 5.4 CallStation Dashboard (if applicable)

| Leak Category | Count | Avg EV (bb) | B0 Count | B0 Avg EV | Interpretation |
|---------------|-------|------------|---------|----------|---------------|
| Slowplay (EQ5+ checked) | | | | | |
| Bluff (EQ0-2 bet into station) | | | | | |
| Thin value (EQ3-4 checked) | | | | | |

---

## 6. Bridge Pain Summary

Run `GTOAgent` with bridge logging enabled against WeirdSizingBot (worst bridge matchup).

```python
from eval_harness.bridge_pain import analyze_bridge_pain, format_pain_map
analysis = analyze_bridge_pain(gto.bridge_log, result.hands, big_blind=20)
print(format_pain_map(analysis))
```

### 6.1 Pain Map Statistics

| Metric | This Branch | B0 Baseline |
|--------|-------------|-------------|
| Total translation events | | |
| Mean translation distance | | |
| Max translation distance | | |

### 6.2 Worst Gaps (Top 5)

| Concrete Ratio | Mapped To | Nominal Ratio | Distance | Phase |
|---------------|-----------|--------------|----------|-------|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

### 6.3 Pain by Phase

| Phase | Count | Avg Distance | B0 Avg Distance |
|-------|-------|-------------|----------------|
| Preflop | | | |
| Flop | | | |
| Turn | | | |
| River | | | |

**Interpretation:** Did the bridge quality change? Worse bridge = more mismatch = more reliance on confidence_nearest blending. If this branch changes the action grid, bridge pain patterns will shift.

---

## 7. Strategy Audit

Run the full strategy audit suite.

```python
from server.gto.exploitability import strategy_audit, river_bluff_audit, allin_audit
audit = strategy_audit(trainer)
bluff = river_bluff_audit(trainer)
allin = allin_audit(trainer)
```

### 7.1 Anomaly Counts

| Anomaly Type | Count | B0 Count | Delta | Severity |
|-------------|-------|---------|-------|----------|
| Premium limp | | | | High if > 0 |
| All-in overuse | | | | Medium |
| Flat strategies (near-uniform) | | | | Medium |
| Frequency anomalies (strong fold) | | | | High |

### 7.2 River Bluff Analysis

| Metric | Value | B0 Baseline | Healthy Range |
|--------|-------|-------------|--------------|
| Overall bluff ratio | | | < 0.40 |
| Diagnosis | | | HEALTHY |
| IP bluff ratio | | | |
| OOP bluff ratio | | | |

### 7.3 Worst Bluffers (Top 3)

| Infoset Key | EQ Bucket | Hand Type | Bet Probability | Position |
|-------------|----------|-----------|----------------|----------|
| | | | | |
| | | | | |
| | | | | |

### 7.4 All-in Audit Summary

| Metric | Value | B0 Baseline |
|--------|-------|-------------|
| All-in nodes examined | | |
| High-regret all-in nodes | | |
| Weak-hand jamming nodes | | |
| Zero-pot issues | | |

---

## 8. Recommendation

### 8.1 Verdict: KEEP / DISCARD / MERGE

Choose one:

- **KEEP as new baseline:** This branch improves on B0 across all key metrics (exploitability, gauntlet average, worst bot). Replace B0 as the reference strategy.
- **MERGE specific changes:** This branch has useful infrastructure/code but the strategy is not strictly better. Merge code changes, keep B0 strategy.
- **DISCARD:** This branch does not improve on B0 and has no useful code changes. Archive results for the record.

### 8.2 Evidence Summary

| Criterion | Met? | Evidence |
|-----------|------|---------|
| Exploitability improved (or not regressed) | | |
| Gauntlet average improved | | |
| No bot regressed > 100 bb/100 | | |
| WeirdSizingBot not worse | | |
| Strategy audit clean (no new anomalies) | | |
| Convergence stable (no cliff or divergence) | | |

### 8.3 Justification

(Write 2-3 sentences explaining why this branch should be kept, merged, or discarded. Reference specific numbers from the sections above.)

### 8.4 Follow-up Actions

(If KEEP or MERGE: what should the next branch build on from this result? If DISCARD: what did we learn that informs future work?)

---

## Appendix A: Raw Data Files

| Artifact | Path |
|----------|------|
| Strategy file | `experiments/v11_{branch}/strategy_{iters}.json` |
| Gauntlet results JSON | `eval_results/v11_{branch}_gauntlet.json` |
| Exploitability data | `eval_results/v11_{branch}_exploitability.json` |
| EV decomposition data | `eval_results/v11_{branch}_ev_decomp.json` |
| Bridge pain data | `eval_results/v11_{branch}_bridge_pain.json` |

## Appendix B: Reproduction Commands

```bash
# Train
venv/bin/python train_gto.py [paste full command]

# Evaluate exploitability
venv/bin/python -c "
from server.gto.cfr import CFRTrainer
from server.gto.exploitability import exploitability_multiseed
trainer = CFRTrainer()
trainer.load('[strategy_path]')
print(exploitability_multiseed(trainer, samples=500, seeds=[42, 123, 456]))
"

# Run gauntlet
venv/bin/python run_eval_harness.py --gauntlet --hands 5000 --seeds 42,123,456 \
    --strategy [strategy_path] --save eval_results/v11_{branch}_gauntlet.json

# EV decomposition (manual script)
venv/bin/python -c "
from server.gto.engine import get_trainer
from eval_harness.match_engine import GTOAgent, HeadsUpMatch
from eval_harness.adversaries import CallStationBot, WeirdSizingBot
from eval_harness.ev_decomposition import decompose_by_street, callstation_dashboard

trainer = get_trainer()
gto = GTOAgent(trainer, mapping='confidence_nearest')

for BotClass in [CallStationBot, WeirdSizingBot]:
    bot = BotClass()
    match = HeadsUpMatch(gto, bot, big_blind=20, seed=42, detailed_tracking=True)
    result = match.play(5000)
    print(f'\n=== vs {bot.name} ===')
    print('Street EV:', decompose_by_street(result.hands, big_blind=20))
    if isinstance(bot, CallStationBot):
        print('Dashboard:', callstation_dashboard(result.hands, big_blind=20))
"
```
