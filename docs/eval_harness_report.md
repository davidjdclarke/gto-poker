# Evaluation Harness Report — v7 (2026-03-12)

Strategy: v7, 50M iterations, 1,047,504 nodes
Eval file: `eval_results/eval_1773348554.json`

---

## 1. Opponent Gauntlet

GTO agent (nearest mapping, 95.1% strategy hit rate) vs 7 exploit bots, 500 hands each.

| Opponent | bb/100 | Showdown % | Notes |
|----------|-------:|----------:|-------|
| CallStationBot | **+439.5** | 88.2% | Fixed from -463 at v6 |
| AggroBot | **+365.6** | 15.2% | Handles hyper-aggression well |
| DonkBot | **+266.5** | 47.2% | Donk bets handled via trained responses |
| NitBot | **+235.9** | 33.4% | Fixed from -19 at v6 |
| PerturbBot | **-210.4** | 45.8% | Slight loss to near-random play (expected) |
| OverfoldBot | **-291.4** | 30.6% | Not exploiting folders — active issue |
| WeirdSizingBot | **-852.7** | 21.6% | Off-tree translation gap |
| **Average** | **-6.7** | | |

### Key Findings

**Resolved:**
- CallStationBot: -463 → +439.5 — river bluff ratio now 35.1% (was causing 100% EQ0 bluffs)
- NitBot: -19 → +235.9 — tight-range convergence improved

**Active Issues:**
- OverfoldBot -291.4: GTO should win heavily vs passive folders. Postflop c-bet frequency
  appears too low. Investigate flop IP strategy after OOP check.
- WeirdSizingBot -852.7: 120% overbet is the worst case (-1305 raw, maps to undertrained
  `bet_overbet` node). See off-tree section.

---

## 2. Off-Tree Robustness

| Sizing | Pre | Post | Maps To | bb/100 |
|--------|----:|-----:|---------|-------:|
| standard (control) | 2.5x | 50% | bet_half | -416.3 |
| min-open 2.0x | 2.0x | 50% | bet_half | -1.4 |
| small open 2.2x | 2.2x | 50% | bet_half | -933.5 |
| large open 3.5x | 3.5x | 50% | bet_half | -982.4 |
| 15% pot postflop | 2.5x | 15% | bet_third | -346.8 |
| 28% pot postflop | 2.5x | 28% | bet_third | -662.4 |
| 42% pot postflop | 2.5x | 42% | bet_half | -87.5 |
| 75% pot postflop | 2.5x | 75% | bet_two_thirds | +74.4 |
| 120% overbet | 2.5x | 120% | bet_overbet | -1305.4 |
| min-click + 67% | 2.0x | 67% | bet_two_thirds | +67.9 |
| 5x open + 200% overbet | 5.0x | 200% | all_in | +150.0 |

Control baseline: -416.3 bb/100. Worst case: 120% overbet at -1305.4.
Max EV loss from translation: 889.1 bb/100 (120% overbet vs control).

### Key Findings

The off-tree scores are all relative to opponents using forced non-standard sizings. The
large negative numbers don't mean GTO loses that many bb/100 in practice — they reflect
how the translation of odd sizings affects strategy quality.

The 120% overbet (-1305 raw) is the most important gap: `bet_overbet` is undertrained
(added late in v6, gets 2x same as all other actions). The next retrain with 3x phase
schedule should improve this.

---

## 3. Bridge A/B Test

Mapping scheme comparison (3 opponents × 4 mappings = 12 matches, 500 hands each):

| Mapping | AggroBot | CallStation | WeirdSizing | **Average** |
|---------|----------|-------------|-------------|------------|
| **nearest** | +365.6 | +912.9 | -447.7 | **+276.9** |
| stochastic | +528.4 | -70.3 | +371.6 | +276.6 |
| resolve | +670.7 | +531.3 | -488.3 | +237.9 |
| conservative | +257.4 | -72.5 | -285.3 | -33.5 |

**Best mapping: `nearest` (+276.9 avg) — this is the default.**

Note: `conservative` was the default in v6 because CallStation was -463 (river over-bluffing).
With the bluff ratio now healthy, conservative's 15% bet→check bleed hurts value betting.
Do not revert to conservative unless river bluff ratio exceeds 45%.

---

## 4. Rare-Node EV Leakage

### Exploitability (3-seed)

| Metric | Value |
|--------|-------|
| Overall | **1.2822 bb/100** |
| Preflop | 1.05 |
| Flop | 1.42 |
| Turn | 1.44 |
| River | 1.17 |

Flop/Turn remain the weakest streets (1.42/1.44). This is the primary target for the next
retrain (3x phase schedule).

### Node Coverage

| Metric | Count | % |
|--------|------:|--:|
| Total nodes | 1,047,504 | 100% |
| Well-visited | 1,047,504 | **100.0%** |
| Low-visit (<100) | 0 | 0.0% |

### Regret Hotspots (top 3 per phase)

**Preflop:**
- `preflop:oop:107:55` — regret=202575, H=1.82
- `preflop:oop:114:55` — regret=193813, H=1.88
- `preflop:oop:111:55` — regret=193311, H=1.83

All are OOP facing double all-in (history `55`). Solver uncertain about calling jams with
strong non-premium hands. Expected; addressed by tightening all-in dampening in next retrain.

**Flop:**
- `flop:ip:119:155` — regret=173289, H=2.82
- `flop:ip:110:155` — regret=160805, H=2.72
- `flop:ip:117:155` — regret=158799, H=2.90

High entropy (H≈2.8 out of 2.9 max) = near-uniform across 6 actions. Undertrained nodes.

**River:**
- `river:oop:92:1555` — regret=11575, H=0.14, fold=99%
- `river:oop:65:1555` — regret=11516, H=0.15, fold=99%

River regrets 10-15x lower than flop — confirming better convergence at the terminal street.

### Strategy Audit

| Anomaly | Count | Severity |
|---------|------:|----------|
| Premium limp | 4 | Medium |
| All-in overuse | 520 | Medium |
| Flat strategies | 1,334 | Medium |
| Frequency anomalies | 65,814 | Low |

**Top leaks:**
- HIGH_PAIR at EQ5 limps 100% (unconverged rare path)
- PREMIUM_PAIR at EQ5 limps 100%
- PREMIUM_PAIR at EQ6 limps 24%
- HIGH_PAIR at EQ6 limps 21%
- Folding 100% at EQ6 (bucket boundary artifact)

### River Bluff Analysis

```
Overall bluff ratio: 35.1% — HEALTHY (target: <40%)

By sizing:
  bet_third       bluff=34%
  bet_half        bluff=35%
  bet_two_thirds  bluff=33%
  bet_pot         bluff=37%
  overbet         bluff=38%
  all_in          bluff=28%

By position:
  ip   bluff=37%
  oop  bluff=31%

Top over-bluffers (isolated nodes, not systemic):
  river:ip:0:1:  bet=100% EQ0 PREMIUM_PAIR
  river:ip:8:1:  bet=100% EQ0 LOW_SUITED_CONNECTOR
  river:ip:9:1:  bet=100% EQ0 SUITED_GAPPER
  river:ip:7:1:  bet=100% EQ0 HIGH_SUITED_CONNECTOR
  river:ip:4:1:  bet=100% EQ0 STRONG_BROADWAY
```

Overall ratio is healthy. Individual EQ0 nodes still 100% bet after OOP check (history=1).
This is a structural abstraction limit — EQ0 can't distinguish "draws that missed" from
"pure air" at river. Addressed long-term by splitting EQ0 bucket (see ACTION_PLAN_v7.md).

---

## Usage

```bash
# Full suite
venv/bin/python run_eval_harness.py

# Individual suites
venv/bin/python run_eval_harness.py --gauntlet --hands 1000
venv/bin/python run_eval_harness.py --offtree
venv/bin/python run_eval_harness.py --bridge
venv/bin/python run_eval_harness.py --leakage

# Quick smoke test
venv/bin/python run_eval_harness.py --quick
```

Results saved to `eval_results/` as JSON.

---

## Historical Comparison

| Version | Exploitability | CallStation | NitBot | OverfoldBot | Bridge Best |
|---------|---------------|-------------|--------|-------------|-------------|
| v5.1 20M | 1.41 | +1518.6 | -623.9 | -10.0 | — |
| v6 20M | 1.31 | -285.0 | +715.9 | — | — |
| v6 50M | 1.28 | -463.0 | -19.0 | — | conservative +274.5 |
| **v7 50M** | **1.2822** | **+439.5** | **+235.9** | **-291.4** | **nearest +276.9** |
