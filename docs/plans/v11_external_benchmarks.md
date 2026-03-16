# V11 Workstream E2: External Benchmark Ladder Design

**Date:** March 15, 2026
**Status:** Design proposal
**Scope:** Benchmarking against external agents to validate solver quality beyond internal gauntlet

---

## 1. Problem Statement

### 1.1 Internal Gauntlet Limitations

The current evaluation suite (`eval_harness/adversaries.py`) contains 7 bots: NitBot, AggroBot, OverfoldBot, CallStationBot, DonkBot, WeirdSizingBot, and PerturbBot. While useful for detecting specific leaks, these bots have fundamental limitations:

1. **No real GTO opponents.** All 7 bots are heuristic-based. Beating a CallStation that never folds does not prove the strategy is close to Nash equilibrium. The exploitability metric (1.22 bb/100) approximates this, but only within our own abstraction.

2. **Circular optimization risk.** Optimizing against known heuristic bots risks overfitting to their specific weaknesses. A strategy that scores +225 bb/100 against 7 known bots might score -50 against a different bot that exploits a blind spot none of the 7 test.

3. **No calibration against known-good solvers.** There is no way to answer "how close are we to Libratus/Pluribus quality?" without an external reference point. Exploitability alone is necessary but not sufficient.

4. **Limited off-tree diversity.** WeirdSizingBot uses 6 fixed off-tree sizes. Real opponents (and professional solvers) produce a continuous spectrum of bet sizes. PerturbBot adds randomness but still uses our own strategy as a base.

5. **No difficulty gradient.** All bots are roughly "weak amateur" level. There is no bot that plays approximately GTO, which would test whether our strategy can achieve near-zero win rate against a strong opponent (the true GTO benchmark).

### 1.2 What We Need

A tiered benchmark system where:
- Tier 1 (internal gauntlet) catches regressions and gross leaks
- Tier 2 (external CFR agents) validates convergence against other solvers
- Tier 3 (public bots) provides real-world calibration against deployed systems

---

## 2. Available External Benchmarks

### 2.1 OpenSpiel HUNL Agents

**What it is:** Google DeepMind's OpenSpiel library includes CFR implementations for heads-up no-limit Texas Hold'em, including:
- Vanilla CFR
- CFR+
- External-sampling MCCFR
- Outcome-sampling MCCFR
- Linear CFR

These can be trained on the same game to produce reference strategies.

**Strengths:**
- Well-tested, peer-reviewed implementations
- Can control abstraction granularity for fair comparison
- Runs locally (no network dependency)
- Multiple algorithm variants for comparative study

**Weaknesses:**
- Their HUNL implementation uses a different abstraction than ours (bucket count, hand type scheme)
- Need to build a translation layer between our game state representation and theirs
- No pre-trained HUNL strategy available -- would need to train from scratch

**Integration effort:** Medium-high. Requires:
- Installing OpenSpiel (C++ core + Python bindings)
- Writing a game state adapter between our `HandContext`/`Deck`/`Card` and OpenSpiel's `State`
- Training an OpenSpiel agent on comparable abstraction
- Building a match engine that can pit our `GTOAgent` against an OpenSpiel policy

### 2.2 Slumbot (Public API)

**What it is:** Slumbot is a public HUNL poker bot accessible via HTTP API at `slumbot.com`. It plays heads-up no-limit Texas Hold'em with real money-equivalent play. Slumbot was developed by Eric Jackson and has been publicly available since ~2018.

**Strengths:**
- Truly external opponent -- no access to its strategy
- Plays full HUNL with real bet sizes (no abstraction translation issues on their end)
- HTTP API enables automated match running
- Widely used as a benchmark in poker AI research
- Free to play against

**Weaknesses:**
- Network latency adds ~100-500ms per action (vs <1ms locally)
- Rate limits may restrict volume (need to check current limits)
- Cannot control opponent's randomness seed (no reproducible matches)
- API format needs adapter to our `HandContext` → Slumbot action format
- Slumbot quality unknown relative to modern solvers (last updated ~2020)

**Integration effort:** Medium. Requires:
- HTTP client for Slumbot API (POST actions, GET game state)
- Adapter: Slumbot game state → `HandContext`
- Adapter: `AgentDecision` → Slumbot action format
- Match orchestrator with retries and rate limiting
- Results parser

### 2.3 ACPC (Annual Computer Poker Competition) Framework

**What it is:** The ACPC provides a standardized protocol for poker agent communication, including:
- TCP socket-based agent communication protocol
- Reference dealer software (deals cards, enforces rules)
- Historical competition logs and agents

**Strengths:**
- Industry-standard evaluation protocol
- Deterministic matchups with shared deck seeds
- Supports large-volume automated play
- Historical comparison: can compare against past ACPC entries (some agents are open-source)

**Weaknesses:**
- ACPC competitions have been discontinued since 2018
- Protocol is TCP-based (need socket adapter)
- Available agents are from 2014-2018 era (pre-Libratus/Pluribus advances)
- Setting up the dealer + agents requires non-trivial C/C++ compilation

**Integration effort:** Medium. Requires:
- ACPC dealer compilation (C source available on GitHub)
- TCP socket adapter for our agent (send/receive ACPC protocol messages)
- Match scheduling and result collection scripts
- Optional: wrap available ACPC agents for our match engine

### 2.4 PokerKit

**What it is:** A Python poker game engine that can serve as an alternative match framework.

**Strengths:**
- Pure Python, easy to integrate
- Handles all game mechanics (betting rounds, pot calculation, hand evaluation)
- Good for validating that our engine mechanics are correct

**Weaknesses:**
- Not a benchmark opponent -- it's a game engine, not an agent
- Would only be useful for cross-validating our `HeadsUpMatch` implementation

**Integration effort:** Low but limited value as a benchmark.

---

## 3. Proposed Ladder

### Tier 1: Internal Gauntlet (Current)

**Purpose:** Regression detection and leak identification.

**Components:**
- 7 heuristic bots (`eval_harness/adversaries.py`)
- 5,000 hands/bot, 3 seeds, `confidence_nearest` mapping
- `run_eval_harness.py --gauntlet --hands 5000 --seeds 42,123,456`

**Pass criteria:**
- Average >= +0 bb/100 (strategy is not exploited by any simple heuristic)
- No bot worse than -300 bb/100 (no catastrophic single-bot failure)
- WeirdSizingBot >= -200 bb/100 (bridge quality maintained)

**When to run:** Every branch, every checkpoint evaluation.

**Metrics tracked:**
- Per-bot bb/100 (mean, std, 95% CI)
- Average bb/100 across all bots
- Strategy hit rate
- Bridge pain map summary (if AIVAT integrated: AIVAT-adjusted results)

### Tier 2: OpenSpiel CFR Agents

**Purpose:** Validate convergence quality by playing against other CFR implementations.

**Components:**
- OpenSpiel CFR+ agent trained on a comparable abstraction (120 buckets, similar action menu)
- OpenSpiel external-sampling MCCFR agent (same abstraction)
- Both trained to similar iteration counts (50M-100M)

**Integration design:**

```
eval_harness/
    external/
        openspiel_adapter.py    # GameState ↔ HandContext translation
        openspiel_agent.py      # OpenSpiel policy wrapped as our Agent
        openspiel_trainer.py    # Train OpenSpiel agents with our abstraction
```

`openspiel_agent.py` would implement our `Agent` protocol:

```python
class OpenSpielAgent(Agent):
    """Wraps an OpenSpiel tabular policy as a match_engine.Agent."""
    name = "OpenSpiel-CFR+"

    def __init__(self, policy, adapter):
        self.policy = policy      # OpenSpiel TabularPolicy
        self.adapter = adapter    # Converts HandContext → OpenSpiel state

    def decide(self, ctx: HandContext) -> AgentDecision:
        state = self.adapter.to_openspiel_state(ctx)
        action_probs = self.policy.action_probabilities(state)
        # Sample and convert back to our AgentDecision
        action = sample_from(action_probs)
        return self.adapter.to_agent_decision(action, ctx)
```

**Pass criteria:**
- Win rate against OpenSpiel CFR+ should be within [-5, +5] bb/100 (both approximate Nash)
- Win rate against OpenSpiel MCCFR should be within [-10, +10] bb/100
- Any larger deviation indicates our abstraction or solver has a systematic issue

**When to run:** Major version releases (v11 baseline, v12 baseline, etc). Not per-branch.

**Metrics tracked:**
- Head-to-head bb/100 (10,000+ hands for tight CIs)
- Exploitability comparison: our strategy vs OpenSpiel strategy on same abstraction
- Per-street EV decomposition to identify where strategies diverge

### Tier 3: Public Bots (Slumbot)

**Purpose:** Real-world calibration against a deployed system.

**Components:**
- Slumbot API integration
- Automated match runner with rate limiting and error handling

**Integration design:**

```
eval_harness/
    external/
        slumbot_client.py       # HTTP client for Slumbot API
        slumbot_match.py        # Orchestrates full matches with result tracking
```

`slumbot_client.py` core interface:

```python
class SlumbotClient:
    BASE_URL = "https://slumbot.com/api"

    def new_hand(self) -> SlumbotHand:
        """Start a new hand. Returns hand state."""

    def act(self, hand_id: str, action: str) -> SlumbotState:
        """Take an action. Returns updated state."""

    def get_result(self, hand_id: str) -> float:
        """Get final result in bb."""
```

`slumbot_match.py` would run full matches:

```python
class SlumbotMatch:
    def __init__(self, our_agent: GTOAgent, client: SlumbotClient,
                 big_blind: int = 100):
        ...

    def play(self, num_hands: int) -> MatchResult:
        """Play num_hands against Slumbot, returning standard MatchResult."""
        # For each hand:
        # 1. Call client.new_hand()
        # 2. Translate Slumbot state → HandContext
        # 3. Call our_agent.decide(ctx)
        # 4. Translate AgentDecision → Slumbot action
        # 5. Call client.act()
        # 6. Repeat until hand complete
        # 7. Record HandRecord
```

**Pass criteria:**
- Win rate >= -10 bb/100 against Slumbot (we should not lose badly to a 2018-era bot)
- Ideally >= 0 bb/100 (our strategy should be competitive or better)
- CIs will be wide due to lower volume (network latency limits throughput)

**When to run:** Quarterly or on major milestones. Not automated in CI.

**Metrics tracked:**
- Head-to-head bb/100 (target: 2,000+ hands for reasonable CIs)
- Per-street EV breakdown (where do we win/lose against Slumbot?)
- Bridge pain analysis (how do Slumbot's bet sizes map to our abstraction?)
- AIVAT-adjusted results (if pilot successful -- critical for Slumbot where volume is limited)

---

## 4. Integration Requirements Summary

### 4.1 Tier 2 (OpenSpiel)

| Requirement | Effort | Priority |
|-------------|--------|----------|
| Install OpenSpiel (pip install open_spiel) | 1 hour | High |
| Game state adapter (our cards/bets ↔ OpenSpiel) | 1 day | High |
| Abstraction alignment (map our 120 buckets to OpenSpiel info sets) | 2 days | High |
| Train OpenSpiel CFR+ on aligned abstraction | 1 day (compute) | High |
| Agent wrapper (OpenSpiel policy → our Agent protocol) | 0.5 days | High |
| Match runner integration with `run_eval_harness.py` | 0.5 days | Medium |
| **Total** | **~5 days** | |

**Dependencies:**
- OpenSpiel Python package (pip installable)
- Shared abstraction definition between our system and OpenSpiel
- Sufficient compute to train OpenSpiel agent (hours, not days)

### 4.2 Tier 3 (Slumbot)

| Requirement | Effort | Priority |
|-------------|--------|----------|
| Slumbot API client (HTTP, JSON parsing) | 0.5 days | Medium |
| State translation (Slumbot format ↔ HandContext) | 1 day | Medium |
| Action translation (AgentDecision ↔ Slumbot actions) | 0.5 days | Medium |
| Match orchestrator with retries/rate limiting | 0.5 days | Medium |
| Results integration with MatchResult format | 0.5 days | Low |
| **Total** | **~3 days** | |

**Dependencies:**
- Network access to slumbot.com
- Understanding of Slumbot API format (documented on their site)
- Rate limit compliance (check current limits before committing to volume targets)

### 4.3 Shared Infrastructure

Both tiers need:

| Component | Purpose |
|-----------|---------|
| `eval_harness/external/__init__.py` | Package for external benchmarks |
| `run_eval_harness.py --tier2` / `--tier3` flags | CLI integration |
| Results schema extension | Store external benchmark results alongside internal gauntlet |
| `docs/results/` template update | Include external benchmark section in branch reports |

---

## 5. Metrics Tracked at Each Tier

| Metric | Tier 1 (Internal) | Tier 2 (OpenSpiel) | Tier 3 (Slumbot) |
|--------|-------------------|-------------------|-------------------|
| bb/100 per opponent | Yes (7 bots) | Yes (2 agents) | Yes (1 agent) |
| 95% CI per opponent | Yes (3+ seeds) | Yes (3+ seeds) | Yes (but wider) |
| Average bb/100 | Yes | Yes | N/A (single opponent) |
| Exploitability | Yes (separate eval) | Yes (head-to-head comparison) | No (black box) |
| AIVAT-adjusted bb/100 | Planned (E1) | Planned | Critical (low volume) |
| Per-street EV decomposition | Yes (detailed_tracking) | Yes | Yes |
| Bridge pain map | Yes | N/A (same abstraction) | Yes (real bet sizes) |
| Strategy hit rate | Yes | Yes | Yes |
| Action distribution comparison | No | Yes (compare trained strategies) | No |

---

## 6. Implementation Roadmap

### Phase 1: Slumbot Integration (Priority: High, Effort: 3 days)

Slumbot is the highest-value external benchmark because:
- It requires no training (already deployed)
- It uses real bet sizes (tests bridge quality in the real world)
- It provides the strongest "reality check" -- are we competitive?

Deliverables:
1. `eval_harness/external/slumbot_client.py` -- API client
2. `eval_harness/external/slumbot_match.py` -- match orchestrator
3. Run 500-hand pilot match, report bb/100
4. Document API format and rate limits

### Phase 2: OpenSpiel Integration (Priority: Medium, Effort: 5 days)

OpenSpiel provides the scientific benchmark:
- Two independent CFR implementations on the same game
- Exploitability cross-validation
- Strategy comparison at the information set level

Deliverables:
1. `eval_harness/external/openspiel_adapter.py` -- state translation
2. `eval_harness/external/openspiel_agent.py` -- agent wrapper
3. `eval_harness/external/openspiel_trainer.py` -- train reference agent
4. Run 5,000-hand match against OpenSpiel CFR+
5. Compare exploitability of both strategies on same abstraction

### Phase 3: CLI Integration (Priority: Low, Effort: 1 day)

Wire Tier 2 and Tier 3 into the eval harness:
1. Add `--tier2`, `--tier3` flags to `run_eval_harness.py`
2. Include external results in JSON output
3. Update branch report template (E3) to include external results

---

## 7. Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Slumbot win rate | >= -10 bb/100 | 2,000+ hands, 95% CI |
| OpenSpiel head-to-head | within +/- 5 bb/100 | 10,000 hands, 3 seeds |
| Exploitability parity | within 2x of OpenSpiel | Same abstraction, same samples |
| Bridge quality under real sizes | avg distance < 0.3 | Slumbot bridge pain map |
| No new leak categories discovered | 0 critical | External benchmark reveals no -500+ bb/100 pattern |

---

## 8. Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Slumbot API changes or goes offline | Medium | Cache API format; have fallback to Tier 2 only |
| OpenSpiel abstraction mismatch invalidates comparison | High | Align abstractions carefully; document any differences |
| Network latency limits Slumbot volume | Medium | Run Slumbot matches overnight; use AIVAT to get more from fewer hands |
| OpenSpiel training produces weak agent (bad abstraction mapping) | Medium | Validate OpenSpiel agent's exploitability independently before comparing |
| External benchmarks reveal our strategy is much weaker than expected | Low (but important) | This is the point -- better to know than not know |
