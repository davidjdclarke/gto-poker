"""
Half-AIVAT variance reduction for GTO agent evaluation.

Uses the GTO agent's own strategy as a zero-mean control variate.
At each GTO decision node, the actual outcome is adjusted by the difference
between what happened and what was expected under the blueprint strategy.

This is the "bucket-EV approximation" variant:
- Continuation values are approximated by a precomputed (phase, eq_bucket) → mean_ev table
- No rollouts required; pure post-processing of existing HandRecord data
- Requires detailed_tracking=True matches (DecisionRecord.strategy must be populated)

Target: 3x variance reduction at 1,000 hands vs 5,000 hands raw.

References:
    Burch et al. "AIVAT: A New Variance Reduction Technique for Agent
    Evaluation in Imperfect Information Games." AAAI 2018.

Usage:
    # Build calibration table (one-time, from GTO self-play)
    from eval_harness.aivat import build_bucket_ev_table, save_bucket_ev_table

    trainer = get_trainer()
    gto0 = GTOAgent(trainer, name="GTO0")
    gto1 = GTOAgent(trainer, name="GTO1")
    match = HeadsUpMatch(gto0, gto1, big_blind=20, seed=42, detailed_tracking=True)
    result = match.play(10000)
    table = build_bucket_ev_table(result.hands, big_blind=20)
    save_bucket_ev_table(table)

    # Apply AIVAT adjustment to a gauntlet match
    from eval_harness.aivat import load_bucket_ev_table, aivat_adjusted_result

    table = load_bucket_ev_table()
    aivat_result = aivat_adjusted_result(match_result, table, gto_player=0)
    print(f"Raw:   {aivat_result.raw_bb100:+.1f} bb/100  (CI: ±{aivat_result.raw_ci_95:.1f})")
    print(f"AIVAT: {aivat_result.aivat_bb100:+.1f} bb/100  (CI: ±{aivat_result.aivat_ci_95:.1f})")
    print(f"Variance reduction: {aivat_result.variance_reduction_pct:.0f}%")
"""
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from eval_harness.match_engine import HandRecord, DecisionRecord, MatchResult, GTOAgent

_DEFAULT_TABLE_PATH = Path(__file__).parent / "bucket_ev_table.json"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class AivatResult:
    """Comparison of raw and AIVAT-adjusted evaluation results."""
    raw_bb100: float
    aivat_bb100: float
    raw_std: float
    aivat_std: float
    raw_ci_95: float
    aivat_ci_95: float
    variance_reduction_pct: float
    num_hands: int
    num_corrected_decisions: int   # decisions where strategy was available


# ---------------------------------------------------------------------------
# Bucket-EV table: build, save, load
# ---------------------------------------------------------------------------
def build_bucket_ev_table(hands: list, big_blind: float = 1.0) -> dict:
    """Build a (phase, eq_bucket) → mean_ev_bb table from GTO self-play hands.

    The table maps each (phase, equity_bucket) combination to the average
    per-hand EV seen by the GTO player (P0) when they held that equity bucket
    at the start of that phase.

    Args:
        hands:      list[HandRecord] with detailed_tracking=True
                    (p0_bucket_per_street must be populated)
        big_blind:  chip value of one big blind (for normalization)

    Returns:
        dict: {"preflop": {0: ev, 1: ev, ...}, "flop": {...}, ...}
    """
    phase_bucket_evs: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    phases = ["preflop", "flop", "turn", "river"]

    for hand in hands:
        p0_net_bb = hand.p0_net / big_blind if big_blind > 0 else hand.p0_net
        for phase in phases:
            bucket = hand.p0_bucket_per_street.get(phase, -1)
            if bucket < 0:
                continue
            eq_bucket = bucket // 15  # equity bucket (0–7)
            phase_bucket_evs[phase][eq_bucket].append(p0_net_bb)

    table = {}
    for phase in phases:
        table[phase] = {}
        for eq_bucket, evs in phase_bucket_evs[phase].items():
            table[phase][int(eq_bucket)] = float(np.mean(evs))

    return table


def save_bucket_ev_table(table: dict,
                          path: Optional[str] = None) -> str:
    """Save bucket-EV table to JSON.

    Args:
        table: from build_bucket_ev_table()
        path:  save path (default: eval_harness/bucket_ev_table.json)

    Returns:
        str: path where file was saved
    """
    save_path = Path(path) if path else _DEFAULT_TABLE_PATH
    # JSON requires string keys
    serializable = {phase: {str(k): v for k, v in buckets.items()}
                    for phase, buckets in table.items()}
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    return str(save_path)


def load_bucket_ev_table(path: Optional[str] = None) -> dict:
    """Load bucket-EV table from JSON.

    Args:
        path: load path (default: eval_harness/bucket_ev_table.json)

    Returns:
        dict: {phase: {eq_bucket_int: mean_ev_bb}}

    Raises:
        FileNotFoundError: if table file does not exist
    """
    load_path = Path(path) if path else _DEFAULT_TABLE_PATH
    if not load_path.exists():
        raise FileNotFoundError(
            f"Bucket-EV table not found at {load_path}. "
            "Run --build-bucket-ev first to generate it."
        )
    with open(load_path) as f:
        raw = json.load(f)
    return {phase: {int(k): float(v) for k, v in buckets.items()}
            for phase, buckets in raw.items()}


def bucket_ev_table_exists(path: Optional[str] = None) -> bool:
    """Return True if a saved bucket-EV table exists."""
    p = Path(path) if path else _DEFAULT_TABLE_PATH
    return p.exists()


# ---------------------------------------------------------------------------
# AIVAT correction
# ---------------------------------------------------------------------------
def compute_aivat_correction(hand: HandRecord, bucket_ev: dict,
                              gto_player: int = 0,
                              big_blind: float = 1.0) -> float:
    """Compute the AIVAT correction value for a single hand.

    Applies one correction per phase (not per decision within a phase).
    For each phase where the GTO player made a decision with known strategy:

        continuation_value = remaining payoff from that phase onward (in bb)
                           = total_payoff_bb - sum(earlier_street_evs_bb)

        correction += continuation_value - bucket_ev[phase][eq_bucket]

    Using per-phase remaining payoff (via hand.street_ev) ensures each
    correction uses an independent estimate of the continuation value,
    which keeps the control variate zero-mean and reduces variance.

    Args:
        hand:       HandRecord with detailed_tracking=True decisions
                    (hand.street_ev must be populated for correct correction)
        bucket_ev:  {phase: {eq_bucket: mean_ev_bb}} from load_bucket_ev_table()
        gto_player: which player index is the GTO agent (0 or 1)
        big_blind:  chip value of one big blind (converts hand.p0_net to bb)

    Returns:
        float: AIVAT correction to subtract from actual payoff (in bb)
    """
    total_bb = hand.p0_net / big_blind

    # Build per-phase remaining payoff using street_ev (chips → bb).
    # remaining[phase] = total payoff from that phase onward.
    # When street_ev is empty (no detailed_tracking), all values = total_bb,
    # which is safe but gives only one meaningful correction (first phase seen).
    _phases_order = ["preflop", "flop", "turn", "river"]
    remaining: dict[str, float] = {}
    cumulative = 0.0
    for ph in _phases_order:
        remaining[ph] = total_bb - cumulative
        cumulative += hand.street_ev.get(ph, 0.0) / big_blind

    correction = 0.0
    phases_corrected: set[str] = set()

    for dec in hand.decisions:
        if dec.player != gto_player:
            continue
        if not dec.strategy:
            continue  # strategy not populated — skip
        if dec.phase in phases_corrected:
            continue  # apply at most once per phase
        if dec.eq_bucket < 0:
            continue

        baseline = bucket_ev.get(dec.phase, {}).get(dec.eq_bucket)
        if baseline is None:
            continue

        # continuation_value = remaining payoff from this phase onward (in bb)
        cont_value = remaining.get(dec.phase, total_bb)
        correction += cont_value - baseline
        phases_corrected.add(dec.phase)

    return correction


def aivat_adjusted_payoff(hand: HandRecord, bucket_ev: dict,
                           gto_player: int = 0,
                           big_blind: float = 1.0) -> float:
    """Return the AIVAT-adjusted per-hand payoff for the GTO player (in bb).

    Args:
        hand:       HandRecord (p0_net in chips; divided by big_blind internally)
        bucket_ev:  from load_bucket_ev_table()
        gto_player: 0 or 1
        big_blind:  chip value of one big blind

    Returns:
        float: AIVAT-adjusted payoff in bb
    """
    correction = compute_aivat_correction(hand, bucket_ev, gto_player, big_blind)
    return hand.p0_net / big_blind - correction


def aivat_adjusted_result(result: MatchResult, bucket_ev: dict,
                           gto_player: int = 0,
                           big_blind: float = 1.0) -> AivatResult:
    """Apply AIVAT corrections to a full MatchResult.

    Args:
        result:     MatchResult from HeadsUpMatch.play() with detailed_tracking=True
        bucket_ev:  from load_bucket_ev_table()
        gto_player: which player index is the GTO agent
        big_blind:  chip value of big blind (only used if p0_net is in chips)

    Returns:
        AivatResult with raw and adjusted statistics
    """
    if not result.hands:
        return AivatResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    n = len(result.hands)
    bb = big_blind if big_blind > 0 else 1.0
    # Normalise to bb (p0_net is stored in chips by HeadsUpMatch)
    raw_payoffs = np.array([h.p0_net / bb for h in result.hands])
    aivat_payoffs = np.array([
        aivat_adjusted_payoff(h, bucket_ev, gto_player, big_blind=bb)
        for h in result.hands
    ])

    # Count decisions with strategy available
    corrected_decisions = sum(
        1 for h in result.hands
        for d in h.decisions
        if d.player == gto_player and bool(d.strategy)
    )

    # Raw statistics (payoffs already in bb)
    raw_mean = float(raw_payoffs.mean()) * 100  # bb/100
    raw_std_per_hand = float(raw_payoffs.std(ddof=1)) if n > 1 else 0.0
    raw_std_bb100 = raw_std_per_hand * 100
    raw_ci = 1.96 * raw_std_per_hand / math.sqrt(n) * 100 if n > 1 else 0.0

    # AIVAT statistics
    aivat_mean = float(aivat_payoffs.mean()) * 100
    aivat_std_per_hand = float(aivat_payoffs.std(ddof=1)) if n > 1 else 0.0
    aivat_std_bb100 = aivat_std_per_hand * 100
    aivat_ci = 1.96 * aivat_std_per_hand / math.sqrt(n) * 100 if n > 1 else 0.0

    # Variance reduction
    raw_var = raw_std_per_hand ** 2
    aivat_var = aivat_std_per_hand ** 2
    if raw_var > 0 and aivat_var > 0:
        reduction_pct = (1.0 - aivat_var / raw_var) * 100.0
    else:
        reduction_pct = 0.0

    return AivatResult(
        raw_bb100=round(raw_mean, 2),
        aivat_bb100=round(aivat_mean, 2),
        raw_std=round(raw_std_bb100, 2),
        aivat_std=round(aivat_std_bb100, 2),
        raw_ci_95=round(raw_ci, 2),
        aivat_ci_95=round(aivat_ci, 2),
        variance_reduction_pct=round(reduction_pct, 1),
        num_hands=n,
        num_corrected_decisions=corrected_decisions,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def variance_reduction_report(raw: list, aivat: list) -> dict:
    """Compute variance reduction statistics from two lists of per-hand payoffs.

    Args:
        raw:   list[float] per-hand raw payoffs (in bb)
        aivat: list[float] per-hand AIVAT-adjusted payoffs (in bb)

    Returns:
        dict with raw_var, aivat_var, reduction_pct, raw_ci_95, aivat_ci_95
    """
    raw_arr = np.array(raw)
    aivat_arr = np.array(aivat)
    n = len(raw_arr)

    raw_var = float(raw_arr.var(ddof=1)) if n > 1 else 0.0
    aivat_var = float(aivat_arr.var(ddof=1)) if n > 1 else 0.0
    raw_ci = 1.96 * math.sqrt(raw_var / n) * 100 if n > 1 else 0.0
    aivat_ci = 1.96 * math.sqrt(aivat_var / n) * 100 if n > 1 else 0.0

    reduction = (1.0 - aivat_var / raw_var) * 100.0 if raw_var > 0 else 0.0

    return {
        "raw_var": round(raw_var, 4),
        "aivat_var": round(aivat_var, 4),
        "reduction_pct": round(reduction, 1),
        "raw_ci_95": round(raw_ci, 2),
        "aivat_ci_95": round(aivat_ci, 2),
        "num_hands": n,
    }


def print_aivat_comparison(result: AivatResult) -> None:
    """Print a formatted AIVAT vs raw comparison table."""
    print(f"\n  AIVAT Variance Reduction:")
    print(f"  {'':20} {'Mean':>10} {'Std':>10} {'95% CI':>10}  {'Decisions':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}  {'-'*10}")
    print(f"  {'Raw bb/100':20} {result.raw_bb100:>+10.1f} {result.raw_std:>10.1f} "
          f"±{result.raw_ci_95:>9.1f}")
    print(f"  {'AIVAT bb/100':20} {result.aivat_bb100:>+10.1f} {result.aivat_std:>10.1f} "
          f"±{result.aivat_ci_95:>9.1f}  {result.num_corrected_decisions:>10,}")
    reduction = result.variance_reduction_pct
    status = "PASS" if reduction >= 200 else "WARN" if reduction >= 100 else "FAIL"
    print(f"\n  [{status}] Variance reduction: {reduction:.0f}%  "
          f"({result.num_hands:,} hands)")
    if reduction < 100:
        print("  [WARN] < 2x reduction — consider upgrading to single-rollout continuation values")


# ---------------------------------------------------------------------------
# Calibration runner (called by --build-bucket-ev)
# ---------------------------------------------------------------------------
def build_and_save_bucket_ev_table(trainer, num_hands: int = 10000,
                                    seed: int = 42, big_blind: int = 20,
                                    save_path: Optional[str] = None) -> dict:
    """Run GTO self-play and build+save the bucket-EV calibration table.

    Args:
        trainer:   CFRTrainer with loaded strategy
        num_hands: calibration hands (default 10,000)
        seed:      RNG seed
        big_blind: big blind chip value
        save_path: where to save the JSON table

    Returns:
        dict: the generated bucket-EV table
    """
    from eval_harness.match_engine import GTOAgent, HeadsUpMatch
    print(f"\n  Building bucket-EV table from {num_hands:,} hands of GTO self-play...")
    gto0 = GTOAgent(trainer, name="GTO0", simulations=80)
    gto1 = GTOAgent(trainer, name="GTO1", simulations=80)
    match = HeadsUpMatch(gto0, gto1, big_blind=big_blind, seed=seed,
                          detailed_tracking=True)
    result = match.play(num_hands)
    table = build_bucket_ev_table(result.hands, big_blind=big_blind)

    path = save_bucket_ev_table(table, path=save_path)
    # Report coverage
    total_entries = sum(len(v) for v in table.values())
    print(f"  Bucket-EV table: {total_entries} entries across "
          f"{len(table)} phases → saved to {path}")
    return table
