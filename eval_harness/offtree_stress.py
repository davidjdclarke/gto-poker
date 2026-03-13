"""
Off-tree action stress tests.

Systematically injects concrete bet sizings that don't map cleanly into
the abstraction and measures:
    - Which abstract action each concrete size maps to
    - How often multiple distinct sizings collapse to the same bucket
    - EV loss from action translation (via match play)
    - Worst-case fragility: lines where translation error is highest

This module provides both:
    1. Static analysis: mapping table showing all translations
    2. Dynamic analysis: play hands with forced off-tree sizings
"""
import random
from dataclasses import dataclass, field
from collections import defaultdict

from eval_harness.match_engine import (
    Agent, AgentDecision, HandContext, HeadsUpMatch, GTOAgent,
    classify_raise, MatchResult,
)
from server.gto.cfr import CFRTrainer


# ---------------------------------------------------------------------------
# Static mapping analysis
# ---------------------------------------------------------------------------
@dataclass
class MappingEntry:
    """How a concrete sizing maps to the abstraction."""
    concrete_size: float        # actual bet as fraction of pot
    concrete_label: str         # human description
    mapped_abstract: str        # which abstract action it becomes
    phase: str
    raise_count: int


def analyze_action_mapping(big_blind: int = 20) -> dict:
    """
    Produce a full mapping table showing how every tested concrete sizing
    maps into the abstract action space.

    Returns:
        {
            "preflop": [MappingEntry, ...],
            "postflop": [MappingEntry, ...],
            "collision_groups": {abstract_action: [concrete_sizes]},
            "summary": str,
        }
    """
    results = {"preflop": [], "postflop": [], "collision_groups": defaultdict(list)}

    # Preflop open sizes (BB multipliers)
    preflop_opens = [
        (2.0, "min-open 2x"),
        (2.2, "small open 2.2x"),
        (2.5, "standard 2.5x"),
        (3.0, "large open 3x"),
        (3.5, "oversize 3.5x"),
        (5.0, "5x open"),
    ]
    pot = big_blind * 1.5  # SB + BB
    current_bet = big_blind

    for mult, label in preflop_opens:
        raise_amount = int(big_blind * mult)
        mapped = classify_raise(raise_amount, pot, current_bet, "preflop", 0, big_blind)
        entry = MappingEntry(mult, label, mapped, "preflop", 0)
        results["preflop"].append(entry)
        results["collision_groups"][f"preflop:{mapped}"].append(mult)

    # Preflop 3-bet sizes (multipliers of open)
    open_to = int(big_blind * 2.5) + big_blind  # standard open = 2.5bb + BB
    pot_after_open = pot + int(big_blind * 2.5)
    threebet_mults = [
        (2.5, "small 3bet 2.5x"),
        (3.0, "standard 3bet 3x"),
        (3.5, "large 3bet 3.5x"),
        (4.0, "oversize 3bet 4x"),
    ]
    for mult, label in threebet_mults:
        raise_amount = int(open_to * mult) - open_to
        mapped = classify_raise(raise_amount, pot_after_open, open_to, "preflop", 1, big_blind)
        entry = MappingEntry(mult, label, mapped, "preflop", 1)
        results["preflop"].append(entry)
        results["collision_groups"][f"preflop_3bet:{mapped}"].append(mult)

    # Postflop bet sizes (pot fractions)
    postflop_pcts = [
        (0.10, "10% pot (micro)"),
        (0.15, "15% pot"),
        (0.20, "20% pot"),
        (0.25, "25% pot"),
        (0.28, "28% pot"),
        (0.33, "33% pot (on-tree)"),
        (0.40, "40% pot"),
        (0.42, "42% pot"),
        (0.50, "50% pot (on-tree)"),
        (0.60, "60% pot"),
        (0.67, "67% pot"),
        (0.75, "75% pot"),
        (0.85, "85% pot"),
        (1.00, "100% pot (on-tree)"),
        (1.20, "120% pot (overbet)"),
        (1.50, "150% pot (large overbet)"),
        (2.00, "200% pot (massive overbet)"),
    ]
    test_pot = 100  # reference pot

    for pct, label in postflop_pcts:
        raise_amount = int(test_pot * pct)
        mapped = classify_raise(raise_amount, test_pot, 0, "flop", 0, big_blind)
        entry = MappingEntry(pct, label, mapped, "flop", 0)
        results["postflop"].append(entry)
        results["collision_groups"][f"postflop:{mapped}"].append(pct)

    # Min-raise postflop (depends on BB)
    min_raise_pct = big_blind / test_pot
    mapped = classify_raise(big_blind, test_pot, 0, "flop", 0, big_blind)
    entry = MappingEntry(min_raise_pct, f"min-raise ({big_blind} into {test_pot})",
                         mapped, "flop", 0)
    results["postflop"].append(entry)

    # Summary
    lines = []
    lines.append("=== OFF-TREE ACTION MAPPING ANALYSIS ===\n")

    lines.append("PREFLOP OPEN SIZING:")
    for e in results["preflop"]:
        if e.raise_count == 0:
            lines.append(f"  {e.concrete_label:<25} -> {e.mapped_abstract}")

    lines.append("\nPREFLOP 3-BET SIZING:")
    for e in results["preflop"]:
        if e.raise_count == 1:
            lines.append(f"  {e.concrete_label:<25} -> {e.mapped_abstract}")

    lines.append("\nPOSTFLOP BET SIZING:")
    for e in results["postflop"]:
        lines.append(f"  {e.concrete_label:<35} -> {e.mapped_abstract}")

    lines.append("\nCOLLISION GROUPS (distinct sizes mapping to same action):")
    for action, sizes in sorted(results["collision_groups"].items()):
        if len(sizes) > 1:
            lines.append(f"  {action}: {sizes}")

    results["summary"] = "\n".join(lines)
    return results


# ---------------------------------------------------------------------------
# Forced-sizing agent for dynamic testing
# ---------------------------------------------------------------------------
class ForcedSizingAgent(Agent):
    """Agent that forces specific off-tree bet sizes to stress-test translation."""

    def __init__(self, name: str, preflop_open_mult: float = 2.5,
                 postflop_bet_pct: float = 0.50):
        self.name = name
        self.preflop_open_mult = preflop_open_mult
        self.postflop_bet_pct = postflop_bet_pct

    def decide(self, ctx: HandContext) -> AgentDecision:
        to_call = ctx.current_bet - ctx.my_bet
        from eval_harness.fast_equity import fast_bot_equity
        try:
            equity = fast_bot_equity(ctx.hole_cards, ctx.community_cards)
        except Exception:
            equity = 0.5

        if ctx.phase == "preflop":
            if equity > 0.50:
                raise_to = ctx.current_bet + max(ctx.min_raise,
                                                  int(ctx.big_blind * self.preflop_open_mult))
                raise_to = min(raise_to, ctx.my_bet + ctx.my_chips)
                if raise_to > ctx.current_bet:
                    return AgentDecision("raise", raise_to)
                return AgentDecision("call" if to_call > 0 else "check")
            elif equity > 0.38:
                return AgentDecision("call" if to_call > 0 else "check")
            return AgentDecision("fold") if to_call > 0 else AgentDecision("check")
        else:
            if equity > 0.40:
                if to_call > 0:
                    if equity > 0.55 and random.random() < 0.3:
                        raise_to = ctx.current_bet + max(ctx.min_raise,
                                                          int(ctx.pot * self.postflop_bet_pct))
                        raise_to = min(raise_to, ctx.my_bet + ctx.my_chips)
                        if raise_to > ctx.current_bet:
                            return AgentDecision("raise", raise_to)
                    return AgentDecision("call")
                else:
                    bet_size = max(ctx.min_raise, int(ctx.pot * self.postflop_bet_pct))
                    raise_to = bet_size
                    raise_to = min(raise_to, ctx.my_bet + ctx.my_chips)
                    if raise_to > 0:
                        return AgentDecision("raise", raise_to)
                    return AgentDecision("check")
            else:
                return AgentDecision("fold") if to_call > 0 else AgentDecision("check")


# ---------------------------------------------------------------------------
# Off-tree stress test suite
# ---------------------------------------------------------------------------
@dataclass
class OffTreeResult:
    """Result from one off-tree sizing test."""
    sizing_label: str
    preflop_mult: float
    postflop_pct: float
    gto_bb_per_100: float
    mapped_to: str           # which abstract action it mapped to
    num_hands: int


def run_offtree_stress_tests(trainer: CFRTrainer, num_hands: int = 500,
                              seed: int = 42, big_blind: int = 20,
                              progress_cb=None) -> dict:
    """
    Run GTO agent against a suite of forced-sizing opponents.

    Tests each problematic sizing individually and measures GTO's bb/100.
    A well-translated strategy should beat all of them; large losses
    indicate translation fragility.
    """
    test_configs = [
        # (label, preflop_mult, postflop_pct, expected_mapping)
        ("standard (control)", 2.5, 0.50, "on-tree"),
        ("min-open 2.0x", 2.0, 0.50, "open_raise"),
        ("small open 2.2x", 2.2, 0.50, "open_raise"),
        ("large open 3.5x", 3.5, 0.50, "open_raise"),
        ("15% pot postflop", 2.5, 0.15, "bet_third"),
        ("28% pot postflop", 2.5, 0.28, "bet_third"),
        ("42% pot postflop", 2.5, 0.42, "bet_half"),
        ("75% pot postflop", 2.5, 0.75, "bet_pot"),
        ("120% overbet", 2.5, 1.20, "all_in"),
        ("min-click + 67%", 2.0, 0.67, "bet_pot"),
        ("5x open + 200% overbet", 5.0, 2.0, "all_in"),
    ]

    results = []
    gto = GTOAgent(trainer, name="GTO")

    for label, pre_mult, post_pct, expected in test_configs:
        opp = ForcedSizingAgent(f"Sizing_{label}", pre_mult, post_pct)

        match = HeadsUpMatch(gto, opp, big_blind=big_blind, seed=seed)
        match_result = match.play(num_hands)

        # Determine actual mapping
        pot_ref = 100
        mapped = classify_raise(int(pot_ref * post_pct), pot_ref, 0, "flop", 0, big_blind)

        results.append(OffTreeResult(
            sizing_label=label,
            preflop_mult=pre_mult,
            postflop_pct=post_pct,
            gto_bb_per_100=match_result.p0_bb_per_100,
            mapped_to=mapped,
            num_hands=num_hands,
        ))

        if progress_cb is not None:
            progress_cb(label)

    # Build report
    report = {
        "results": results,
        "summary": _format_offtree_report(results),
    }
    return report


def _format_offtree_report(results: list[OffTreeResult]) -> str:
    lines = []
    lines.append("=== OFF-TREE ROBUSTNESS REPORT ===\n")
    lines.append(f"{'Sizing':<30} {'Pre':<6} {'Post':<6} {'Mapped':<12} {'bb/100':>8}")
    lines.append("-" * 72)

    control_ev = None
    for r in results:
        if "control" in r.sizing_label:
            control_ev = r.gto_bb_per_100

        flag = ""
        if control_ev is not None and r.gto_bb_per_100 < control_ev - 5:
            flag = " [!]"
        elif r.gto_bb_per_100 < -5:
            flag = " [!!]"

        lines.append(
            f"{r.sizing_label:<30} {r.preflop_mult:<6.1f} {r.postflop_pct:<6.2f} "
            f"{r.mapped_to:<12} {r.gto_bb_per_100:>+8.1f}{flag}"
        )

    if control_ev is not None:
        lines.append(f"\nControl baseline: {control_ev:+.1f} bb/100")
        worst = min(results, key=lambda r: r.gto_bb_per_100)
        lines.append(f"Worst sizing:     {worst.sizing_label} at {worst.gto_bb_per_100:+.1f} bb/100")
        ev_loss = control_ev - worst.gto_bb_per_100
        lines.append(f"Max EV loss from translation: {ev_loss:.1f} bb/100")

    return "\n".join(lines)
