"""
EV decomposition tools for diagnosing strategy leaks.

Requires hands played with `detailed_tracking=True` on HeadsUpMatch.
All EV values are in raw chips (divide by big_blind for bb units).
"""
from collections import defaultdict
from eval_harness.match_engine import HandRecord, DecisionRecord


def decompose_by_street(hands: list[HandRecord], big_blind: float = 1.0) -> dict:
    """Per-street P0 EV in bb/100.

    Returns dict: {phase: bb_per_100} aggregated across all hands.
    """
    totals = defaultdict(float)
    counts = defaultdict(int)
    for h in hands:
        for phase, ev in h.street_ev.items():
            totals[phase] += ev
            counts[phase] += 1
    num_hands = len(hands) if hands else 1
    return {
        phase: (totals[phase] / big_blind / num_hands) * 100
        for phase in totals
    }


def decompose_by_action_family(hands: list[HandRecord],
                                phase: str | None = None,
                                big_blind: float = 1.0) -> dict:
    """P0 EV contribution by action family (fold/check/call/bet/raise).

    Groups P0's decisions by action family and sums the hand-level p0_net
    for hands where P0 took that action on the given phase (or any phase).

    Returns: {action_family: {total_ev_bb, count, avg_ev_bb}}
    """
    families = {"fold": [], "check": [], "call": [], "bet": []}

    for h in hands:
        p0_actions_in_phase = [
            d for d in h.decisions
            if d.player == 0 and (phase is None or d.phase == phase)
        ]
        for d in p0_actions_in_phase:
            family = _action_to_family(d.action)
            families[family].append(h.p0_net / big_blind)

    result = {}
    for fam, values in families.items():
        if values:
            result[fam] = {
                "total_ev_bb": sum(values),
                "count": len(values),
                "avg_ev_bb": sum(values) / len(values),
            }
        else:
            result[fam] = {"total_ev_bb": 0.0, "count": 0, "avg_ev_bb": 0.0}
    return result


def decompose_by_bucket(hands: list[HandRecord],
                         phase: str | None = None,
                         big_blind: float = 1.0) -> dict:
    """P0 EV by equity bucket (0-7).

    Returns: {eq_bucket: {total_ev_bb, count, avg_ev_bb}}
    """
    buckets = defaultdict(list)
    for h in hands:
        if phase is not None:
            bucket = h.p0_bucket_per_street.get(phase, -1)
            if bucket < 0:
                continue
            eq = bucket // 15
        else:
            # Use the last phase's bucket as representative
            if not h.p0_bucket_per_street:
                continue
            last_phase = h.phases_reached[-1] if h.phases_reached else "preflop"
            bucket = h.p0_bucket_per_street.get(last_phase, -1)
            if bucket < 0:
                continue
            eq = bucket // 15
        buckets[eq].append(h.p0_net / big_blind)

    result = {}
    for eq in range(8):
        values = buckets.get(eq, [])
        if values:
            result[eq] = {
                "total_ev_bb": sum(values),
                "count": len(values),
                "avg_ev_bb": sum(values) / len(values),
            }
        else:
            result[eq] = {"total_ev_bb": 0.0, "count": 0, "avg_ev_bb": 0.0}
    return result


def callstation_dashboard(hands: list[HandRecord],
                           big_blind: float = 1.0) -> dict:
    """Diagnose EV leaks specific to the CallStation matchup.

    Classifies hands into leak categories:
    - slowplay_leak: P0 checked with strong hand (EQ5+), opponent checked behind
    - bluff_leak: P0 bet with weak hand (EQ0-2), opponent called
    - thin_value_leak: P0 checked with medium hand (EQ3-5) that could have bet for value

    Returns: {category: {count, total_ev_bb, avg_ev_bb, example_hands}}
    """
    slowplay = []
    bluff = []
    thin_value = []

    for h in hands:
        p0_decisions = [d for d in h.decisions if d.player == 0]
        if not p0_decisions:
            continue

        for d in p0_decisions:
            if d.eq_bucket < 0:
                continue
            family = _action_to_family(d.action)

            # Slowplay: strong hand (EQ5+) checked when could have bet
            if d.eq_bucket >= 5 and family == "check":
                slowplay.append({
                    "hand_num": h.hand_num,
                    "phase": d.phase,
                    "eq_bucket": d.eq_bucket,
                    "p0_net_bb": h.p0_net / big_blind,
                })

            # Bluff: weak hand (EQ0-2) bet into a station
            if d.eq_bucket <= 2 and family == "bet":
                bluff.append({
                    "hand_num": h.hand_num,
                    "phase": d.phase,
                    "eq_bucket": d.eq_bucket,
                    "p0_net_bb": h.p0_net / big_blind,
                    "amount": d.amount,
                })

            # Thin value: medium hand (EQ3-4) checked — potential missed value
            if 3 <= d.eq_bucket <= 4 and family == "check":
                thin_value.append({
                    "hand_num": h.hand_num,
                    "phase": d.phase,
                    "eq_bucket": d.eq_bucket,
                    "p0_net_bb": h.p0_net / big_blind,
                })

    def _summarize(entries):
        if not entries:
            return {"count": 0, "total_ev_bb": 0.0, "avg_ev_bb": 0.0,
                    "by_phase": {}, "by_eq": {}}
        total = sum(e["p0_net_bb"] for e in entries)
        # Break down by phase
        by_phase = defaultdict(list)
        by_eq = defaultdict(list)
        for e in entries:
            by_phase[e["phase"]].append(e["p0_net_bb"])
            by_eq[e["eq_bucket"]].append(e["p0_net_bb"])
        return {
            "count": len(entries),
            "total_ev_bb": total,
            "avg_ev_bb": total / len(entries),
            "by_phase": {p: {"count": len(v), "avg_ev_bb": sum(v)/len(v)}
                         for p, v in by_phase.items()},
            "by_eq": {eq: {"count": len(v), "avg_ev_bb": sum(v)/len(v)}
                      for eq, v in by_eq.items()},
        }

    return {
        "slowplay_leak": _summarize(slowplay),
        "bluff_leak": _summarize(bluff),
        "thin_value_leak": _summarize(thin_value),
        "num_hands": len(hands),
        "overall_bb_per_100": (sum(h.p0_net for h in hands) / big_blind / max(len(hands), 1)) * 100,
    }


def format_dashboard(dashboard: dict) -> str:
    """Format callstation_dashboard output for display."""
    lines = []
    lines.append(f"=== CallStation Dashboard ({dashboard['num_hands']} hands) ===")
    lines.append(f"Overall: {dashboard['overall_bb_per_100']:.1f} bb/100\n")

    for category in ["slowplay_leak", "bluff_leak", "thin_value_leak"]:
        d = dashboard[category]
        label = category.replace("_", " ").title()
        lines.append(f"--- {label} ---")
        lines.append(f"  Count: {d['count']}")
        lines.append(f"  Total EV: {d['total_ev_bb']:.1f} bb")
        lines.append(f"  Avg EV: {d['avg_ev_bb']:.1f} bb")
        if d.get("by_phase"):
            for phase, stats in sorted(d["by_phase"].items()):
                lines.append(f"    {phase}: {stats['count']} decisions, "
                             f"avg {stats['avg_ev_bb']:.1f} bb")
        if d.get("by_eq"):
            for eq, stats in sorted(d["by_eq"].items()):
                lines.append(f"    EQ{eq}: {stats['count']} decisions, "
                             f"avg {stats['avg_ev_bb']:.1f} bb")
        lines.append("")

    return "\n".join(lines)


def _action_to_family(action: str) -> str:
    """Map concrete action string to family."""
    if action == "fold":
        return "fold"
    if action == "check":
        return "check"
    if action == "call":
        return "call"
    # Everything else is a bet/raise
    return "bet"
