"""
Bridge pain map: diagnose where off-tree action translation loses EV.

Analyzes the gap between concrete bet ratios opponents use and the abstract
actions they get mapped to. Large gaps indicate translation error hotspots.
"""
from collections import defaultdict
from eval_harness.match_engine import HandRecord
from server.gto.abstraction import Action, ACTION_NAMES


# Abstract action -> nominal pot fraction
_ABSTRACT_SIZES = {
    int(Action.BET_QUARTER_POT): 0.25,
    int(Action.BET_THIRD_POT): 0.33,
    int(Action.BET_HALF_POT): 0.50,
    int(Action.BET_TWO_THIRDS_POT): 0.67,
    int(Action.BET_THREE_QUARTER_POT): 0.75,
    int(Action.BET_POT): 1.00,
    int(Action.BET_OVERBET): 1.25,
    int(Action.BET_DOUBLE_POT): 2.00,
    int(Action.ALL_IN): 3.00,
    int(Action.DONK_SMALL): 0.25,
    int(Action.DONK_MEDIUM): 0.50,
    int(Action.OPEN_RAISE): 0.33,  # approx preflop
    int(Action.THREE_BET): 0.50,
    int(Action.FOUR_BET): 0.67,
}


def analyze_bridge_pain(bridge_log: list[tuple],
                         hands: list[HandRecord],
                         big_blind: float = 1.0) -> dict:
    """Analyze bridge translation quality from GTOAgent.bridge_log.

    Args:
        bridge_log: list of (concrete_ratio, mapped_action, phase, bucket)
        hands: HandRecord list for EV context
        big_blind: for normalization

    Returns dict with:
        - entries: list of per-event records
        - by_mapped_action: {action_id: {count, avg_distance, concrete_range}}
        - by_phase: {phase: {count, avg_distance}}
        - worst_gaps: top 10 largest translation distances
    """
    entries = []
    by_action = defaultdict(list)
    by_phase = defaultdict(list)

    for concrete_ratio, mapped_action, phase, bucket in bridge_log:
        nominal = _ABSTRACT_SIZES.get(mapped_action, 0.5)
        distance = abs(concrete_ratio - nominal)
        entry = {
            "concrete_ratio": round(concrete_ratio, 3),
            "mapped_action": mapped_action,
            "mapped_name": ACTION_NAMES.get(mapped_action, f"action_{mapped_action}"),
            "nominal_ratio": nominal,
            "distance": round(distance, 3),
            "phase": phase,
            "bucket": bucket,
        }
        entries.append(entry)
        by_action[mapped_action].append(entry)
        by_phase[phase].append(entry)

    # Summarize by mapped action
    action_summary = {}
    for action_id, evts in by_action.items():
        ratios = [e["concrete_ratio"] for e in evts]
        distances = [e["distance"] for e in evts]
        action_summary[ACTION_NAMES.get(action_id, f"action_{action_id}")] = {
            "count": len(evts),
            "avg_distance": round(sum(distances) / len(distances), 3),
            "max_distance": round(max(distances), 3),
            "concrete_range": (round(min(ratios), 3), round(max(ratios), 3)),
            "nominal": _ABSTRACT_SIZES.get(action_id, 0.5),
        }

    # Summarize by phase
    phase_summary = {}
    for phase, evts in by_phase.items():
        distances = [e["distance"] for e in evts]
        phase_summary[phase] = {
            "count": len(evts),
            "avg_distance": round(sum(distances) / len(distances), 3),
        }

    # Worst gaps
    worst = sorted(entries, key=lambda e: e["distance"], reverse=True)[:10]

    return {
        "total_events": len(entries),
        "by_mapped_action": action_summary,
        "by_phase": phase_summary,
        "worst_gaps": worst,
    }


def summarize_pain_zones(bridge_log: list[tuple]) -> list[dict]:
    """Identify concrete ratio ranges that suffer the worst translation.

    Bins concrete ratios into 0.1-wide bands and reports which bands
    have the highest average translation distance.

    Returns sorted list of {ratio_band, count, avg_distance, mapped_to_actions}.
    """
    bands = defaultdict(list)
    for concrete_ratio, mapped_action, phase, bucket in bridge_log:
        band = round(concrete_ratio * 10) / 10  # bin to nearest 0.1
        nominal = _ABSTRACT_SIZES.get(mapped_action, 0.5)
        distance = abs(concrete_ratio - nominal)
        bands[band].append({
            "distance": distance,
            "mapped_action": ACTION_NAMES.get(mapped_action, f"action_{mapped_action}"),
        })

    zones = []
    for band, evts in bands.items():
        actions_used = list(set(e["mapped_action"] for e in evts))
        zones.append({
            "ratio_band": band,
            "count": len(evts),
            "avg_distance": round(sum(e["distance"] for e in evts) / len(evts), 3),
            "mapped_to_actions": actions_used,
        })

    zones.sort(key=lambda z: z["avg_distance"], reverse=True)
    return zones


def format_pain_map(analysis: dict) -> str:
    """Format bridge pain analysis for display."""
    lines = [f"=== Bridge Pain Map ({analysis['total_events']} translation events) ===\n"]

    lines.append("By Mapped Action:")
    for name, stats in sorted(analysis["by_mapped_action"].items(),
                              key=lambda x: x[1]["avg_distance"], reverse=True):
        lines.append(f"  {name:20s}  count={stats['count']:4d}  "
                     f"avg_dist={stats['avg_distance']:.3f}  "
                     f"max_dist={stats['max_distance']:.3f}  "
                     f"concrete={stats['concrete_range']}  "
                     f"nominal={stats['nominal']:.2f}")

    lines.append("\nBy Phase:")
    for phase, stats in sorted(analysis["by_phase"].items()):
        lines.append(f"  {phase:10s}  count={stats['count']:4d}  "
                     f"avg_dist={stats['avg_distance']:.3f}")

    lines.append("\nWorst Gaps (top 10):")
    for g in analysis["worst_gaps"]:
        lines.append(f"  concrete={g['concrete_ratio']:.3f} -> "
                     f"{g['mapped_name']} (nominal={g['nominal_ratio']:.2f}) "
                     f"dist={g['distance']:.3f} "
                     f"phase={g['phase']} bucket={g['bucket']}")

    return "\n".join(lines)
