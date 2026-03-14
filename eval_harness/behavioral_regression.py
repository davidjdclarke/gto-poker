"""
Behavioral regression suite for GTO solver ablation experiments.

Defines 6 strategically meaningful node families and audits their
action distributions across experiments. Produces comparable metrics
for each family: action distribution, entropy, fold/call/raise split,
and delta vs a pinned baseline.
"""
import json
import math
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from server.gto.cfr import CFRTrainer
from server.gto.abstraction import (
    Action, HandType, decode_bucket, get_available_actions,
    count_raises, NUM_HAND_TYPES,
)

PHASES = ['preflop', 'flop', 'turn', 'river']

BET_ACTIONS = {
    int(Action.BET_QUARTER_POT), int(Action.BET_THIRD_POT),
    int(Action.BET_HALF_POT),
    int(Action.BET_TWO_THIRDS_POT), int(Action.BET_THREE_QUARTER_POT),
    int(Action.BET_POT),
    int(Action.BET_OVERBET), int(Action.BET_DOUBLE_POT),
    int(Action.ALL_IN),
    int(Action.OPEN_RAISE), int(Action.THREE_BET), int(Action.FOUR_BET),
    int(Action.DONK_SMALL), int(Action.DONK_MEDIUM),
}


def _parse_key(key: str):
    """Parse infoset key -> (phase, position, bucket, eq_bucket, hand_type, history)."""
    parts = key.split(':')
    if len(parts) < 3:
        return None
    phase = parts[0]
    if len(parts) == 4:
        position = parts[1]
        bucket = int(parts[2])
        history_str = parts[3]
    else:
        position = None
        bucket = int(parts[1])
        history_str = parts[-1]
    eq_bucket, hand_type = decode_bucket(bucket)
    history = tuple(int(c) for c in history_str) if history_str else ()
    return phase, position, bucket, eq_bucket, hand_type, history


def _has_bet(history):
    """Check if last action in history is a bet/raise."""
    if not history:
        return False
    return history[-1] in BET_ACTIONS


def _entropy(probs):
    """Shannon entropy of a probability distribution."""
    return -sum(p * math.log2(p) for p in probs if p > 1e-10)


@dataclass
class FamilyReport:
    name: str
    num_nodes: int = 0
    action_dist: dict = field(default_factory=dict)  # action_name -> mean prob
    entropy: float = 0.0
    fold_pct: float = 0.0
    call_pct: float = 0.0
    raise_pct: float = 0.0
    delta: dict = field(default_factory=dict)  # action_name -> delta vs baseline


# --- Node family filters ---

def _filter_preflop_premium(phase, position, eq_bucket, hand_type, history):
    """Preflop unopened premium hands (OOP)."""
    return (phase == 'preflop' and
            position == 'oop' and
            not history and
            hand_type in (int(HandType.PREMIUM_PAIR), int(HandType.HIGH_PAIR)) and
            eq_bucket >= 5)


def _filter_jam_over_open(phase, position, eq_bucket, hand_type, history):
    """Preflop jam-over-open spots (IP facing open raise)."""
    return (phase == 'preflop' and
            position == 'ip' and
            len(history) >= 1 and
            history[-1] == int(Action.OPEN_RAISE))


def _filter_flop_low_eq_probe(phase, position, eq_bucket, hand_type, history):
    """Flop low-equity probe spots (OOP, donk-eligible)."""
    return (phase == 'flop' and
            position == 'oop' and
            not history and
            eq_bucket <= 1)


def _filter_river_bluff(phase, position, eq_bucket, hand_type, history):
    """River bluff-capable spots (low equity, facing or making bet)."""
    return (phase == 'river' and
            eq_bucket <= 2 and
            len(history) >= 1)


def _filter_river_thin_value(phase, position, eq_bucket, hand_type, history):
    """River thin-value spots (medium-high equity)."""
    return (phase == 'river' and
            eq_bucket in (4, 5) and
            len(history) >= 1)


def _filter_overbet_defense(phase, position, eq_bucket, hand_type, history):
    """Overbet-defense spots (facing BET_OVERBET)."""
    return (phase in ('flop', 'turn', 'river') and
            len(history) >= 1 and
            history[-1] == int(Action.BET_OVERBET))


NODE_FAMILIES = [
    ("preflop_premium", _filter_preflop_premium),
    ("jam_over_open", _filter_jam_over_open),
    ("flop_low_eq_probe", _filter_flop_low_eq_probe),
    ("river_bluff", _filter_river_bluff),
    ("river_thin_value", _filter_river_thin_value),
    ("overbet_defense", _filter_overbet_defense),
]


def _audit_family(trainer: CFRTrainer, name: str, filter_fn) -> FamilyReport:
    """Audit a single node family."""
    report = FamilyReport(name=name)

    # Collect action distributions across matching nodes
    action_sums = {}  # action_int -> sum of probs
    total_entropy = 0.0
    fold_sum = 0.0
    call_sum = 0.0
    raise_sum = 0.0
    count = 0

    for key, node in trainer.nodes.items():
        parsed = _parse_key(key)
        if parsed is None:
            continue
        phase, position, bucket, eq_bucket, hand_type, history = parsed

        # Skip rarely visited
        visit_weight = float(node.strategy_sum.sum())
        if visit_weight < 100:
            continue

        if not filter_fn(phase, position, eq_bucket, hand_type, history):
            continue

        has_bet_flag = _has_bet(history)
        raise_count = count_raises(history, phase)
        can_raise = raise_count < 4
        actions = get_available_actions(has_bet_flag, can_raise, phase,
                                        raise_count,
                                        history_len=len(history),
                                        eq_bucket=eq_bucket)
        avg = node.get_average_strategy()

        # Handle action/node size mismatch (training may have used different actions)
        if len(actions) != len(avg):
            # Try with FOLD prepended (common mismatch)
            if len(avg) == len(actions) + 1:
                actions = [Action.FOLD] + list(actions)
            else:
                continue

        probs = [float(avg[i]) for i in range(len(actions))]
        total_entropy += _entropy(probs)

        for i, action in enumerate(actions):
            a = int(action)
            action_sums[a] = action_sums.get(a, 0.0) + probs[i]
            if a == int(Action.FOLD):
                fold_sum += probs[i]
            elif a == int(Action.CHECK_CALL):
                call_sum += probs[i]
            else:
                raise_sum += probs[i]

        count += 1

    if count == 0:
        return report

    report.num_nodes = count
    report.entropy = total_entropy / count

    # Normalize action distribution
    action_names = {int(a): a.name for a in Action}
    for a, s in action_sums.items():
        aname = action_names.get(a, f"action_{a}")
        report.action_dist[aname] = round(s / count, 4)

    total = fold_sum + call_sum + raise_sum
    if total > 0:
        report.fold_pct = round(fold_sum / total * 100, 1)
        report.call_pct = round(call_sum / total * 100, 1)
        report.raise_pct = round(raise_sum / total * 100, 1)

    return report


def run_behavioral_regression(trainer: CFRTrainer,
                               baseline_path: str = None) -> dict:
    """
    Run behavioral regression across all 6 node families.

    Args:
        trainer: Trained CFRTrainer to audit.
        baseline_path: Path to baseline behavioral_regression.json.
            If provided, computes deltas for each family.

    Returns:
        Dict with family_name -> FamilyReport (as dict).
    """
    # Load baseline if provided
    baseline = {}
    if baseline_path and Path(baseline_path).exists():
        with open(baseline_path) as f:
            baseline = json.load(f)

    results = {}
    for name, filter_fn in NODE_FAMILIES:
        report = _audit_family(trainer, name, filter_fn)

        # Compute deltas vs baseline
        if name in baseline:
            base = baseline[name]
            base_dist = base.get('action_dist', {})
            for aname, prob in report.action_dist.items():
                base_prob = base_dist.get(aname, 0.0)
                report.delta[aname] = round(prob - base_prob, 4)

        results[name] = asdict(report)

    return results


def save_behavioral_baseline(trainer: CFRTrainer, path: str):
    """Save behavioral regression results as a baseline snapshot."""
    results = run_behavioral_regression(trainer)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    return results


def print_behavioral_report(results: dict):
    """Print a formatted behavioral regression report."""
    print("\n╔══════════════════════════════════════════════════╗")
    print("║         Behavioral Regression Report             ║")
    print("╠══════════════════════════════════════════════════╣")
    for name, report in results.items():
        n = report.get('num_nodes', 0)
        ent = report.get('entropy', 0)
        fold = report.get('fold_pct', 0)
        call = report.get('call_pct', 0)
        raise_pct = report.get('raise_pct', 0)
        print(f"\n  {name} ({n} nodes, entropy={ent:.2f})")
        print(f"    Fold/Call/Raise: {fold:.1f}% / {call:.1f}% / {raise_pct:.1f}%")

        # Action distribution
        dist = report.get('action_dist', {})
        delta = report.get('delta', {})
        if dist:
            top = sorted(dist.items(), key=lambda x: -x[1])[:5]
            parts = []
            for aname, prob in top:
                d = delta.get(aname)
                if d is not None and abs(d) > 0.005:
                    parts.append(f"{aname}={prob:.3f}({d:+.3f})")
                else:
                    parts.append(f"{aname}={prob:.3f}")
            print(f"    Top actions: {', '.join(parts)}")

    print("\n╚══════════════════════════════════════════════════╝")
