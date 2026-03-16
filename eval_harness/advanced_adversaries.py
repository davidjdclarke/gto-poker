"""
Advanced gauntlet: policy-distorted bots for deeper GTO evaluation.

Each bot starts from the trained GTO base strategy and applies multiplicative
reweighting over action families before sampling. This preserves strategic
coherence (action support, relative ordering) while introducing controlled bias.

4 styles × 3 intensities = 12 bots.

Usage:
    from eval_harness.advanced_adversaries import get_advanced_adversaries
    bots = get_advanced_adversaries(trainer)   # list of 12 Agent instances

CLI:
    venv/bin/python run_eval_harness.py --gauntlet advanced --hands 2000 --seeds 42,123,456
"""
import random
import math
from dataclasses import dataclass, field

from server.gto.abstraction import Action
from eval_harness.match_engine import Agent, AgentDecision, HandContext, GTOAgent


# ---------------------------------------------------------------------------
# Action family definitions
# ---------------------------------------------------------------------------
# Maps family name → list of Action int values.
# Used for per-family multiplier lookup.
_FAMILY_MAP: dict[str, list[int]] = {
    "fold":          [int(Action.FOLD)],
    "passive":       [int(Action.CHECK_CALL)],
    "small_bet":     [int(Action.BET_QUARTER_POT), int(Action.BET_THIRD_POT),
                      int(Action.DONK_SMALL)],
    "medium_bet":    [int(Action.BET_HALF_POT), int(Action.BET_TWO_THIRDS_POT),
                      int(Action.BET_THREE_QUARTER_POT), int(Action.DONK_MEDIUM)],
    "large_bet":     [int(Action.BET_POT), int(Action.BET_OVERBET),
                      int(Action.BET_DOUBLE_POT)],
    "all_in":        [int(Action.ALL_IN)],
    "preflop_raise": [int(Action.OPEN_RAISE), int(Action.THREE_BET),
                      int(Action.FOUR_BET)],
}

# Reverse lookup: action_int → family name
_ACTION_TO_FAMILY: dict[int, str] = {}
for _family, _actions in _FAMILY_MAP.items():
    for _a in _actions:
        _ACTION_TO_FAMILY[_a] = _family


def _action_to_family(action_id: int) -> str:
    return _ACTION_TO_FAMILY.get(action_id, "passive")


# ---------------------------------------------------------------------------
# Distortion profiles
# ---------------------------------------------------------------------------
# Strong-intensity multipliers per style.
# Mild  = base + 0.4 × (strong_mult - 1.0)
# Medium = base + 0.7 × (strong_mult - 1.0)
# Strong = as listed below

_STRONG_MULTS: dict[str, dict[str, float]] = {
    "aggressive": {
        "fold":          0.50,
        "passive":       0.70,
        "small_bet":     0.80,
        "medium_bet":    1.20,
        "large_bet":     1.80,
        "all_in":        1.40,
        "preflop_raise": 1.80,
    },
    "nit": {
        "fold":          2.50,
        "passive":       1.20,
        "small_bet":     1.00,
        "medium_bet":    0.60,
        "large_bet":     0.30,
        "all_in":        0.20,
        "preflop_raise": 0.40,
    },
    "station": {
        "fold":          0.20,
        "passive":       2.00,
        "small_bet":     1.20,
        "medium_bet":    1.10,
        "large_bet":     0.80,
        "all_in":        0.60,
        "preflop_raise": 0.60,
    },
    "overfolder": {
        "fold":          3.00,
        "passive":       0.80,
        "small_bet":     0.70,
        "medium_bet":    0.70,
        "large_bet":     0.70,
        "all_in":        0.50,
        "preflop_raise": 1.00,
    },
}

_INTENSITY_SCALE = {"mild": 0.4, "medium": 0.7, "strong": 1.0}


def _build_multipliers(style: str, intensity: str) -> dict[str, float]:
    """Compute per-family multiplier dict for a given style and intensity."""
    scale = _INTENSITY_SCALE[intensity]
    strong = _STRONG_MULTS[style]
    return {
        family: 1.0 + scale * (mult - 1.0)
        for family, mult in strong.items()
    }


# ---------------------------------------------------------------------------
# Distortion math
# ---------------------------------------------------------------------------
_MIN_SUPPORT = 0.01


def _distort(strategy: dict, mults: dict, phase: str) -> dict:
    """Apply multiplicative reweighting to a strategy dict.

    Args:
        strategy: {action_int: prob}
        mults:    {family: multiplier}
        phase:    "preflop" | "flop" | "turn" | "river"

    Returns:
        Renormalized dict with same keys. Actions with prob=0 stay at 0.
        Actions with prob>0 are floored at _MIN_SUPPORT after reweighting.
    """
    out = {}
    for action_id, prob in strategy.items():
        if prob <= 0.0:
            out[action_id] = 0.0
            continue
        family = _action_to_family(action_id)
        w = mults.get(family, 1.0)
        out[action_id] = max(prob * w, _MIN_SUPPORT)

    total = sum(out.values())
    if total <= 0.0:
        # All-zero fallback: uniform over positive-prob actions
        positives = [a for a, p in strategy.items() if p > 0.0]
        if positives:
            u = 1.0 / len(positives)
            return {a: (u if a in positives else 0.0) for a in strategy}
        return strategy
    return {a: p / total for a, p in out.items()}


# ---------------------------------------------------------------------------
# PolicyDistortedBot
# ---------------------------------------------------------------------------
class PolicyDistortedBot(Agent):
    """GTO-distorted bot that biases action frequencies via multiplicative reweighting.

    Holds an internal GTOAgent to compute the base strategy, then applies
    distortion before sampling. The external interface is a plain Agent.

    Attributes:
        style:     "aggressive" | "nit" | "station" | "overfolder"
        intensity: "mild" | "medium" | "strong"
        mults:     per-family multipliers (logged for auditability)
    """

    def __init__(self, trainer, style: str, intensity: str,
                 base_mapping: str = "confidence_nearest"):
        self.style = style
        self.intensity = intensity
        self.mults = _build_multipliers(style, intensity)
        self.name = f"{style.capitalize()}_{intensity}"
        self._inner = GTOAgent(trainer, name=f"_inner_{style}", mapping=base_mapping)
        # Diagnostics
        self._decisions_made = 0
        self._family_freq: dict[str, int] = {f: 0 for f in _FAMILY_MAP}

    def decide(self, ctx: HandContext) -> AgentDecision:
        base_strategy = self._inner.compute_strategy(ctx)
        distorted = _distort(base_strategy, self.mults, ctx.phase)

        action_ids = list(distorted.keys())
        weights = [max(0.0, distorted[a]) for a in action_ids]
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(action_ids)] * len(action_ids)

        chosen = random.choices(action_ids, weights=weights, k=1)[0]

        # Track family frequencies for diagnostics
        fam = _action_to_family(chosen)
        self._family_freq[fam] = self._family_freq.get(fam, 0) + 1
        self._decisions_made += 1

        from server.gto.abstraction import Action as A
        from eval_harness.match_engine import _abstract_to_concrete
        return _abstract_to_concrete(A(chosen), ctx)

    def profile_summary(self) -> dict:
        """Return per-family frequency summary (for logging)."""
        if self._decisions_made == 0:
            return {}
        return {
            "name": self.name,
            "style": self.style,
            "intensity": self.intensity,
            "multipliers": dict(self.mults),
            "family_frequencies": {
                f: round(c / self._decisions_made, 3)
                for f, c in self._family_freq.items()
                if c > 0
            },
            "decisions": self._decisions_made,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_advanced_adversaries(trainer, base_mapping: str = "confidence_nearest"
                             ) -> list[PolicyDistortedBot]:
    """Return the 12-bot advanced adversary pool.

    4 styles × 3 intensities = 12 PolicyDistortedBot instances.
    All share the same trainer but have independent GTOAgent instances.

    Args:
        trainer:      CFRTrainer with loaded strategy
        base_mapping: mapping mode for the inner GTOAgent (default: confidence_nearest)

    Returns:
        list of 12 PolicyDistortedBot instances, ordered style-major, intensity-minor
    """
    styles = ("aggressive", "nit", "station", "overfolder")
    intensities = ("mild", "medium", "strong")
    return [
        PolicyDistortedBot(trainer, style, intensity, base_mapping=base_mapping)
        for style in styles
        for intensity in intensities
    ]


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def robustness_score(bot_results: dict[str, float]) -> float:
    """Compute mean - 0.5 * std across all advanced bot EVs.

    Args:
        bot_results: {bot_name: bb_per_100}

    Returns:
        float: robustness score (higher = more robust)
    """
    values = list(bot_results.values())
    if not values:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    return mean - 0.5 * std


def advanced_gauntlet_summary(bot_results: dict[str, float]) -> dict:
    """Compute comprehensive summary metrics for the advanced gauntlet.

    Args:
        bot_results: {bot_name: bb_per_100} — bot_name is "Style_intensity"

    Returns:
        dict with: avg, worst_case, fraction_beaten, robustness_score,
                   by_style, by_intensity
    """
    if not bot_results:
        return {}

    values = list(bot_results.values())
    avg = sum(values) / len(values)
    worst = min(values)
    fraction_beaten = sum(1 for v in values if v > 0) / len(values)

    by_style: dict[str, list[float]] = {}
    by_intensity: dict[str, list[float]] = {}
    for name, ev in bot_results.items():
        # Name format: "Style_intensity" e.g. "Aggressive_mild"
        parts = name.split("_", 1)
        if len(parts) == 2:
            style, intensity = parts[0].lower(), parts[1].lower()
            by_style.setdefault(style, []).append(ev)
            by_intensity.setdefault(intensity, []).append(ev)

    return {
        "avg": round(avg, 1),
        "worst_case": round(worst, 1),
        "fraction_beaten": round(fraction_beaten, 3),
        "robustness_score": round(robustness_score(bot_results), 1),
        "by_style": {s: round(sum(vs) / len(vs), 1) for s, vs in by_style.items()},
        "by_intensity": {i: round(sum(vs) / len(vs), 1)
                         for i, vs in by_intensity.items()},
    }
