"""
Translation-confidence blending for off-tree action handling.

When facing off-tree actions, the solver's trained strategy may be unreliable
because the mapped abstract action differs from what the opponent actually did.
This module computes a confidence score and blends the trained strategy with
a simple equity-based heuristic.

The blend formula:
    pi_final = (1 - alpha) * pi_trained + alpha * pi_heuristic

where alpha in [0, MAX_ALPHA] grows with:
    - larger action-size mismatch (opponent's bet doesn't match any abstract size)
    - lower node visit count (undertrained region)
    - higher strategy entropy (uncertain trained strategy)
"""
import math
from server.gto.abstraction import Action

# Maximum blend weight toward heuristic
MAX_ALPHA = 0.50

# Visit count threshold for full confidence (log scale)
VISIT_CONFIDENCE_CENTER = 10.0  # log(visit_count) at 50% confidence
VISIT_CONFIDENCE_SLOPE = 2.0    # steepness of sigmoid

# Bet size ratios for each abstract action (fraction of pot)
ABSTRACT_BET_SIZES = {
    int(Action.BET_THIRD_POT): 0.33,
    int(Action.BET_HALF_POT): 0.50,
    int(Action.BET_TWO_THIRDS_POT): 0.67,
    int(Action.BET_POT): 1.00,
    int(Action.BET_OVERBET): 1.25,
    int(Action.DONK_SMALL): 0.25,
    int(Action.DONK_MEDIUM): 0.50,
    int(Action.ALL_IN): 3.00,  # approximate
}


def compute_confidence(strategy: dict[int, float],
                       visit_count: float,
                       concrete_bet_ratio: float = None,
                       mapped_action: int = None) -> float:
    """
    Compute alpha (blend weight toward heuristic).

    Args:
        strategy: Trained strategy dict {action_int: probability}.
        visit_count: Node visit weight (strategy_sum.sum()).
        concrete_bet_ratio: Actual bet as fraction of pot (None if no bet).
        mapped_action: Abstract action the concrete bet was mapped to.

    Returns:
        alpha in [0, MAX_ALPHA]. 0 = fully trust trained, MAX_ALPHA = max heuristic.
    """
    # 1. Visit confidence: sigmoid of log(visit_count)
    if visit_count > 0:
        log_visits = math.log(visit_count + 1)
        visit_confidence = 1.0 / (1.0 + math.exp(
            -(log_visits - VISIT_CONFIDENCE_CENTER) / VISIT_CONFIDENCE_SLOPE))
    else:
        visit_confidence = 0.0

    # 2. Strategy entropy factor: high entropy = uncertain
    probs = [p for p in strategy.values() if p > 1e-10]
    if probs:
        max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
        entropy = -sum(p * math.log2(p) for p in probs)
        entropy_factor = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        entropy_factor = 1.0  # no strategy = max uncertainty

    # 3. Action mismatch: how far is the concrete bet from the mapped abstract bet
    mismatch_penalty = 0.0
    if concrete_bet_ratio is not None and mapped_action is not None:
        abstract_ratio = ABSTRACT_BET_SIZES.get(mapped_action, None)
        if abstract_ratio is not None:
            # Relative difference
            diff = abs(concrete_bet_ratio - abstract_ratio) / max(abstract_ratio, 0.1)
            mismatch_penalty = min(diff, 2.0) / 2.0  # normalize to [0, 1]

    # Combine: alpha grows when any component indicates low confidence
    alpha = mismatch_penalty * (1.0 - visit_confidence) * (0.5 + 0.5 * entropy_factor)

    # Also add a baseline from visit confidence alone
    alpha += (1.0 - visit_confidence) * 0.1

    return min(alpha, MAX_ALPHA)


def equity_heuristic(eq_bucket: int, has_bet: bool,
                     phase: str) -> dict[int, float]:
    """
    Simple equity-based heuristic strategy.

    Maps equity to action: fold trash, check/call medium, bet strong.
    This is deliberately simple — it's the fallback when trained strategy
    is unreliable, not a replacement.

    Args:
        eq_bucket: Equity bucket 0-7 (0=weakest, 7=strongest).
        has_bet: Whether there's a bet to call.
        phase: Game phase (preflop, flop, turn, river).

    Returns:
        Strategy dict {action_int: probability}.
    """
    if has_bet:
        if eq_bucket <= 1:
            # Trash: mostly fold, small bluff raise
            return {
                int(Action.FOLD): 0.85,
                int(Action.CHECK_CALL): 0.10,
                int(Action.BET_HALF_POT): 0.05,  # bluff raise
            }
        elif eq_bucket <= 3:
            # Weak: lean fold, some call
            return {
                int(Action.FOLD): 0.50,
                int(Action.CHECK_CALL): 0.45,
                int(Action.BET_HALF_POT): 0.05,
            }
        elif eq_bucket <= 5:
            # Medium: mostly call, some raise
            return {
                int(Action.FOLD): 0.05,
                int(Action.CHECK_CALL): 0.75,
                int(Action.BET_HALF_POT): 0.15,
                int(Action.BET_POT): 0.05,
            }
        else:
            # Strong: raise for value
            return {
                int(Action.FOLD): 0.0,
                int(Action.CHECK_CALL): 0.30,
                int(Action.BET_POT): 0.50,
                int(Action.ALL_IN): 0.20,
            }
    else:
        if eq_bucket <= 1:
            # Trash: check, occasional bluff
            return {
                int(Action.CHECK_CALL): 0.85,
                int(Action.BET_THIRD_POT): 0.15,
            }
        elif eq_bucket <= 3:
            # Weak: mostly check
            return {
                int(Action.CHECK_CALL): 0.70,
                int(Action.BET_THIRD_POT): 0.20,
                int(Action.BET_HALF_POT): 0.10,
            }
        elif eq_bucket <= 5:
            # Medium: bet for value
            return {
                int(Action.CHECK_CALL): 0.40,
                int(Action.BET_HALF_POT): 0.35,
                int(Action.BET_POT): 0.25,
            }
        else:
            # Strong: bet big for value
            return {
                int(Action.CHECK_CALL): 0.15,
                int(Action.BET_POT): 0.50,
                int(Action.BET_OVERBET): 0.20,
                int(Action.ALL_IN): 0.15,
            }


def blend_strategies(trained: dict[int, float],
                     heuristic: dict[int, float],
                     alpha: float) -> dict[int, float]:
    """
    Blend trained strategy with heuristic.

    Actions in heuristic but not in trained are added.
    Actions in trained but not in heuristic keep their weight.
    Result is renormalized.
    """
    all_actions = set(trained.keys()) | set(heuristic.keys())
    blended = {}
    for a in all_actions:
        t = trained.get(a, 0.0)
        h = heuristic.get(a, 0.0)
        blended[a] = (1.0 - alpha) * t + alpha * h

    # Renormalize
    total = sum(blended.values())
    if total > 0:
        blended = {a: p / total for a, p in blended.items()}
    return blended
