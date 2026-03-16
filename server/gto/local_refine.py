"""
Local refinement for off-tree action handling (v11 Workstream C2).

When the opponent bets a size not in the abstract action set, the blueprint
strategy answers the wrong question. Local refinement runs a small CFR solve
at the current decision point using the blueprint as a prior, adapting to the
actual bet size.

Scope: turn and river only, triggered by high bridge-pain mismatch.

References:
    Brown & Sandholm, "Safe and Nested Subgame Solving for Imperfect-Information
    Games", AAAI 2017 — safe subgame solving with blueprint gadget.
"""
import math
import random
import numpy as np
from server.gto.abstraction import (
    Action, InfoSet, get_available_actions, count_raises,
    decode_bucket, make_bucket, NUM_HAND_TYPES,
)
from server.gto.cfr import CFRNode


# --- Configuration ---

# Maximum CFR iterations per refinement (controls latency)
MAX_REFINE_ITERS = 200

# Minimum mismatch to trigger refinement (fraction of pot)
# Raised from 0.20 to 0.40 after v11 gauntlet showed over-triggering
MISMATCH_THRESHOLD = 0.40

# Minimum visit-weight below which refinement is worth the cost
# Raised from 50 to 200 — trust well-trained nodes more
VISIT_THRESHOLD = 200.0

# Phases where refinement is allowed
REFINE_PHASES = ('turn', 'river')

# Number of opponent buckets to sample during refinement
OPP_SAMPLES = 30

# How much to blend refinement vs blueprint (0=pure blueprint, 1=pure refine)
REFINE_BLEND_ALPHA = 0.5


def should_refine(concrete_bet_ratio: float,
                  mapped_action: int,
                  phase: str,
                  visit_count: float,
                  pain_families: set[int] | None = None) -> bool:
    """Decide whether to invoke local refinement for this decision.

    Trigger policy (v11-C3): invoke only when mismatch is large enough
    to justify the compute cost and the blueprint is undertrained.

    Args:
        concrete_bet_ratio: Actual bet as fraction of pot
        mapped_action: Abstract action the bet was mapped to
        phase: Game phase
        visit_count: Node visit weight from strategy_sum
        pain_families: Set of action IDs from top bridge pain families
                       (precomputed from bridge_pain.summarize_pain_zones).
                       If provided, actions in this set get lower thresholds.
    """
    if phase not in REFINE_PHASES:
        return False

    if concrete_bet_ratio is None or concrete_bet_ratio <= 0:
        return False

    # Check mismatch
    from eval_harness.confidence import ABSTRACT_BET_SIZES
    nominal = ABSTRACT_BET_SIZES.get(mapped_action)
    if nominal is None:
        return False

    distance = abs(concrete_bet_ratio - nominal)

    # Lower threshold for known high-pain action families
    threshold = MISMATCH_THRESHOLD
    if pain_families and mapped_action in pain_families:
        threshold = MISMATCH_THRESHOLD * 0.5  # More aggressive for known pain spots

    if distance < threshold:
        return False

    # Only refine when the node is relatively undertrained
    if visit_count > VISIT_THRESHOLD:
        # Well-trained node: blueprint is likely reliable despite mismatch
        # Still refine if mismatch is extreme
        if distance < 0.50:
            return False

    return True


def build_pain_families(bridge_log: list[tuple],
                        top_n: int = 5) -> set[int]:
    """Extract the top-N highest-pain action families from bridge log.

    Used by the trigger policy to give preferential refinement treatment
    to actions that historically produce the worst translation errors.

    Args:
        bridge_log: list of (concrete_ratio, mapped_action, phase, bucket)
        top_n: Number of top pain families to include

    Returns:
        Set of action IDs (ints) that are in the top pain families
    """
    from collections import defaultdict
    from eval_harness.confidence import ABSTRACT_BET_SIZES

    pain_by_action = defaultdict(list)
    for concrete_ratio, mapped_action, phase, bucket in bridge_log:
        if phase not in REFINE_PHASES:
            continue
        nominal = ABSTRACT_BET_SIZES.get(mapped_action, 0.5)
        distance = abs(concrete_ratio - nominal)
        pain_by_action[mapped_action].append(distance)

    # Rank by average pain
    ranked = []
    for action_id, distances in pain_by_action.items():
        avg_pain = sum(distances) / len(distances)
        ranked.append((action_id, avg_pain, len(distances)))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return {r[0] for r in ranked[:top_n]}


class LocalRefiner:
    """Runs a miniature CFR solve at a single decision point.

    The solver treats the current decision as the root of a tiny subgame.
    The opponent's range is approximated by sampling buckets. The blueprint
    strategy initializes regrets (warm start).

    This is a heuristic approximation of safe subgame solving, not a full
    implementation. It trades theoretical safety guarantees for simplicity
    and speed.
    """

    def __init__(self, trainer, num_iters: int = MAX_REFINE_ITERS,
                 opp_samples: int = OPP_SAMPLES):
        self.trainer = trainer
        self.num_iters = num_iters
        self.opp_samples = opp_samples
        # Diagnostics
        self.refine_count = 0
        self.total_iters = 0

    def refine_strategy(self, phase: str, bucket: int,
                        history: tuple, position: str,
                        concrete_bet_ratio: float,
                        available_actions: list) -> dict[int, float]:
        """Compute a refined strategy for a single decision point.

        Instead of looking up the blueprint for the *mapped* action,
        we run a mini-CFR at this node that accounts for the actual
        pot odds created by the concrete bet.

        Args:
            phase: 'turn' or 'river'
            bucket: Player's 2D bucket (0-119)
            history: Abstract history tuple leading to this point
            position: 'oop' or 'ip'
            concrete_bet_ratio: Actual bet as fraction of pot
            available_actions: List of Action enums available

        Returns:
            Strategy dict {action_int: probability}
        """
        self.refine_count += 1
        n_actions = len(available_actions)
        action_ints = [int(a) for a in available_actions]

        # Initialize regrets from blueprint (warm start)
        info_set = InfoSet(bucket, phase, history, position=position)
        key = info_set.key
        node = self.trainer.nodes.get(key)

        regret_sum = np.zeros(n_actions, dtype=np.float64)
        strategy_sum = np.zeros(n_actions, dtype=np.float64)

        if node is not None:
            avg = node.get_average_strategy()
            if len(avg) == n_actions:
                # Seed strategy_sum from blueprint
                strategy_sum = avg * 100.0  # bootstrap weight

        # Sample opponent buckets for counterfactual computation
        eq_bucket, hand_type = decode_bucket(bucket)
        opp_buckets = self._sample_opponent_buckets(eq_bucket)

        # Mini-CFR loop
        for t in range(self.num_iters):
            # Current strategy via regret matching
            pos_regrets = np.maximum(regret_sum, 0)
            total = pos_regrets.sum()
            if total > 0:
                strategy = pos_regrets / total
            else:
                strategy = np.ones(n_actions) / n_actions

            # Compute action utilities via simplified payoff model
            utilities = np.zeros(n_actions, dtype=np.float64)
            for opp_b in opp_buckets:
                action_values = self._compute_action_values(
                    action_ints, phase, bucket, opp_b,
                    history, concrete_bet_ratio)
                utilities += action_values

            utilities /= len(opp_buckets)

            # Node utility
            node_util = (strategy * utilities).sum()

            # Update regrets (CFR+)
            regrets = utilities - node_util
            regret_sum = np.maximum(regret_sum + regrets, 0)

            # Accumulate strategy with linear weighting
            weight = max(t - 10, 0)  # small delay
            strategy_sum += weight * strategy

        self.total_iters += self.num_iters

        # Return average strategy
        total = strategy_sum.sum()
        if total > 0:
            avg_strat = strategy_sum / total
        else:
            avg_strat = np.ones(n_actions) / n_actions

        return {action_ints[i]: float(avg_strat[i]) for i in range(n_actions)}

    def _sample_opponent_buckets(self, our_eq: int) -> list[int]:
        """Sample plausible opponent buckets given our equity bucket."""
        buckets = []
        for _ in range(self.opp_samples):
            # Opponent equity is roughly inverse-correlated with ours
            # but with significant noise
            opp_eq = random.randint(0, 7)
            opp_ht = random.randint(0, NUM_HAND_TYPES - 1)
            buckets.append(make_bucket(opp_eq, opp_ht))
        return buckets

    def _compute_action_values(self, actions: list[int],
                               phase: str, our_bucket: int,
                               opp_bucket: int, history: tuple,
                               concrete_bet_ratio: float) -> np.ndarray:
        """Estimate action values using simplified payoff model.

        This is a heuristic approximation: we use equity comparison
        and pot odds to estimate the value of each action, adjusted
        for the actual concrete bet size.

        Args:
            actions: Available action ints
            phase: Game phase
            our_bucket: Our 2D bucket
            opp_bucket: Opponent's 2D bucket
            history: Current history
            concrete_bet_ratio: Actual bet as fraction of pot

        Returns:
            np.array of estimated values per action
        """
        our_eq = our_bucket // NUM_HAND_TYPES
        opp_eq = opp_bucket // NUM_HAND_TYPES
        n = len(actions)
        values = np.zeros(n, dtype=np.float64)

        # Win probability (simplified from equity comparison)
        if our_eq > opp_eq:
            win_prob = 0.65 + 0.05 * (our_eq - opp_eq)
        elif our_eq < opp_eq:
            win_prob = 0.35 - 0.05 * (opp_eq - our_eq)
        else:
            win_prob = 0.50
        win_prob = max(0.05, min(0.95, win_prob))

        # Pot odds from the concrete bet
        # pot_odds = amount_to_call / (pot + amount_to_call)
        pot_odds = concrete_bet_ratio / (1.0 + concrete_bet_ratio)

        for i, action in enumerate(actions):
            if action == int(Action.FOLD):
                # Lose current investment (normalized)
                values[i] = -0.5
            elif action == int(Action.CHECK_CALL):
                # Call: pay concrete_bet_ratio, win (1 + 2*bet) with win_prob
                call_cost = concrete_bet_ratio
                pot_if_win = 1.0 + 2.0 * concrete_bet_ratio
                values[i] = win_prob * pot_if_win - call_cost
            elif action == int(Action.ALL_IN):
                # All-in: high variance, good with strong equity
                allin_size = 3.0  # approximate
                values[i] = win_prob * (1.0 + 2.0 * allin_size) - allin_size
                # Fold equity (opponent folds sometimes)
                opp_fold_prob = max(0, 0.4 - 0.05 * opp_eq)
                values[i] = opp_fold_prob * 0.5 + (1 - opp_fold_prob) * values[i]
            else:
                # Bet/raise actions: use the action's pot fraction
                bet_sizes = {
                    int(Action.BET_QUARTER_POT): 0.25,
                    int(Action.BET_THIRD_POT): 0.33,
                    int(Action.BET_HALF_POT): 0.50,
                    int(Action.BET_TWO_THIRDS_POT): 0.67,
                    int(Action.BET_THREE_QUARTER_POT): 0.75,
                    int(Action.BET_POT): 1.00,
                    int(Action.BET_OVERBET): 1.25,
                    int(Action.BET_DOUBLE_POT): 2.00,
                    int(Action.DONK_SMALL): 0.25,
                    int(Action.DONK_MEDIUM): 0.50,
                }
                bet_frac = bet_sizes.get(action, 0.5)

                # When we raise: opponent can fold, call, or re-raise
                opp_fold_prob = max(0, 0.3 - 0.04 * opp_eq + 0.1 * bet_frac)
                opp_call_prob = 1.0 - opp_fold_prob

                fold_value = 0.5 + concrete_bet_ratio  # win current pot
                call_value = win_prob * (1.0 + 2.0 * (concrete_bet_ratio + bet_frac)) - (concrete_bet_ratio + bet_frac)

                values[i] = opp_fold_prob * fold_value + opp_call_prob * call_value

        return values


def refine_or_blueprint(trainer, refiner: LocalRefiner | None,
                        phase: str, bucket: int,
                        history: tuple, position: str,
                        strategy: dict[int, float],
                        concrete_bet_ratio: float | None,
                        mapped_action: int | None,
                        visit_count: float,
                        available_actions: list) -> dict[int, float]:
    """Main entry point: decide whether to refine or use blueprint.

    Called from GTOAgent.decide() when mapping is "refine".

    Args:
        trainer: CFRTrainer with loaded strategy
        refiner: LocalRefiner instance (None to skip refinement)
        phase: Game phase
        bucket: Player's 2D bucket
        history: Abstract history
        position: 'oop' or 'ip'
        strategy: Blueprint strategy dict
        concrete_bet_ratio: Actual bet as fraction of pot (None if no bet)
        mapped_action: Abstract action the bet was mapped to (None if no bet)
        visit_count: Node visit weight
        available_actions: List of Action enums

    Returns:
        Strategy dict (refined or original blueprint)
    """
    if refiner is None:
        return strategy

    if concrete_bet_ratio is None or mapped_action is None:
        return strategy

    if not should_refine(concrete_bet_ratio, mapped_action, phase, visit_count):
        return strategy

    refined = refiner.refine_strategy(
        phase, bucket, history, position,
        concrete_bet_ratio, available_actions)

    # Blend refined with blueprint instead of full replacement
    # This preserves the trained strategy's strength in well-understood spots
    alpha = REFINE_BLEND_ALPHA
    all_actions = set(strategy.keys()) | set(refined.keys())
    blended = {}
    for a in all_actions:
        s = strategy.get(a, 0.0)
        r = refined.get(a, 0.0)
        blended[a] = (1.0 - alpha) * s + alpha * r

    total = sum(blended.values())
    if total > 0:
        blended = {a: p / total for a, p in blended.items()}
    else:
        # Edge case: all zeros — fall back to uniform over available actions
        n = len(blended)
        blended = {a: 1.0 / n for a in blended} if n > 0 else strategy
    return blended
