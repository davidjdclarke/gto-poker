"""
Local refinement for off-tree action handling (v12 Workstream 1 — Refine 2.0).

When the opponent bets a size not in the abstract action set, the blueprint
strategy answers the wrong question. Local refinement runs a small CFR solve
at the current decision point using the blueprint as a prior, adapting to the
actual bet size.

Scope: turn and river only, triggered by high bridge-pain mismatch.

v12 improvements over v11:
  - Blueprint CFV payoffs replace heuristic win-probability estimates
  - Adaptive trigger threshold (visit count + entropy + board texture)
  - Board texture modulates the blend alpha between refine and blueprint

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

# Base mismatch threshold (fraction of pot) — adaptive threshold uses this as start
MISMATCH_THRESHOLD = 0.40

# Minimum/maximum for the adaptive threshold clamp
_THRESHOLD_MIN = 0.20
_THRESHOLD_MAX = 0.60

# Minimum visit-weight below which refinement is worth the cost
VISIT_THRESHOLD = 200.0

# Phases where refinement is allowed
REFINE_PHASES = ('turn', 'river')

# Number of opponent buckets to sample during refinement
OPP_SAMPLES = 30

# Base blend weight (0=pure blueprint, 1=pure refine); adjusted by board texture
REFINE_BLEND_ALPHA = 0.5

# Max refine triggers per match (budget cap to avoid latency spikes)
_MAX_REFINE_BUDGET = 100


# ---------------------------------------------------------------------------
# Change B — Adaptive threshold
# ---------------------------------------------------------------------------
def compute_adaptive_threshold(visit_count: float,
                                entropy: float,
                                mismatch: float,
                                board_texture=None) -> float:
    """Compute an adaptive mismatch threshold for the refine trigger.

    Lowers threshold (triggers more aggressively) when:
    - Low visit count (blueprint undertrained at this node)
    - High entropy (strategy uncertain / not converged)
    - Dynamic board (DRAW_HEAVY or CONNECTED)

    Raises threshold (trusts blueprint more) when:
    - High visit count
    - Low entropy (strategy confident)
    - Dry board

    Args:
        visit_count:   strategy_sum.sum() for this node (0 if absent)
        entropy:       Shannon entropy of the blueprint strategy, normalised 0–1
        mismatch:      |concrete - nominal| (unused directly; kept for signature compat)
        board_texture: BoardTexture enum value or None

    Returns:
        float threshold in [_THRESHOLD_MIN, _THRESHOLD_MAX]
    """
    base = MISMATCH_THRESHOLD

    # Visit-count adjustment: log-sigmoid centred around VISIT_THRESHOLD
    if visit_count > 0:
        log_v = math.log(visit_count + 1)
        log_thresh = math.log(VISIT_THRESHOLD + 1)
        visit_adj = 0.10 * (log_v / log_thresh - 1.0)   # +/-0.10 range
    else:
        visit_adj = -0.10   # very undertrained → lower threshold

    # Entropy adjustment: high entropy → lower threshold (need more refinement)
    entropy_adj = -0.08 * (entropy - 0.5)

    # Board texture adjustment
    texture_adj = 0.0
    if board_texture is not None:
        try:
            from server.gto.board_texture import BoardTexture
            if board_texture in (BoardTexture.DRAW_HEAVY, BoardTexture.CONNECTED):
                texture_adj = -0.08   # dynamic board → trigger more eagerly
            elif board_texture == BoardTexture.DRY:
                texture_adj = +0.08   # dry board → trust blueprint more
            elif board_texture == BoardTexture.MONOTONE:
                texture_adj = -0.04
        except ImportError:
            pass

    threshold = base + visit_adj + entropy_adj + texture_adj
    return max(_THRESHOLD_MIN, min(_THRESHOLD_MAX, threshold))


def _strategy_entropy(strategy: dict) -> float:
    """Compute normalised Shannon entropy of a strategy dict (0 = uniform, 1 = pure)."""
    probs = [p for p in strategy.values() if p > 0]
    if not probs:
        return 0.0
    n = len(probs)
    if n <= 1:
        return 0.0
    raw = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(n)
    return raw / max_entropy if max_entropy > 0 else 0.0


def should_refine(concrete_bet_ratio: float,
                  mapped_action: int,
                  phase: str,
                  visit_count: float,
                  pain_families: set | None = None,
                  strategy: dict | None = None,
                  board_texture=None) -> bool:
    """Decide whether to invoke local refinement for this decision.

    v12: Uses adaptive threshold based on visit count, entropy, and board texture.
    New args (strategy, board_texture) are optional for backward compatibility.

    Args:
        concrete_bet_ratio: Actual bet as fraction of pot
        mapped_action:      Abstract action the bet was mapped to
        phase:              Game phase
        visit_count:        Node visit weight from strategy_sum
        pain_families:      Set of action IDs with lower pain threshold
        strategy:           Blueprint strategy dict (for entropy computation)
        board_texture:      BoardTexture enum or None
    """
    if phase not in REFINE_PHASES:
        return False

    if concrete_bet_ratio is None or concrete_bet_ratio <= 0:
        return False

    from eval_harness.confidence import ABSTRACT_BET_SIZES
    nominal = ABSTRACT_BET_SIZES.get(mapped_action)
    if nominal is None:
        return False

    distance = abs(concrete_bet_ratio - nominal)

    # Compute adaptive threshold
    entropy = _strategy_entropy(strategy) if strategy else 0.5
    threshold = compute_adaptive_threshold(visit_count, entropy, distance, board_texture)

    # Pain family override: halve the threshold for historically painful actions
    if pain_families and mapped_action in pain_families:
        threshold = max(_THRESHOLD_MIN, threshold * 0.5)

    return distance >= threshold


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
    """Runs a miniature CFR solve at a single decision point (v12: Refine 2.0).

    v12 changes vs v11:
    - _compute_action_values_v2(): uses blueprint counterfactual values (1-ply
      backup from the trained strategy) instead of heuristic win-probability.
      Falls back to heuristic when the opponent node is absent.
    - Diagnostics: blueprint_cfv_count / heuristic_fallback_count track which
      path fires for each action.
    - Per-match refine budget cap (_MAX_REFINE_BUDGET).
    """

    def __init__(self, trainer, num_iters: int = MAX_REFINE_ITERS,
                 opp_samples: int = OPP_SAMPLES):
        self.trainer = trainer
        self.num_iters = num_iters
        self.opp_samples = opp_samples
        # Diagnostics
        self.refine_count = 0
        self.total_iters = 0
        self.blueprint_cfv_count = 0    # actions evaluated with blueprint CFV
        self.heuristic_fallback_count = 0  # actions that fell back to heuristic

    def refine_strategy(self, phase: str, bucket: int,
                        history: tuple, position: str,
                        concrete_bet_ratio: float,
                        available_actions: list) -> dict:
        """Compute a refined strategy for a single decision point.

        Uses mini-CFR (200 iters) with blueprint warm-start.
        Action values computed via blueprint CFV (falls back to heuristic).

        Returns:
            dict {action_int: probability}
        """
        self.refine_count += 1
        n_actions = len(available_actions)
        action_ints = [int(a) for a in available_actions]

        # Warm-start from blueprint
        info_set = InfoSet(bucket, phase, history, position=position)
        key = info_set.key
        node = self.trainer.nodes.get(key)

        regret_sum = np.zeros(n_actions, dtype=np.float64)
        strategy_sum = np.zeros(n_actions, dtype=np.float64)

        if node is not None:
            avg = node.get_average_strategy()
            if len(avg) == n_actions:
                strategy_sum = avg * 100.0

        # Sample opponent buckets
        eq_bucket, _ = decode_bucket(bucket)
        opp_buckets = self._sample_opponent_buckets(eq_bucket)

        # Mini-CFR loop
        for t in range(self.num_iters):
            pos_regrets = np.maximum(regret_sum, 0)
            total = pos_regrets.sum()
            strategy = pos_regrets / total if total > 0 else np.ones(n_actions) / n_actions

            utilities = np.zeros(n_actions, dtype=np.float64)
            for opp_b in opp_buckets:
                utilities += self._compute_action_values_v2(
                    action_ints, phase, bucket, opp_b,
                    history, position, concrete_bet_ratio)
            utilities /= len(opp_buckets)

            node_util = (strategy * utilities).sum()
            regret_sum = np.maximum(regret_sum + (utilities - node_util), 0)
            weight = max(t - 10, 0)
            strategy_sum += weight * strategy

        self.total_iters += self.num_iters

        total = strategy_sum.sum()
        avg_strat = strategy_sum / total if total > 0 else np.ones(n_actions) / n_actions
        return {action_ints[i]: float(avg_strat[i]) for i in range(n_actions)}

    def _sample_opponent_buckets(self, our_eq: int) -> list:
        """Sample plausible opponent buckets given our equity bucket."""
        buckets = []
        for _ in range(self.opp_samples):
            opp_eq = random.randint(0, 7)
            opp_ht = random.randint(0, NUM_HAND_TYPES - 1)
            buckets.append(make_bucket(opp_eq, opp_ht))
        return buckets

    def _compute_action_values_v2(self, actions: list,
                                   phase: str, our_bucket: int,
                                   opp_bucket: int, history: tuple,
                                   our_position: str,
                                   concrete_bet_ratio: float) -> np.ndarray:
        """Estimate action values using blueprint CFV (1-ply backup).

        For each candidate action `a`:
        1. Construct the opponent's next infoset key after we take `a`.
        2. If the key exists in the blueprint, use the opponent's average
           strategy to compute a weighted sum of terminal payoffs.
        3. If the key is absent, fall back to the heuristic model.

        Args:
            actions:            Available action ints
            phase:              Game phase
            our_bucket:         Our 2D bucket (0–119)
            opp_bucket:         Opponent's sampled bucket
            history:            Current abstract history tuple
            our_position:       'oop' or 'ip' for the current actor
            concrete_bet_ratio: Actual bet as fraction of pot

        Returns:
            np.ndarray of estimated values (one per action)
        """
        our_eq = our_bucket // NUM_HAND_TYPES
        opp_eq = opp_bucket // NUM_HAND_TYPES
        n = len(actions)
        values = np.zeros(n, dtype=np.float64)

        # Opponent position at the next node (flipped)
        opp_position = 'ip' if our_position == 'oop' else 'oop'

        for i, action in enumerate(actions):
            v = self._blueprint_cfv_for_action(
                action, phase, opp_bucket, history, opp_position,
                our_eq, opp_eq, concrete_bet_ratio)
            values[i] = v

        return values

    def _blueprint_cfv_for_action(self,
                                   action: int,
                                   phase: str,
                                   opp_bucket: int,
                                   history: tuple,
                                   opp_position: str,
                                   our_eq: int,
                                   opp_eq: int,
                                   concrete_bet_ratio: float) -> float:
        """Compute our EV for taking `action` using a 1-ply blueprint backup.

        Returns our expected value from this action, approximated by:
            EV = sum_r(pi_opp(r) * terminal_approx(action, opp_r))

        Falls back to heuristic payoff if the opponent node is absent.
        """
        next_history = history + (action,)
        opp_key = InfoSet(opp_bucket, phase, next_history,
                          position=opp_position).key

        opp_node = self.trainer.nodes.get(opp_key)
        if opp_node is None:
            self.heuristic_fallback_count += 1
            return self._heuristic_action_value(
                action, our_eq, opp_eq, concrete_bet_ratio)

        opp_avg = opp_node.get_average_strategy()
        # Determine opponent available actions at the next node
        # The opponent faces what we just did; reconstruct their menu
        opp_has_bet = self._has_bet_in_history(next_history, phase)
        opp_raise_count = self._count_raises_in_history(next_history, phase)
        opp_can_raise = opp_raise_count < 4
        try:
            opp_actions = get_available_actions(
                opp_has_bet, opp_can_raise, phase, opp_raise_count,
                history_len=len(next_history),
                eq_bucket=opp_eq)
        except Exception:
            self.heuristic_fallback_count += 1
            return self._heuristic_action_value(
                action, our_eq, opp_eq, concrete_bet_ratio)

        if len(opp_avg) != len(opp_actions):
            self.heuristic_fallback_count += 1
            return self._heuristic_action_value(
                action, our_eq, opp_eq, concrete_bet_ratio)

        self.blueprint_cfv_count += 1

        # 1-ply terminal approximation for each opponent response
        ev = 0.0
        for j, opp_a in enumerate(opp_actions):
            pi = float(opp_avg[j])
            if pi <= 0:
                continue
            terminal_ev = self._terminal_approx(
                action, int(opp_a), our_eq, opp_eq, concrete_bet_ratio)
            ev += pi * terminal_ev

        return ev

    def _terminal_approx(self, our_action: int, opp_response: int,
                          our_eq: int, opp_eq: int,
                          concrete_bet_ratio: float) -> float:
        """Approximate terminal EV for (our_action, opp_response) pair."""
        # Win probability from equity comparison
        eq_diff = (our_eq - opp_eq) / 7.0  # normalised to [-1, 1]
        win_prob = max(0.05, min(0.95, 0.50 + 0.35 * eq_diff))

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
            int(Action.ALL_IN): 3.00,
        }

        # Opponent FOLDS: we win the current pot (normalized ~1.0)
        if opp_response == int(Action.FOLD):
            if our_action == int(Action.FOLD):
                return -0.5   # both fold — we lose our investment
            return 0.5 + concrete_bet_ratio   # win pot + their bet

        # Opponent CALLS / CHECKS our bet
        if opp_response == int(Action.CHECK_CALL):
            our_bet = bet_sizes.get(our_action, 0.0)
            if our_action in (int(Action.FOLD),):
                return -0.5
            if our_action == int(Action.CHECK_CALL):
                # Both check: showdown at current pot
                return win_prob * 1.0 - (1.0 - win_prob) * 1.0  # net
            # We bet, they call
            total_pot = 1.0 + 2.0 * (concrete_bet_ratio + our_bet)
            return win_prob * total_pot - (concrete_bet_ratio + our_bet)

        # Opponent RAISES back at us: we lose our investment vs their raise
        opp_bet = bet_sizes.get(opp_response, 0.5)
        # Simplified: we fold to the raise (conservative)
        invested = concrete_bet_ratio + bet_sizes.get(our_action, 0.0)
        return -invested * 0.5   # lose half of invested (as if we sometimes continue)

    @staticmethod
    def _has_bet_in_history(history: tuple, phase: str) -> bool:
        """Check if the last action in history is a bet/raise."""
        if not history:
            return phase == 'preflop'
        bet_actions = {
            int(Action.BET_QUARTER_POT), int(Action.BET_THIRD_POT),
            int(Action.BET_HALF_POT), int(Action.BET_TWO_THIRDS_POT),
            int(Action.BET_THREE_QUARTER_POT), int(Action.BET_POT),
            int(Action.BET_OVERBET), int(Action.BET_DOUBLE_POT),
            int(Action.ALL_IN), int(Action.OPEN_RAISE),
            int(Action.THREE_BET), int(Action.FOUR_BET),
            int(Action.DONK_SMALL), int(Action.DONK_MEDIUM),
        }
        return history[-1] in bet_actions

    @staticmethod
    def _count_raises_in_history(history: tuple, phase: str) -> int:
        """Count raise actions in history for the current street."""
        return count_raises(history, phase)

    def _heuristic_action_value(self, action: int,
                                 our_eq: int, opp_eq: int,
                                 concrete_bet_ratio: float) -> float:
        """Legacy heuristic payoff (v11). Used as fallback when blueprint absent."""
        eq_diff = our_eq - opp_eq
        win_prob = max(0.05, min(0.95, 0.50 + 0.05 * eq_diff))
        opp_fold_prob = max(0, 0.4 - 0.05 * opp_eq)

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

        if action == int(Action.FOLD):
            return -0.5
        if action == int(Action.CHECK_CALL):
            call_cost = concrete_bet_ratio
            pot_if_win = 1.0 + 2.0 * concrete_bet_ratio
            return win_prob * pot_if_win - call_cost
        if action == int(Action.ALL_IN):
            allin_size = 3.0
            base = win_prob * (1.0 + 2.0 * allin_size) - allin_size
            return opp_fold_prob * 0.5 + (1 - opp_fold_prob) * base

        bet_frac = bet_sizes.get(action, 0.5)
        opp_call_prob = 1.0 - opp_fold_prob
        fold_value = 0.5 + concrete_bet_ratio
        call_value = (win_prob * (1.0 + 2.0 * (concrete_bet_ratio + bet_frac))
                      - (concrete_bet_ratio + bet_frac))
        return opp_fold_prob * fold_value + opp_call_prob * call_value


# ---------------------------------------------------------------------------
# Change C — Board-texture-aware blend alpha
# ---------------------------------------------------------------------------
def _compute_blend_alpha(board_texture, base: float = REFINE_BLEND_ALPHA) -> float:
    """Adjust blend alpha based on board texture.

    Dynamic boards → trust refine output more (higher alpha).
    Dry boards → trust blueprint more (lower alpha).

    Returns:
        float alpha in [0.30, 0.70]
    """
    if board_texture is None:
        return base
    try:
        from server.gto.board_texture import BoardTexture
        if board_texture in (BoardTexture.DRAW_HEAVY, BoardTexture.CONNECTED):
            return min(base + 0.10, 0.70)
        elif board_texture == BoardTexture.DRY:
            return max(base - 0.10, 0.30)
        elif board_texture == BoardTexture.MONOTONE:
            return min(base + 0.05, 0.65)
    except ImportError:
        pass
    return base


def refine_or_blueprint(trainer, refiner,
                        phase: str, bucket: int,
                        history: tuple, position: str,
                        strategy: dict,
                        concrete_bet_ratio,
                        mapped_action,
                        visit_count: float,
                        available_actions: list,
                        community_cards=None) -> dict:
    """Main entry point: decide whether to refine or use blueprint.

    Called from GTOAgent.decide() when mapping is "refine".

    v12 additions:
    - board_texture derived from community_cards modulates threshold + alpha
    - should_refine() receives strategy (for entropy) and board_texture

    Args:
        trainer:            CFRTrainer with loaded strategy
        refiner:            LocalRefiner instance (None to skip refinement)
        phase:              Game phase
        bucket:             Player's 2D bucket
        history:            Abstract history
        position:           'oop' or 'ip'
        strategy:           Blueprint strategy dict
        concrete_bet_ratio: Actual bet as fraction of pot (None if no bet)
        mapped_action:      Abstract action the bet was mapped to (None if no bet)
        visit_count:        Node visit weight
        available_actions:  List of Action enums
        community_cards:    list[Card] from ctx.community_cards (None allowed)

    Returns:
        Strategy dict (refined or original blueprint)
    """
    if refiner is None:
        return strategy

    if concrete_bet_ratio is None or mapped_action is None:
        return strategy

    # Budget cap: avoid runaway latency
    if refiner.refine_count >= _MAX_REFINE_BUDGET:
        return strategy

    # Classify board texture for adaptive threshold and blend
    board_texture = None
    if community_cards:
        try:
            from server.gto.board_texture import classify_board_texture
            board_texture = classify_board_texture(community_cards)
        except Exception:
            pass

    if not should_refine(concrete_bet_ratio, mapped_action, phase, visit_count,
                         strategy=strategy, board_texture=board_texture):
        return strategy

    refined = refiner.refine_strategy(
        phase, bucket, history, position,
        concrete_bet_ratio, available_actions)

    # Blend with board-texture-aware alpha
    alpha = _compute_blend_alpha(board_texture)
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
        n = len(blended)
        blended = {a: 1.0 / n for a in blended} if n > 0 else strategy
    return blended
