"""
A/B action translation evaluator.

Tests multiple live-action mapping schemes to reveal how much performance
is lost in the bridge vs the blueprint.

Mapping schemes:
    nearest     - map to closest abstract sizing (current default)
    conservative - always map to the smaller abstract sizing
    stochastic  - interpolate between two nearest abstract sizes randomly
    resolve     - use a simple heuristic re-solve at decision point

Each scheme is implemented as a GTOAgent variant. We run each against the
same opponent suite and compare bb/100.
"""
import random
import math
from dataclasses import dataclass, field
from collections import defaultdict

from server.gto.cfr import CFRTrainer
from server.gto.abstraction import (
    Action, ACTION_NAMES, InfoSet, NUM_BUCKETS,
    get_available_actions, count_raises, decode_bucket,
)
from server.gto.equity import hand_strength_bucket
from eval_harness.fast_equity import fast_bucket
from server.gto.abstraction import concrete_to_abstract_history as _concrete_to_abstract_history
from eval_harness.match_engine import (
    Agent, AgentDecision, HandContext, HeadsUpMatch, GTOAgent,
    _has_bet_to_call, MatchResult,
)


# ---------------------------------------------------------------------------
# Alternate mapping agents
# ---------------------------------------------------------------------------
class ConservativeGTOAgent(Agent):
    """Maps off-tree actions to the smaller/safer abstract action.
    When facing a bet between 1/3 pot and 1/2 pot, treats it as 1/3 pot."""

    def __init__(self, trainer: CFRTrainer, name: str = "GTO-Conservative"):
        self.trainer = trainer
        self.name = name

    def decide(self, ctx: HandContext) -> AgentDecision:
        bucket = fast_bucket(ctx.hole_cards, ctx.community_cards,
                             simulations=80)

        history = _concrete_to_abstract_history(ctx.betting_history, ctx.phase)
        pos_from_hist = 'oop' if len(history) % 2 == 0 else 'ip'

        has_bet = _has_bet_to_call(history, ctx.phase)
        raise_count = count_raises(history, ctx.phase)
        can_raise = raise_count < 4
        actions = get_available_actions(has_bet, can_raise, ctx.phase, raise_count,
                                       history_len=len(history),
                                       eq_bucket=decode_bucket(bucket)[0])

        info_set = InfoSet(bucket, ctx.phase, history, position=pos_from_hist)
        key = info_set.key

        if key in self.trainer.nodes:
            node = self.trainer.nodes[key]
            avg_strategy = node.get_average_strategy()
            strategy = {int(actions[i]): float(avg_strategy[i])
                        for i in range(len(actions))}
        else:
            uniform = 1.0 / len(actions)
            strategy = {int(a): uniform for a in actions}

        # Conservative adjustment: boost check/call and fold, reduce aggression
        # This simulates "when in doubt, take the safer line"
        if int(Action.CHECK_CALL) in strategy:
            boost = 0.0
            for a in [int(Action.BET_THIRD_POT), int(Action.BET_HALF_POT),
                       int(Action.BET_TWO_THIRDS_POT), int(Action.BET_POT),
                       int(Action.BET_OVERBET),
                       int(Action.DONK_SMALL), int(Action.DONK_MEDIUM)]:
                if a in strategy:
                    steal = strategy[a] * 0.15
                    strategy[a] -= steal
                    boost += steal
            strategy[int(Action.CHECK_CALL)] += boost

        action_ids = list(strategy.keys())
        weights = [max(0, strategy[a]) for a in action_ids]
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(action_ids)] * len(action_ids)

        chosen = random.choices(action_ids, weights=weights, k=1)[0]
        return _agent_abstract_to_concrete(Action(chosen), ctx)


class StochasticGTOAgent(Agent):
    """Interpolates between adjacent abstract sizes stochastically.
    When facing a 42% pot bet (between 33% and 50%), mixes the strategies
    for both neighboring sizes proportionally."""

    def __init__(self, trainer: CFRTrainer, name: str = "GTO-Stochastic"):
        self.trainer = trainer
        self.name = name

    def decide(self, ctx: HandContext) -> AgentDecision:
        bucket = fast_bucket(ctx.hole_cards, ctx.community_cards,
                             simulations=80)

        history = _concrete_to_abstract_history(ctx.betting_history, ctx.phase)
        pos_from_hist = 'oop' if len(history) % 2 == 0 else 'ip'

        has_bet = _has_bet_to_call(history, ctx.phase)
        raise_count = count_raises(history, ctx.phase)
        can_raise = raise_count < 4
        actions = get_available_actions(has_bet, can_raise, ctx.phase, raise_count,
                                       history_len=len(history),
                                       eq_bucket=decode_bucket(bucket)[0])

        info_set = InfoSet(bucket, ctx.phase, history, position=pos_from_hist)
        key = info_set.key

        if key in self.trainer.nodes:
            node = self.trainer.nodes[key]
            avg_strategy = node.get_average_strategy()
            strategy = {int(actions[i]): float(avg_strategy[i])
                        for i in range(len(actions))}
        else:
            uniform = 1.0 / len(actions)
            strategy = {int(a): uniform for a in actions}

        # Stochastic interpolation: look at neighboring histories
        # If the concrete action that led here was between two abstract sizes,
        # blend the strategies for both interpretations
        if len(history) > 0 and ctx.phase != 'preflop':
            last_action = history[-1]
            neighbors = _get_neighbor_histories(history, ctx.phase)
            if neighbors:
                for alt_history, weight in neighbors:
                    alt_key = InfoSet(bucket, ctx.phase, alt_history,
                                      position=pos_from_hist).key
                    if alt_key in self.trainer.nodes:
                        alt_node = self.trainer.nodes[alt_key]
                        alt_actions = get_available_actions(
                            _has_bet_to_call(alt_history, ctx.phase),
                            count_raises(alt_history, ctx.phase) < 4,
                            ctx.phase,
                            count_raises(alt_history, ctx.phase),
                            history_len=len(alt_history),
                            eq_bucket=decode_bucket(bucket)[0],
                        )
                        alt_strat = alt_node.get_average_strategy()
                        # Blend: mix with weight
                        for i, a in enumerate(alt_actions):
                            a_int = int(a)
                            if a_int in strategy and i < len(alt_strat):
                                strategy[a_int] = (
                                    (1 - weight) * strategy.get(a_int, 0) +
                                    weight * float(alt_strat[i])
                                )

        action_ids = list(strategy.keys())
        weights = [max(0, strategy[a]) for a in action_ids]
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(action_ids)] * len(action_ids)

        chosen = random.choices(action_ids, weights=weights, k=1)[0]
        return _agent_abstract_to_concrete(Action(chosen), ctx)


class ResolveGTOAgent(Agent):
    """Simple heuristic re-solve: adjusts strategy based on pot odds
    and actual bet size faced, not just the abstract bucket."""

    def __init__(self, trainer: CFRTrainer, name: str = "GTO-Resolve"):
        self.trainer = trainer
        self.name = name

    def decide(self, ctx: HandContext) -> AgentDecision:
        bucket = fast_bucket(ctx.hole_cards, ctx.community_cards,
                             simulations=80)

        history = _concrete_to_abstract_history(ctx.betting_history, ctx.phase)
        pos_from_hist = 'oop' if len(history) % 2 == 0 else 'ip'

        has_bet = _has_bet_to_call(history, ctx.phase)
        raise_count = count_raises(history, ctx.phase)
        can_raise = raise_count < 4
        actions = get_available_actions(has_bet, can_raise, ctx.phase, raise_count,
                                       history_len=len(history),
                                       eq_bucket=decode_bucket(bucket)[0])

        info_set = InfoSet(bucket, ctx.phase, history, position=pos_from_hist)
        key = info_set.key

        if key in self.trainer.nodes:
            node = self.trainer.nodes[key]
            avg_strategy = node.get_average_strategy()
            strategy = {int(actions[i]): float(avg_strategy[i])
                        for i in range(len(actions))}
        else:
            uniform = 1.0 / len(actions)
            strategy = {int(a): uniform for a in actions}

        # Heuristic resolve: adjust based on actual pot odds
        to_call = ctx.current_bet - ctx.my_bet
        if to_call > 0 and ctx.pot > 0:
            pot_odds = to_call / (ctx.pot + to_call)
            abstract_pot_odds = self._abstract_pot_odds(history, ctx.phase)

            if abstract_pot_odds > 0:
                # If actual pot odds are better (lower) than abstract,
                # we should call more and fold less
                odds_ratio = pot_odds / abstract_pot_odds
                if odds_ratio < 0.85:
                    # Getting a better price: call more
                    fold_action = int(Action.FOLD)
                    call_action = int(Action.CHECK_CALL)
                    if fold_action in strategy and call_action in strategy:
                        shift = strategy[fold_action] * 0.3
                        strategy[fold_action] -= shift
                        strategy[call_action] += shift
                elif odds_ratio > 1.15:
                    # Getting a worse price: fold more
                    fold_action = int(Action.FOLD)
                    call_action = int(Action.CHECK_CALL)
                    if fold_action in strategy and call_action in strategy:
                        shift = strategy[call_action] * 0.2
                        strategy[call_action] -= shift
                        strategy[fold_action] += shift

        action_ids = list(strategy.keys())
        weights = [max(0, strategy[a]) for a in action_ids]
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(action_ids)] * len(action_ids)

        chosen = random.choices(action_ids, weights=weights, k=1)[0]
        return _agent_abstract_to_concrete(Action(chosen), ctx)

    def _abstract_pot_odds(self, history: tuple, phase: str) -> float:
        """Compute pot odds implied by the abstract action."""
        if not history:
            return 0.0

        last = history[-1]
        # Approximate pot odds for each abstract bet size
        pot_odds_map = {
            int(Action.DONK_SMALL): 0.25 / (1.0 + 0.25),       # ~20%
            int(Action.DONK_MEDIUM): 0.50 / (1.0 + 0.50),      # ~33%
            int(Action.BET_THIRD_POT): 0.33 / (1.0 + 0.33),    # ~25%
            int(Action.BET_HALF_POT): 0.50 / (1.0 + 0.50),     # ~33%
            int(Action.BET_TWO_THIRDS_POT): 0.67 / (1.0 + 0.67),  # ~40%
            int(Action.BET_POT): 1.0 / (1.0 + 1.0),            # ~50%
            int(Action.BET_OVERBET): 1.25 / (1.0 + 1.25),      # ~56%
            int(Action.ALL_IN): 0.6,                             # varies
            int(Action.OPEN_RAISE): 0.4,
            int(Action.THREE_BET): 0.35,
            int(Action.FOUR_BET): 0.3,
        }
        return pot_odds_map.get(last, 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_neighbor_histories(history: tuple, phase: str) -> list:
    """Find nearby abstract histories by swapping the last bet action
    to an adjacent sizing."""
    if not history:
        return []

    last = history[-1]
    neighbors = []

    # Postflop sizing neighbors
    sizing_chain = [
        int(Action.DONK_SMALL),
        int(Action.BET_THIRD_POT),
        int(Action.DONK_MEDIUM),
        int(Action.BET_HALF_POT),
        int(Action.BET_TWO_THIRDS_POT),
        int(Action.BET_POT),
        int(Action.BET_OVERBET),
    ]

    if last in sizing_chain:
        idx = sizing_chain.index(last)
        if idx > 0:
            alt = history[:-1] + (sizing_chain[idx - 1],)
            neighbors.append((alt, 0.25))
        if idx < len(sizing_chain) - 1:
            alt = history[:-1] + (sizing_chain[idx + 1],)
            neighbors.append((alt, 0.25))

    return neighbors


def _agent_abstract_to_concrete(action: Action, ctx: HandContext) -> AgentDecision:
    """Convert abstract action to concrete decision (shared by all agents)."""
    to_call = ctx.current_bet - ctx.my_bet

    if action == Action.FOLD:
        return AgentDecision("fold") if to_call > 0 else AgentDecision("check")
    if action == Action.CHECK_CALL:
        return AgentDecision("call" if to_call > 0 else "check")
    if action == Action.ALL_IN:
        total = ctx.my_bet + ctx.my_chips
        if total <= ctx.current_bet:
            return AgentDecision("call")
        return AgentDecision("raise", total)

    sizing = {
        Action.OPEN_RAISE: max(ctx.min_raise, int(ctx.big_blind * 2.5)),
        Action.THREE_BET: max(ctx.min_raise, ctx.current_bet * 3),
        Action.FOUR_BET: max(ctx.min_raise, int(ctx.current_bet * 2.2)),
        Action.BET_THIRD_POT: max(ctx.min_raise, ctx.pot // 3),
        Action.BET_HALF_POT: max(ctx.min_raise, ctx.pot // 2),
        Action.BET_TWO_THIRDS_POT: max(ctx.min_raise, int(ctx.pot * 2 / 3)),
        Action.BET_POT: max(ctx.min_raise, ctx.pot),
        Action.BET_OVERBET: max(ctx.min_raise, int(ctx.pot * 1.25)),
        Action.DONK_SMALL: max(ctx.min_raise, ctx.pot // 4),
        Action.DONK_MEDIUM: max(ctx.min_raise, ctx.pot // 2),
    }
    raise_size = sizing.get(action, ctx.min_raise)
    raise_to = int(ctx.current_bet + max(raise_size, ctx.min_raise))
    raise_to = min(raise_to, ctx.my_bet + ctx.my_chips)

    if raise_to <= ctx.current_bet:
        return AgentDecision("call" if to_call > 0 else "check")
    return AgentDecision("raise", raise_to)


# ---------------------------------------------------------------------------
# A/B test runner
# ---------------------------------------------------------------------------
@dataclass
class TranslationResult:
    """Result for one mapping scheme against one opponent."""
    mapping_name: str
    opponent_name: str
    bb_per_100: float
    num_hands: int


def run_translation_ab(trainer: CFRTrainer, num_hands: int = 500,
                        seed: int = 42, big_blind: int = 20,
                        progress_cb=None) -> dict:
    """
    Run all translation schemes against a shared opponent set.

    Returns comparison showing which mapping performs best.
    """
    from eval_harness.adversaries import get_all_adversaries

    # Build agents for each mapping
    mapping_agents = {
        "nearest": GTOAgent(trainer, name="GTO-Nearest"),
        "conservative": ConservativeGTOAgent(trainer),
        "stochastic": StochasticGTOAgent(trainer),
        "resolve": ResolveGTOAgent(trainer),
    }

    # Test opponents: use a subset for speed
    opponents = [
        get_all_adversaries(trainer)[1],   # AggroBot
        get_all_adversaries(trainer)[3],   # CallStationBot
        get_all_adversaries(trainer)[5],   # WeirdSizingBot
    ]

    results = []
    for map_name, agent in mapping_agents.items():
        for opp in opponents:
            match = HeadsUpMatch(agent, opp, big_blind=big_blind, seed=seed)
            match_result = match.play(num_hands)
            results.append(TranslationResult(
                mapping_name=map_name,
                opponent_name=opp.name,
                bb_per_100=match_result.p0_bb_per_100,
                num_hands=num_hands,
            ))
            if progress_cb is not None:
                progress_cb(map_name, opp.name)

    report = {
        "results": results,
        "summary": _format_translation_report(results, mapping_agents.keys()),
    }
    return report


def _format_translation_report(results: list[TranslationResult],
                                mappings) -> str:
    lines = []
    lines.append("=== LIVE BRIDGE LOSS REPORT (A/B Translation) ===\n")

    # Group by mapping
    by_mapping = defaultdict(list)
    for r in results:
        by_mapping[r.mapping_name].append(r)

    # Header
    opp_names = sorted(set(r.opponent_name for r in results))
    header = f"{'Mapping':<20}"
    for opp in opp_names:
        header += f" {opp:>14}"
    header += f" {'Average':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    best_avg = -999
    best_mapping = ""

    for map_name in mappings:
        row = f"{map_name:<20}"
        total = 0
        for opp in opp_names:
            match = [r for r in by_mapping[map_name] if r.opponent_name == opp]
            if match:
                bb = match[0].bb_per_100
                row += f" {bb:>+14.1f}"
                total += bb
            else:
                row += f" {'N/A':>14}"
        avg = total / len(opp_names) if opp_names else 0
        row += f" {avg:>+10.1f}"
        if avg > best_avg:
            best_avg = avg
            best_mapping = map_name
        lines.append(row)

    lines.append(f"\nBest mapping: {best_mapping} ({best_avg:+.1f} bb/100 avg)")

    # Compute bridge loss
    nearest_results = by_mapping.get("nearest", [])
    if nearest_results:
        nearest_avg = sum(r.bb_per_100 for r in nearest_results) / len(nearest_results)
        lines.append(f"Nearest (default): {nearest_avg:+.1f} bb/100 avg")
        if best_avg > nearest_avg:
            lines.append(f"Bridge improvement possible: {best_avg - nearest_avg:.1f} bb/100")

    return "\n".join(lines)
