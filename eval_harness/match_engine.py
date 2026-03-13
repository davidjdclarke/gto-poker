"""
Heads-up match engine for concrete-card evaluation.

Deals real cards, maps to abstract strategy, plays full hands, and tracks
bb/100, EV loss from abstraction mismatch, and per-line diagnostics.

This is the less-coupled evaluation environment: real private hands + boards,
concrete play, and a real evaluator scoring the outcome.
"""
import random
from dataclasses import dataclass, field
from collections import defaultdict

from server.deck import Deck, Card
from server.evaluator import best_hand, determine_winners
from server.gto.equity import hand_strength_bucket
from server.gto.abstraction import (
    Action, ACTION_NAMES, InfoSet, NUM_BUCKETS,
    get_available_actions, count_raises, decode_bucket,
    concrete_to_abstract_history,
)
from server.gto.cfr import CFRTrainer
from server.gto.opponent_model import OpponentProfile
from eval_harness.fast_equity import fast_bucket


# ---------------------------------------------------------------------------
# Agent protocol
# ---------------------------------------------------------------------------
@dataclass
class AgentDecision:
    """What an agent wants to do."""
    action: str       # "fold", "check", "call", "raise"
    amount: int = 0   # raise-to amount (0 for non-raises)


class Agent:
    """Base class for match participants."""
    name: str = "BaseAgent"

    def decide(self, ctx: "HandContext") -> AgentDecision:
        raise NotImplementedError

    def notify_result(self, ctx: "HandContext", net_chips: float):
        """Called after hand completes. Override for learning agents."""
        pass


# ---------------------------------------------------------------------------
# Hand context (visible to agents)
# ---------------------------------------------------------------------------
@dataclass
class HandContext:
    """All public + private info available to the acting player."""
    hole_cards: list[Card]
    community_cards: list[Card]
    pot: int
    current_bet: int
    my_bet: int
    my_chips: int
    opp_chips: int
    min_raise: int
    big_blind: int
    phase: str                     # preflop/flop/turn/river
    is_ip: bool                    # am I in position?
    betting_history: list[str]     # concrete actions this street
    hand_number: int
    street_pot_start: int          # pot at start of this street


# ---------------------------------------------------------------------------
# GTO Agent — uses the trained strategy
# ---------------------------------------------------------------------------
class GTOAgent(Agent):
    """Plays according to trained CFR strategy with configurable action mapping."""

    def __init__(self, trainer: CFRTrainer, name: str = "GTO",
                 mapping: str = "nearest", simulations: int = 80,
                 opponent_profile: "OpponentProfile | None" = None):
        self.trainer = trainer
        self.name = name
        self.mapping = mapping  # nearest, conservative, stochastic, resolve
        self.simulations = simulations
        self.opponent_profile = opponent_profile
        # Diagnostics
        self.lookup_hits = 0
        self.lookup_misses = 0
        self.abstraction_mismatches = 0

    def decide(self, ctx: HandContext) -> AgentDecision:
        bucket = fast_bucket(ctx.hole_cards, ctx.community_cards,
                             simulations=self.simulations)

        history = _concrete_to_abstract_history(ctx.betting_history, ctx.phase)
        position = 'oop' if not ctx.is_ip else 'ip'

        # Fix: for abstract lookup, position is encoded by history length
        # P0 (oop) acts at even history, P1 (ip) at odd history
        # But in our match, SB=oop acts first preflop
        # Position from history length:
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
            self.lookup_hits += 1
        else:
            uniform = 1.0 / len(actions)
            strategy = {int(a): uniform for a in actions}
            self.lookup_misses += 1

        # Apply mapping adjustments before sampling
        if self.mapping == "conservative":
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

        # Apply opponent-adaptive adjustments (EQ0-3 bluff nodes only)
        if self.opponent_profile is not None:
            eq_bucket, _ = decode_bucket(bucket)
            adjustments = self.opponent_profile.compute_adjustments(ctx.phase, eq_bucket)
            if adjustments:
                for a_id in list(strategy.keys()):
                    strategy[a_id] = max(0.0, strategy[a_id] * adjustments.get(a_id, 1.0))

        # Sample action
        action_ids = list(strategy.keys())
        weights = [max(0, strategy[a]) for a in action_ids]
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(action_ids)] * len(action_ids)

        chosen = random.choices(action_ids, weights=weights, k=1)[0]
        return _abstract_to_concrete(Action(chosen), ctx)


# ---------------------------------------------------------------------------
# Abstract ↔ Concrete conversion
# ---------------------------------------------------------------------------
def _concrete_to_abstract_history(history: list[str], phase: str) -> tuple:
    """Convert concrete action names to abstract history tuple.

    Delegates to the canonical implementation in abstraction.py so that
    match_engine.py and engine.py are guaranteed to use identical mappings.
    """
    return concrete_to_abstract_history(history, phase)


def _has_bet_to_call(history: tuple, phase: str) -> bool:
    """Check if there's a bet to call given abstract history."""
    if not history:
        return phase == 'preflop'  # BB is a forced bet
    bet_actions = {
        int(Action.BET_THIRD_POT), int(Action.BET_HALF_POT),
        int(Action.BET_TWO_THIRDS_POT), int(Action.BET_POT),
        int(Action.BET_OVERBET), int(Action.ALL_IN),
        int(Action.OPEN_RAISE), int(Action.THREE_BET),
        int(Action.FOUR_BET),
        int(Action.DONK_SMALL), int(Action.DONK_MEDIUM),
    }
    return history[-1] in bet_actions


def _abstract_to_concrete(action: Action, ctx: HandContext) -> AgentDecision:
    """Convert abstract action to concrete game decision."""
    to_call = ctx.current_bet - ctx.my_bet

    if action == Action.FOLD:
        if to_call == 0:
            return AgentDecision("check")
        return AgentDecision("fold")

    if action == Action.CHECK_CALL:
        if to_call == 0:
            return AgentDecision("check")
        return AgentDecision("call")

    if action == Action.ALL_IN:
        total = ctx.my_bet + ctx.my_chips
        if total <= ctx.current_bet:
            return AgentDecision("call")
        return AgentDecision("raise", total)

    # Sized raises
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
        if to_call == 0:
            return AgentDecision("check")
        return AgentDecision("call")

    return AgentDecision("raise", raise_to)


# ---------------------------------------------------------------------------
# Classify concrete raise into abstract action name
# ---------------------------------------------------------------------------
def classify_raise(raise_amount: int, pot: int, current_bet: int,
                   phase: str, raise_count: int, big_blind: int) -> str:
    """Classify a concrete raise into an abstract action name."""
    if phase == "preflop":
        if raise_count == 0:
            return "open_raise"
        elif raise_count == 1:
            return "three_bet"
        elif raise_count == 2:
            return "four_bet"
        else:
            return "all_in"

    if pot <= 0:
        return "bet_third"
    ratio = raise_amount / pot
    if ratio >= 2.5:
        return "all_in"
    elif ratio >= 1.1:
        return "bet_overbet"
    elif ratio >= 0.8:
        return "bet_pot"
    elif ratio >= 0.56:
        return "bet_two_thirds"
    elif ratio >= 0.4:
        return "bet_half"
    else:
        return "bet_third"


# ---------------------------------------------------------------------------
# Match result tracking
# ---------------------------------------------------------------------------
@dataclass
class HandRecord:
    """Record of a single hand played."""
    hand_num: int
    p0_cards: list[str]
    p1_cards: list[str]
    community: list[str]
    pot: int
    winner: int              # 0, 1, or -1 for split
    p0_net: float            # chips won/lost by p0
    phases_reached: list[str]
    went_to_showdown: bool
    actions: list[str]


@dataclass
class MatchResult:
    """Aggregate results from a match."""
    p0_name: str
    p1_name: str
    num_hands: int
    p0_total_won: float = 0.0
    p1_total_won: float = 0.0
    hands: list[HandRecord] = field(default_factory=list)

    # Derived metrics
    showdown_count: int = 0
    phase_counts: dict = field(default_factory=lambda: defaultdict(int))

    @property
    def p0_bb_per_100(self) -> float:
        if self.num_hands == 0:
            return 0.0
        return (self.p0_total_won / self.num_hands) * 100

    @property
    def p1_bb_per_100(self) -> float:
        if self.num_hands == 0:
            return 0.0
        return (self.p1_total_won / self.num_hands) * 100


# ---------------------------------------------------------------------------
# Match engine
# ---------------------------------------------------------------------------
class HeadsUpMatch:
    """
    Plays a heads-up match between two agents using real cards.

    Alternates button each hand. Tracks bb/100 and per-hand records.
    """

    def __init__(self, agent0: Agent, agent1: Agent,
                 starting_chips: int = 10000, big_blind: int = 20,
                 seed: int = None):
        self.agents = [agent0, agent1]
        self.starting_chips = starting_chips
        self.big_blind = big_blind
        self.small_blind = big_blind // 2
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def play(self, num_hands: int) -> MatchResult:
        result = MatchResult(
            p0_name=self.agents[0].name,
            p1_name=self.agents[1].name,
            num_hands=num_hands,
        )

        for h in range(num_hands):
            # Alternate button: even hands p0 is SB/BTN, odd hands p1 is SB/BTN
            sb_player = h % 2
            bb_player = 1 - sb_player

            record = self._play_hand(h, sb_player, bb_player)
            result.hands.append(record)
            result.p0_total_won += record.p0_net
            result.p1_total_won += -record.p0_net
            if record.went_to_showdown:
                result.showdown_count += 1
            for ph in record.phases_reached:
                result.phase_counts[ph] += 1

        # Normalize to bb
        bb = self.big_blind
        result.p0_total_won /= bb
        result.p1_total_won /= bb

        return result

    def _play_hand(self, hand_num: int, sb_idx: int, bb_idx: int) -> HandRecord:
        deck = Deck()
        chips = [self.starting_chips, self.starting_chips]
        bets = [0, 0]
        pot = 0
        community = []
        actions_log = []
        phases_reached = ["preflop"]

        # Deal hole cards
        hole_cards = [deck.deal(2), deck.deal(2)]

        # Post blinds
        sb_amount = min(self.small_blind, chips[sb_idx])
        bb_amount = min(self.big_blind, chips[bb_idx])
        chips[sb_idx] -= sb_amount
        bets[sb_idx] = sb_amount
        chips[bb_idx] -= bb_amount
        bets[bb_idx] = bb_amount
        current_bet = bb_amount
        min_raise = self.big_blind

        # Preflop betting
        folded = [False, False]
        street_history = []
        raise_count = 0

        # Preflop: SB acts first
        acting_order = [sb_idx, bb_idx]

        pot, folded, street_history, raise_count = self._betting_round(
            acting_order, chips, bets, pot, current_bet, min_raise,
            community, hole_cards, "preflop", hand_num, folded,
            sb_idx, actions_log,
        )

        if folded[0] or folded[1]:
            return self._finish_hand(
                hand_num, hole_cards, community, pot, bets, chips,
                folded, phases_reached, False, actions_log, sb_idx,
            )

        # Collect bets
        pot += bets[0] + bets[1]
        bets = [0, 0]
        current_bet = 0
        min_raise = self.big_blind

        # Postflop streets
        for street_name, num_cards in [("flop", 3), ("turn", 1), ("river", 1)]:
            phases_reached.append(street_name)
            deck.burn()
            community.extend(deck.deal(num_cards))

            # Postflop: BB (OOP) acts first
            acting_order = [bb_idx, sb_idx]

            pot, folded, street_history, raise_count = self._betting_round(
                acting_order, chips, bets, pot, current_bet, min_raise,
                community, hole_cards, street_name, hand_num, folded,
                sb_idx, actions_log,
            )

            if folded[0] or folded[1]:
                return self._finish_hand(
                    hand_num, hole_cards, community, pot, bets, chips,
                    folded, phases_reached, False, actions_log, sb_idx,
                )

            pot += bets[0] + bets[1]
            bets = [0, 0]
            current_bet = 0
            min_raise = self.big_blind

        # Showdown
        return self._finish_hand(
            hand_num, hole_cards, community, pot, bets, chips,
            folded, phases_reached, True, actions_log, sb_idx,
        )

    def _betting_round(self, acting_order, chips, bets, pot, current_bet,
                       min_raise, community, hole_cards, phase, hand_num,
                       folded, sb_idx, actions_log):
        """Run one street of betting. Returns (pot, folded, history, raise_count)."""
        street_history = []
        raise_count_in = 0 if phase != "preflop" else 0
        needs_to_act = set(acting_order)
        max_actions = 12
        action_count = 0

        while needs_to_act and action_count < max_actions:
            for player_idx in acting_order:
                if player_idx not in needs_to_act:
                    continue
                if folded[player_idx]:
                    needs_to_act.discard(player_idx)
                    continue
                if chips[player_idx] <= 0:
                    needs_to_act.discard(player_idx)
                    continue

                is_ip = (player_idx == sb_idx)  # SB = BTN = IP in HU

                ctx = HandContext(
                    hole_cards=hole_cards[player_idx],
                    community_cards=list(community),
                    pot=pot + bets[0] + bets[1],
                    current_bet=current_bet,
                    my_bet=bets[player_idx],
                    my_chips=chips[player_idx],
                    opp_chips=chips[1 - player_idx],
                    min_raise=min_raise,
                    big_blind=self.big_blind,
                    phase=phase,
                    is_ip=is_ip,
                    betting_history=list(street_history),
                    hand_number=hand_num,
                    street_pot_start=pot,
                )

                decision = self.agents[player_idx].decide(ctx)
                to_call = current_bet - bets[player_idx]

                if decision.action == "fold":
                    if to_call <= 0:
                        decision = AgentDecision("check")
                        street_history.append("check")
                        actions_log.append(f"p{player_idx}:check")
                    else:
                        folded[player_idx] = True
                        street_history.append("fold")
                        actions_log.append(f"p{player_idx}:fold")
                    needs_to_act.discard(player_idx)

                elif decision.action == "check":
                    street_history.append("check")
                    actions_log.append(f"p{player_idx}:check")
                    needs_to_act.discard(player_idx)

                elif decision.action == "call":
                    actual = min(to_call, chips[player_idx])
                    chips[player_idx] -= actual
                    bets[player_idx] += actual
                    street_history.append("call")
                    actions_log.append(f"p{player_idx}:call")
                    needs_to_act.discard(player_idx)

                elif decision.action == "raise":
                    raise_to = decision.amount
                    raise_to = min(raise_to, bets[player_idx] + chips[player_idx])
                    if raise_to <= current_bet:
                        # Can't raise, just call
                        actual = min(to_call, chips[player_idx])
                        chips[player_idx] -= actual
                        bets[player_idx] += actual
                        street_history.append("call")
                        actions_log.append(f"p{player_idx}:call")
                        needs_to_act.discard(player_idx)
                    else:
                        actual_raise = raise_to - bets[player_idx]
                        chips[player_idx] -= actual_raise
                        raise_amount = raise_to - current_bet
                        label = classify_raise(
                            raise_amount, pot + bets[0] + bets[1],
                            current_bet, phase, raise_count_in, self.big_blind,
                        )
                        street_history.append(label)
                        actions_log.append(f"p{player_idx}:{label}:{raise_to}")
                        bets[player_idx] = raise_to
                        min_raise = max(min_raise, raise_amount)
                        current_bet = raise_to
                        raise_count_in += 1
                        needs_to_act.discard(player_idx)
                        # Opponent needs to act again
                        opp = 1 - player_idx
                        if not folded[opp] and chips[opp] > 0:
                            needs_to_act.add(opp)

                action_count += 1

                # Record this player's action into the *other* agent's opponent profile
                other_agent = self.agents[1 - player_idx]
                if (hasattr(other_agent, 'opponent_profile')
                        and other_agent.opponent_profile is not None):
                    other_agent.opponent_profile.record(phase, decision.action)
                if folded[0] or folded[1]:
                    break

            if folded[0] or folded[1]:
                break

        return pot, folded, street_history, raise_count_in

    def _finish_hand(self, hand_num, hole_cards, community, pot, bets, chips,
                     folded, phases_reached, showdown, actions_log, sb_idx):
        """Resolve hand and compute net chips."""
        # Collect remaining bets
        total_pot = pot + bets[0] + bets[1]

        if folded[0]:
            winner = 1
        elif folded[1]:
            winner = 0
        elif showdown and len(community) == 5:
            h0 = best_hand(hole_cards[0], community)
            h1 = best_hand(hole_cards[1], community)
            hands = [(0, h0), (1, h1)]
            winners = determine_winners(hands)
            if len(winners) == 1:
                winner = winners[0]
            else:
                winner = -1  # split
        else:
            winner = -1  # split (shouldn't normally happen)

        # Compute net for p0
        # Each player started with starting_chips
        # p0 invested: starting_chips - chips[0]
        # p1 invested: starting_chips - chips[1]
        p0_invested = self.starting_chips - chips[0]
        p1_invested = self.starting_chips - chips[1]

        if winner == 0:
            p0_net = float(p1_invested)  # p0 wins what p1 put in
        elif winner == 1:
            p0_net = float(-p0_invested)  # p0 loses what they put in
        else:
            # Split pot
            p0_net = float(p1_invested - p0_invested) / 2.0

        # Notify agents
        for i, agent in enumerate(self.agents):
            net = p0_net if i == 0 else -p0_net
            ctx = HandContext(
                hole_cards=hole_cards[i], community_cards=community,
                pot=total_pot, current_bet=0, my_bet=0,
                my_chips=chips[i], opp_chips=chips[1-i],
                min_raise=0, big_blind=self.big_blind, phase=phases_reached[-1],
                is_ip=(i == sb_idx), betting_history=[],
                hand_number=hand_num, street_pot_start=0,
            )
            agent.notify_result(ctx, net)

        return HandRecord(
            hand_num=hand_num,
            p0_cards=[str(c) for c in hole_cards[0]],
            p1_cards=[str(c) for c in hole_cards[1]],
            community=[str(c) for c in community],
            pot=total_pot,
            winner=winner,
            p0_net=p0_net,
            phases_reached=phases_reached,
            went_to_showdown=showdown,
            actions=actions_log,
        )
