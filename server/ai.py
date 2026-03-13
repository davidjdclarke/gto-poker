import random
from server.player import Player
from server.evaluator import best_hand, HandResult
from server.gto.engine import gto_decide, GTODecision

# Simplified starting hand strength (Chen formula approximation)
# Maps (high_rank, low_rank, suited) -> score 0-20
def starting_hand_score(hole_cards) -> float:
    c1, c2 = hole_cards
    high = max(c1.rank, c2.rank)
    low = min(c1.rank, c2.rank)
    suited = c1.suit == c2.suit

    # Base score from high card
    score_map = {14: 10, 13: 8, 12: 7, 11: 6}
    score = score_map.get(high, high / 2.0)

    # Pair bonus
    if high == low:
        score = max(5, score * 2)
        if score < 10:
            score = max(score, 5)

    # Suited bonus
    if suited:
        score += 2

    # Closeness bonus (connectors)
    gap = high - low
    if gap == 1:
        score += 1
    elif gap == 2:
        score += 0.5
    elif gap >= 5:
        score -= (gap - 4)

    return max(0, score)


def hand_strength_estimate(hand_result: HandResult) -> float:
    """Convert a hand result to a 0-1 strength estimate."""
    # Rough mapping of hand rank to strength
    base = {
        0: 0.1,   # High card
        1: 0.3,   # One pair
        2: 0.5,   # Two pair
        3: 0.65,  # Three of a kind
        4: 0.75,  # Straight
        5: 0.8,   # Flush
        6: 0.88,  # Full house
        7: 0.95,  # Four of a kind
        8: 0.99,  # Straight flush
        9: 1.0,   # Royal flush
    }
    strength = base.get(hand_result.rank, 0.1)
    # Add some variation based on kickers
    strength += random.uniform(-0.05, 0.05)
    return min(1.0, max(0.0, strength))


class AIDecision:
    """Represents an AI player's decision."""

    def __init__(self, action: str, amount: int = 0):
        self.action = action  # "fold", "check", "call", "raise"
        self.amount = amount


def decide(player: Player, community_cards: list, pot: int,
           current_bet: int, min_raise: int, big_blind: int,
           num_opponents: int = 1, betting_history: list[str] = None,
           is_in_position: bool = True) -> AIDecision:
    """Make an AI decision based on player style and game state."""
    style = player.ai_style

    # GTO players use the CFR engine
    if style == "gto":
        try:
            gto = gto_decide(player, community_cards, pot, current_bet,
                             min_raise, big_blind, num_opponents,
                             betting_history, is_in_position=is_in_position)
            return AIDecision(gto.action, gto.amount)
        except Exception:
            # Fallback to balanced heuristic if GTO fails
            pass

    to_call = current_bet - player.current_bet

    # Pre-flop (no community cards)
    if not community_cards:
        return _preflop_decision(player, style, to_call, min_raise, pot, big_blind)

    # Post-flop
    hand_result = best_hand(player.hole_cards, community_cards)
    strength = hand_strength_estimate(hand_result)

    return _postflop_decision(player, style, strength, to_call, min_raise, pot, big_blind)


def _preflop_decision(player, style, to_call, min_raise, pot, big_blind):
    score = starting_hand_score(player.hole_cards)

    # Thresholds based on style
    thresholds = {
        "tight":      {"play": 7, "raise": 10},
        "balanced":   {"play": 5, "raise": 8},
        "loose":      {"play": 3, "raise": 7},
        "aggressive": {"play": 4, "raise": 6},
    }
    t = thresholds.get(style, thresholds["balanced"])

    # Add some randomness
    score += random.uniform(-1, 1)

    if score < t["play"]:
        if to_call == 0:
            return AIDecision("check")
        return AIDecision("fold")

    if score >= t["raise"]:
        raise_amount = min_raise
        if style == "aggressive":
            raise_amount = min_raise * random.choice([1, 2, 3])
        raise_amount = min(raise_amount, player.chips)
        if raise_amount > 0 and player.chips > to_call:
            return AIDecision("raise", current_bet_total(player, to_call, raise_amount))
        elif to_call == 0:
            return AIDecision("check")
        else:
            return AIDecision("call")

    # Playable but not raise-worthy
    if to_call == 0:
        return AIDecision("check")
    if to_call <= big_blind * 3:
        return AIDecision("call")
    # Big raise to call with mediocre hand
    if random.random() < 0.3:
        return AIDecision("call")
    return AIDecision("fold")


def _postflop_decision(player, style, strength, to_call, min_raise, pot, big_blind):
    # Pot odds consideration
    pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0

    # Style modifiers
    aggression = {"tight": 0.0, "balanced": 0.1, "loose": 0.15, "aggressive": 0.25}
    bluff_chance = {"tight": 0.02, "balanced": 0.08, "loose": 0.15, "aggressive": 0.2}

    mod = aggression.get(style, 0.1)
    bluff = bluff_chance.get(style, 0.08)

    effective_strength = strength + mod + random.uniform(-0.1, 0.1)

    # Strong hand - raise
    if effective_strength > 0.7:
        if player.chips > to_call and random.random() < 0.6 + mod:
            raise_amt = int(min_raise * random.uniform(1, 2.5))
            raise_amt = min(raise_amt, player.chips - to_call)
            if raise_amt >= min_raise:
                return AIDecision("raise", current_bet_total(player, to_call, raise_amt))
        if to_call == 0:
            return AIDecision("check")
        return AIDecision("call")

    # Medium hand - call if price is right
    if effective_strength > 0.4:
        if to_call == 0:
            # Occasionally bet with medium hands
            if random.random() < 0.3 + mod:
                raise_amt = min(min_raise, player.chips)
                if raise_amt >= min_raise:
                    return AIDecision("raise", current_bet_total(player, to_call, raise_amt))
            return AIDecision("check")
        if pot_odds < effective_strength:
            return AIDecision("call")
        if random.random() < 0.2:
            return AIDecision("call")
        return AIDecision("fold")

    # Weak hand
    if to_call == 0:
        # Bluff sometimes
        if random.random() < bluff:
            raise_amt = min(min_raise, player.chips)
            if raise_amt >= min_raise:
                return AIDecision("raise", current_bet_total(player, to_call, raise_amt))
        return AIDecision("check")

    # Bluff call/raise
    if random.random() < bluff:
        return AIDecision("call")

    return AIDecision("fold")


def current_bet_total(player, to_call, raise_amount):
    """Return the total raise amount (what the new current bet should be)."""
    return player.current_bet + to_call + raise_amount
