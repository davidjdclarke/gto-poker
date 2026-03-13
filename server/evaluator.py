from itertools import combinations
from collections import Counter
from server.deck import Card

HAND_RANKS = {
    "Royal Flush": 9,
    "Straight Flush": 8,
    "Four of a Kind": 7,
    "Full House": 6,
    "Flush": 5,
    "Straight": 4,
    "Three of a Kind": 3,
    "Two Pair": 2,
    "One Pair": 1,
    "High Card": 0,
}


class HandResult:
    __slots__ = ("rank", "tiebreakers", "name", "best_five")

    def __init__(self, rank: int, tiebreakers: tuple, name: str, best_five: list[Card]):
        self.rank = rank
        self.tiebreakers = tiebreakers
        self.name = name
        self.best_five = best_five

    def __lt__(self, other):
        return (self.rank, self.tiebreakers) < (other.rank, other.tiebreakers)

    def __eq__(self, other):
        return (self.rank, self.tiebreakers) == (other.rank, other.tiebreakers)

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other


def evaluate_five(cards: list[Card]) -> HandResult:
    ranks = sorted([c.rank for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    rank_counts = Counter(ranks)
    is_flush = len(set(suits)) == 1

    # Check straight
    is_straight = False
    straight_high = 0
    unique_ranks = sorted(set(ranks), reverse=True)

    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        # Ace-low straight (wheel): A-2-3-4-5
        if unique_ranks == [14, 5, 4, 3, 2]:
            is_straight = True
            straight_high = 5

    if is_straight and is_flush:
        name = "Royal Flush" if straight_high == 14 else "Straight Flush"
        rank = HAND_RANKS[name]
        return HandResult(rank, (straight_high,), name, cards)

    # Four of a kind
    counts_desc = rank_counts.most_common()
    if counts_desc[0][1] == 4:
        quad_rank = counts_desc[0][0]
        kicker = counts_desc[1][0]
        return HandResult(7, (quad_rank, kicker), "Four of a Kind", cards)

    # Full house
    if counts_desc[0][1] == 3 and counts_desc[1][1] == 2:
        trip_rank = counts_desc[0][0]
        pair_rank = counts_desc[1][0]
        return HandResult(6, (trip_rank, pair_rank), "Full House", cards)

    if is_flush:
        return HandResult(5, tuple(ranks), "Flush", cards)

    if is_straight:
        return HandResult(4, (straight_high,), "Straight", cards)

    # Three of a kind
    if counts_desc[0][1] == 3:
        trip_rank = counts_desc[0][0]
        kickers = sorted([r for r, c in counts_desc if c == 1], reverse=True)
        return HandResult(3, (trip_rank, *kickers), "Three of a Kind", cards)

    # Two pair
    if counts_desc[0][1] == 2 and counts_desc[1][1] == 2:
        pairs = sorted([r for r, c in counts_desc if c == 2], reverse=True)
        kicker = [r for r, c in counts_desc if c == 1][0]
        return HandResult(2, (*pairs, kicker), "Two Pair", cards)

    # One pair
    if counts_desc[0][1] == 2:
        pair_rank = counts_desc[0][0]
        kickers = sorted([r for r, c in counts_desc if c == 1], reverse=True)
        return HandResult(1, (pair_rank, *kickers), "One Pair", cards)

    # High card
    return HandResult(0, tuple(ranks), "High Card", cards)


def best_hand(hole_cards: list[Card], community: list[Card]) -> HandResult:
    all_cards = hole_cards + community
    best = None
    for combo in combinations(all_cards, 5):
        result = evaluate_five(list(combo))
        if best is None or result > best:
            best = result
    return best


def determine_winners(players_hands: list[tuple[int, HandResult]]) -> list[int]:
    """Given list of (player_index, HandResult), return indices of winners."""
    if not players_hands:
        return []
    best_result = max(players_hands, key=lambda x: x[1])[1]
    return [idx for idx, result in players_hands if result == best_result]
