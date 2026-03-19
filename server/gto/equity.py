"""
Hand equity and strength evaluation for GTO abstraction.

Provides Monte Carlo equity calculation and 2D hand bucketing using:
- Equity dimension: Expected Hand Strength Squared (E[HS^2])
  [Zinkevich et al. 2007, Section 4.1]
- Hand type dimension: structural classification (pairs, suited connectors,
  broadway, etc.) that preserves strategic distinctions beyond raw equity.

References:
  [1] Zinkevich et al. "Regret Minimization in Games with Incomplete
      Information." NIPS 2007, Section 4.1.
"""
import random
from itertools import combinations
from server.deck import Card, SUITS
from server.evaluator import best_hand
from server.gto.abstraction import (
    classify_hand_type, make_bucket, NUM_EQUITY_BUCKETS, HandType
)


# Module-level toggle for suit isomorphism (WS3).
# Set to False to disable canonicalization (e.g. for ablation testing).
SUIT_ISO_ENABLED = True

# WS4: EMD clustering mode. When enabled, equity bucketing uses
# precomputed EMD cluster centroids instead of uniform bands.
EMD_MODE_ENABLED = False
_EMD_CENTROIDS = None    # Lazy-loaded: {street: np.ndarray(K, N_BINS)}
_EMD_PREFLOP_TABLE = None  # Lazy-loaded: {canonical_key: cluster_id}


def make_deck() -> list[Card]:
    return [Card(rank, suit) for suit in SUITS for rank in range(2, 15)]


def hand_equity(hole_cards: list[Card], community: list[Card],
                num_opponents: int = 1, simulations: int = 500) -> float:
    """
    Estimate equity (win probability) of hole_cards given community cards
    via Monte Carlo simulation.

    Returns float between 0.0 and 1.0.
    """
    dead = set((c.rank, c.suit) for c in hole_cards + community)
    remaining = [c for c in make_deck() if (c.rank, c.suit) not in dead]

    cards_needed = (5 - len(community))  # community cards to deal
    opp_cards_needed = 2 * num_opponents

    wins = 0
    ties = 0
    total = 0

    for _ in range(simulations):
        random.shuffle(remaining)
        idx = 0

        # Deal remaining community
        sim_community = list(community) + remaining[idx:idx + cards_needed]
        idx += cards_needed

        # Deal opponent hands
        opp_hands = []
        for _ in range(num_opponents):
            opp_hands.append(remaining[idx:idx + 2])
            idx += 2

        # Evaluate
        my_hand = best_hand(hole_cards, sim_community)
        best_opp = None
        for oh in opp_hands:
            opp_result = best_hand(oh, sim_community)
            if best_opp is None or opp_result > best_opp:
                best_opp = opp_result

        if best_opp is None or my_hand > best_opp:
            wins += 1
        elif my_hand == best_opp:
            ties += 1
        total += 1

    if total == 0:
        return 0.5
    return (wins + ties * 0.5) / total


def hand_strength_squared(hole_cards: list[Card], community: list[Card],
                          num_opponents: int = 1,
                          simulations: int = 300) -> float:
    """
    Compute Expected Hand Strength Squared (E[HS^2]).

    E[HS^2] is the expected square of the hand strength after the next
    community card is revealed. It captures both current strength and
    the potential for improvement (variance), making it a better metric
    for card abstraction than raw hand strength.

    For river (5 community cards), this equals equity^2 since no more
    cards are dealt.

    [Zinkevich et al. 2007, Section 4.1]
    """
    if len(community) >= 5:
        # River: no more cards, E[HS^2] = HS^2
        eq = hand_equity(hole_cards, community, num_opponents, simulations)
        return eq * eq

    dead = set((c.rank, c.suit) for c in hole_cards + community)
    remaining = [c for c in make_deck() if (c.rank, c.suit) not in dead]

    # Sample future boards and compute squared equity at each
    sum_hs_sq = 0.0
    num_rollouts = min(simulations, len(remaining))

    sampled_cards = random.sample(remaining, num_rollouts)
    for next_card in sampled_cards:
        future_community = list(community) + [next_card]
        eq = hand_equity(hole_cards, future_community, num_opponents,
                         simulations=max(50, simulations // 5))
        sum_hs_sq += eq * eq

    return sum_hs_sq / num_rollouts if num_rollouts > 0 else 0.25


# ---------------------------------------------------------------------------
# WS3: Suit isomorphism canonicalization (Waugh 2013)
# ---------------------------------------------------------------------------

# Canonical suit ordering: map suits to indices for lexicographic comparison
_SUIT_ORD = {'c': 0, 'd': 1, 'h': 2, 's': 3}
_SUITS_LIST = ['c', 'd', 'h', 's']


def canonicalize_hand_board(hole_cards: list[Card],
                            community: list[Card]) -> tuple[list[Card], list[Card]]:
    """Canonicalize hole cards + board under suit isomorphism.

    Two hand-board combinations are suit-isomorphic if one can be obtained
    from the other by a permutation of suits. This function returns the
    lexicographically smallest representation, ensuring isomorphic hands
    map to the same canonical form.

    Algorithm (Waugh 2013):
    1. Build the suit signature: for each suit, record its occurrences on the
       board as a sorted tuple of ranks.
    2. Group suits with identical board signatures (these are freely
       interchangeable).
    3. Within each equivalence class, try all permutations and pick the one
       that produces the lexicographically smallest (hole, board) tuple.

    For preflop (empty community), this reduces to: suited pairs are
    canonical as-is (c,c), offsuit pairs are canonical as (c,d).

    Args:
        hole_cards: 2-element list of Card
        community:  0-5 element list of Card

    Returns:
        (canonical_hole, canonical_board) with remapped suits
    """
    from itertools import permutations

    if not community:
        # Preflop: simple canonicalization
        c1, c2 = hole_cards[0], hole_cards[1]
        if c1.suit == c2.suit:
            return [Card(c1.rank, 'c'), Card(c2.rank, 'c')], []
        else:
            # Offsuit: map to (c, d)
            r1, r2 = max(c1.rank, c2.rank), min(c1.rank, c2.rank)
            return [Card(r1, 'c'), Card(r2, 'd')], []

    # Build board suit signature: suit -> sorted tuple of ranks on board
    board_sig = {}
    for suit in _SUITS_LIST:
        ranks = sorted(c.rank for c in community if c.suit == suit)
        board_sig[suit] = tuple(ranks)

    # Group suits by their board signature
    sig_to_suits = {}
    for suit, sig in board_sig.items():
        sig_to_suits.setdefault(sig, []).append(suit)

    # Get the distinct signature groups, sorted by signature for determinism
    sig_groups = sorted(sig_to_suits.values(),
                        key=lambda g: board_sig[g[0]], reverse=True)

    # Generate all valid suit permutations:
    # Within each signature group, any permutation is valid (these suits
    # are interchangeable on the board). Across groups, no swapping.
    def _generate_perms(groups):
        if not groups:
            yield {}
            return
        group = groups[0]
        rest = groups[1:]
        for perm in permutations(group):
            mapping = dict(zip(group, perm))
            for rest_mapping in _generate_perms(rest):
                full = {**mapping, **rest_mapping}
                yield full

    def _apply(mapping, cards):
        return tuple((c.rank, mapping.get(c.suit, c.suit)) for c in cards)

    def _card_key(rank_suit):
        return (rank_suit[0], _SUIT_ORD.get(rank_suit[1], 0))

    best = None
    best_mapping = None

    for mapping in _generate_perms(sig_groups):
        hole_mapped = _apply(mapping, hole_cards)
        board_mapped = _apply(mapping, community)
        # Create a sortable key: board first (sorted), then hole (sorted)
        key = (tuple(sorted(board_mapped, key=_card_key)),
               tuple(sorted(hole_mapped, key=_card_key)))
        if best is None or key < best:
            best = key
            best_mapping = mapping

    # Apply best mapping
    canon_hole = [Card(c.rank, best_mapping.get(c.suit, c.suit))
                  for c in hole_cards]
    canon_board = [Card(c.rank, best_mapping.get(c.suit, c.suit))
                   for c in community]
    return canon_hole, canon_board


def _load_emd_data():
    """Lazy-load EMD centroids and preflop table on first use."""
    global _EMD_CENTROIDS, _EMD_PREFLOP_TABLE
    if _EMD_CENTROIDS is None:
        from server.gto.emd_clustering import load_centroids, load_preflop_table
        _EMD_CENTROIDS = load_centroids()
        _EMD_PREFLOP_TABLE = load_preflop_table()


def emd_equity_bucket(hole_cards: list[Card], community: list[Card],
                      simulations: int = 300) -> int:
    """Compute equity bucket using EMD cluster assignment (WS4).

    Preflop: direct lookup in precomputed table (0ms).
    Postflop: compute raw equity, map to nearest centroid via precomputed
    boundaries. Same speed as standard bucketing — no histogram or EMD needed.
    """
    from server.gto.emd_clustering import (
        load_equity_boundaries, fast_emd_bucket
    )
    _load_emd_data()

    if len(community) == 0:
        # Preflop: direct lookup
        c1, c2 = hole_cards[0], hole_cards[1]
        high = max(c1.rank, c2.rank)
        low = min(c1.rank, c2.rank)
        suited = c1.suit == c2.suit
        if high == low:
            key = f"{high}_{low}"
        else:
            s = 's' if suited else 'o'
            key = f"{high}_{low}_{s}"
        if key in _EMD_PREFLOP_TABLE:
            return _EMD_PREFLOP_TABLE[key]

    # Postflop (or preflop fallback): compute raw equity, use boundary lookup
    street = {0: 'preflop', 3: 'flop', 4: 'turn', 5: 'river'}.get(
        len(community), 'river')
    boundaries = load_equity_boundaries()[street]

    # Use fast equity (Cython if available)
    if len(community) < 5:
        score = hand_strength_squared(hole_cards, community, 1, simulations)
        score = score ** 0.5
    else:
        score = hand_equity(hole_cards, community, 1, simulations)

    return fast_emd_bucket(score, boundaries)


def hand_strength_bucket(hole_cards: list[Card], community: list[Card],
                         num_opponents: int = 1, num_buckets: int = None,
                         simulations: int = 300,
                         use_ehs2: bool = True) -> int:
    """
    Return a 2D bucket index combining equity and hand type.

    The bucket encodes two dimensions:
    - Equity bucket (0 to NUM_EQUITY_BUCKETS-1): based on E[HS^2]
    - Hand type: structural classification (pair, suited connector, etc.)

    This separates hands like AKo vs QJs that have similar equity but
    very different strategic properties (blocker effects, playability).

    When use_ehs2=True, uses Expected Hand Strength Squared for the equity
    dimension [Zinkevich et al. 2007, Section 4.1].
    """
    # WS3: canonicalize under suit isomorphism before computing equity
    # This ensures isomorphic hands always map to the same bucket.
    if SUIT_ISO_ENABLED:
        canon_hole, canon_board = canonicalize_hand_board(hole_cards, community)
    else:
        canon_hole, canon_board = hole_cards, community

    # Equity dimension
    if EMD_MODE_ENABLED:
        # WS4: EMD cluster assignment replaces uniform bands
        eq_bucket = emd_equity_bucket(canon_hole, canon_board, simulations)
    elif use_ehs2 and len(canon_board) < 5:
        score = hand_strength_squared(
            canon_hole, canon_board, num_opponents, simulations)
        score = score ** 0.5
        eq_bucket = int(score * NUM_EQUITY_BUCKETS)
        eq_bucket = min(eq_bucket, NUM_EQUITY_BUCKETS - 1)
    else:
        score = hand_equity(
            canon_hole, canon_board, num_opponents, simulations)
        eq_bucket = int(score * NUM_EQUITY_BUCKETS)
        eq_bucket = min(eq_bucket, NUM_EQUITY_BUCKETS - 1)

    # Hand type dimension (uses original cards — rank+suited doesn't change)
    c1, c2 = hole_cards[0], hole_cards[1]
    suited = c1.suit == c2.suit
    hand_type = classify_hand_type(c1.rank, c2.rank, suited)

    return make_bucket(eq_bucket, int(hand_type))


def preflop_equity_table(simulations: int = 1000) -> dict[str, float]:
    """
    Pre-compute equity for all 169 canonical starting hands.
    Returns dict mapping hand string (e.g., "AKs", "TT", "72o") to equity.
    """
    table = {}
    ranks = list(range(2, 15))

    for i, r1 in enumerate(ranks):
        for j, r2 in enumerate(ranks):
            if j > i:
                continue

            if r1 == r2:
                # Pair
                key = _rank_char(r1) + _rank_char(r2)
                cards = [Card(r1, 'h'), Card(r2, 'd')]
            elif j < i:
                # r1 > r2 (since we iterate high to low effectively)
                high, low = max(r1, r2), min(r1, r2)
                # Suited
                key_s = _rank_char(high) + _rank_char(low) + 's'
                cards_s = [Card(high, 'h'), Card(low, 'h')]
                table[key_s] = hand_equity(cards_s, [], 1, simulations)

                # Offsuit
                key_o = _rank_char(high) + _rank_char(low) + 'o'
                cards_o = [Card(high, 'h'), Card(low, 'd')]
                table[key_o] = hand_equity(cards_o, [], 1, simulations)
                continue

            table[key] = hand_equity(cards, [], 1, simulations)

    return table


def _rank_char(rank: int) -> str:
    m = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T'}
    return m.get(rank, str(rank))


def canonicalize_hand(hole_cards: list[Card]) -> str:
    """Convert two hole cards to canonical form like 'AKs', 'TT', '72o'."""
    c1, c2 = hole_cards
    high = max(c1.rank, c2.rank)
    low = min(c1.rank, c2.rank)
    suited = c1.suit == c2.suit

    if high == low:
        return _rank_char(high) + _rank_char(low)
    suffix = 's' if suited else 'o'
    return _rank_char(high) + _rank_char(low) + suffix
