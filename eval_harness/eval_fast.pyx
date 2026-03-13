# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Fast hand evaluation and equity calculation in Cython.

Replaces the pure-Python best_hand() and hand_equity() for gauntlet evaluation.
Provides ~10-20x speedup by avoiding Python object overhead.

Cards are encoded as integers: card = rank * 4 + suit_idx
  rank: 2-14 (14=Ace)
  suit_idx: 0=h, 1=d, 2=c, 3=s
"""

from libc.stdlib cimport rand, srand, RAND_MAX

# ── Card encoding ──
# card = rank * 4 + suit_idx
# rank: 0-12 mapped from 2-14 (subtract 2)
# suit: 0-3

cdef inline int card_rank(int card) noexcept nogil:
    return card >> 2  # card // 4  (0-12, where 0=2, 12=Ace)

cdef inline int card_suit(int card) noexcept nogil:
    return card & 3   # card % 4


# ── Hand evaluation ──
# Returns a single int encoding hand rank + tiebreakers.
# Format: hand_category * (15^5) + kicker encoding
# Higher value = better hand.

cdef int MULT = 15  # base for kicker encoding (ranks 0-14 fit in 15)
cdef int CAT_SHIFT = MULT * MULT * MULT * MULT * MULT  # 15^5 = 759375

cdef int evaluate_5(int* cards) noexcept nogil:
    """Evaluate exactly 5 cards. Returns comparable int (higher = better)."""
    cdef int ranks[5]
    cdef int suits[5]
    cdef int rank_count[15]  # index by rank (0-14 to be safe)
    cdef int i, j, temp
    cdef bint is_flush, is_straight
    cdef int straight_high
    cdef int unique[5]
    cdef int n_unique

    # Extract and sort ranks descending
    for i in range(5):
        ranks[i] = card_rank(cards[i])
        suits[i] = card_suit(cards[i])

    # Bubble sort descending (only 5 elements)
    for i in range(4):
        for j in range(i + 1, 5):
            if ranks[j] > ranks[i]:
                temp = ranks[i]; ranks[i] = ranks[j]; ranks[j] = temp
                temp = suits[i]; suits[i] = suits[j]; suits[j] = temp

    # Count ranks
    for i in range(15):
        rank_count[i] = 0
    for i in range(5):
        rank_count[ranks[i]] += 1

    # Check flush
    is_flush = (suits[0] == suits[1] == suits[2] == suits[3] == suits[4])

    # Check straight - get unique ranks
    n_unique = 0
    for i in range(5):
        if n_unique == 0 or unique[n_unique - 1] != ranks[i]:
            unique[n_unique] = ranks[i]
            n_unique += 1

    is_straight = False
    straight_high = 0
    if n_unique == 5:
        if unique[0] - unique[4] == 4:
            is_straight = True
            straight_high = unique[0]
        # Wheel: A-5-4-3-2 (Ace=12, 5=3, 4=2, 3=1, 2=0)
        if unique[0] == 12 and unique[1] == 3 and unique[2] == 2 and unique[3] == 1 and unique[4] == 0:
            is_straight = True
            straight_high = 3  # 5-high straight

    # Straight flush / Royal flush (category 8)
    if is_straight and is_flush:
        return 8 * CAT_SHIFT + straight_high

    # Find counts for pattern matching
    cdef int quad_rank = -1, trip_rank = -1
    cdef int pair1_rank = -1, pair2_rank = -1
    cdef int kickers[5]
    cdef int n_kickers = 0

    # Scan ranks descending to find patterns
    for i in range(12, -1, -1):  # 12 down to 0 (Ace down to 2)
        pass  # We'll use a different approach

    # Count-based pattern detection
    cdef int fours = -1, threes = -1, pair_hi = -1, pair_lo = -1
    n_kickers = 0

    # Scan from highest rank down
    for i in range(12, -1, -1):
        if rank_count[i] == 4:
            fours = i
        elif rank_count[i] == 3:
            threes = i
        elif rank_count[i] == 2:
            if pair_hi < 0:
                pair_hi = i
            else:
                pair_lo = i
        elif rank_count[i] == 1:
            kickers[n_kickers] = i
            n_kickers += 1

    # Four of a kind (category 7)
    if fours >= 0:
        # Kicker is the remaining card
        temp = 0
        for i in range(15):
            if rank_count[i] > 0 and i != fours:
                temp = i
        return 7 * CAT_SHIFT + fours * MULT + temp

    # Full house (category 6)
    if threes >= 0 and pair_hi >= 0:
        return 6 * CAT_SHIFT + threes * MULT + pair_hi

    # Flush (category 5)
    if is_flush:
        return 5 * CAT_SHIFT + (ranks[0] * MULT*MULT*MULT*MULT +
                                  ranks[1] * MULT*MULT*MULT +
                                  ranks[2] * MULT*MULT +
                                  ranks[3] * MULT +
                                  ranks[4])

    # Straight (category 4)
    if is_straight:
        return 4 * CAT_SHIFT + straight_high

    # Three of a kind (category 3)
    if threes >= 0:
        return 3 * CAT_SHIFT + (threes * MULT * MULT +
                                 kickers[0] * MULT +
                                 kickers[1])

    # Two pair (category 2)
    if pair_hi >= 0 and pair_lo >= 0:
        return 2 * CAT_SHIFT + (pair_hi * MULT * MULT +
                                 pair_lo * MULT +
                                 kickers[0])

    # One pair (category 1)
    if pair_hi >= 0:
        return 1 * CAT_SHIFT + (pair_hi * MULT * MULT * MULT +
                                 kickers[0] * MULT * MULT +
                                 kickers[1] * MULT +
                                 kickers[2])

    # High card (category 0)
    return (ranks[0] * MULT*MULT*MULT*MULT +
            ranks[1] * MULT*MULT*MULT +
            ranks[2] * MULT*MULT +
            ranks[3] * MULT +
            ranks[4])


cdef int best_hand_7(int* all_cards) noexcept nogil:
    """Find best 5-card hand from 7 cards. Returns evaluation score."""
    cdef int combo[5]
    cdef int best = -1
    cdef int val
    cdef int i, j, k, l, m, idx

    # All C(7,5) = 21 combinations
    for i in range(3):
        for j in range(i + 1, 4):
            for k in range(j + 1, 5):
                for l in range(k + 1, 6):
                    for m in range(l + 1, 7):
                        combo[0] = all_cards[i]
                        combo[1] = all_cards[j]
                        combo[2] = all_cards[k]
                        combo[3] = all_cards[l]
                        combo[4] = all_cards[m]
                        val = evaluate_5(combo)
                        if val > best:
                            best = val
    return best


# ── Fisher-Yates shuffle ──

cdef inline unsigned int xorshift32(unsigned int* state) noexcept nogil:
    """Simple xorshift32 PRNG."""
    cdef unsigned int s = state[0]
    s ^= s << 13
    s ^= s >> 17
    s ^= s << 5
    state[0] = s
    return s

cdef void shuffle_array(int* arr, int n, unsigned int* rng) noexcept nogil:
    """Fisher-Yates shuffle."""
    cdef int i, j, temp
    for i in range(n - 1, 0, -1):
        j = xorshift32(rng) % (i + 1)
        temp = arr[i]
        arr[i] = arr[j]
        arr[j] = temp


# ── Monte Carlo equity ──
def hand_equity_fast(list hole_cards_py, list community_py,
                     int simulations=100, unsigned int seed=0) -> float:
    """
    Fast Monte Carlo equity calculation.

    Args:
        hole_cards_py: list of Card objects (2 cards)
        community_py: list of Card objects (0-5 cards)
        simulations: number of MC simulations
        seed: RNG seed (0 = use default)

    Returns:
        float equity in [0, 1]
    """
    cdef unsigned int rng_state = seed if seed > 0 else 42

    cdef int hole[2]
    cdef int community[5]
    cdef int n_community = len(community_py)
    cdef int remaining[52]
    cdef int n_remaining = 0
    cdef int dead[9]  # max 2 hole + 5 community + 2 buffer
    cdef int n_dead = 0
    cdef int i, j
    cdef bint is_dead

    # Suit mapping
    cdef dict suit_map = {'h': 0, 'd': 1, 'c': 2, 's': 3}

    # Encode hole cards
    for i in range(2):
        c = hole_cards_py[i]
        hole[i] = (c.rank - 2) * 4 + suit_map[c.suit]
        dead[n_dead] = hole[i]
        n_dead += 1

    # Encode community
    for i in range(n_community):
        c = community_py[i]
        community[i] = (c.rank - 2) * 4 + suit_map[c.suit]
        dead[n_dead] = community[i]
        n_dead += 1

    # Build remaining deck
    for i in range(52):
        is_dead = False
        for j in range(n_dead):
            if i == dead[j]:
                is_dead = True
                break
        if not is_dead:
            remaining[n_remaining] = i
            n_remaining += 1

    cdef int cards_needed = 5 - n_community
    cdef int wins = 0
    cdef int ties_count = 0
    cdef int total = simulations
    cdef int all_cards[7]
    cdef int opp_cards[7]
    cdef int my_val, opp_val
    cdef int idx

    # Pre-fill hole cards
    all_cards[0] = hole[0]
    all_cards[1] = hole[1]

    for i in range(simulations):
        shuffle_array(remaining, n_remaining, &rng_state)
        idx = 0

        # Fill community
        for j in range(n_community):
            all_cards[2 + j] = community[j]
        for j in range(cards_needed):
            all_cards[2 + n_community + j] = remaining[idx]
            idx += 1

        # Opponent cards
        opp_cards[0] = remaining[idx]
        opp_cards[1] = remaining[idx + 1]
        idx += 2

        # Fill opponent community (same board)
        for j in range(n_community):
            opp_cards[2 + j] = community[j]
        for j in range(cards_needed):
            opp_cards[2 + n_community + j] = all_cards[2 + n_community + j]

        my_val = best_hand_7(all_cards)
        opp_val = best_hand_7(opp_cards)

        if my_val > opp_val:
            wins += 1
        elif my_val == opp_val:
            ties_count += 1

    if total == 0:
        return 0.5
    return (<double>wins + <double>ties_count * 0.5) / <double>total


def set_seed(unsigned int seed):
    """Set the RNG seed (no-op, seed is now per-call via hand_equity_fast)."""
    pass


def evaluate_hand(list hole_cards_py, list community_py) -> int:
    """
    Evaluate best 5-card hand from hole + community cards.
    Returns comparable integer (higher = better).
    """
    cdef int all_cards[7]
    cdef int n = 0
    cdef dict suit_map = {'h': 0, 'd': 1, 'c': 2, 's': 3}
    cdef int combo[5]
    cdef int best = -1
    cdef int val
    cdef int i, j, k, l, m

    for c in hole_cards_py:
        all_cards[n] = (c.rank - 2) * 4 + suit_map[c.suit]
        n += 1
    for c in community_py:
        all_cards[n] = (c.rank - 2) * 4 + suit_map[c.suit]
        n += 1

    if n == 7:
        return best_hand_7(all_cards)
    elif n == 5:
        return evaluate_5(all_cards)
    else:
        # For 6 cards, enumerate C(6,5) = 6 combos
        for i in range(n - 4):
            for j in range(i + 1, n - 3):
                for k in range(j + 1, n - 2):
                    for l in range(k + 1, n - 1):
                        for m in range(l + 1, n):
                            combo[0] = all_cards[i]
                            combo[1] = all_cards[j]
                            combo[2] = all_cards[k]
                            combo[3] = all_cards[l]
                            combo[4] = all_cards[m]
                            val = evaluate_5(combo)
                            if val > best:
                                best = val
        return best
