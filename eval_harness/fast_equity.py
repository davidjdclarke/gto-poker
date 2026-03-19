"""
Fast equity and bucketing for evaluation harness.

The standard hand_strength_bucket() does nested Monte Carlo:
  80 outer sims x 50 inner sims = 4,000 best_hand calls per decision (~340ms).

This module provides cached and approximate alternatives:

  1. Preflop bucket cache: only 169 canonical hands, computed once.
  2. Fast postflop EHS2: fewer sims + result caching by (hole, board) key.
  3. Direct equity bucketing: skip EHS2 nesting, use raw equity for bucketing.

Speedup: ~20-50x for preflop, ~5-10x for postflop.
"""
import random
from functools import lru_cache
from server.deck import Card, SUITS
from server.evaluator import best_hand
from server.gto.equity import hand_equity, make_deck
from server.gto.abstraction import (
    classify_hand_type, make_bucket, NUM_EQUITY_BUCKETS, NUM_HAND_TYPES,
)

# Try to use Cython-accelerated equity (~47x faster)
try:
    from eval_harness.eval_fast import hand_equity_fast as _cy_equity
    _HAS_FAST_EVAL = True
except ImportError:
    _HAS_FAST_EVAL = False


# ---------------------------------------------------------------------------
# Preflop bucket cache
# ---------------------------------------------------------------------------
_PREFLOP_BUCKET_CACHE: dict[str, int] = {}
_PREFLOP_CACHE_BUILT = False


def _canonical_key(hole_cards: list[Card]) -> str:
    """Canonical hand key: 'high_low_s' or 'high_low_o' or 'rank_rank'."""
    c1, c2 = hole_cards
    high, low = max(c1.rank, c2.rank), min(c1.rank, c2.rank)
    if high == low:
        return f"{high}_{low}"
    suited = 's' if c1.suit == c2.suit else 'o'
    return f"{high}_{low}_{suited}"


def _ehs2_fast(hole_cards, community, simulations=300):
    """Compute E[HS^2] using Cython equity if available."""
    import random as _rng
    if len(community) >= 5:
        if _HAS_FAST_EVAL:
            eq = _cy_equity(hole_cards, community, simulations=simulations)
        else:
            eq = hand_equity(hole_cards, community, num_opponents=1,
                             simulations=simulations)
        return eq * eq

    dead = set((c.rank, c.suit) for c in hole_cards + community)
    remaining = [c for c in make_deck() if (c.rank, c.suit) not in dead]
    num_rollouts = min(simulations, len(remaining))
    sampled_cards = _rng.sample(remaining, num_rollouts)

    sum_hs_sq = 0.0
    inner_sims = max(50, simulations // 5)
    for next_card in sampled_cards:
        future_community = list(community) + [next_card]
        if _HAS_FAST_EVAL:
            eq = _cy_equity(hole_cards, future_community,
                            simulations=inner_sims)
        else:
            eq = hand_equity(hole_cards, future_community,
                             num_opponents=1, simulations=inner_sims)
        sum_hs_sq += eq * eq

    return sum_hs_sq / num_rollouts if num_rollouts > 0 else 0.0


def build_preflop_cache(simulations: int = 500):
    """
    Build bucket cache for all 169 canonical preflop hands.
    With Cython eval: ~1s. Without: ~30s. Only needs to run once per session.
    """
    global _PREFLOP_CACHE_BUILT
    if _PREFLOP_CACHE_BUILT:
        return

    for r1 in range(2, 15):
        for r2 in range(2, r1 + 1):
            for suited in ([True, False] if r1 != r2 else [False]):
                s1, s2 = ('h', 'h') if suited else ('h', 'd')
                cards = [Card(r1, s1), Card(r2, s2)]
                key = _canonical_key(cards)
                if key in _PREFLOP_BUCKET_CACHE:
                    continue

                ehs2 = _ehs2_fast(cards, [], simulations=simulations)
                score = ehs2 ** 0.5
                eq_bucket = min(int(score * NUM_EQUITY_BUCKETS),
                                NUM_EQUITY_BUCKETS - 1)

                hand_type = classify_hand_type(r1, r2, suited)
                bucket = make_bucket(eq_bucket, int(hand_type))
                _PREFLOP_BUCKET_CACHE[key] = bucket

    _PREFLOP_CACHE_BUILT = True


def get_preflop_bucket(hole_cards: list[Card]) -> int:
    """Look up preflop bucket from cache. Falls back to computation if needed."""
    key = _canonical_key(hole_cards)
    if key in _PREFLOP_BUCKET_CACHE:
        return _PREFLOP_BUCKET_CACHE[key]

    # Cache miss — compute and store
    ehs2 = _ehs2_fast(hole_cards, [], simulations=300)
    score = ehs2 ** 0.5
    eq_bucket = min(int(score * NUM_EQUITY_BUCKETS), NUM_EQUITY_BUCKETS - 1)
    c1, c2 = hole_cards
    hand_type = classify_hand_type(c1.rank, c2.rank, c1.suit == c2.suit)
    bucket = make_bucket(eq_bucket, int(hand_type))
    _PREFLOP_BUCKET_CACHE[key] = bucket
    return bucket


# ---------------------------------------------------------------------------
# Fast postflop bucketing (skip nested MC, use direct equity)
# ---------------------------------------------------------------------------
def fast_postflop_bucket(hole_cards: list[Card], community: list[Card],
                         simulations: int = 100) -> int:
    """
    Compute postflop bucket using direct equity instead of E[HS^2].

    Standard E[HS^2] does nested MC (outer sim x inner sim). This uses
    raw equity with more sims for similar accuracy at ~10x speed.

    The bucket won't exactly match training buckets, but for evaluation
    purposes (where we need speed over exact match) this is fine.
    """
    if _HAS_FAST_EVAL:
        eq = _cy_equity(hole_cards, community, simulations=simulations)
    else:
        eq = hand_equity(hole_cards, community, num_opponents=1,
                         simulations=simulations)
    eq_bucket = min(int(eq * NUM_EQUITY_BUCKETS), NUM_EQUITY_BUCKETS - 1)

    c1, c2 = hole_cards
    hand_type = classify_hand_type(c1.rank, c2.rank, c1.suit == c2.suit)
    return make_bucket(eq_bucket, int(hand_type))


# ---------------------------------------------------------------------------
# Cached postflop EHS2 (still nested MC but with caching)
# ---------------------------------------------------------------------------
_POSTFLOP_CACHE: dict[str, int] = {}
_POSTFLOP_CACHE_MAX = 50000  # Limit cache size


def _board_key(hole_cards: list[Card], community: list[Card]) -> str:
    """Unique key for (hole, board) combination."""
    h = tuple(sorted((c.rank, c.suit) for c in hole_cards))
    b = tuple(sorted((c.rank, c.suit) for c in community))
    return f"{h}:{b}"


def cached_postflop_bucket(hole_cards: list[Card], community: list[Card],
                            simulations: int = 40) -> int:
    """
    Postflop bucket with LRU-style cache. Same board+hole = same bucket.
    Uses reduced sims (40 vs 80) since this is eval, not training.
    """
    key = _board_key(hole_cards, community)

    if key in _POSTFLOP_CACHE:
        return _POSTFLOP_CACHE[key]

    # Use fast direct equity instead of nested EHS2
    bucket = fast_postflop_bucket(hole_cards, community, simulations)

    if len(_POSTFLOP_CACHE) < _POSTFLOP_CACHE_MAX:
        _POSTFLOP_CACHE[key] = bucket

    return bucket


# ---------------------------------------------------------------------------
# Fast equity float (for embedding model feature extraction)
# ---------------------------------------------------------------------------
def fast_equity_float(hole_cards: list[Card], community: list[Card],
                      simulations: int = 100) -> float:
    """Return raw equity float in [0, 1] for embedding feature extraction.

    Preflop: approximates from cached bucket (avoids MC).
    Postflop: uses Cython equity if available, else Python fallback.
    """
    if not community:
        # Approximate from cached preflop bucket
        bucket = get_preflop_bucket(hole_cards)
        eq_bucket = bucket // NUM_HAND_TYPES
        return (eq_bucket + 0.5) / NUM_EQUITY_BUCKETS
    if _HAS_FAST_EVAL:
        return float(_cy_equity(hole_cards, community, simulations=simulations))
    return float(hand_equity(hole_cards, community, num_opponents=1,
                             simulations=simulations))


# ---------------------------------------------------------------------------
# Unified fast bucket function
# ---------------------------------------------------------------------------
def fast_bucket(hole_cards: list[Card], community: list[Card],
                simulations: int = 100) -> int:
    """
    Fast bucket computation for eval harness.

    Preflop: uses cache (0ms after warmup).
    Postflop: uses direct equity (~5-10ms) instead of nested EHS2 (~340ms).
    EMD mode: delegates to equity.py EMD bucketing for full histogram accuracy.
    """
    from server.gto.equity import EMD_MODE_ENABLED
    if EMD_MODE_ENABLED:
        from server.gto.emd_clustering import load_equity_boundaries, fast_emd_bucket, load_preflop_table
        if len(community) == 0:
            # Preflop: direct table lookup
            table = load_preflop_table()
            c1, c2 = hole_cards[0], hole_cards[1]
            high, low = max(c1.rank, c2.rank), min(c1.rank, c2.rank)
            if high == low:
                pkey = f"{high}_{low}"
            else:
                pkey = f"{high}_{low}_{'s' if c1.suit == c2.suit else 'o'}"
            eq_cluster = table.get(pkey, 0)
        else:
            # Postflop: fast equity → boundary lookup (no histogram)
            street = {3: 'flop', 4: 'turn', 5: 'river'}.get(len(community), 'river')
            if _HAS_FAST_EVAL:
                eq = _cy_equity(hole_cards, community, simulations=simulations)
            else:
                eq = hand_equity(hole_cards, community, num_opponents=1,
                                 simulations=simulations)
            boundaries = load_equity_boundaries()[street]
            eq_cluster = fast_emd_bucket(eq, boundaries)
        hand_type = classify_hand_type(hole_cards[0].rank, hole_cards[1].rank,
                                       hole_cards[0].suit == hole_cards[1].suit)
        return make_bucket(eq_cluster, int(hand_type))
    if len(community) == 0:
        return get_preflop_bucket(hole_cards)
    return cached_postflop_bucket(hole_cards, community, simulations)


# ---------------------------------------------------------------------------
# Fast equity for adversary bots
# ---------------------------------------------------------------------------
_BOT_EQUITY_CACHE: dict[str, float] = {}


def fast_bot_equity(hole_cards: list[Card], community: list[Card]) -> float:
    """
    Fast equity for adversary bot decisions.
    Preflop: cached (169 entries).
    Postflop: 30 sims, no cache (boards vary too much).
    """
    if len(community) == 0:
        key = _canonical_key(hole_cards)
        if key not in _BOT_EQUITY_CACHE:
            if _HAS_FAST_EVAL:
                _BOT_EQUITY_CACHE[key] = _cy_equity(
                    hole_cards, [], simulations=200)
            else:
                _BOT_EQUITY_CACHE[key] = hand_equity(
                    hole_cards, [], num_opponents=1, simulations=200)
        return _BOT_EQUITY_CACHE[key]
    if _HAS_FAST_EVAL:
        return _cy_equity(hole_cards, community, simulations=30)
    return hand_equity(hole_cards, community, num_opponents=1, simulations=30)
