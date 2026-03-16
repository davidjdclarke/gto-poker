"""
Board texture classification for structured abstraction (v11 Workstream D3).

Adds a board-texture signal to the abstraction without expanding the bucket
space. Instead of broadening the bucket grid (which would explode node count),
the texture class is used as an auxiliary input to the confidence blending
system and local refinement trigger policy.

Board textures:
    0 = DRY       : no flush draw, no straight draw, unpaired board
    1 = MONOTONE  : 3+ cards of same suit (flush possible)
    2 = DRAW_HEAVY: open-ended straight draw or flush draw possible
    3 = PAIRED    : board has a pair (full house/trips possible)
    4 = CONNECTED : 3+ cards within 4 ranks (strong straight potential)

Usage:
    from server.gto.board_texture import classify_board_texture, BoardTexture

    texture = classify_board_texture(community_cards)
    # Returns BoardTexture enum value
"""
from enum import IntEnum
from collections import Counter


class BoardTexture(IntEnum):
    DRY = 0
    MONOTONE = 1
    DRAW_HEAVY = 2
    PAIRED = 3
    CONNECTED = 4


def classify_board_texture(community_cards) -> BoardTexture:
    """Classify the board texture from community cards.

    Args:
        community_cards: list of Card objects with .rank and .suit attributes

    Returns:
        BoardTexture enum value
    """
    if not community_cards:
        return BoardTexture.DRY

    ranks = [c.rank for c in community_cards]
    suits = [c.suit for c in community_cards]

    # Check for monotone (3+ same suit)
    suit_counts = Counter(suits)
    max_suit = max(suit_counts.values())
    if max_suit >= 3:
        return BoardTexture.MONOTONE

    # Check for paired board
    rank_counts = Counter(ranks)
    if max(rank_counts.values()) >= 2:
        return BoardTexture.PAIRED

    # Check for connected board (3+ cards within 4-rank window)
    sorted_ranks = sorted(set(ranks))
    if len(sorted_ranks) >= 3:
        for i in range(len(sorted_ranks) - 2):
            window = sorted_ranks[i + 2] - sorted_ranks[i]
            if window <= 4:
                return BoardTexture.CONNECTED

    # Check for draw-heavy (flush draw or OESD possible)
    if max_suit >= 2 and len(community_cards) >= 3:
        # Two-suit with 3+ board cards = flush draw possible
        if max_suit >= 2:
            # Check if there are enough cards for a flush draw
            for suit, count in suit_counts.items():
                if count >= 2:
                    # Two suited cards on board + player could have 2 more
                    # Check if board has straight draw potential too
                    if _has_straight_draw(sorted_ranks):
                        return BoardTexture.DRAW_HEAVY

    # Check for straight draw potential alone
    if _has_straight_draw(sorted_ranks):
        return BoardTexture.DRAW_HEAVY

    return BoardTexture.DRY


def _has_straight_draw(sorted_ranks: list[int]) -> bool:
    """Check if the board has open-ended straight draw potential."""
    if len(sorted_ranks) < 2:
        return False

    # Check for consecutive or near-consecutive cards
    gaps = 0
    for i in range(len(sorted_ranks) - 1):
        diff = sorted_ranks[i + 1] - sorted_ranks[i]
        if diff <= 2:
            gaps += 1

    # 2+ close gaps = draw-heavy
    return gaps >= 2


def texture_name(texture: BoardTexture) -> str:
    """Human-readable name for a board texture."""
    return {
        BoardTexture.DRY: "dry",
        BoardTexture.MONOTONE: "monotone",
        BoardTexture.DRAW_HEAVY: "draw-heavy",
        BoardTexture.PAIRED: "paired",
        BoardTexture.CONNECTED: "connected",
    }.get(texture, "unknown")


def texture_adjustments(texture: BoardTexture,
                        eq_bucket: int) -> dict[str, float]:
    """Suggest strategy adjustments based on board texture.

    Returns multipliers for action categories. Used by confidence blending
    to adjust the heuristic strategy based on board texture.

    Args:
        texture: BoardTexture classification
        eq_bucket: Player's equity bucket (0-7)

    Returns:
        Dict with 'bet_mult' and 'check_mult' adjustment factors.
    """
    if texture == BoardTexture.MONOTONE:
        # Monotone boards: more checking, less betting (many draws)
        if eq_bucket <= 3:
            return {'bet_mult': 0.7, 'check_mult': 1.3}
        else:
            return {'bet_mult': 1.1, 'check_mult': 0.9}

    elif texture == BoardTexture.PAIRED:
        # Paired boards: polarize (bet big or check, less medium bets)
        if eq_bucket <= 2:
            return {'bet_mult': 1.2, 'check_mult': 0.8}  # more bluffs
        elif eq_bucket <= 5:
            return {'bet_mult': 0.8, 'check_mult': 1.2}  # trap more
        else:
            return {'bet_mult': 1.3, 'check_mult': 0.7}  # value bet

    elif texture == BoardTexture.DRAW_HEAVY:
        # Draw-heavy: bet more to protect equity, deny draws
        if eq_bucket >= 4:
            return {'bet_mult': 1.3, 'check_mult': 0.7}
        else:
            return {'bet_mult': 0.9, 'check_mult': 1.1}

    elif texture == BoardTexture.CONNECTED:
        # Connected: similar to draw-heavy but less extreme
        if eq_bucket >= 4:
            return {'bet_mult': 1.15, 'check_mult': 0.85}
        else:
            return {'bet_mult': 0.95, 'check_mult': 1.05}

    # DRY: no adjustment
    return {'bet_mult': 1.0, 'check_mult': 1.0}
