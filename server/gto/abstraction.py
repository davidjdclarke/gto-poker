"""
Game state abstraction for CFR.

Abstracts the poker game into a tractable form:
- Card abstraction: 2D bucketing by (equity_bucket, hand_type)
  Separates hands with similar equity but different strategic properties.
- Action abstraction: phase-aware bet sizing menus
  Preflop uses realistic open/3bet/4bet sizes; postflop uses pot-fraction bets.
- Position encoding: early (EP/MP) vs late (CO/BTN) vs blinds
"""
from enum import IntEnum


# --- Action Abstraction ---

class Action(IntEnum):
    FOLD = 0
    CHECK_CALL = 1       # Check if no bet to call, otherwise call
    BET_THIRD_POT = 2    # Bet/raise ~1/3 pot (postflop, thin value/probes)
    BET_HALF_POT = 3     # Bet/raise half pot (postflop)
    BET_POT = 4          # Bet/raise full pot (postflop)
    ALL_IN = 5           # All-in
    # Preflop-specific actions
    OPEN_RAISE = 6       # Open raise (~2.5bb)
    THREE_BET = 7        # 3-bet (~3x open)
    FOUR_BET = 8         # 4-bet / squeeze
    # v6 additions
    BET_TWO_THIRDS_POT = 9   # Bet/raise ~2/3 pot (fills 50-100% gap)
    BET_OVERBET = 10          # Overbet ~1.25x pot (110-150% region)
    DONK_SMALL = 11           # Donk lead ~1/4 pot (flop/turn OOP probe)
    DONK_MEDIUM = 12          # Donk lead ~1/2 pot (flop/turn OOP lead)


ACTION_NAMES = {
    Action.FOLD: "fold",
    Action.CHECK_CALL: "check/call",
    Action.BET_THIRD_POT: "bet 1/3 pot",
    Action.BET_HALF_POT: "bet half pot",
    Action.BET_TWO_THIRDS_POT: "bet 2/3 pot",
    Action.BET_POT: "bet pot",
    Action.BET_OVERBET: "overbet 1.25x",
    Action.ALL_IN: "all-in",
    Action.OPEN_RAISE: "open raise",
    Action.THREE_BET: "3-bet",
    Action.FOUR_BET: "4-bet",
    Action.DONK_SMALL: "donk small",
    Action.DONK_MEDIUM: "donk medium",
}


# --- Hand Type Classification ---

class HandType(IntEnum):
    """Structural hand categories that affect strategy beyond raw equity.

    v5: expanded from 11 to 15 types. Splits target the most lossy
    categories where strategic differences were being merged:
    - Broadway split: AK/AQ play very differently from KQ/QJ
    - Suited connector split: high vs low playability differs
    - Offsuit ace split: ATo+ are playable, A9o- are marginal
    - Suited trash: suited non-connectors have flush equity
    """
    PREMIUM_PAIR = 0         # AA, KK, QQ
    HIGH_PAIR = 1            # JJ, TT
    MID_PAIR = 2             # 99-66
    LOW_PAIR = 3             # 55-22
    STRONG_BROADWAY = 4      # AK, AQ (strong blocker + high equity)
    BROADWAY = 5             # KQ, KJ, QJ, QT, JT (weaker broadway)
    SUITED_ACE = 6           # Axs (nut flush + blocker)
    HIGH_SUITED_CONNECTOR = 7  # T9s, 98s, 87s (high playability)
    LOW_SUITED_CONNECTOR = 8   # 76s, 65s, 54s (speculative)
    SUITED_GAPPER = 9        # One-gap suited (T8s, 97s, etc.)
    STRONG_OFFSUIT_ACE = 10  # ATo, AJo (playable offsuit aces)
    WEAK_OFFSUIT_ACE = 11    # A9o- (marginal offsuit aces)
    OFFSUIT_BROADWAY = 12    # Offsuit broadway not covered above
    SUITED_TRASH = 13        # Suited non-connectors (flush equity)
    TRASH = 14               # Offsuit junk


NUM_HAND_TYPES = len(HandType)
NUM_EQUITY_BUCKETS = 8   # Equity dimension of 2D bucket
NUM_BUCKETS = NUM_EQUITY_BUCKETS * NUM_HAND_TYPES  # Total bucket count


def classify_hand_type(rank1: int, rank2: int, suited: bool) -> HandType:
    """
    Classify two hole cards into a structural hand type.

    Args:
        rank1, rank2: Card ranks (2-14, where 14=Ace)
        suited: Whether the cards share a suit
    """
    high = max(rank1, rank2)
    low = min(rank1, rank2)
    is_pair = (rank1 == rank2)
    gap = high - low - 1  # 0 = connected, 1 = one-gap, etc.

    if is_pair:
        if high >= 12:  # QQ+
            return HandType.PREMIUM_PAIR
        elif high >= 10:  # JJ-TT
            return HandType.HIGH_PAIR
        elif high >= 6:  # 99-66
            return HandType.MID_PAIR
        else:  # 55-22
            return HandType.LOW_PAIR

    is_broadway = (high >= 10 and low >= 10)

    if suited:
        if high == 14:  # Axs
            return HandType.SUITED_ACE
        if is_broadway:
            # AK/AQ suited already captured by SUITED_ACE check above
            # Remaining broadway suited: KQs, KJs, QJs, etc.
            return HandType.BROADWAY
        if gap == 0:
            if low >= 7:  # T9s, 98s, 87s
                return HandType.HIGH_SUITED_CONNECTOR
            else:  # 76s, 65s, 54s, 43s, 32s
                return HandType.LOW_SUITED_CONNECTOR
        if gap <= 1 and low >= 5:  # One-gap suited
            return HandType.SUITED_GAPPER
        # Suited but not connected/gapped/broadway/ace
        return HandType.SUITED_TRASH
    else:
        if high == 14 and low >= 10:  # AKo, AQo, AJo, ATo
            return HandType.STRONG_BROADWAY
        if is_broadway:  # KQo, KJo, QJo, KTo, QTo, JTo
            return HandType.OFFSUIT_BROADWAY
        if high == 14:
            if low >= 7:  # A9o, A8o, A7o
                return HandType.STRONG_OFFSUIT_ACE
            else:  # A6o-A2o
                return HandType.WEAK_OFFSUIT_ACE
        return HandType.TRASH


def make_bucket(equity_bucket: int, hand_type: int) -> int:
    """Combine equity bucket and hand type into a single bucket index."""
    return equity_bucket * NUM_HAND_TYPES + hand_type


def decode_bucket(bucket: int) -> tuple[int, int]:
    """Decode a combined bucket into (equity_bucket, hand_type)."""
    return bucket // NUM_HAND_TYPES, bucket % NUM_HAND_TYPES


# --- Information Set ---

class InfoSet:
    """
    An information set represents what a player knows:
    - Their hand bucket (2D: equity x hand_type)
    - The game phase (preflop, flop, turn, river)
    - Their position (ip = in position, oop = out of position)
    - The betting history in the current round
    """

    def __init__(self, bucket: int, phase: str, history: tuple[int, ...],
                 position: str = ''):
        self.bucket = bucket
        self.phase = phase
        self.history = history
        self.position = position  # 'ip' or 'oop'

    @property
    def key(self) -> str:
        """Unique string key for this information set.

        Format: {phase}:{position}:{bucket}:{history}
        Position is always included for v5+ strategies.
        """
        hist_str = ''.join(str(a) for a in self.history)
        if self.position:
            return f"{self.phase}:{self.position}:{self.bucket}:{hist_str}"
        return f"{self.phase}:{self.bucket}:{hist_str}"

    def __eq__(self, other):
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)


# --- Phase-Aware Action Menus ---

def get_available_actions(has_bet_to_call: bool, can_raise: bool,
                          phase: str = 'postflop',
                          raise_count: int = 0, history_len: int = -1,
                          eq_bucket: int = -1) -> list[Action]:
    """
    Return available actions given game state.

    Preflop uses distinct open/3bet/4bet actions.
    Postflop uses pot-fraction bet sizing with v6 additions:
    - 2/3 pot and overbet sizes for all postflop spots
    - Donk-bet options on flop/turn when OOP leads (first action)

    raise_count tracks how many raises have occurred this street.
    history_len: length of current street history (-1 = unknown).
    """
    if phase == 'preflop':
        return _preflop_actions(has_bet_to_call, can_raise, raise_count)
    else:
        is_donk_eligible = (
            phase in ('flop', 'turn') and
            history_len == 0 and
            not has_bet_to_call
        )
        return _postflop_actions(has_bet_to_call, can_raise, is_donk_eligible)


def _preflop_actions(has_bet_to_call: bool, can_raise: bool,
                     raise_count: int) -> list[Action]:
    """
    Preflop action abstraction with realistic sizing.

    Action tree:
    - Facing blind only (no raises): fold, call (limp), open_raise, all-in
    - Facing open: fold, call, 3-bet, all-in
    - Facing 3-bet: fold, call, 4-bet, all-in
    - Facing 4-bet: fold, call, all-in
    - BB after limp (no raises, no bet): check or raise
    """
    if not has_bet_to_call:
        # BB checked to (after limps, no raise)
        actions = [Action.CHECK_CALL]  # Check
        if can_raise:
            actions.append(Action.OPEN_RAISE)
            actions.append(Action.ALL_IN)
        return actions

    if raise_count == 0:
        # Facing the big blind only — no one has raised yet
        # Can fold, call (limp), open raise, or jam
        actions = [Action.FOLD, Action.CHECK_CALL]
        if can_raise:
            actions.append(Action.OPEN_RAISE)
            actions.append(Action.ALL_IN)
        return actions

    # Facing a real raise
    actions = [Action.FOLD, Action.CHECK_CALL]
    if can_raise:
        if raise_count <= 1:
            # Facing open -> can 3-bet
            actions.append(Action.THREE_BET)
            actions.append(Action.ALL_IN)
        elif raise_count == 2:
            # Facing 3-bet -> can 4-bet
            actions.append(Action.FOUR_BET)
            actions.append(Action.ALL_IN)
        else:
            # Facing 4-bet+ -> can only jam
            actions.append(Action.ALL_IN)
    return actions


def _postflop_actions(has_bet_to_call: bool, can_raise: bool,
                      is_donk_eligible: bool = False) -> list[Action]:
    """Postflop action abstraction with pot-fraction sizing.

    v6: Added 2/3 pot, overbet (1.25x pot), and donk-bet branches.
    Donk bets available on flop/turn first action (OOP leading).
    """
    if has_bet_to_call:
        actions = [Action.FOLD, Action.CHECK_CALL]
    else:
        actions = [Action.CHECK_CALL]  # Check

    if can_raise:
        if is_donk_eligible:
            # Flop/turn OOP lead: donk-specific sizes + standard big sizes
            actions.extend([Action.DONK_SMALL, Action.DONK_MEDIUM,
                            Action.BET_TWO_THIRDS_POT, Action.BET_POT,
                            Action.ALL_IN])
        else:
            # Standard postflop: full sizing menu
            actions.extend([Action.BET_THIRD_POT, Action.BET_HALF_POT,
                            Action.BET_TWO_THIRDS_POT, Action.BET_POT,
                            Action.BET_OVERBET, Action.ALL_IN])

    return actions


def get_position(acting_player: int, phase: str) -> str:
    """
    Return position label for a player in heads-up poker.

    In heads-up:
    - Preflop: P0=SB acts first (ip preflop), P1=BB acts second (oop preflop)
      BUT strategically SB is out of position postflop, so we use postflop
      convention: P0=oop, P1=ip across all streets for consistency.
    - Postflop: P0=SB acts first (oop), P1=BB acts second (ip)

    Using a consistent mapping (P0=oop, P1=ip) across all streets simplifies
    the solver and matches how poker players think about position.
    """
    return 'ip' if acting_player == 1 else 'oop'


def count_raises(history: tuple[int, ...], phase: str = 'postflop') -> int:
    """Count the number of raises in the current action sequence."""
    raise_actions = {
        int(Action.OPEN_RAISE), int(Action.THREE_BET), int(Action.FOUR_BET),
        int(Action.BET_THIRD_POT), int(Action.BET_HALF_POT),
        int(Action.BET_TWO_THIRDS_POT), int(Action.BET_POT),
        int(Action.BET_OVERBET), int(Action.ALL_IN),
        int(Action.DONK_SMALL), int(Action.DONK_MEDIUM),
    }
    return sum(1 for a in history if a in raise_actions)


# ---------------------------------------------------------------------------
# Canonical concrete→abstract history mapping (C1 fix — single source of truth)
# ---------------------------------------------------------------------------

# All known concrete action name aliases, covering:
#   - legacy server-style names  (raise_small, raise, raise_big, raise_half, raise_overbet)
#   - eval harness / match_engine style names  (open_raise, three_bet, bet_half, …)
_PREFLOP_CONCRETE_MAP: dict[str, int] = {
    "fold":        int(Action.FOLD),
    "check":       int(Action.CHECK_CALL),
    "call":        int(Action.CHECK_CALL),
    # Legacy server names
    "raise_small": int(Action.OPEN_RAISE),
    "raise":       int(Action.THREE_BET),
    "raise_big":   int(Action.FOUR_BET),
    "all_in":      int(Action.ALL_IN),
    # Eval harness / match_engine names
    "open_raise":  int(Action.OPEN_RAISE),
    "three_bet":   int(Action.THREE_BET),
    "four_bet":    int(Action.FOUR_BET),
}

_POSTFLOP_CONCRETE_MAP: dict[str, int] = {
    "fold":          int(Action.FOLD),
    "check":         int(Action.CHECK_CALL),
    "call":          int(Action.CHECK_CALL),
    # Legacy server names
    "raise_small":   int(Action.BET_THIRD_POT),
    "raise_half":    int(Action.BET_HALF_POT),
    "raise":         int(Action.BET_TWO_THIRDS_POT),
    "raise_big":     int(Action.BET_POT),
    "raise_overbet": int(Action.BET_OVERBET),
    "donk_small":    int(Action.DONK_SMALL),
    "donk_medium":   int(Action.DONK_MEDIUM),
    "all_in":        int(Action.ALL_IN),
    # Eval harness / match_engine names
    "bet_third":      int(Action.BET_THIRD_POT),
    "bet_half":       int(Action.BET_HALF_POT),
    "bet_two_thirds": int(Action.BET_TWO_THIRDS_POT),
    "bet_pot":        int(Action.BET_POT),
    "bet_overbet":    int(Action.BET_OVERBET),
}


def concrete_to_abstract_history(history: list[str], phase: str) -> tuple:
    """Convert a list of concrete action strings to an abstract history tuple.

    Single source of truth for string→Action mapping used by both the live
    engine (engine.py) and the eval harness (match_engine.py).

    Unknown action strings are skipped with a warning rather than silently
    dropped or replaced with CHECK_CALL (either alternative corrupts position
    tracking).

    Args:
        history: concrete action strings for the current street, e.g.
                 ["check", "bet_half", "call"]
        phase:   "preflop" | "flop" | "turn" | "river"

    Returns:
        Tuple of Action int values, e.g. (1, 3, 1)
    """
    import logging
    _log = logging.getLogger(__name__)

    mapping = _PREFLOP_CONCRETE_MAP if phase == 'preflop' else _POSTFLOP_CONCRETE_MAP
    result = []
    for action in history:
        if not isinstance(action, str):
            _log.warning(f"concrete_to_abstract_history: non-string action {action!r} in {phase} history — skipped")
            continue
        abstract = mapping.get(action)
        if abstract is None:
            _log.warning(f"concrete_to_abstract_history: unknown action {action!r} in {phase} history — skipped")
            continue
        result.append(abstract)
    return tuple(result)
