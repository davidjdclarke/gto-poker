"""
Opponent modeling — post-lookup probability adjustment layer.

Tracks per-phase fold/call rates over a rolling window and applies
multipliers to bluff-range actions (EQ0-3) after CFR strategy lookup.
The trained strategy.json is never modified.
"""

import collections

# Action ints that can be scaled (bet/donk actions only, not fold/check/jam)
_BET_ACTION_IDS = frozenset([2, 3, 4, 9, 10, 11, 12])
# Action ints:
#   2 = BET_THIRD_POT, 3 = BET_HALF_POT, 4 = BET_POT,
#   9 = BET_TWO_THIRDS_POT, 10 = BET_OVERBET,
#   11 = DONK_SMALL, 12 = DONK_MEDIUM
#   0=FOLD, 1=CHECK_CALL, 5=ALL_IN, 6=OPEN_RAISE, 7=THREE_BET, 8=FOUR_BET — untouched

_PHASES = ('preflop', 'flop', 'turn', 'river')


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class OpponentProfile:
    """
    Rolling-window opponent profile. Records observed opponent actions per
    phase and computes bet-frequency adjustment multipliers.

    Usage:
        profile = OpponentProfile()
        # After each opponent action:
        profile.record('flop', 'fold')
        # Before sampling from GTO strategy:
        adjustments = profile.compute_adjustments('flop', eq_bucket=1)
        if adjustments:
            for a_id, mult in adjustments.items():
                strategy[a_id] = max(0.0, strategy[a_id] * mult)
            # renormalize as usual
    """

    def __init__(self, window: int = 50, min_samples: int = 8):
        self._window = window
        self._min_samples = min_samples
        # Each deque stores 'fold', 'call', or 'raise' per decision
        self._decisions: dict[str, collections.deque] = {
            phase: collections.deque(maxlen=window) for phase in _PHASES
        }

    def record(self, phase: str, action_str: str) -> None:
        """Record an opponent action for the given phase.

        action_str: concrete action string from match_engine / game.py
            'fold'               → fold
            'call', 'check'      → call
            anything else        → raise (includes all bet labels)
        """
        if phase not in self._decisions:
            return
        if action_str == 'fold':
            category = 'fold'
        elif action_str in ('call', 'check'):
            category = 'call'
        else:
            category = 'raise'
        self._decisions[phase].append(category)

    def fold_rate(self, phase: str) -> float | None:
        """Fraction of decisions that were folds. None if < min_samples."""
        decisions = self._decisions.get(phase)
        if decisions is None or len(decisions) < self._min_samples:
            return None
        return decisions.count('fold') / len(decisions)

    def call_rate(self, phase: str) -> float | None:
        """Fraction of decisions that were calls/checks. None if < min_samples."""
        decisions = self._decisions.get(phase)
        if decisions is None or len(decisions) < self._min_samples:
            return None
        return decisions.count('call') / len(decisions)

    def compute_adjustments(self, phase: str, eq_bucket: int) -> dict[int, float] | None:
        """Return per-action-id multipliers, or None if no adjustment applies.

        Only adjusts bet actions for bluff buckets (EQ0-3).
        Medium+ hands (EQ4+) are never touched — they have showdown value.

        Fold-heavy opponents: increase bet multiplier (more bluffing is profitable)
        Call-heavy opponents: decrease bet multiplier (stop bluffing into stations)
        """
        # Never adjust medium+ hands (EQ4+); only bluff buckets adapt
        if eq_bucket > 3:
            return None

        fr = self.fold_rate(phase)
        cr = self.call_rate(phase)

        if fr is None and cr is None:
            return None

        # Ramps: 0.0 at neutral (0.50 rate), 1.0 at extreme (0.90 rate)
        # fold_ramp → multiplier [1.0, 2.0]  (more betting vs folders)
        # call_ramp → multiplier [1.0, 0.05] (less betting vs callers)
        fold_mult = 1.0
        call_mult = 1.0

        if fr is not None:
            fold_ramp = _clamp((fr - 0.50) / 0.40, 0.0, 1.0)
            fold_mult = 1.0 + fold_ramp * 1.0  # max 2.0

        if cr is not None:
            call_ramp = _clamp((cr - 0.50) / 0.40, 0.0, 1.0)
            call_mult = 1.0 - call_ramp * 0.95  # min 0.05

        combined = fold_mult * call_mult
        if abs(combined - 1.0) < 0.02:
            return None  # negligible adjustment — skip

        return {a_id: combined for a_id in _BET_ACTION_IDS}

    def decision_count(self, phase: str) -> int:
        """Number of decisions recorded for this phase."""
        decisions = self._decisions.get(phase)
        return len(decisions) if decisions else 0
