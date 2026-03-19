"""
Slumbot API client for external GTO benchmarking.

Slumbot is a strong, publicly available heads-up hold'em bot.
Playing against it provides an external ground-truth reference
beyond the internal handcrafted gauntlet.

API: POST https://slumbot.com/api/new_hand (empty JSON body)
     POST https://slumbot.com/api/act     (JSON: {"token": ..., "incr": ...})

Actual API format (verified 2026-03-16 via live probing):

  new_hand response:
    {"action": "b200", "client_pos": 0, "hole_cards": ["Tc","9s"],
     "board": [], "token": "<uuid>", "old_action": ""}

  act body:  {"token": "<uuid>", "incr": "<action>"}
    action = "k" (check) | "c" (call) | "f" (fold) | "b{amount}" (bet/raise)

  terminal state: response includes "winnings" key
    {"winnings": 1200, "won_pot": 2400, "bot_hole_cards": [...], ...}

  Blind structure: SB=50, BB=100 chips.
  "b{x}" in action string = total commitment by this player for the street.
  "/" separates streets. client_pos=0 means we are OOP (BB).

Note: The Slumbot API is not officially documented. This client is based on
live probing of the web interface and may break if the API changes.
"""
import time
from typing import Optional

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

from eval_harness.match_engine import HandContext, AgentDecision

# Slumbot blind structure (verified 2026-03-16)
_SLUMBOT_SB = 50
_SLUMBOT_BB = 100
_SLUMBOT_STARTING_STACK = 10000

_SLUMBOT_RANK_MAP = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}


def parse_slumbot_cards(raw_cards: list) -> list:
    """Convert Slumbot card strings (e.g. ['Tc', '9s']) to Card objects."""
    from server.deck import Card
    result = []
    for s in raw_cards:
        try:
            rank = _SLUMBOT_RANK_MAP[s[0].upper()]
            suit = s[1].lower()
            result.append(Card(rank=rank, suit=suit))
        except (KeyError, IndexError):
            pass
    return result


# ---------------------------------------------------------------------------
# Action-string parser
# ---------------------------------------------------------------------------
def _parse_action_state(action_str: str, client_pos: int) -> dict:
    """Parse a Slumbot action string into HandContext fields.

    Args:
        action_str:  cumulative action string from API state (e.g. "b200c/kb200")
        client_pos:  0 = we are OOP (BB), 1 = we are IP (SB/button)

    Returns:
        dict with keys: phase, pot, current_bet, my_bet, street_pot_start,
                        is_ip, betting_history
    """
    streets = action_str.split("/")
    n_streets = len(streets)
    phase_names = ["preflop", "flop", "turn", "river"]
    phase = phase_names[min(n_streets - 1, 3)]

    sb = _SLUMBOT_SB
    bb = _SLUMBOT_BB

    # Total pot accumulation across completed streets + current
    total_pot = 0

    # --- Preflop ---
    # Slumbot (SB/button) acts first. client_pos=0 → we are BB.
    # Both players post blinds implicitly before preflop actions.
    # p0 = our position, p1 = Slumbot.
    # b{x} = total amount committed by this player this street.
    p0_pf = bb  # BB starts committed at bb
    p1_pf = sb  # SB starts committed at sb

    actor = 1  # SB (Slumbot) acts first preflop
    s = streets[0]
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "b":
            j = i + 1
            while j < len(s) and s[j].isdigit():
                j += 1
            amount = int(s[i + 1:j])
            if actor == 1:
                p1_pf = amount
            else:
                p0_pf = amount
            actor ^= 1
            i = j
        elif ch == "c":
            if actor == 1:
                p1_pf = p0_pf
            else:
                p0_pf = p1_pf
            actor ^= 1
            i += 1
        else:  # k / f
            actor ^= 1
            i += 1

    total_pot = p0_pf + p1_pf

    # --- Postflop streets ---
    # OOP (client_pos=0) acts first on each postflop street.
    for street_idx in range(1, n_streets):
        s = streets[street_idx]
        p0_s = 0
        p1_s = 0
        actor = 0  # OOP acts first postflop (matches client_pos=0 = OOP)
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == "b":
                j = i + 1
                while j < len(s) and s[j].isdigit():
                    j += 1
                amount = int(s[i + 1:j])
                if actor == 0:
                    p0_s = amount
                else:
                    p1_s = amount
                actor ^= 1
                i = j
            elif ch == "c":
                if actor == 0:
                    p0_s = p1_s
                else:
                    p1_s = p0_s
                actor ^= 1
                i += 1
            else:
                actor ^= 1
                i += 1
        total_pot += p0_s + p1_s

    # Current street uncommitted (last partial street)
    # Re-parse the current (last) street to get my_bet and current_bet
    cur_s = streets[n_streets - 1]
    p0_cur = 0
    p1_cur = 0
    if n_streets > 1:
        # postflop: OOP (p0) acts first
        actor = 0
    else:
        # preflop: SB (p1) acts first
        actor = 1
    i = 0
    while i < len(cur_s):
        ch = cur_s[i]
        if ch == "b":
            j = i + 1
            while j < len(cur_s) and cur_s[j].isdigit():
                j += 1
            amount = int(cur_s[i + 1:j])
            if actor == 0:
                p0_cur = amount
            else:
                p1_cur = amount
            actor ^= 1
            i = j
        elif ch == "c":
            if actor == 0:
                p0_cur = p1_cur
            else:
                p1_cur = p0_cur
            actor ^= 1
            i += 1
        else:
            actor ^= 1
            i += 1

    # After API response it's always our turn (client pos 0 = p0)
    # current_bet = Slumbot's street commitment we need to match
    # my_bet = our current street commitment
    if client_pos == 0:
        current_bet = p1_cur   # what Slumbot put in (we need to call this)
        my_bet = p0_cur        # what we've already put in this street
    else:
        current_bet = p0_cur
        my_bet = p1_cur

    # pot includes both players' current-street bets
    street_pot_start = total_pot - p0_cur - p1_cur

    return {
        "phase": phase,
        "pot": total_pot,
        "current_bet": current_bet,
        "my_bet": my_bet,
        "street_pot_start": street_pot_start,
        "is_ip": client_pos == 1,
    }


def is_terminal(state: dict) -> bool:
    """Return True if the state represents a completed hand."""
    return "winnings" in state


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class SlumbotClient:
    """HTTP client for the Slumbot web API.

    All methods raise SlumbotAPIError on network or protocol errors.

    Verified API format (2026-03-16):
      POST /api/new_hand  body: {} → returns state with token + Slumbot's first action
      POST /api/act       body: {"token": <token>, "incr": <action>}
      Terminal:           response contains "winnings" key
    """

    BASE_URL = "https://slumbot.com/api"
    DEFAULT_TIMEOUT = 10.0
    DEFAULT_RATE_LIMIT_S = 0.5

    def __init__(self, timeout: float = DEFAULT_TIMEOUT,
                 rate_limit_s: float = DEFAULT_RATE_LIMIT_S):
        if not _REQUESTS_AVAILABLE:
            raise ImportError(
                "slumbot_client requires the 'requests' package. "
                "Install with: venv/bin/pip install requests"
            )
        self.timeout = timeout
        self.rate_limit_s = rate_limit_s
        self._last_request_time = 0.0
        self._session = requests.Session()

    def _throttle(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_s:
            time.sleep(self.rate_limit_s - elapsed)
        self._last_request_time = time.time()

    def new_hand(self) -> dict:
        """Start a new hand against Slumbot.

        Returns:
            dict with keys: token, action, client_pos, hole_cards, board.
            If state is immediately terminal (Slumbot folded preflop), also
            includes: winnings, won_pot, bot_hole_cards.

        Raises:
            SlumbotAPIError: on HTTP or protocol error
        """
        self._throttle()
        try:
            resp = self._session.post(
                f"{self.BASE_URL}/new_hand",
                json={},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise SlumbotAPIError(f"new_hand failed: {e}") from e

    def act(self, token: str, incr: str) -> dict:
        """Submit an action for the current hand.

        Args:
            token: session token from new_hand() response
            incr:  our action increment:
                     "k"        = check
                     "c"        = call
                     "f"        = fold
                     "b{amount}" = bet/raise (amount = total chips committed
                                   by us this street, e.g. "b300")

        Returns:
            dict: Updated state. If terminal, includes "winnings" key.

        Raises:
            SlumbotAPIError: on HTTP or protocol error or illegal action
        """
        self._throttle()
        try:
            resp = self._session.post(
                f"{self.BASE_URL}/act",
                json={"token": token, "incr": incr},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if "error_msg" in data:
                raise SlumbotAPIError(f"Illegal action '{incr}': {data['error_msg']}")
            return data
        except SlumbotAPIError:
            raise
        except Exception as e:
            raise SlumbotAPIError(f"act failed: {e}") from e

    def get_result(self, state: dict) -> float:
        """Extract the net chip payoff from a terminal state.

        Args:
            state: terminal game state (must have "winnings" key)

        Returns:
            float: net chips won (positive) or lost (negative).
                   NOT normalized to bb — divide by big_blind for bb units.

        Raises:
            SlumbotAPIError: if state is not terminal
        """
        if not is_terminal(state):
            raise SlumbotAPIError("get_result called on non-terminal state")
        return float(state["winnings"])

    @staticmethod
    def state_to_hand_context(state: dict, our_cards: list,
                               big_blind: int = _SLUMBOT_BB) -> HandContext:
        """Translate a Slumbot API state dict into a HandContext.

        Parses the cumulative action string to derive pot, current_bet,
        phase, and position. Accurate for standard 1-raise-per-street sequences;
        approximates deeper stacks.

        Args:
            state:     API state dict from new_hand() or act()
            our_cards: list of Card objects for our hole cards
            big_blind: big blind chip value (default 100, Slumbot's native BB)
        """
        action_str = state.get("action", "")
        client_pos = int(state.get("client_pos", 0))

        parsed = _parse_action_state(action_str, client_pos)

        community_raw = state.get("board", [])
        community = parse_slumbot_cards(community_raw)

        starting_stack = _SLUMBOT_STARTING_STACK
        my_chips = max(0, starting_stack - parsed["pot"] + parsed["my_bet"])
        opp_chips = max(0, starting_stack - parsed["pot"] + parsed["current_bet"])
        to_call = parsed["current_bet"] - parsed["my_bet"]
        min_raise = max(big_blind, to_call if to_call > 0 else big_blind)

        return HandContext(
            hole_cards=our_cards,
            community_cards=community,
            pot=parsed["pot"],
            current_bet=parsed["current_bet"],
            my_bet=parsed["my_bet"],
            my_chips=my_chips,
            opp_chips=opp_chips,
            min_raise=min_raise,
            big_blind=big_blind,
            phase=parsed["phase"],
            is_ip=parsed["is_ip"],
            betting_history=[],
            hand_number=int(state.get("session_num_hands", 0)),
            street_pot_start=parsed["street_pot_start"],
        )

    @staticmethod
    def decision_to_action(decision: AgentDecision, state: dict,
                            big_blind: int = _SLUMBOT_BB) -> str:
        """Translate an AgentDecision into a Slumbot incr action string.

        Args:
            decision:  AgentDecision from GTOAgent.decide()
            state:     current game state (for parsing current commitments)
            big_blind: big blind size

        Returns:
            str: Slumbot incr string ("f", "c", "k", "b{total_street_commit}")
        """
        if decision.action == "fold":
            return "f"
        if decision.action == "check":
            return "k"
        if decision.action == "call":
            return "c"
        if decision.action == "raise":
            # decision.amount is a raise-to chip amount (total stack committed)
            # Convert to Slumbot's "b{total_street_commit}" format.
            # We need to express as total chips committed THIS STREET by us.
            action_str = state.get("action", "")
            client_pos = int(state.get("client_pos", 0))
            parsed = _parse_action_state(action_str, client_pos)
            # decision.amount is typically a raise-to amount (total chips in pot
            # including our commitment). For Slumbot, b{x} = total we commit this
            # street. Approximate as decision.amount (raise-to amount).
            return f"b{int(decision.amount)}"
        return "c"  # fallback


# ---------------------------------------------------------------------------
# Error class
# ---------------------------------------------------------------------------
class SlumbotAPIError(Exception):
    """Raised on Slumbot API communication or protocol errors."""
    pass
