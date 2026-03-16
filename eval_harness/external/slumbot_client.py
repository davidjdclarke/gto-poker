"""
Slumbot API client for external GTO benchmarking.

Slumbot is a strong, publicly available heads-up hold'em bot.
Playing against it provides an external ground-truth reference
beyond the internal handcrafted gauntlet.

API reference: https://www.slumbot.com/about (unofficial API, no public docs)

Usage:
    client = SlumbotClient()
    state = client.new_hand()
    ctx = SlumbotClient.state_to_hand_context(state, our_cards, big_blind=100)
    decision = gto_agent.decide(ctx)
    action_str = SlumbotClient.decision_to_action(decision, state)
    state = client.act(state["hand_id"], action_str)
    result = client.get_result(state)

Note: The Slumbot API is not officially documented. This client is based on
reverse-engineering the web interface. It may break if the API changes.
See docs/plans/v11_external_benchmarks.md for the full integration design.
"""
import json
import time
from typing import Optional

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

from eval_harness.match_engine import HandContext, AgentDecision


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class SlumbotClient:
    """HTTP client for the Slumbot web API.

    All methods raise SlumbotAPIError on network or protocol errors.
    """

    BASE_URL = "https://slumbot.com/api"
    DEFAULT_TIMEOUT = 10.0
    DEFAULT_RATE_LIMIT_S = 0.5  # seconds between requests

    def __init__(self, timeout: float = DEFAULT_TIMEOUT,
                 rate_limit_s: float = DEFAULT_RATE_LIMIT_S):
        if not _REQUESTS_AVAILABLE:
            raise ImportError(
                "slumbot_client requires the 'requests' package. "
                "Install with: pip install requests"
            )
        self.timeout = timeout
        self.rate_limit_s = rate_limit_s
        self._last_request_time = 0.0
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def _throttle(self):
        """Enforce rate limit between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_s:
            time.sleep(self.rate_limit_s - elapsed)
        self._last_request_time = time.time()

    def new_hand(self) -> dict:
        """Start a new hand against Slumbot.

        Returns:
            dict: API state dict containing hand_id, hole_cards, and
                  current game state fields.

        Raises:
            SlumbotAPIError: on HTTP or protocol error
        """
        self._throttle()
        try:
            resp = self._session.post(
                f"{self.BASE_URL}/new_hand",
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise SlumbotAPIError(f"new_hand failed: {e}") from e

    def act(self, hand_id: str, action: str) -> dict:
        """Submit an action for the current hand.

        Args:
            hand_id: ID from new_hand() response
            action:  Slumbot action string (e.g. "b200", "c", "f", "k")
                     Format: "b{amount}" = bet/raise to amount
                             "c"          = call
                             "f"          = fold
                             "k"          = check

        Returns:
            dict: Updated game state

        Raises:
            SlumbotAPIError: on HTTP or protocol error
        """
        self._throttle()
        try:
            resp = self._session.post(
                f"{self.BASE_URL}/act",
                data=json.dumps({"hand_id": hand_id, "action": action}),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise SlumbotAPIError(f"act failed: {e}") from e

    def get_result(self, state: dict) -> float:
        """Extract the final bb-normalized payoff from a terminal state.

        Args:
            state: terminal game state dict from act()

        Returns:
            float: net chips won (positive) or lost (negative) for our player,
                   NOT normalized to bb (raw chips).

        Raises:
            SlumbotAPIError: if state is not terminal
        """
        if not state.get("terminal", False):
            raise SlumbotAPIError("get_result called on non-terminal state")
        # Field name depends on which player we are ("p0_win", "p1_win", etc.)
        # Full mapping documented in v11_external_benchmarks.md §3.2
        winnings = state.get("winnings", state.get("p0_win", None))
        if winnings is None:
            raise SlumbotAPIError(f"No winnings field in terminal state: {list(state.keys())}")
        return float(winnings)

    @staticmethod
    def state_to_hand_context(state: dict, our_cards: list,
                               big_blind: int) -> HandContext:
        """Translate a Slumbot API state dict into a HandContext.

        Args:
            state:     API state dict from new_hand() or act()
            our_cards: list of Card objects for our hole cards
            big_blind: big blind size in chips

        Returns:
            HandContext suitable for passing to GTOAgent.decide()

        Note: This translation is approximate. Slumbot uses a fixed blind
        structure (100/200 by default). Adjust big_blind accordingly.
        Full field mapping is in docs/plans/v11_external_benchmarks.md §3.1.
        """
        # Slumbot API field names (inferred from web interface)
        pot = int(state.get("pot", big_blind * 3))
        current_bet = int(state.get("current_bet", 0))
        my_bet = int(state.get("our_bet", 0))
        my_chips = int(state.get("our_chips", big_blind * 500))
        opp_chips = int(state.get("opp_chips", big_blind * 500))
        min_raise = max(big_blind, current_bet - my_bet)
        phase_map = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
        phase = phase_map.get(int(state.get("street", 0)), "preflop")
        is_ip = bool(state.get("our_position", 0))  # 0=OOP, 1=IP
        community_raw = state.get("board", [])
        # community_raw are string card representations; caller passes real Card objects
        # For simplicity, keep as empty list here — caller should enrich if needed
        from server.deck import Card
        try:
            community = [Card.from_str(c) for c in community_raw]
        except Exception:
            community = []

        betting_history = state.get("betting_history", [])

        return HandContext(
            hole_cards=our_cards,
            community_cards=community,
            pot=pot,
            current_bet=current_bet,
            my_bet=my_bet,
            my_chips=my_chips,
            opp_chips=opp_chips,
            min_raise=min_raise,
            big_blind=big_blind,
            phase=phase,
            is_ip=is_ip,
            betting_history=betting_history,
            hand_number=int(state.get("hand_number", 0)),
            street_pot_start=int(state.get("street_pot", pot)),
        )

    @staticmethod
    def decision_to_action(decision: AgentDecision, state: dict,
                            big_blind: int = 200) -> str:
        """Translate an AgentDecision into a Slumbot action string.

        Args:
            decision:  AgentDecision from GTOAgent.decide()
            state:     current game state (for context)
            big_blind: big blind size (needed to compute min raise)

        Returns:
            str: Slumbot action string ("f", "c", "k", "b{amount}")
        """
        if decision.action == "fold":
            return "f"
        if decision.action == "check":
            return "k"
        if decision.action == "call":
            return "c"
        if decision.action == "raise":
            # Slumbot uses "b{raise_to}" format
            return f"b{int(decision.amount)}"
        return "c"  # fallback


# ---------------------------------------------------------------------------
# Error class
# ---------------------------------------------------------------------------
class SlumbotAPIError(Exception):
    """Raised on Slumbot API communication or protocol errors."""
    pass
