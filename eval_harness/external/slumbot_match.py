"""
Slumbot match orchestrator.

Plays a series of hands against the Slumbot API and aggregates results
into the standard MatchResult format for compatibility with existing
eval infrastructure.

Usage:
    from eval_harness.external.slumbot_client import SlumbotClient
    from eval_harness.external.slumbot_match import SlumbotMatch
    from eval_harness.match_engine import GTOAgent

    gto = GTOAgent(trainer, name="GTO", mapping="refine")
    client = SlumbotClient()
    match = SlumbotMatch(gto, client, big_blind=200)
    result = match.play(num_hands=100)
    print(f"GTO vs Slumbot: {result.p0_bb_per_100:+.1f} bb/100")
"""
import random
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from eval_harness.match_engine import GTOAgent, HandContext, MatchResult, HandRecord
from eval_harness.external.slumbot_client import SlumbotClient, SlumbotAPIError


@dataclass
class SlumbotMatchConfig:
    """Configuration for a Slumbot match."""
    big_blind: int = 200
    starting_chips: int = 20000
    rate_limit_s: float = 0.5
    max_retries: int = 3
    retry_delay_s: float = 2.0
    seed: Optional[int] = None


class SlumbotMatch:
    """Orchestrates a series of hands against the Slumbot API.

    Handles:
    - Rate limiting and retries
    - State translation (API state ↔ HandContext)
    - Result aggregation into MatchResult format

    Note: This is an asymmetric match — we always play as P0 (GTO agent)
    against Slumbot. Button alternation is handled by the API.
    """

    def __init__(self, agent: GTOAgent, client: SlumbotClient,
                 config: Optional[SlumbotMatchConfig] = None):
        self.agent = agent
        self.client = client
        self.config = config or SlumbotMatchConfig()
        if self.config.seed is not None:
            random.seed(self.config.seed)

    def play(self, num_hands: int) -> MatchResult:
        """Play num_hands against Slumbot and return aggregated MatchResult.

        Args:
            num_hands: number of hands to play

        Returns:
            MatchResult with p0 = our GTO agent, p1 = Slumbot
        """
        result = MatchResult(
            p0_name=self.agent.name,
            p1_name="Slumbot",
            num_hands=num_hands,
        )
        bb = self.config.big_blind
        failed_hands = 0

        for hand_num in range(num_hands):
            try:
                record = self._play_one_hand(hand_num)
                result.hands.append(record)
                result.p0_total_won += record.p0_net
                result.p1_total_won -= record.p0_net
                if record.went_to_showdown:
                    result.showdown_count += 1
                for ph in record.phases_reached:
                    result.phase_counts[ph] += 1
            except SlumbotAPIError as e:
                failed_hands += 1
                if failed_hands > num_hands * 0.1:
                    raise RuntimeError(
                        f"Too many Slumbot API failures ({failed_hands}): {e}"
                    ) from e
                # Record a zero-EV placeholder for failed hands
                result.hands.append(HandRecord(
                    hand_num=hand_num,
                    p0_cards=[], p1_cards=[], community=[],
                    pot=0, winner=-1, p0_net=0.0,
                    phases_reached=["preflop"],
                    went_to_showdown=False,
                    actions=[f"API_ERROR:{e}"],
                ))

        # Normalize to bb
        result.p0_total_won /= bb
        result.p1_total_won /= bb

        if failed_hands > 0:
            print(f"  [WARN] {failed_hands}/{num_hands} hands failed due to API errors")

        return result

    def _play_one_hand(self, hand_num: int) -> HandRecord:
        """Play a single hand against Slumbot.

        Retries on transient API errors up to config.max_retries times.
        """
        for attempt in range(self.config.max_retries):
            try:
                return self._play_hand_attempt(hand_num)
            except SlumbotAPIError:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_s)
                else:
                    raise

    def _play_hand_attempt(self, hand_num: int) -> HandRecord:
        """Single attempt to play one hand.

        The Slumbot API is synchronous: each response already includes
        Slumbot's counter-action. After new_hand() or act(), if not terminal,
        it is always our turn to act next.
        """
        from eval_harness.external.slumbot_client import is_terminal
        state = self.client.new_hand()
        token = state.get("token", "")

        from eval_harness.external.slumbot_client import parse_slumbot_cards
        our_cards_raw = state.get("hole_cards", [])
        our_cards = parse_slumbot_cards(our_cards_raw)

        actions_log = []
        phases_reached = ["preflop"]
        p0_net = 0.0
        went_to_showdown = False
        _max_iterations = 50  # guard against infinite loop on unexpected API state
        _iteration = 0

        while not is_terminal(state) and _iteration < _max_iterations:
            _iteration += 1

            ctx = SlumbotClient.state_to_hand_context(
                state, our_cards, self.config.big_blind
            )
            # Track phases reached
            if ctx.phase not in phases_reached:
                phases_reached.append(ctx.phase)

            decision = self.agent.decide(ctx)
            action_str = SlumbotClient.decision_to_action(
                decision, state, self.config.big_blind
            )
            actions_log.append(f"gto:{action_str}")
            state = self.client.act(token, action_str)
            # Update token in case it rotates (API may issue a new one)
            if "token" in state:
                token = state["token"]

        # Extract result
        if is_terminal(state):
            p0_net = self.client.get_result(state)
            went_to_showdown = bool(state.get("bot_hole_cards"))

        return HandRecord(
            hand_num=hand_num,
            p0_cards=[str(c) for c in our_cards],
            p1_cards=[str(c) for c in state.get("bot_hole_cards", [])],
            community=[str(c) for c in state.get("board", [])],
            pot=int(state.get("won_pot", 0)),
            winner=0 if p0_net > 0 else (1 if p0_net < 0 else -1),
            p0_net=p0_net,
            phases_reached=phases_reached,
            went_to_showdown=went_to_showdown,
            actions=actions_log,
        )
