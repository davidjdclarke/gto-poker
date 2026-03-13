import asyncio
import random
from server.deck import Deck
from server.player import Player
from server.evaluator import best_hand, determine_winners
from server.ai import decide
from server.gto.engine import get_trainer, _get_phase
from server.gto.equity import hand_strength_bucket, hand_equity, hand_strength_squared
from server.gto.abstraction import Action, ACTION_NAMES, NUM_BUCKETS, decode_bucket
from server.gto.opponent_model import OpponentProfile

AI_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eddie",
    "Fiona", "George", "Hannah", "Ivan"
]
AI_STYLES = ["tight", "balanced", "loose", "aggressive", "gto"]


class Game:
    def __init__(self, human_name: str, num_players: int, starting_chips: int,
                 send_callback, small_blind: int = 10):
        self.send = send_callback  # async function to send state to client
        self.small_blind = small_blind
        self.big_blind = small_blind * 2
        self.hand_number = 0
        self.round_history: list[str] = []  # Abstract action history for current betting round

        # Create players
        self.players: list[Player] = []
        # Human at seat 0
        self.players.append(Player(
            id="human", name=human_name, chips=starting_chips,
            seat=0, is_human=True
        ))
        # AI players
        names = random.sample(AI_NAMES, min(num_players - 1, len(AI_NAMES)))
        for i, name in enumerate(names):
            self.players.append(Player(
                id=f"ai_{i+1}", name=name, chips=starting_chips,
                seat=i + 1, is_human=False,
                ai_style=random.choice(AI_STYLES)
            ))

        self.dealer_index = 0
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        self.side_pots = []
        self.phase = "waiting"
        self.current_player_index = 0
        self.current_bet = 0
        self.min_raise = self.big_blind
        self.last_raiser_index = -1
        self.human_action_event = asyncio.Event()
        self.human_action = None
        self.running = True
        self.winners_info = None
        # Session-level opponent profile for adaptive GTO suggestions
        self.opponent_profile = OpponentProfile(window=50, min_samples=15)

    def active_players(self) -> list[Player]:
        return [p for p in self.players if not p.is_folded and p.chips + p.current_bet > 0]

    def players_in_hand(self) -> list[Player]:
        return [p for p in self.players if not p.is_folded]

    def non_folded_non_allin(self) -> list[Player]:
        return [p for p in self.players if not p.is_folded and not p.is_all_in]

    async def run(self):
        """Main game loop."""
        while self.running:
            # Remove busted players
            alive = [p for p in self.players if p.chips > 0 or p.current_bet > 0]
            if len(alive) < 2:
                await self.send_state(game_over=True)
                break

            await self.play_hand()

            # Check if human is busted
            human = self.players[0]
            if human.chips <= 0:
                await self.send_state(game_over=True)
                break

            # Brief pause between hands
            await asyncio.sleep(2)

            # Advance dealer
            self.advance_dealer()

    def advance_dealer(self):
        n = len(self.players)
        idx = self.dealer_index
        for _ in range(n):
            idx = (idx + 1) % n
            if self.players[idx].chips > 0:
                self.dealer_index = idx
                return

    async def play_hand(self):
        self.hand_number += 1
        self.deck.reset()
        self.community_cards = []
        self.pot = 0
        self.side_pots = []
        self.winners_info = None

        for p in self.players:
            p.reset_for_hand()

        # Skip players with no chips
        active_seats = [i for i, p in enumerate(self.players) if p.chips > 0]
        if len(active_seats) < 2:
            return

        # Mark folded for busted players
        for p in self.players:
            if p.chips <= 0:
                p.is_folded = True

        # Post blinds
        sb_idx = self._next_active(self.dealer_index)
        bb_idx = self._next_active(sb_idx)
        self._sb_idx = sb_idx
        self._bb_idx = bb_idx

        sb_player = self.players[sb_idx]
        bb_player = self.players[bb_idx]

        sb_amount = sb_player.place_bet(self.small_blind)
        bb_amount = bb_player.place_bet(self.big_blind)
        self.pot = sb_amount + bb_amount

        # Deal hole cards
        for p in self.players:
            if not p.is_folded:
                p.hole_cards = self.deck.deal(2)

        # Pre-flop
        self.phase = "preflop"
        self.current_bet = self.big_blind
        self.min_raise = self.big_blind
        self.round_history = []
        first_to_act = self._next_active(bb_idx)
        await self.send_state()
        await self.betting_round(first_to_act, preflop=True, bb_idx=bb_idx)

        if self._check_single_winner():
            return

        # Flop
        self.phase = "flop"
        self.deck.burn()
        self.community_cards.extend(self.deck.deal(3))
        await self._reset_and_bet()

        if self._check_single_winner():
            return

        # Turn
        self.phase = "turn"
        self.deck.burn()
        self.community_cards.extend(self.deck.deal(1))
        await self._reset_and_bet()

        if self._check_single_winner():
            return

        # River
        self.phase = "river"
        self.deck.burn()
        self.community_cards.extend(self.deck.deal(1))
        await self._reset_and_bet()

        if self._check_single_winner():
            return

        # Showdown
        await self.showdown()

    async def _reset_and_bet(self):
        self._reset_bets()
        await self.send_state()
        first = self._next_active(self.dealer_index)
        await self.betting_round(first)

    def _reset_bets(self):
        for p in self.players:
            p.current_bet = 0
            p.has_acted = False
        self.current_bet = 0
        self.min_raise = self.big_blind
        self.last_raiser_index = -1
        self.round_history = []

    def _next_active(self, from_idx: int) -> int:
        n = len(self.players)
        idx = from_idx
        for _ in range(n):
            idx = (idx + 1) % n
            if not self.players[idx].is_folded and not self.players[idx].is_all_in and self.players[idx].chips > 0:
                return idx
        return -1

    def _check_single_winner(self) -> bool:
        """Check if only one player remains."""
        remaining = [p for p in self.players if not p.is_folded]
        if len(remaining) == 1:
            winner = remaining[0]
            winner.chips += self.pot
            self.winners_info = [{
                "name": winner.name,
                "chips_won": self.pot,
                "hand_name": "Everyone folded",
            }]
            self.phase = "result"
            asyncio.get_event_loop().create_task(self._show_result())
            return True
        return False

    async def _show_result(self):
        await self.send_state()
        await asyncio.sleep(3)

    async def betting_round(self, first_idx: int, preflop: bool = False, bb_idx: int = -1):
        n = len(self.players)
        idx = first_idx

        if idx == -1:
            return

        # Track who needs to act
        players_to_act = set()
        for i, p in enumerate(self.players):
            if not p.is_folded and not p.is_all_in and p.chips > 0:
                players_to_act.add(i)

        # In preflop, BB gets option to check/raise even if no one raised
        bb_had_option = False

        iterations = 0
        max_iterations = n * 10  # Safety limit

        while players_to_act and iterations < max_iterations:
            iterations += 1

            if idx not in players_to_act:
                idx = self._find_next_in_set(idx, players_to_act)
                if idx == -1:
                    break

            player = self.players[idx]

            if player.is_folded or player.is_all_in or player.chips <= 0:
                players_to_act.discard(idx)
                idx = (idx + 1) % n
                continue

            self.current_player_index = idx
            await self.send_state()

            # Get action
            if player.is_human:
                action, amount = await self.get_human_action()
            else:
                await asyncio.sleep(random.uniform(0.8, 2.0))
                # In heads-up, SB/dealer is IP (acts last postflop)
                is_ip = (idx == self._sb_idx) if len(self.players) == 2 else (idx != self._bb_idx)
                decision = decide(player, self.community_cards, self.pot,
                                  self.current_bet, self.min_raise, self.big_blind,
                                  is_in_position=is_ip)
                action, amount = decision.action, decision.amount
                # Record AI action so the opponent model can adapt GTO suggestions
                self.opponent_profile.record(self.phase, action)

            # Validate and apply action
            self._apply_action(idx, action, amount)
            players_to_act.discard(idx)

            # If a raise occurred, everyone else needs to act again
            if action == "raise":
                for i, p in enumerate(self.players):
                    if i != idx and not p.is_folded and not p.is_all_in and p.chips > 0:
                        players_to_act.add(i)

            # Pre-flop BB option
            if preflop and idx == bb_idx and not bb_had_option and action != "raise":
                bb_had_option = True

            idx = (idx + 1) % n

        # Collect bets into pot
        for p in self.players:
            self.pot += p.current_bet

    def _find_next_in_set(self, from_idx: int, player_set: set) -> int:
        n = len(self.players)
        for offset in range(1, n + 1):
            check = (from_idx + offset) % n
            if check in player_set:
                return check
        return -1

    def _classify_bet_size(self, raise_amount: int) -> str:
        """Classify a raise into an abstract action for GTO history tracking."""
        if self.phase == "preflop":
            # Count existing raises in this round's history
            preflop_raises = ["open_raise", "three_bet", "four_bet"]
            raise_count = sum(1 for a in self.round_history if a in preflop_raises)
            if raise_count == 0:
                return "open_raise"
            elif raise_count == 1:
                return "three_bet"
            elif raise_count == 2:
                return "four_bet"
            else:
                return "all_in"

        # Postflop: classify by pot ratio
        total_pot = self.pot + sum(p.current_bet for p in self.players)
        if total_pot <= 0:
            total_pot = self.big_blind
        ratio = raise_amount / total_pot
        if ratio >= 1.5:
            return "all_in"
        elif ratio >= 1.1:
            return "bet_overbet"
        elif ratio >= 0.8:
            return "bet_pot"
        elif ratio >= 0.56:
            return "bet_two_thirds"
        elif ratio >= 0.4:
            return "bet_half"
        else:
            return "bet_third"

    def _apply_action(self, player_idx: int, action: str, amount: int = 0):
        player = self.players[player_idx]
        to_call = self.current_bet - player.current_bet

        if action == "fold":
            player.is_folded = True
            self.round_history.append("fold")

        elif action == "check":
            self.round_history.append("check")

        elif action == "call":
            actual = player.place_bet(to_call)
            self.round_history.append("call")

        elif action == "raise":
            # Amount is the total new bet level
            if amount <= self.current_bet:
                # Invalid raise, treat as call
                player.place_bet(to_call)
                self.round_history.append("call")
                return

            raise_by = amount - self.current_bet
            if raise_by < self.min_raise and player.chips > to_call + self.min_raise:
                # Minimum raise not met, adjust
                amount = self.current_bet + self.min_raise

            needed = amount - player.current_bet
            actual = player.place_bet(needed)
            new_bet = player.current_bet

            if new_bet > self.current_bet:
                actual_raise = new_bet - self.current_bet
                self.round_history.append(self._classify_bet_size(actual_raise))
                self.min_raise = max(self.min_raise, new_bet - self.current_bet)
                self.current_bet = new_bet
                self.last_raiser_index = player_idx
            else:
                self.round_history.append("call")

        player.has_acted = True

    async def showdown(self):
        self.phase = "showdown"
        await self.send_state()
        await asyncio.sleep(1)

        # Evaluate hands
        contenders = [(i, p) for i, p in enumerate(self.players) if not p.is_folded]
        if not contenders:
            return

        hands = []
        for idx, player in contenders:
            result = best_hand(player.hole_cards, self.community_cards)
            hands.append((idx, result))

        winner_indices = determine_winners(hands)
        share = self.pot // len(winner_indices)
        remainder = self.pot % len(winner_indices)

        self.winners_info = []
        for i, widx in enumerate(winner_indices):
            winner = self.players[widx]
            won = share + (1 if i < remainder else 0)
            winner.chips += won
            hand_result = next(r for idx, r in hands if idx == widx)
            self.winners_info.append({
                "name": winner.name,
                "chips_won": won,
                "hand_name": hand_result.name,
                "best_cards": [str(c) for c in hand_result.best_five],
            })

        self.phase = "result"
        await self.send_state()
        await asyncio.sleep(4)

    async def get_human_action(self):
        self.human_action_event.clear()
        self.human_action = None
        await self.send_state()
        await self.human_action_event.wait()
        return self.human_action

    def receive_human_action(self, action: str, amount: int = 0):
        self.human_action = (action, amount)
        self.human_action_event.set()

    def get_valid_actions(self) -> dict:
        """Get valid actions for the current human player."""
        human = self.players[0]
        to_call = self.current_bet - human.current_bet
        can_check = to_call == 0
        can_call = to_call > 0 and human.chips > 0
        can_raise = human.chips > to_call
        min_raise_to = self.current_bet + self.min_raise

        return {
            "can_check": can_check,
            "can_call": can_call,
            "can_raise": can_raise,
            "call_amount": min(to_call, human.chips),
            "min_raise": min_raise_to,
            "max_raise": human.current_bet + human.chips,
        }

    def _get_abstract_history(self) -> tuple:
        """
        Compress the multiplayer round history into a 2-player abstract
        history matching the CFR training model.

        The CFR model is strictly 2-player with alternating actions.
        We extract the significant action sequence:
        - Raises are preserved in order (they form the escalation tree)
        - Passive actions (check/call) before the first raise become one CHECK_CALL
        - Calls between raises are included (they represent the call response)
        - Folds are skipped (other players folding doesn't affect the tree)

        Examples (preflop multiplayer → 2-player abstract):
          [check, open_raise, fold] → (1, 6)  # limped then raised
          [open_raise, three_bet, fold] → (6, 7)  # raise then 3bet
          [open_raise, call, fold, three_bet] → (6, 7)  # raise, call, 3bet
          [check, call, fold] → (1,)  # just passive action
        """
        mapping = {
            "check": int(Action.CHECK_CALL),
            "call": int(Action.CHECK_CALL),
            "bet_third": int(Action.BET_THIRD_POT),
            "bet_half": int(Action.BET_HALF_POT),
            "bet_two_thirds": int(Action.BET_TWO_THIRDS_POT),
            "bet_pot": int(Action.BET_POT),
            "bet_overbet": int(Action.BET_OVERBET),
            "donk_small": int(Action.DONK_SMALL),
            "donk_medium": int(Action.DONK_MEDIUM),
            "all_in": int(Action.ALL_IN),
            "open_raise": int(Action.OPEN_RAISE),
            "three_bet": int(Action.THREE_BET),
            "four_bet": int(Action.FOUR_BET),
        }

        raise_actions = {"open_raise", "three_bet", "four_bet",
                         "bet_third", "bet_half", "bet_two_thirds",
                         "bet_pot", "bet_overbet",
                         "donk_small", "donk_medium", "all_in"}

        if not self.round_history:
            return ()

        # Build 2-player abstract sequence:
        # 1. Collect all raises in order (the escalation tree)
        # 2. If there were passive actions before the first raise, prepend CHECK_CALL
        # Raises go directly adjacent in the model (e.g., open→3bet = (6,7))
        raises = []
        had_passive_before_first_raise = False
        for action in self.round_history:
            if action == "fold":
                continue
            if action in raise_actions:
                raises.append(mapping[action])
            elif action in ("check", "call"):
                if not raises:
                    had_passive_before_first_raise = True

        if not raises:
            # Only passive actions — model as single check
            return (int(Action.CHECK_CALL),)

        result = []
        if had_passive_before_first_raise:
            result.append(int(Action.CHECK_CALL))
        result.extend(raises)

        return tuple(result)

    def _compute_gto_suggestions_sync(self, hole_cards, community_cards,
                                       num_opponents, is_in_position=True) -> dict | None:
        """Compute GTO strategy suggestions (CPU-bound, run in executor)."""
        try:
            trainer = get_trainer()
            phase = _get_phase(community_cards)

            # Compute raw equity
            equity = hand_equity(hole_cards, community_cards,
                                 num_opponents=num_opponents, simulations=150)

            # Compute E[HS^2] and bucket using improved metric
            bucket = hand_strength_bucket(
                hole_cards, community_cards,
                num_opponents=num_opponents,
                num_buckets=NUM_BUCKETS,
                simulations=150,
                use_ehs2=True
            )

            # Build abstract history from round actions
            history = self._get_abstract_history()

            # Use actual table position rather than history-length parity.
            # At a multi-player table, the compressed abstract history length
            # does not reliably reflect the human's IP/OOP status.
            position = 'ip' if is_in_position else 'oop'

            from server.gto.abstraction import InfoSet
            debug_key = InfoSet(bucket, phase, history, position=position).key
            found = debug_key in trainer.nodes
            print(f"[GTO] key={debug_key} found={found} "
                  f"hist={history} strat_preview=", end="", flush=True)

            player_strategy = trainer.get_strategy(phase, bucket, history,
                                                   position=position)

            # Apply opponent-adaptive adjustments (EQ0-3 bluff nodes only)
            eq_bucket, _ = decode_bucket(bucket)
            adjustments = self.opponent_profile.compute_adjustments(phase, eq_bucket)
            if adjustments:
                player_strategy = dict(player_strategy)
                for a_id in list(player_strategy.keys()):
                    player_strategy[a_id] = max(0.0, player_strategy[a_id]
                                                * adjustments.get(a_id, 1.0))
                total_p = sum(player_strategy.values())
                if total_p > 0:
                    player_strategy = {k: v / total_p for k, v in player_strategy.items()}

            # Log the actual strategy returned
            strat_str = ", ".join(
                f"{ACTION_NAMES.get(Action(a), '?')}:{p:.0%}"
                for a, p in player_strategy.items() if p > 0.01
            )
            print(strat_str, flush=True)
            player_strat_display = {
                ACTION_NAMES[Action(a)]: round(p * 100, 1)
                for a, p in player_strategy.items()
            }

            # Determine recommended action (highest probability)
            best_action = max(player_strategy, key=player_strategy.get)
            recommended = ACTION_NAMES[Action(best_action)]

            # Get strategies for ALL buckets in this phase
            all_buckets = []
            for b in range(NUM_BUCKETS):
                strat = trainer.get_strategy(phase, b, history,
                                             position=position)
                all_buckets.append({
                    "bucket": b,
                    "strategy": {
                        ACTION_NAMES[Action(a)]: round(p * 100, 1)
                        for a, p in strat.items()
                    }
                })

            return {
                "equity": round(equity * 100, 1),
                "bucket": bucket,
                "phase": phase,
                "position": position,
                "your_strategy": player_strat_display,
                "recommended": recommended,
                "all_buckets": all_buckets,
                "algorithm": {
                    "name": "CFR+",
                    "iterations": trainer.iterations,
                    "sampling": "external",
                    "abstraction": "E[HS²]",
                },
            }
        except Exception as e:
            print(f"GTO suggestion error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def send_state(self, game_over: bool = False):
        human = self.players[0]
        is_human_turn = (self.current_player_index == 0
                         and not human.is_folded
                         and not human.is_all_in
                         and self.phase not in ("waiting", "result", "showdown"))

        reveal = self.phase in ("showdown", "result")

        # Compute GTO suggestions in a thread to avoid blocking the event loop
        gto_suggestions = None
        if is_human_turn and human.hole_cards:
            print(f"[GTO] Computing suggestions for phase={self.phase}, cards={human.hole_cards}", flush=True)
            try:
                loop = asyncio.get_event_loop()
                num_opponents = len([p for p in self.players if not p.is_folded and not p.is_human]) or 1
                is_ip = (0 == self._sb_idx) if len(self.players) == 2 else (0 != self._bb_idx)
                gto_suggestions = await loop.run_in_executor(
                    None,
                    self._compute_gto_suggestions_sync,
                    list(human.hole_cards),
                    list(self.community_cards),
                    num_opponents,
                    is_ip,
                )
                print(f"[GTO] Result: equity={gto_suggestions.get('equity') if gto_suggestions else 'None'}", flush=True)
            except Exception as e:
                print(f"[GTO] ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()

        state = {
            "type": "state",
            "phase": self.phase,
            "pot": self.pot + sum(p.current_bet for p in self.players),
            "community_cards": [str(c) for c in self.community_cards],
            "players": [p.to_dict(reveal=reveal, is_self=p.is_human) for p in self.players],
            "dealer_index": self.dealer_index,
            "current_player_index": self.current_player_index,
            "your_turn": is_human_turn,
            "valid_actions": self.get_valid_actions() if is_human_turn else None,
            "hand_number": self.hand_number,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "game_over": game_over,
            "winners": self.winners_info,
            "gto_suggestions": gto_suggestions,
        }

        await self.send(state)
