#!/usr/bin/env python3
"""
Self-play simulation framework: GTO engine vs AI players.

Runs hands automatically, monitors GTO strategy lookups, and detects
anomalies like uniform distributions (strategy miss), unrealistic
bet patterns, and position mismatches.

Usage:
    python simulate.py                    # 100 hands, default settings
    python simulate.py --hands 1000       # 1000 hands
    python simulate.py --verbose          # Show every decision
    python simulate.py --players 2        # Heads-up only
"""
import argparse
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

from server.player import Player
from server.deck import Deck, Card
from server.evaluator import best_hand, determine_winners
from server.gto.engine import get_trainer, _get_phase
from server.gto.equity import hand_strength_bucket
from server.gto.abstraction import (
    Action, ACTION_NAMES, InfoSet, NUM_BUCKETS,
    get_available_actions, count_raises, decode_bucket,
)
from server.gto.cfr import CFRTrainer
from server.ai import decide as ai_decide, AIDecision


# ---------------------------------------------------------------------------
# Diagnostics collector
# ---------------------------------------------------------------------------
@dataclass
class StrategyLookup:
    """Record of a single GTO strategy lookup."""
    hand_num: int
    phase: str
    position: str
    bucket: int
    history: tuple
    key: str
    found: bool
    strategy: dict  # {action_int: prob}
    is_uniform: bool
    chosen_action: int
    player_name: str


@dataclass
class HandResult:
    """Summary of a completed hand."""
    hand_num: int
    winner_names: list[str]
    pot: int
    went_to_showdown: bool
    phases_reached: list[str]
    actions_taken: int


@dataclass
class SimulationStats:
    """Aggregate statistics across all hands."""
    lookups: list[StrategyLookup] = field(default_factory=list)
    hand_results: list[HandResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Counters
    total_lookups: int = 0
    found_lookups: int = 0
    uniform_lookups: int = 0

    # Per-phase tracking
    phase_lookups: dict = field(default_factory=lambda: defaultdict(int))
    phase_found: dict = field(default_factory=lambda: defaultdict(int))
    phase_uniform: dict = field(default_factory=lambda: defaultdict(int))

    # Action distribution
    action_counts: dict = field(default_factory=lambda: defaultdict(int))

    # Position tracking
    position_counts: dict = field(default_factory=lambda: defaultdict(int))
    position_found: dict = field(default_factory=lambda: defaultdict(int))

    # History length tracking
    history_miss_examples: list = field(default_factory=list)

    def record_lookup(self, lookup: StrategyLookup):
        self.lookups.append(lookup)
        self.total_lookups += 1
        self.phase_lookups[lookup.phase] += 1
        self.position_counts[lookup.position] += 1
        self.action_counts[lookup.chosen_action] += 1

        if lookup.found:
            self.found_lookups += 1
            self.phase_found[lookup.phase] += 1
            self.position_found[lookup.position] += 1

        if lookup.is_uniform:
            self.uniform_lookups += 1
            self.phase_uniform[lookup.phase] += 1

        if not lookup.found and len(self.history_miss_examples) < 50:
            self.history_miss_examples.append(lookup)


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------
class PokerSimulation:
    """Headless poker game for GTO self-play testing."""

    def __init__(self, num_players: int = 6, starting_chips: int = 1000,
                 small_blind: int = 10, verbose: bool = False):
        self.num_players = num_players
        self.starting_chips = starting_chips
        self.small_blind = small_blind
        self.big_blind = small_blind * 2
        self.verbose = verbose
        self.stats = SimulationStats()

        # Create players: mix of GTO and heuristic AI
        styles = ["gto", "tight", "balanced", "loose", "aggressive"]
        self.players = []
        for i in range(num_players):
            style = styles[i % len(styles)]
            p = Player(
                id=f"sim_{i}",
                name=f"Player_{i}_{style}",
                chips=starting_chips,
                seat=i,
                is_human=False,
                ai_style=style,
            )
            self.players.append(p)

        self.deck = Deck()
        self.trainer = get_trainer()
        self.dealer_index = 0
        self.hand_num = 0

    def run(self, num_hands: int = 100):
        """Run the full simulation."""
        print(f"\n{'='*60}")
        print(f"  GTO Self-Play Simulation")
        print(f"  Players: {self.num_players} | Hands: {num_hands}")
        print(f"  Styles: {', '.join(p.ai_style for p in self.players)}")
        print(f"  Strategy: {self.trainer.iterations:,} iterations, "
              f"{len(self.trainer.nodes):,} nodes")
        print(f"{'='*60}\n")

        t_start = time.time()

        for h in range(num_hands):
            self.hand_num = h + 1
            self._play_hand()

            # Reset busted players
            for p in self.players:
                if p.chips <= 0:
                    p.chips = self.starting_chips

            self.dealer_index = (self.dealer_index + 1) % self.num_players

            if self.verbose and h % 10 == 0:
                self._print_progress(h, num_hands)

        elapsed = time.time() - t_start
        self._print_report(num_hands, elapsed)

    def _play_hand(self):
        """Play a single hand."""
        # Reset
        self.deck.reset()
        self.pot = 0
        self.community_cards = []
        self.current_bet = 0
        self.min_raise = self.big_blind
        self.round_history = []
        self.actions_taken = 0
        self.phases_reached = ["preflop"]

        for p in self.players:
            p.reset_for_hand()

        # Post blinds
        n = self.num_players
        sb_idx = (self.dealer_index + 1) % n
        bb_idx = (self.dealer_index + 2) % n
        if n == 2:
            sb_idx = self.dealer_index
            bb_idx = (self.dealer_index + 1) % n

        self.players[sb_idx].place_bet(self.small_blind)
        self.players[bb_idx].place_bet(self.big_blind)
        self.current_bet = self.big_blind
        self.sb_idx = sb_idx
        self.bb_idx = bb_idx

        # Deal hole cards
        for p in self.players:
            p.hole_cards = self.deck.deal(2)

        # Preflop
        first = (bb_idx + 1) % n
        self.phase = "preflop"
        self._betting_round(first, preflop=True)

        if self._count_active() <= 1:
            self._finish_hand(showdown=False)
            return

        # Flop
        self.phase = "flop"
        self.phases_reached.append("flop")
        self.deck.burn()
        self.community_cards.extend(self.deck.deal(3))
        self._reset_bets()
        first = self._next_active(self.dealer_index)
        if first >= 0:
            self._betting_round(first)

        if self._count_active() <= 1:
            self._finish_hand(showdown=False)
            return

        # Turn
        self.phase = "turn"
        self.phases_reached.append("turn")
        self.deck.burn()
        self.community_cards.extend(self.deck.deal(1))
        self._reset_bets()
        first = self._next_active(self.dealer_index)
        if first >= 0:
            self._betting_round(first)

        if self._count_active() <= 1:
            self._finish_hand(showdown=False)
            return

        # River
        self.phase = "river"
        self.phases_reached.append("river")
        self.deck.burn()
        self.community_cards.extend(self.deck.deal(1))
        self._reset_bets()
        first = self._next_active(self.dealer_index)
        if first >= 0:
            self._betting_round(first)

        self._finish_hand(showdown=True)

    def _betting_round(self, first_idx: int, preflop: bool = False):
        """Run a complete betting round."""
        n = self.num_players
        idx = first_idx
        self.round_history = []

        # Track who still needs to act
        players_to_act = set()
        for i, p in enumerate(self.players):
            if not p.is_folded and not p.is_all_in and p.chips > 0:
                players_to_act.add(i)

        max_actions = n * 6  # Safety limit
        action_count = 0

        while players_to_act and action_count < max_actions:
            player = self.players[idx]

            if idx in players_to_act:
                # Get decision
                decision = self._get_decision(idx)
                self._apply_action(idx, decision.action, decision.amount)
                self.actions_taken += 1
                action_count += 1

                players_to_act.discard(idx)

                # If raise, everyone else needs to act again
                if decision.action == "raise":
                    for i, p in enumerate(self.players):
                        if (i != idx and not p.is_folded and
                                not p.is_all_in and p.chips > 0):
                            players_to_act.add(i)

            idx = (idx + 1) % n

    def _get_decision(self, idx: int) -> AIDecision:
        """Get an AI decision, with GTO monitoring for GTO-style players."""
        player = self.players[idx]

        if player.ai_style == "gto":
            return self._gto_decision_with_monitoring(idx)

        # Use the standard AI for non-GTO players
        is_ip = (idx == self.sb_idx) if self.num_players == 2 else (idx != self.bb_idx)
        return ai_decide(
            player, self.community_cards, self.pot,
            self.current_bet, self.min_raise, self.big_blind,
            is_in_position=is_ip,
        )

    def _gto_decision_with_monitoring(self, idx: int) -> AIDecision:
        """Make a GTO decision and record diagnostics."""
        player = self.players[idx]
        phase = self.phase

        # Compute bucket
        try:
            bucket = hand_strength_bucket(
                player.hole_cards, self.community_cards,
                num_opponents=self._count_active() - 1,
                simulations=100,
                use_ehs2=True,
            )
        except Exception as e:
            self.stats.errors.append(f"Hand {self.hand_num}: bucket error: {e}")
            return AIDecision("check" if self.current_bet <= player.current_bet else "fold")

        # Build abstract history
        history = self._get_abstract_history()

        # Position from history length
        position = 'oop' if len(history) % 2 == 0 else 'ip'

        # Look up strategy
        has_bet = self._has_bet_to_call(history)
        raise_count = count_raises(history, phase)
        can_raise = raise_count < 4
        actions = get_available_actions(has_bet, can_raise, phase, raise_count,
                                        eq_bucket=decode_bucket(bucket)[0])

        info_set = InfoSet(bucket, phase, history, position=position)
        key = info_set.key
        found = key in self.trainer.nodes

        if found:
            node = self.trainer.nodes[key]
            avg_strategy = node.get_average_strategy()
            strategy = {int(actions[i]): float(avg_strategy[i])
                        for i in range(len(actions))}
        else:
            uniform = 1.0 / len(actions)
            strategy = {int(a): uniform for a in actions}

        # Check if uniform
        probs = list(strategy.values())
        is_uniform = (len(set(round(p, 4) for p in probs)) <= 1)

        # Sample action
        action_ids = list(strategy.keys())
        weights = list(strategy.values())
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(action_ids)] * len(action_ids)

        chosen = random.choices(action_ids, weights=weights, k=1)[0]

        # Record lookup
        lookup = StrategyLookup(
            hand_num=self.hand_num,
            phase=phase,
            position=position,
            bucket=bucket,
            history=history,
            key=key,
            found=found,
            strategy=strategy,
            is_uniform=is_uniform,
            chosen_action=chosen,
            player_name=player.name,
        )
        self.stats.record_lookup(lookup)

        if self.verbose:
            status = "HIT" if found else "MISS"
            strat_str = ", ".join(
                f"{ACTION_NAMES.get(Action(a), '?')}:{p:.0%}"
                for a, p in strategy.items() if p > 0.01
            )
            print(f"  [{status}] {player.name} {phase} {position} "
                  f"b={bucket} h={history} -> {strat_str} "
                  f"=> {ACTION_NAMES.get(Action(chosen), '?')}")

        # Convert to concrete action
        return self._abstract_to_concrete(Action(chosen), player)

    def _abstract_to_concrete(self, action: Action, player: Player) -> AIDecision:
        """Convert abstract CFR action to concrete game action."""
        to_call = self.current_bet - player.current_bet

        if action == Action.FOLD:
            if to_call == 0:
                return AIDecision("check")
            return AIDecision("fold")

        if action == Action.CHECK_CALL:
            if to_call == 0:
                return AIDecision("check")
            return AIDecision("call")

        # Raise actions
        if action == Action.ALL_IN:
            total = player.current_bet + player.chips
            return AIDecision("raise", total)

        # Size the raise based on action type
        sizing = {
            Action.OPEN_RAISE: 2.5 * self.big_blind,
            Action.THREE_BET: 3.0 * self.current_bet,
            Action.FOUR_BET: 2.5 * self.current_bet,
            Action.BET_QUARTER_POT: max(self.pot * 0.25, self.min_raise),
            Action.BET_THIRD_POT: max(self.pot * 0.33, self.min_raise),
            Action.BET_HALF_POT: max(self.pot * 0.5, self.min_raise),
            Action.BET_TWO_THIRDS_POT: max(self.pot * 0.67, self.min_raise),
            Action.BET_THREE_QUARTER_POT: max(self.pot * 0.75, self.min_raise),
            Action.BET_POT: max(self.pot, self.min_raise),
            Action.BET_OVERBET: max(self.pot * 1.25, self.min_raise),
            Action.BET_DOUBLE_POT: max(self.pot * 2.0, self.min_raise),
            Action.DONK_SMALL: max(self.pot * 0.25, self.min_raise),
            Action.DONK_MEDIUM: max(self.pot * 0.5, self.min_raise),
        }
        raise_size = sizing.get(action, self.min_raise)
        raise_to = int(self.current_bet + max(raise_size, self.min_raise))
        raise_to = min(raise_to, player.current_bet + player.chips)

        if raise_to <= self.current_bet:
            # Can't raise, just call/check
            if to_call == 0:
                return AIDecision("check")
            return AIDecision("call")

        return AIDecision("raise", raise_to)

    def _get_abstract_history(self) -> tuple:
        """Same logic as game.py — compress multiplayer to 2-player."""
        mapping = {
            "check": int(Action.CHECK_CALL),
            "call": int(Action.CHECK_CALL),
            "bet_quarter": int(Action.BET_QUARTER_POT),
            "bet_third": int(Action.BET_THIRD_POT),
            "bet_half": int(Action.BET_HALF_POT),
            "bet_two_thirds": int(Action.BET_TWO_THIRDS_POT),
            "bet_three_quarter": int(Action.BET_THREE_QUARTER_POT),
            "bet_pot": int(Action.BET_POT),
            "bet_overbet": int(Action.BET_OVERBET),
            "bet_double_pot": int(Action.BET_DOUBLE_POT),
            "donk_small": int(Action.DONK_SMALL),
            "donk_medium": int(Action.DONK_MEDIUM),
            "all_in": int(Action.ALL_IN),
            "open_raise": int(Action.OPEN_RAISE),
            "three_bet": int(Action.THREE_BET),
            "four_bet": int(Action.FOUR_BET),
        }
        raise_actions = {"open_raise", "three_bet", "four_bet",
                         "bet_quarter", "bet_third", "bet_half",
                         "bet_two_thirds", "bet_three_quarter",
                         "bet_pot", "bet_overbet", "bet_double_pot",
                         "donk_small", "donk_medium", "all_in"}

        if not self.round_history:
            return ()

        raises = []
        had_passive = False
        for a in self.round_history:
            if a == "fold":
                continue
            if a in raise_actions:
                raises.append(mapping[a])
            elif a in ("check", "call"):
                if not raises:
                    had_passive = True

        if not raises:
            return (int(Action.CHECK_CALL),)

        result = []
        if had_passive:
            result.append(int(Action.CHECK_CALL))
        result.extend(raises)
        return tuple(result)

    def _has_bet_to_call(self, history: tuple) -> bool:
        """Check if the last action in history is a bet/raise."""
        if not history:
            return False
        bet_actions = {
            int(Action.BET_QUARTER_POT), int(Action.BET_THIRD_POT),
            int(Action.BET_HALF_POT),
            int(Action.BET_TWO_THIRDS_POT), int(Action.BET_THREE_QUARTER_POT),
            int(Action.BET_POT),
            int(Action.BET_OVERBET), int(Action.BET_DOUBLE_POT),
            int(Action.ALL_IN),
            int(Action.OPEN_RAISE), int(Action.THREE_BET),
            int(Action.FOUR_BET),
            int(Action.DONK_SMALL), int(Action.DONK_MEDIUM),
        }
        return history[-1] in bet_actions

    def _classify_bet_size(self, raise_amount: int) -> str:
        """Classify raise into abstract action (same as game.py)."""
        if self.phase == "preflop":
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

    def _apply_action(self, idx: int, action: str, amount: int = 0):
        """Apply an action (same logic as game.py)."""
        player = self.players[idx]
        to_call = self.current_bet - player.current_bet

        if action == "fold":
            player.is_folded = True
            self.round_history.append("fold")

        elif action == "check":
            self.round_history.append("check")

        elif action == "call":
            player.place_bet(to_call)
            self.round_history.append("call")

        elif action == "raise":
            if amount <= self.current_bet:
                player.place_bet(to_call)
                self.round_history.append("call")
                return

            needed = amount - player.current_bet
            player.place_bet(needed)
            new_bet = player.current_bet

            if new_bet > self.current_bet:
                actual_raise = new_bet - self.current_bet
                self.round_history.append(self._classify_bet_size(actual_raise))
                self.min_raise = max(self.min_raise, actual_raise)
                self.current_bet = new_bet
            else:
                self.round_history.append("call")

        player.has_acted = True

    def _reset_bets(self):
        """Collect bets and reset for new street."""
        for p in self.players:
            self.pot += p.current_bet
            p.current_bet = 0
            p.has_acted = False
        self.current_bet = 0
        self.min_raise = self.big_blind
        self.round_history = []

    def _count_active(self) -> int:
        return sum(1 for p in self.players if not p.is_folded)

    def _next_active(self, from_idx: int) -> int:
        n = self.num_players
        for offset in range(1, n + 1):
            check = (from_idx + offset) % n
            if not self.players[check].is_folded and not self.players[check].is_all_in:
                return check
        return -1

    def _finish_hand(self, showdown: bool):
        """Award pot and record results."""
        # Collect remaining bets
        for p in self.players:
            self.pot += p.current_bet
            p.current_bet = 0

        active = [(i, p) for i, p in enumerate(self.players) if not p.is_folded]

        if len(active) == 1:
            winner = active[0][1]
            winner.chips += self.pot
            winner_names = [winner.name]
        elif showdown and self.community_cards:
            # Evaluate hands
            hands = []
            for i, p in active:
                hr = best_hand(p.hole_cards, self.community_cards)
                hands.append((i, hr))

            winner_indices = determine_winners(hands)
            share = self.pot // len(winner_indices)
            winner_names = []
            for wi in winner_indices:
                self.players[wi].chips += share
                winner_names.append(self.players[wi].name)
        else:
            # Split pot (shouldn't happen often)
            share = self.pot // len(active)
            winner_names = [p.name for _, p in active]
            for _, p in active:
                p.chips += share

        self.stats.hand_results.append(HandResult(
            hand_num=self.hand_num,
            winner_names=winner_names,
            pot=self.pot,
            went_to_showdown=showdown and len(active) > 1,
            phases_reached=self.phases_reached,
            actions_taken=self.actions_taken,
        ))

    def _print_progress(self, hand: int, total: int):
        hit_rate = (self.stats.found_lookups / self.stats.total_lookups * 100
                    if self.stats.total_lookups > 0 else 0)
        print(f"  Hand {hand}/{total} | "
              f"Lookups: {self.stats.total_lookups} | "
              f"Hit rate: {hit_rate:.0f}%")

    def _print_report(self, num_hands: int, elapsed: float):
        """Print comprehensive diagnostics report."""
        s = self.stats
        print(f"\n{'='*60}")
        print(f"  SIMULATION REPORT")
        print(f"  {num_hands} hands in {elapsed:.1f}s "
              f"({elapsed/num_hands*1000:.0f}ms/hand)")
        print(f"{'='*60}")

        # Overall hit rate
        hit_rate = s.found_lookups / s.total_lookups * 100 if s.total_lookups else 0
        uniform_rate = s.uniform_lookups / s.total_lookups * 100 if s.total_lookups else 0
        print(f"\n  STRATEGY LOOKUPS")
        print(f"  {'Total lookups:':<30} {s.total_lookups:>8}")
        print(f"  {'Found (trained):':<30} {s.found_lookups:>8} ({hit_rate:.1f}%)")
        print(f"  {'Uniform (miss):':<30} {s.uniform_lookups:>8} ({uniform_rate:.1f}%)")

        # Per-phase breakdown
        print(f"\n  PER-PHASE HIT RATE")
        for phase in ["preflop", "flop", "turn", "river"]:
            total = s.phase_lookups.get(phase, 0)
            found = s.phase_found.get(phase, 0)
            uniform = s.phase_uniform.get(phase, 0)
            if total > 0:
                pct = found / total * 100
                upct = uniform / total * 100
                print(f"  {phase:<12} {found:>5}/{total:<5} = {pct:5.1f}% hit | "
                      f"{uniform:>5} uniform ({upct:.1f}%)")

        # Per-position breakdown
        print(f"\n  PER-POSITION HIT RATE")
        for pos in ["oop", "ip"]:
            total = s.position_counts.get(pos, 0)
            found = s.position_found.get(pos, 0)
            if total > 0:
                pct = found / total * 100
                print(f"  {pos:<12} {found:>5}/{total:<5} = {pct:5.1f}% hit")

        # Action distribution
        print(f"\n  ACTION DISTRIBUTION (GTO players)")
        total_actions = sum(s.action_counts.values())
        for a_int, count in sorted(s.action_counts.items()):
            name = ACTION_NAMES.get(Action(a_int), f"action_{a_int}")
            pct = count / total_actions * 100 if total_actions else 0
            bar = "#" * int(pct / 2)
            print(f"  {name:<16} {count:>6} ({pct:5.1f}%) {bar}")

        # Hand outcomes
        showdowns = sum(1 for hr in s.hand_results if hr.went_to_showdown)
        avg_pot = (sum(hr.pot for hr in s.hand_results) / len(s.hand_results)
                   if s.hand_results else 0)
        print(f"\n  HAND OUTCOMES")
        print(f"  {'Went to showdown:':<30} {showdowns:>8} "
              f"({showdowns/num_hands*100:.1f}%)")
        print(f"  {'Average pot:':<30} {avg_pot:>8.0f}")

        # Phase reach frequency
        phase_reach = defaultdict(int)
        for hr in s.hand_results:
            for ph in hr.phases_reached:
                phase_reach[ph] += 1
        print(f"\n  PHASE REACH FREQUENCY")
        for phase in ["preflop", "flop", "turn", "river"]:
            count = phase_reach.get(phase, 0)
            pct = count / num_hands * 100
            print(f"  {phase:<12} {count:>5}/{num_hands} ({pct:.1f}%)")

        # Anomaly detection
        print(f"\n  ANOMALY DETECTION")
        anomalies = []

        if hit_rate < 50:
            anomalies.append(
                f"LOW HIT RATE: {hit_rate:.1f}% of strategy lookups miss "
                f"trained nodes (expected >80%)")

        if uniform_rate > 30:
            anomalies.append(
                f"HIGH UNIFORM RATE: {uniform_rate:.1f}% lookups return "
                f"uniform distribution (>30% threshold)")

        # Check for phase-specific issues
        for phase in ["preflop", "flop", "turn", "river"]:
            total = s.phase_lookups.get(phase, 0)
            found = s.phase_found.get(phase, 0)
            if total > 10 and found / total < 0.5:
                anomalies.append(
                    f"PHASE MISS: {phase} hit rate is only "
                    f"{found/total*100:.1f}% ({found}/{total})")

        # Check fold rate
        fold_count = s.action_counts.get(int(Action.FOLD), 0)
        if total_actions > 0:
            fold_rate = fold_count / total_actions * 100
            if fold_rate > 60:
                anomalies.append(
                    f"HIGH FOLD RATE: GTO players fold {fold_rate:.1f}% "
                    f"of the time (>60% threshold)")
            if fold_rate < 10:
                anomalies.append(
                    f"LOW FOLD RATE: GTO players fold only {fold_rate:.1f}% "
                    f"of the time (<10% threshold)")

        # Check for all-in overuse
        allin_count = s.action_counts.get(int(Action.ALL_IN), 0)
        if total_actions > 0 and allin_count / total_actions > 0.15:
            anomalies.append(
                f"ALL-IN OVERUSE: GTO players go all-in "
                f"{allin_count/total_actions*100:.1f}% (>15% threshold)")

        if anomalies:
            for a in anomalies:
                print(f"  [!] {a}")
        else:
            print(f"  [OK] No anomalies detected")

        # Show miss examples
        if s.history_miss_examples:
            print(f"\n  SAMPLE STRATEGY MISSES (first {min(10, len(s.history_miss_examples))})")
            seen_keys = set()
            for lookup in s.history_miss_examples[:20]:
                if lookup.key in seen_keys:
                    continue
                seen_keys.add(lookup.key)
                print(f"  key={lookup.key} "
                      f"round_hist_len={len(lookup.history)}")
                if len(seen_keys) >= 10:
                    break

        # Errors
        if s.errors:
            print(f"\n  ERRORS ({len(s.errors)})")
            for err in s.errors[:10]:
                print(f"  {err}")

        print()


def main():
    parser = argparse.ArgumentParser(description="GTO Self-Play Simulation")
    parser.add_argument("--hands", type=int, default=100, help="Number of hands")
    parser.add_argument("--players", type=int, default=6, help="Number of players")
    parser.add_argument("--chips", type=int, default=1000, help="Starting chips")
    parser.add_argument("--blind", type=int, default=10, help="Small blind")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show every decision")
    args = parser.parse_args()

    sim = PokerSimulation(
        num_players=args.players,
        starting_chips=args.chips,
        small_blind=args.blind,
        verbose=args.verbose,
    )
    sim.run(args.hands)


if __name__ == "__main__":
    main()
