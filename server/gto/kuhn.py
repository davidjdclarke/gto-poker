"""
Kuhn Poker implementation for CFR+ correctness benchmarking.

Kuhn Poker is a minimal poker game with known Nash Equilibrium:
- 3 cards: J(0), Q(1), K(2)
- 2 players, 1 chip ante each
- One betting round: check/bet, then check/call/fold
- Nash: P1 bets K always, bluffs J at 1/3, checks Q. P2 calls K always,
  calls Q vs bet 1/3, folds J always.
- Game value = -1/18 for P1

Used to verify the CFR+ solver converges to correct equilibrium.
"""
import numpy as np
from itertools import permutations
from server.gto.cfr import CFRNode

CARDS = [0, 1, 2]  # J, Q, K
CARD_NAMES = {0: 'J', 1: 'Q', 2: 'K'}
ACTIONS = [0, 1]  # 0=pass(check/fold), 1=bet(bet/call)


class KuhnPoker:
    """Kuhn Poker game logic."""

    @staticmethod
    def is_terminal(history: tuple) -> bool:
        if len(history) < 2:
            return False
        # Length 2 terminals: pass-pass, bet-pass(fold), bet-bet(call)
        if len(history) == 2 and history != (0, 1):
            return True
        # Length 3 terminals: pass-bet-pass(fold), pass-bet-bet(call)
        if len(history) == 3:
            return True
        return False

    @staticmethod
    def terminal_utility(history: tuple, cards: tuple[int, int]) -> float:
        """Returns utility for player 0."""
        if len(history) == 2:
            if history == (0, 0):
                # pass-pass: showdown, pot = 2 (ante 1 each)
                return 1.0 if cards[0] > cards[1] else -1.0
            elif history == (1, 0):
                # bet-fold: P0 wins P1's ante
                return 1.0
            elif history == (1, 1):
                # bet-call: showdown, pot = 4 (2 each)
                return 2.0 if cards[0] > cards[1] else -2.0
        elif len(history) == 3:
            if history == (0, 1, 0):
                # pass-bet-fold: P0 folds, loses ante
                return -1.0
            elif history == (0, 1, 1):
                # pass-bet-call: showdown, pot = 4
                return 2.0 if cards[0] > cards[1] else -2.0
        raise ValueError(f"Not terminal: {history}")

    @staticmethod
    def active_player(history: tuple) -> int:
        return len(history) % 2

    @staticmethod
    def get_info_set_key(history: tuple, cards: tuple[int, int], player: int) -> str:
        card_name = CARD_NAMES[cards[player]]
        hist_str = ''.join('p' if a == 0 else 'b' for a in history)
        return f"{card_name}:{hist_str}"

    @staticmethod
    def get_actions(history: tuple) -> list[int]:
        return [0, 1]

    @staticmethod
    def all_card_deals() -> list[tuple[int, int]]:
        return [(a, b) for a, b in permutations(CARDS, 2)]


class KuhnCFRTrainer:
    """CFR+ trainer for Kuhn Poker, reusing CFRNode."""

    def __init__(self):
        self.nodes: dict[str, CFRNode] = {}
        self.game = KuhnPoker()
        self.iterations = 0

    def train(self, num_iterations: int) -> list[float]:
        """Train CFR+ on Kuhn Poker. Returns game values per iteration."""
        game_values = []
        deals = self.game.all_card_deals()

        for i in range(num_iterations):
            t = self.iterations + i + 1
            weight = max(t - num_iterations // 4, 0)

            iter_value = 0.0
            for cards in deals:
                iter_value += self._cfr(cards, (), 1.0, 1.0, weight)
            iter_value /= len(deals)
            game_values.append(iter_value)
            self.iterations += 1

        return game_values

    def _cfr(self, cards: tuple[int, int], history: tuple,
             p0: float, p1: float, weight: float) -> float:
        """Vanilla CFR+ traversal. Returns utility for P0."""
        if self.game.is_terminal(history):
            return self.game.terminal_utility(history, cards)

        player = self.game.active_player(history)
        info_key = self.game.get_info_set_key(history, cards, player)
        actions = self.game.get_actions(history)

        if info_key not in self.nodes:
            self.nodes[info_key] = CFRNode(len(actions))
        node = self.nodes[info_key]

        strategy = node.get_strategy()
        reach = p0 if player == 0 else p1
        node.accumulate_strategy(strategy, reach * weight)

        action_utilities = np.zeros(len(actions))
        for i, action in enumerate(actions):
            new_history = history + (action,)
            if player == 0:
                action_utilities[i] = self._cfr(
                    cards, new_history, p0 * strategy[i], p1, weight)
            else:
                action_utilities[i] = self._cfr(
                    cards, new_history, p0, p1 * strategy[i], weight)

        node_utility = strategy @ action_utilities

        # CFR+ regret update
        opponent_reach = p1 if player == 0 else p0
        sign = 1.0 if player == 0 else -1.0
        node.update_regrets(sign * opponent_reach * (action_utilities - node_utility))

        return node_utility

    def get_average_strategy(self) -> dict[str, np.ndarray]:
        """Return average strategies for all info sets."""
        return {key: node.get_average_strategy()
                for key, node in self.nodes.items()}

    def get_strategy_readable(self) -> dict[str, dict[str, float]]:
        """Human-readable strategies: {info_set: {pass: p, bet: p}}."""
        result = {}
        for key, node in self.nodes.items():
            avg = node.get_average_strategy()
            result[key] = {'pass': float(avg[0]), 'bet': float(avg[1])}
        return result
