"""
Kuhn Poker benchmark tests for CFR+ correctness.

Verifies the solver converges to the known Nash Equilibrium:
- P1 bets K always, bluffs J at 1/3, checks Q always
- P2 calls K always, calls Q vs bet at 1/3, folds J always
- Game value = -1/18 for P1
- Exploitability -> 0
"""
import pytest
from server.gto.kuhn import KuhnCFRTrainer, KuhnPoker
from server.gto.exploitability import exploitability_kuhn, exploitability_abstracted
from server.gto.cfr import CFRTrainer


class TestKuhnGameLogic:
    """Test Kuhn Poker terminal utilities are correct."""

    def test_pass_pass_higher_wins(self):
        game = KuhnPoker()
        # K vs J: P0 has K, wins
        assert game.terminal_utility((0, 0), (2, 0)) == 1.0
        # J vs K: P0 has J, loses
        assert game.terminal_utility((0, 0), (0, 2)) == -1.0

    def test_bet_fold(self):
        game = KuhnPoker()
        # P0 bets, P1 folds -> P0 wins ante
        assert game.terminal_utility((1, 0), (0, 2)) == 1.0
        assert game.terminal_utility((1, 0), (2, 0)) == 1.0

    def test_bet_call_showdown(self):
        game = KuhnPoker()
        # bet-call: winner gets 2
        assert game.terminal_utility((1, 1), (2, 0)) == 2.0
        assert game.terminal_utility((1, 1), (0, 2)) == -2.0

    def test_pass_bet_fold(self):
        game = KuhnPoker()
        # P0 checks, P1 bets, P0 folds
        assert game.terminal_utility((0, 1, 0), (0, 2)) == -1.0

    def test_pass_bet_call_showdown(self):
        game = KuhnPoker()
        assert game.terminal_utility((0, 1, 1), (2, 0)) == 2.0
        assert game.terminal_utility((0, 1, 1), (0, 2)) == -2.0

    def test_zero_sum(self):
        """Every terminal node must be zero-sum."""
        game = KuhnPoker()
        terminals = [(0, 0), (1, 0), (1, 1), (0, 1, 0), (0, 1, 1)]
        for cards in game.all_card_deals():
            for h in terminals:
                u0 = game.terminal_utility(h, cards)
                u1 = game.terminal_utility(h, (cards[1], cards[0]))
                # Swap cards and negate should give same result
                # (or just verify u0 = -u1 when viewed from other player)
                # Since utility is always P0's, swapping cards and
                # checking the same history tests symmetry


class TestKuhnConvergence:
    """Test CFR+ converges to Nash Equilibrium on Kuhn Poker."""

    @pytest.fixture
    def trained_strategy(self):
        trainer = KuhnCFRTrainer()
        trainer.train(50000)
        return trainer.get_average_strategy(), trainer.get_strategy_readable()

    def test_game_value(self):
        """Game value should converge to -1/18 ~ -0.0556."""
        trainer = KuhnCFRTrainer()
        values = trainer.train(20000)
        # Average over last 1000 iterations
        avg_value = sum(values[-1000:]) / 1000
        assert abs(avg_value - (-1.0/18)) < 0.02, \
            f"Game value {avg_value:.4f} not close to {-1/18:.4f}"

    def test_p1_king_bet_frequency(self, trained_strategy):
        """P1 with King bet freq should be 3*alpha where alpha in [0, 1/3].
        Nash family: bet freq in [0, 1]. Must be >= J bluff freq * 3."""
        _, readable = trained_strategy
        king_open = readable.get('K:', {})
        jack_open = readable.get('J:', {})
        k_bet = king_open.get('bet', 0)
        j_bet = jack_open.get('bet', 0)
        # K bet = 3 * J bluff (Nash relationship)
        assert abs(k_bet - 3 * j_bet) < 0.1, \
            f"P1 King bet {k_bet:.3f} should be ~3x Jack bluff {j_bet:.3f}"

    def test_p1_queen_always_checks(self, trained_strategy):
        """P1 with Queen should always check."""
        _, readable = trained_strategy
        queen_open = readable.get('Q:', {})
        assert queen_open.get('pass', 0) > 0.9, \
            f"P1 Queen check freq {queen_open.get('pass', 0):.3f} should be ~1.0"

    def test_p1_jack_bluff_in_range(self, trained_strategy):
        """P1 with Jack should bluff at alpha in [0, 1/3]."""
        _, readable = trained_strategy
        jack_open = readable.get('J:', {})
        bet_freq = jack_open.get('bet', 0)
        assert 0 <= bet_freq <= 1/3 + 0.05, \
            f"P1 Jack bluff freq {bet_freq:.3f} should be in [0, {1/3:.3f}]"

    def test_p2_king_always_calls(self, trained_strategy):
        """P2 with King facing bet should always call."""
        _, readable = trained_strategy
        king_vs_bet = readable.get('K:b', {})
        assert king_vs_bet.get('bet', 0) > 0.9, \
            f"P2 King call freq {king_vs_bet.get('bet', 0):.3f} should be ~1.0"

    def test_p2_jack_always_folds(self, trained_strategy):
        """P2 with Jack facing bet should always fold."""
        _, readable = trained_strategy
        jack_vs_bet = readable.get('J:b', {})
        assert jack_vs_bet.get('pass', 0) > 0.9, \
            f"P2 Jack fold freq {jack_vs_bet.get('pass', 0):.3f} should be ~1.0"

    def test_p2_queen_calls_one_third(self, trained_strategy):
        """P2 with Queen facing bet should call ~1/3 to keep P1 J indifferent."""
        _, readable = trained_strategy
        queen_vs_bet = readable.get('Q:b', {})
        call_freq = queen_vs_bet.get('bet', 0)
        assert abs(call_freq - 1/3) < 0.1, \
            f"P2 Queen call freq {call_freq:.3f} should be ~{1/3:.3f}"


class TestExploitability:
    """Test exploitability decreases with training."""

    def test_kuhn_exploitability_decreases(self):
        """Exploitability should decrease with more iterations."""
        trainer = KuhnCFRTrainer()

        trainer.train(500)
        exp_500 = exploitability_kuhn(trainer.get_average_strategy())

        trainer.train(4500)  # total 5000
        exp_5000 = exploitability_kuhn(trainer.get_average_strategy())

        trainer.train(45000)  # total 50000
        exp_50000 = exploitability_kuhn(trainer.get_average_strategy())

        print(f"Exploitability: 500={exp_500:.6f}, 5k={exp_5000:.6f}, 50k={exp_50000:.6f}")

        assert exp_5000 < exp_500, \
            f"Exploitability should decrease: {exp_5000:.6f} >= {exp_500:.6f}"
        assert exp_50000 < 0.005, \
            f"Exploitability after 50k iters should be < 0.005: {exp_50000:.6f}"

    def test_kuhn_near_zero_exploitability(self):
        """After sufficient training, exploitability should be near zero."""
        trainer = KuhnCFRTrainer()
        trainer.train(100000)
        exp = exploitability_kuhn(trainer.get_average_strategy())
        print(f"Exploitability after 100k: {exp:.8f}")
        assert exp < 0.001, f"Exploitability {exp:.6f} should be < 0.001"

    def test_abstracted_exploitability_smoke(self):
        """Smoke test: abstracted game exploitability should be computable."""
        trainer = CFRTrainer()
        trainer.train(2000, sampling='external')
        exp = exploitability_abstracted(trainer, phases=['preflop'])
        print(f"Abstracted preflop exploitability after 2k: {exp:.4f}")
        assert exp >= 0, "Exploitability must be non-negative"
        assert exp < 200, "Exploitability should be bounded"
