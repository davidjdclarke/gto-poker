#!/usr/bin/env python3
"""
Toy-game validation for solver dynamics (v11 Workstream B3).

Validates weight schedule variants on Kuhn Poker before committing to
expensive HUNL training. Kuhn has a known Nash equilibrium (game value
= -1/18), so we can measure exact convergence quality.

Usage:
    venv/bin/python run_toy_validation.py
    venv/bin/python run_toy_validation.py --iters 50000
    venv/bin/python run_toy_validation.py --plot  # save convergence plot
"""
import sys
import time
import numpy as np
from server.gto.kuhn import KuhnPoker, KuhnCFRTrainer
from server.gto.cfr import CFRNode

NASH_GAME_VALUE = -1 / 18  # ≈ -0.0556


class ScheduledKuhnTrainer(KuhnCFRTrainer):
    """Kuhn CFR trainer with configurable weight and discount schedules.

    Extends KuhnCFRTrainer to support:
    - Linear (standard CFR+)
    - Polynomial (t^power weighting)
    - Scheduled DCFR (time-varying discount + DCFR weights)
    """

    def __init__(self, weight_mode: str = 'linear',
                 weight_param: float = 1.0,
                 regret_discount: float = 1.0):
        super().__init__()
        self.weight_mode = weight_mode
        self.weight_param = weight_param
        self.regret_discount = regret_discount

    def train(self, num_iterations: int) -> list[float]:
        game_values = []
        deals = self.game.all_card_deals()
        delay = num_iterations // 4

        for i in range(num_iterations):
            t = self.iterations + i + 1

            # Compute weight based on mode
            if t <= delay:
                weight = 0.0
            elif self.weight_mode == 'polynomial':
                weight = (t - delay) ** self.weight_param
            elif self.weight_mode == 'scheduled':
                # Scheduled DCFR
                gamma = self.weight_param if self.weight_param != 1.0 else 2.0
                weight = (t / (t + 1.0)) ** gamma
                # Time-varying regret discount
                alpha = 1.5
                self.regret_discount = (t ** alpha) / (t ** alpha + 1.0)
            else:
                # Linear (standard)
                weight = max(t - delay, 0)

            iter_value = 0.0
            for cards in deals:
                iter_value += self._cfr_scheduled(cards, (), 1.0, 1.0, weight)
            iter_value /= len(deals)
            game_values.append(iter_value)
            self.iterations += 1

        return game_values

    def _cfr_scheduled(self, cards, history, p0, p1, weight):
        """CFR traversal with configurable regret discount."""
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
                action_utilities[i] = self._cfr_scheduled(
                    cards, new_history, p0 * strategy[i], p1, weight)
            else:
                action_utilities[i] = self._cfr_scheduled(
                    cards, new_history, p0, p1 * strategy[i], weight)

        node_utility = strategy @ action_utilities

        opponent_reach = p1 if player == 0 else p0
        sign = 1.0 if player == 0 else -1.0
        regrets = sign * opponent_reach * (action_utilities - node_utility)
        node.update_regrets(regrets, discount=self.regret_discount)

        return node_utility


def compute_exploitability(trainer) -> float:
    """Exact exploitability for Kuhn Poker."""
    game = trainer.game
    strategies = {}
    for key, node in trainer.nodes.items():
        strategies[key] = node.get_average_strategy()

    def best_response_value(cards, history, br_player, pi_opponent):
        if game.is_terminal(history):
            util = game.terminal_utility(history, cards)
            return util if br_player == 0 else -util

        player = game.active_player(history)
        info_key = game.get_info_set_key(history, cards, player)
        actions = game.get_actions(history)

        if player == br_player:
            # BR player: take best action
            best = float('-inf')
            for i, action in enumerate(actions):
                val = best_response_value(cards, history + (action,),
                                          br_player, pi_opponent)
                best = max(best, val)
            return best
        else:
            # Opponent: play according to average strategy
            strat = strategies.get(info_key, np.ones(len(actions)) / len(actions))
            val = 0.0
            for i, action in enumerate(actions):
                val += strat[i] * best_response_value(
                    cards, history + (action,), br_player,
                    pi_opponent * strat[i])
            return val

    deals = game.all_card_deals()
    br0 = sum(best_response_value(c, (), 0, 1.0) for c in deals) / len(deals)
    br1 = sum(best_response_value(c, (), 1, 1.0) for c in deals) / len(deals)
    return (br0 + br1) / 2.0


class Zhang2026KuhnTrainer(ScheduledKuhnTrainer):
    """Kuhn CFR trainer with Zhang et al. (AAAI 2026) automated schedule.

    discount[t] = t^alpha / (t^alpha + c)
    weight[t]   = t^beta
    """

    def __init__(self, alpha: float = 1.5, beta: float = 2.0, c: float = 1.0):
        super().__init__(weight_mode='zhang2026', weight_param=beta, regret_discount=1.0)
        self.zhang_alpha = alpha
        self.zhang_beta = beta
        self.zhang_c = c

    def train(self, num_iterations: int) -> list[float]:
        game_values = []
        deals = self.game.all_card_deals()
        delay = num_iterations // 4

        for i in range(num_iterations):
            t = self.iterations + i + 1

            # Zhang 2026 schedule
            t_pow_alpha = t ** self.zhang_alpha
            self.regret_discount = t_pow_alpha / (t_pow_alpha + self.zhang_c)

            if t <= delay:
                weight = 0.0
            else:
                weight = (t - delay) ** self.zhang_beta

            iter_value = 0.0
            for cards in deals:
                iter_value += self._cfr_scheduled(cards, (), 1.0, 1.0, weight)
            iter_value /= len(deals)
            game_values.append(iter_value)
            self.iterations += 1

        return game_values


def run_validation(num_iterations: int = 20000, save_plot: bool = False):
    """Run all schedule variants on Kuhn Poker and compare convergence."""
    configs = [
        ('linear (CFR+)', 'linear', 1.0, 1.0),
        ('poly(1.25)', 'polynomial', 1.25, 1.0),
        ('poly(1.5)', 'polynomial', 1.5, 1.0),
        ('poly(2.0)', 'polynomial', 2.0, 1.0),
        ('scheduled(2.0)', 'scheduled', 2.0, 1.0),
        ('scheduled(3.0)', 'scheduled', 3.0, 1.0),
        ('DCFR(0.995)', 'linear', 1.0, 0.995),
    ]
    # WS2c: Zhang 2026 schedule configs — sweep alpha, beta, c
    zhang_configs = [
        ('zhang(1.5,2.0,1)', 1.5, 2.0, 1.0),
        ('zhang(2.0,2.0,1)', 2.0, 2.0, 1.0),
        ('zhang(1.5,2.0,10)', 1.5, 2.0, 10.0),
        ('zhang(2.0,3.0,1)', 2.0, 3.0, 1.0),
        ('zhang(1.5,3.0,10)', 1.5, 3.0, 10.0),
        ('zhang(3.0,2.0,1)', 3.0, 2.0, 1.0),
    ]

    print(f"{'='*70}")
    print(f"  Kuhn Poker Solver Dynamics Validation ({num_iterations:,} iterations)")
    print(f"  Nash game value: {NASH_GAME_VALUE:.6f}")
    print(f"{'='*70}\n")

    results = {}
    checkpoints = [1000, 2000, 5000, 10000, num_iterations]
    checkpoints = [c for c in checkpoints if c <= num_iterations]

    for name, mode, param, discount in configs:
        t0 = time.time()
        trainer = ScheduledKuhnTrainer(
            weight_mode=mode,
            weight_param=param,
            regret_discount=discount)
        game_values = trainer.train(num_iterations)
        elapsed = time.time() - t0

        # Measure exploitability at checkpoints
        exploit_at = {}
        for ckpt in checkpoints:
            # Retrain to exact checkpoint
            t2 = ScheduledKuhnTrainer(mode, param, discount)
            t2.train(ckpt)
            exploit_at[ckpt] = compute_exploitability(t2)

        final_exploit = exploit_at[num_iterations]
        final_value_error = abs(game_values[-1] - NASH_GAME_VALUE)

        results[name] = {
            'exploit_at': exploit_at,
            'final_exploit': final_exploit,
            'value_error': final_value_error,
            'elapsed': elapsed,
            'game_values': game_values,
        }

        status = 'OK' if final_exploit < 0.01 else 'WARN' if final_exploit < 0.05 else 'FAIL'
        print(f"  [{status:>4}] {name:25s}  exploit={final_exploit:.6f}  "
              f"val_err={final_value_error:.6f}  time={elapsed:.2f}s")

    # WS2c: Zhang 2026 schedule sweep
    print(f"\n  --- Zhang 2026 Schedule Sweep ---")
    for name, alpha, beta, c in zhang_configs:
        t0 = time.time()
        trainer = Zhang2026KuhnTrainer(alpha=alpha, beta=beta, c=c)
        game_values = trainer.train(num_iterations)
        elapsed = time.time() - t0

        exploit_at = {}
        for ckpt in checkpoints:
            t2 = Zhang2026KuhnTrainer(alpha=alpha, beta=beta, c=c)
            t2.train(ckpt)
            exploit_at[ckpt] = compute_exploitability(t2)

        final_exploit = exploit_at[num_iterations]
        final_value_error = abs(game_values[-1] - NASH_GAME_VALUE)

        results[name] = {
            'exploit_at': exploit_at,
            'final_exploit': final_exploit,
            'value_error': final_value_error,
            'elapsed': elapsed,
            'game_values': game_values,
        }

        status = 'OK' if final_exploit < 0.01 else 'WARN' if final_exploit < 0.05 else 'FAIL'
        print(f"  [{status:>4}] {name:25s}  exploit={final_exploit:.6f}  "
              f"val_err={final_value_error:.6f}  time={elapsed:.2f}s")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  Exploitability at Checkpoints")
    print(f"{'='*70}")

    header = f"  {'Config':25s}"
    for ckpt in checkpoints:
        header += f"  {ckpt:>8,}"
    print(header)
    print(f"  {'-'*25}" + "  " + "  ".join(['-'*8] * len(checkpoints)))

    for name, data in results.items():
        row = f"  {name:25s}"
        for ckpt in checkpoints:
            row += f"  {data['exploit_at'][ckpt]:>8.5f}"
        print(row)

    # Recommendation
    print(f"\n{'='*70}")
    print(f"  Recommendation")
    print(f"{'='*70}")

    baseline_exploit = results['linear (CFR+)']['final_exploit']
    improvements = []
    for name, data in results.items():
        if name == 'linear (CFR+)':
            continue
        delta = baseline_exploit - data['final_exploit']
        # Check if it converges faster at 5k checkpoint
        fast_ckpt = min(5000, num_iterations)
        baseline_5k = results['linear (CFR+)']['exploit_at'].get(fast_ckpt, baseline_exploit)
        variant_5k = data['exploit_at'].get(fast_ckpt, data['final_exploit'])
        speed_delta = baseline_5k - variant_5k

        improvements.append((name, delta, speed_delta, data['final_exploit']))

    improvements.sort(key=lambda x: x[1], reverse=True)

    for name, delta, speed_delta, final in improvements:
        direction = '+' if delta > 0 else ''
        speed_dir = '+' if speed_delta > 0 else ''
        print(f"  {name:25s}  final_delta={direction}{delta:.6f}  "
              f"speed_delta={speed_dir}{speed_delta:.6f}  "
              f"final={final:.6f}")

    best = improvements[0] if improvements else None
    if best and best[1] > 0:
        print(f"\n  Best variant: {best[0]} "
              f"({'+' if best[1] > 0 else ''}{best[1]:.6f} exploitability improvement)")
    else:
        print(f"\n  No variant improves over baseline CFR+ at {num_iterations:,} iters.")

    if save_plot:
        _save_plot(results, checkpoints, num_iterations)

    return results


def _save_plot(results, checkpoints, num_iterations):
    """Save convergence plot to docs/results/."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available, skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Exploitability convergence
    for name, data in results.items():
        xs = sorted(data['exploit_at'].keys())
        ys = [data['exploit_at'][x] for x in xs]
        ax1.plot(xs, ys, 'o-', label=name, markersize=4)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Exploitability')
    ax1.set_title('Kuhn Poker: Exploitability Convergence')
    ax1.legend(fontsize=8)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Game value convergence
    window = max(100, num_iterations // 100)
    for name, data in results.items():
        values = data['game_values']
        if len(values) > window:
            smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
            ax2.plot(range(window, len(values)+1), smoothed, label=name, alpha=0.7)
    ax2.axhline(y=NASH_GAME_VALUE, color='k', linestyle='--', alpha=0.5, label='Nash')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Game Value')
    ax2.set_title('Kuhn Poker: Game Value Convergence')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = 'docs/results/v11_toy_validation.png'
    plt.savefig(path, dpi=150)
    print(f"  Plot saved to {path}")


if __name__ == '__main__':
    iters = 20000
    save_plot = False
    for arg in sys.argv[1:]:
        if arg == '--plot':
            save_plot = True
        elif arg.startswith('--iters'):
            iters = int(sys.argv[sys.argv.index(arg) + 1])
        elif arg.isdigit():
            iters = int(arg)

    run_validation(iters, save_plot)
