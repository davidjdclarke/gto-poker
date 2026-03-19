#!/usr/bin/env python3
"""Evaluate EMD+texture strategy: H2H vs B0 and gauntlet."""
import sys
import time
import numpy as np

# Enable EMD mode BEFORE any imports that use abstraction
from server.gto.abstraction import enable_emd_mode
enable_emd_mode()
from server.gto import equity
equity.EMD_MODE_ENABLED = True

from server.gto.cfr import CFRTrainer
from eval_harness.match_engine import HeadsUpMatch, GTOAgent
from eval_harness.adversaries import get_all_adversaries

EMD_STRATEGY = "server/gto/strategy.json"
B0_STRATEGY = "experiments/best/v9_B0_100M_allbots_positive.json"
SEEDS = [42, 123, 456]


def run_h2h(hands=50000):
    """H2H: EMD+texture (P0) vs B0+pseudo_harmonic (P1)."""
    print("=" * 60)
    print("  H2H: EMD+texture vs B0+pseudo_harmonic")
    print(f"  {hands} hands × {len(SEEDS)} seeds")
    print("=" * 60)

    # Load EMD+texture
    emd_trainer = CFRTrainer()
    emd_trainer.load(EMD_STRATEGY)
    print(f"  EMD+texture: {len(emd_trainer.nodes)} nodes")

    # Load B0 (need separate trainer without EMD mode for B0)
    # B0 uses standard bucketing — but EMD_MODE is globally set.
    # For H2H, both agents use their own trainer but share the eval harness.
    # The GTOAgent uses fast_bucket() which delegates to hand_strength_bucket()
    # when EMD_MODE_ENABLED. For B0, we need EMD off.
    # Workaround: B0 agent uses pseudo_harmonic mapping with its own trainer.
    b0_trainer = CFRTrainer()
    b0_trainer.load(B0_STRATEGY)
    print(f"  B0: {len(b0_trainer.nodes)} nodes")

    results = []
    for seed in SEEDS:
        emd_agent = GTOAgent(emd_trainer, name="EMD+tex", mapping="pseudo_harmonic",
                             simulations=80)
        b0_agent = GTOAgent(b0_trainer, name="B0", mapping="pseudo_harmonic",
                            simulations=80)

        match = HeadsUpMatch(emd_agent, b0_agent, big_blind=20, seed=seed)
        t0 = time.time()
        result = match.play(hands)
        elapsed = time.time() - t0

        bb100 = result.p0_bb_per_100
        results.append(bb100)
        print(f"  Seed {seed}: EMD+tex {bb100:+.1f} bb/100 ({elapsed:.0f}s)")

    mean = np.mean(results)
    std = np.std(results, ddof=1)
    print(f"\n  H2H result: {mean:+.1f} ± {std:.1f} bb/100")
    print(f"  Per seed: {[f'{r:+.1f}' for r in results]}")
    return mean


def run_gauntlet(hands=10000):
    """Classic gauntlet with EMD+texture strategy."""
    print("\n" + "=" * 60)
    print("  Classic Gauntlet: EMD+texture + pseudo_harmonic")
    print(f"  {hands} hands × {len(SEEDS)} seeds")
    print("=" * 60)

    emd_trainer = CFRTrainer()
    emd_trainer.load(EMD_STRATEGY)
    bots = get_all_adversaries(trainer=emd_trainer)

    all_results = {}
    for bot in bots:
        bot_results = []
        for seed in SEEDS:
            gto = GTOAgent(emd_trainer, name="EMD+tex", mapping="pseudo_harmonic",
                           simulations=80)
            match = HeadsUpMatch(gto, bot, big_blind=20, seed=seed)
            result = match.play(hands)
            bot_results.append(result.p0_bb_per_100)

        mean = np.mean(bot_results)
        std = np.std(bot_results, ddof=1)
        all_results[bot.name] = mean
        print(f"  {bot.name:20s}: {mean:+8.1f} ± {std:.1f} bb/100")

    avg = np.mean(list(all_results.values()))
    print(f"\n  Gauntlet average: {avg:+.1f} bb/100")
    return all_results


if __name__ == '__main__':
    h2h = run_h2h(hands=5000)
    gauntlet = run_gauntlet(hands=5000)
