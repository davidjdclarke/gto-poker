#!/usr/bin/env python3
"""
Variance study: measure stability of Monte Carlo best-response estimates.

Trains a strategy to a fixed iteration count, then runs the BR estimator
multiple times with different seeds to measure mean and std of exploitability.

Usage:
    python variance_study.py [iterations] [num_runs] [samples_per_run]
    python variance_study.py 5000 10 500
    python variance_study.py 5000 10 2000
"""
import sys
import time
import random
import json
import numpy as np
from pathlib import Path
from server.gto.cfr import CFRTrainer
from server.gto.exploitability import exploitability_abstracted

RESULTS_FILE = Path(__file__).parent / "variance_results.json"


def run_variance_study(iterations: int = 5000,
                       num_runs: int = 10,
                       samples_per_run: int = 500):
    """Train once, then measure exploitability num_runs times."""

    print(f"Training {iterations} iterations...")
    trainer = CFRTrainer()
    t0 = time.time()
    trainer.train(iterations, averaging_delay=iterations // 4,
                  sampling='external')
    train_time = time.time() - t0
    print(f"Training done in {train_time:.1f}s ({len(trainer.nodes)} nodes)")

    results_500 = []
    results_2k = []

    # Run at specified sample count
    print(f"\n--- {num_runs} BR runs at {samples_per_run} samples ---")
    print(f"{'Run':>4} {'Exploitability':>15}")
    print("-" * 25)

    for i in range(num_runs):
        random.seed(i * 1000 + 42)
        np.random.seed(i * 1000 + 42)
        exp = exploitability_abstracted(trainer, samples=samples_per_run)
        results_500.append(exp)
        print(f"{i+1:>4} {exp:>15.4f}")

    mean_500 = np.mean(results_500)
    std_500 = np.std(results_500)
    print(f"\nMean: {mean_500:.4f}  Std: {std_500:.4f}  "
          f"CV: {std_500/mean_500*100:.1f}%")

    # Also run at 2000 samples for comparison
    samples_hi = 2000
    print(f"\n--- {num_runs} BR runs at {samples_hi} samples ---")
    print(f"{'Run':>4} {'Exploitability':>15}")
    print("-" * 25)

    for i in range(num_runs):
        random.seed(i * 2000 + 99)
        np.random.seed(i * 2000 + 99)
        exp = exploitability_abstracted(trainer, samples=samples_hi)
        results_2k.append(exp)
        print(f"{i+1:>4} {exp:>15.4f}")

    mean_2k = np.mean(results_2k)
    std_2k = np.std(results_2k)
    print(f"\nMean: {mean_2k:.4f}  Std: {std_2k:.4f}  "
          f"CV: {std_2k/mean_2k*100:.1f}%")

    # Save
    output = {
        'iterations': iterations,
        'num_nodes': len(trainer.nodes),
        'runs': {
            f'{samples_per_run}_samples': {
                'values': results_500,
                'mean': float(mean_500),
                'std': float(std_500),
                'cv_pct': float(std_500 / mean_500 * 100),
            },
            f'{samples_hi}_samples': {
                'values': results_2k,
                'mean': float(mean_2k),
                'std': float(std_2k),
                'cv_pct': float(std_2k / mean_2k * 100),
            },
        },
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Verdict
    print("\n--- Verdict ---")
    if std_500 / mean_500 > 0.15:
        print(f"⚠ High variance at {samples_per_run} samples "
              f"(CV={std_500/mean_500*100:.1f}%) - increase sample count")
    elif std_500 / mean_500 > 0.05:
        print(f"~ Moderate variance at {samples_per_run} samples "
              f"(CV={std_500/mean_500*100:.1f}%) - usable but noisy")
    else:
        print(f"✓ Low variance at {samples_per_run} samples "
              f"(CV={std_500/mean_500*100:.1f}%) - metric is stable")


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    samp = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    run_variance_study(iters, runs, samp)
