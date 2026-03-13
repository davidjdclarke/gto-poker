#!/usr/bin/env python3
"""
Convergence study: measure exploitability vs iterations.

Separates convergence problems from abstraction problems.
Plots exploitability curve on log-scale x-axis.

Usage:
    python convergence_study.py [max_iterations] [sampling]
    python convergence_study.py 100000 external
"""
import sys
import time
import json
from pathlib import Path
from server.gto.cfr import CFRTrainer
from server.gto.exploitability import exploitability_abstracted

RESULTS_FILE = Path(__file__).parent / "convergence_results.json"


def run_convergence_study(max_iterations: int = 100000,
                          sampling: str = 'external'):
    """Train and measure exploitability at checkpoints."""

    checkpoints = []
    # Log-spaced checkpoints: 1k, 2k, 5k, 10k, 20k, 50k, 100k, ...
    i = 1000
    while i <= max_iterations:
        checkpoints.append(i)
        if i * 2 <= max_iterations:
            checkpoints.append(i * 2)
        if i * 5 <= max_iterations:
            checkpoints.append(i * 5)
        i *= 10
    checkpoints = sorted(set(c for c in checkpoints if c <= max_iterations))

    if max_iterations not in checkpoints:
        checkpoints.append(max_iterations)

    trainer = CFRTrainer()
    results = []
    trained_so_far = 0

    print(f"Convergence study: {max_iterations} iterations, {sampling} sampling")
    print(f"Checkpoints: {checkpoints}")
    print(f"{'Iterations':>12} {'Exploit':>10} {'Exploit_PF':>10} "
          f"{'Nodes':>8} {'Time(s)':>8} {'Trend':>8}")
    print("-" * 70)

    for checkpoint in checkpoints:
        iters_needed = checkpoint - trained_so_far
        if iters_needed <= 0:
            continue

        t0 = time.time()
        averaging_delay = checkpoint // 4
        trainer.train(iters_needed, averaging_delay=averaging_delay,
                      sampling=sampling)
        train_time = time.time() - t0
        trained_so_far = checkpoint

        # Measure exploitability
        t0 = time.time()
        exp_all = exploitability_abstracted(trainer)
        exp_preflop = exploitability_abstracted(trainer, phases=['preflop'])
        eval_time = time.time() - t0

        # Trend indicator
        trend = ""
        if results:
            prev = results[-1]['exploitability']
            if exp_all < prev * 0.9:
                trend = "↓↓"
            elif exp_all < prev:
                trend = "↓"
            elif exp_all > prev * 1.1:
                trend = "↑↑"
            else:
                trend = "→"

        result = {
            'iterations': checkpoint,
            'exploitability': exp_all,
            'exploitability_preflop': exp_preflop,
            'num_nodes': len(trainer.nodes),
            'train_time': train_time,
            'eval_time': eval_time,
        }
        results.append(result)

        print(f"{checkpoint:>12,} {exp_all:>10.4f} {exp_preflop:>10.4f} "
              f"{len(trainer.nodes):>8,} {train_time:>8.1f} {trend:>8}")

    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump({
            'sampling': sampling,
            'max_iterations': max_iterations,
            'results': results,
        }, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Save trained weights for reuse
    weights_file = Path(__file__).parent / f"strategy_{max_iterations // 1000}k.json"
    trainer.save(str(weights_file))
    print(f"Strategy saved to {weights_file}")

    # Summary
    print("\n--- Summary ---")
    if len(results) >= 2:
        first = results[0]
        last = results[-1]
        reduction = (1 - last['exploitability'] / first['exploitability']) * 100
        print(f"Exploitability: {first['exploitability']:.4f} → "
              f"{last['exploitability']:.4f} ({reduction:+.1f}%)")

        # Check if plateauing
        if len(results) >= 3:
            recent = results[-3:]
            recent_change = abs(recent[-1]['exploitability'] -
                                recent[0]['exploitability'])
            avg_exp = sum(r['exploitability'] for r in recent) / 3
            if avg_exp > 0 and recent_change / avg_exp < 0.05:
                print("⚠ Exploitability appears PLATEAUED - "
                      "abstraction quality is likely the bottleneck")
            else:
                print("✓ Exploitability still decreasing - "
                      "more iterations may help")

    return results


if __name__ == "__main__":
    max_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    samp = sys.argv[2] if len(sys.argv) > 2 else 'external'
    run_convergence_study(max_iters, samp)
