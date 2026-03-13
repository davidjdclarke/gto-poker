#!/usr/bin/env python3
"""
Experiment runner: train, evaluate, and log results to the experiment matrix.

Runs a named experiment with configurable parameters and appends results
to experiments/matrix.json for easy comparison.

Usage:
    python run_experiment.py --name baseline_v5 --iterations 100000
    python run_experiment.py --name position_test --iterations 50000 --notes "testing position encoding"
    python run_experiment.py --list                    # Show all experiments
    python run_experiment.py --compare exp1 exp2       # Compare two experiments
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import random as rng

from server.gto.cfr import CFRTrainer, STRATEGY_VERSION
from server.gto.abstraction import NUM_EQUITY_BUCKETS, NUM_HAND_TYPES, NUM_BUCKETS
from server.gto.exploitability import (
    exploitability_abstracted, exploitability_breakdown, strategy_audit,
)

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
MATRIX_FILE = EXPERIMENTS_DIR / "matrix.json"


def load_matrix() -> list[dict]:
    """Load the experiment matrix."""
    if MATRIX_FILE.exists():
        with open(MATRIX_FILE) as f:
            return json.load(f)
    return []


def save_matrix(matrix: list[dict]):
    """Save the experiment matrix."""
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    with open(MATRIX_FILE, 'w') as f:
        json.dump(matrix, f, indent=2)


def run_experiment(name: str, iterations: int = 100000,
                   sampling: str = 'external',
                   notes: str = '') -> dict:
    """Run a full experiment: train + evaluate."""

    print(f"=== Experiment: {name} ===")
    print(f"  Iterations: {iterations:,}")
    print(f"  Sampling: {sampling}")
    print(f"  Buckets: {NUM_BUCKETS} ({NUM_EQUITY_BUCKETS} eq x {NUM_HAND_TYPES} ht)")
    print(f"  Strategy version: {STRATEGY_VERSION}")

    # Train
    trainer = CFRTrainer()
    delay = iterations // 4
    print(f"\nTraining...")
    t0 = time.time()
    trainer.train(iterations, averaging_delay=delay, sampling=sampling)
    train_time = time.time() - t0
    print(f"  Done in {train_time:.1f}s ({len(trainer.nodes):,} nodes)")

    # Evaluate exploitability (3 seeds)
    print("\nEvaluating exploitability (500 samples, 3 seeds)...")
    exp_values = []
    exp_pf_values = []
    for seed in [42, 123, 456]:
        rng.seed(seed)
        np.random.seed(seed)
        exp = exploitability_abstracted(trainer, samples=500)
        exp_pf = exploitability_abstracted(trainer, phases=['preflop'], samples=500)
        exp_values.append(exp)
        exp_pf_values.append(exp_pf)

    exp_mean = float(np.mean(exp_values))
    exp_std = float(np.std(exp_values, ddof=1))
    exp_pf_mean = float(np.mean(exp_pf_values))

    print(f"  All phases: {exp_mean:.4f} ± {exp_std:.4f}")
    print(f"  Preflop:    {exp_pf_mean:.4f}")

    # Strategy audit
    print("\nRunning strategy audit...")
    audit = strategy_audit(trainer)
    print(f"  Premium limp issues: {audit['counts']['premium_limp']}")
    print(f"  All-in overuse:     {audit['counts']['allin_overuse']}")
    print(f"  Flat strategies:    {audit['counts']['flat_strategies']}")

    # Top leaks
    if audit['leak_summary']:
        print("  Top leaks:")
        for leak in audit['leak_summary'][:3]:
            print(f"    [{leak['severity']}] {leak['description']}")

    # Save strategy
    strategy_file = EXPERIMENTS_DIR / f"{name}_strategy.json"
    trainer.save(str(strategy_file))

    # Build result
    result = {
        "name": name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": notes,
        "config": {
            "iterations": iterations,
            "sampling": sampling,
            "strategy_version": STRATEGY_VERSION,
            "equity_buckets": NUM_EQUITY_BUCKETS,
            "hand_types": NUM_HAND_TYPES,
            "total_buckets": NUM_BUCKETS,
        },
        "results": {
            "num_nodes": len(trainer.nodes),
            "train_time_s": round(train_time, 1),
            "ms_per_iteration": round(train_time / iterations * 1000, 2),
            "exploitability_mean": round(exp_mean, 4),
            "exploitability_std": round(exp_std, 4),
            "exploitability_preflop": round(exp_pf_mean, 4),
        },
        "audit_counts": audit["counts"],
        "strategy_file": str(strategy_file),
    }

    return result


def list_experiments():
    """Print the experiment matrix as a table."""
    matrix = load_matrix()
    if not matrix:
        print("No experiments recorded yet.")
        return

    print(f"{'Name':<25} {'Iters':>10} {'Nodes':>8} {'Exploit':>10} "
          f"{'PF Exploit':>10} {'ms/iter':>8} {'Time':>8}")
    print("-" * 90)

    for exp in matrix:
        r = exp.get("results", {})
        c = exp.get("config", {})
        print(f"{exp['name']:<25} {c.get('iterations',0):>10,} "
              f"{r.get('num_nodes',0):>8,} "
              f"{r.get('exploitability_mean',0):>10.4f} "
              f"{r.get('exploitability_preflop',0):>10.4f} "
              f"{r.get('ms_per_iteration',0):>8.2f} "
              f"{r.get('train_time_s',0):>7.0f}s")


def compare_experiments(name1: str, name2: str):
    """Compare two experiments side by side."""
    matrix = load_matrix()
    exp1 = next((e for e in matrix if e['name'] == name1), None)
    exp2 = next((e for e in matrix if e['name'] == name2), None)

    if not exp1:
        print(f"Experiment not found: {name1}")
        return
    if not exp2:
        print(f"Experiment not found: {name2}")
        return

    r1, r2 = exp1["results"], exp2["results"]
    c1, c2 = exp1["config"], exp2["config"]

    print(f"\n{'Metric':<25} {name1:>20} {name2:>20} {'Delta':>15}")
    print("-" * 85)

    metrics = [
        ("Iterations", c1.get("iterations"), c2.get("iterations")),
        ("Buckets", c1.get("total_buckets"), c2.get("total_buckets")),
        ("Nodes", r1.get("num_nodes"), r2.get("num_nodes")),
        ("Exploitability", r1.get("exploitability_mean"), r2.get("exploitability_mean")),
        ("Exploit (preflop)", r1.get("exploitability_preflop"), r2.get("exploitability_preflop")),
        ("ms/iteration", r1.get("ms_per_iteration"), r2.get("ms_per_iteration")),
        ("Train time (s)", r1.get("train_time_s"), r2.get("train_time_s")),
    ]

    for label, v1, v2 in metrics:
        if v1 is None or v2 is None:
            continue
        if isinstance(v1, float):
            delta = v2 - v1
            pct = (delta / v1 * 100) if v1 != 0 else 0
            print(f"{label:<25} {v1:>20.4f} {v2:>20.4f} {delta:>+10.4f} ({pct:+.1f}%)")
        else:
            delta = v2 - v1
            print(f"{label:<25} {v1:>20,} {v2:>20,} {delta:>+15,}")


def main():
    parser = argparse.ArgumentParser(description="Run and track solver experiments")
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--iterations", type=int, default=100000,
                        help="Training iterations (default: 100000)")
    parser.add_argument("--sampling", type=str, default="external",
                        choices=["external", "vanilla"])
    parser.add_argument("--notes", type=str, default="",
                        help="Notes about this experiment")
    parser.add_argument("--list", action="store_true",
                        help="List all experiments")
    parser.add_argument("--compare", nargs=2, metavar=("EXP1", "EXP2"),
                        help="Compare two experiments")
    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    if args.compare:
        compare_experiments(args.compare[0], args.compare[1])
        return

    if not args.name:
        print("Must specify --name for experiment")
        sys.exit(1)

    # Check for duplicate name
    matrix = load_matrix()
    if any(e['name'] == args.name for e in matrix):
        print(f"Experiment '{args.name}' already exists. Use a different name.")
        sys.exit(1)

    result = run_experiment(args.name, args.iterations, args.sampling, args.notes)

    # Append to matrix
    matrix.append(result)
    save_matrix(matrix)
    print(f"\nExperiment '{args.name}' saved to {MATRIX_FILE}")

    # Show comparison table
    print("\n--- All Experiments ---")
    list_experiments()


if __name__ == "__main__":
    main()
