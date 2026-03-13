#!/usr/bin/env python3
"""
Reproducible benchmark: train, evaluate, and compare strategies.

Runs the full pipeline:
  1. Train (or load) a strategy
  2. Exploitability eval at 500 and 2000 BR samples
  3. Multi-seed variance check (5 seeds)
  4. Representative strategy tables (preflop + postflop)
  5. Output summary JSON to benchmarks/

Usage:
    python run_benchmark.py                          # Load existing strategy
    python run_benchmark.py --train 100000           # Train 100k fresh
    python run_benchmark.py --load strategy_1m.json  # Load specific file
    python run_benchmark.py --name my_experiment     # Custom benchmark name
    python run_benchmark.py --compare baseline_1m    # Compare against baseline
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from server.gto.cfr import CFRTrainer
from server.gto.abstraction import (
    Action, ACTION_NAMES, NUM_EQUITY_BUCKETS, NUM_HAND_TYPES,
    HandType, make_bucket,
)
from server.gto.exploitability import (
    exploitability_abstracted, exploitability_breakdown,
    exploitability_multiseed, strategy_audit,
)

BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"
BENCHMARKS_DIR.mkdir(exist_ok=True)


def collect_strategy_tables(trainer: CFRTrainer) -> dict:
    """Collect representative strategy tables for preflop and postflop."""
    tables = {"preflop": {}, "flop": {}}

    # Preflop: all hand types at key equity levels, both positions
    for ht in HandType:
        ht_strategies = {}
        for eq in [0, 2, 4, 6, 7]:
            bucket = make_bucket(eq, int(ht))
            for pos in ['oop', 'ip']:
                strategy = trainer.get_strategy('preflop', bucket, (),
                                                position=pos)
                ht_strategies[f"eq_{eq}_{pos}"] = {
                    ACTION_NAMES.get(Action(a), str(a)): round(p, 4)
                    for a, p in strategy.items()
                }
        tables["preflop"][ht.name] = ht_strategies

    # Flop: sample across equity and hand types, both positions
    for eq in [0, 3, 5, 7]:
        for ht in [HandType.PREMIUM_PAIR, HandType.SUITED_CONNECTOR, HandType.TRASH]:
            bucket = make_bucket(eq, int(ht))
            for pos in ['oop', 'ip']:
                strategy = trainer.get_strategy('flop', bucket, (),
                                                position=pos)
                key = f"eq_{eq}_{ht.name}_{pos}"
                tables["flop"][key] = {
                    ACTION_NAMES.get(Action(a), str(a)): round(p, 4)
                    for a, p in strategy.items()
                }

    return tables


def run_exploitability(trainer: CFRTrainer, samples: int,
                       seeds: list[int] = None) -> dict:
    """Run exploitability with optional multi-seed averaging."""
    if seeds is None:
        seeds = [None]

    all_results = []
    preflop_results = []

    for seed in seeds:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        exp_all = exploitability_abstracted(trainer, samples=samples)
        exp_pf = exploitability_abstracted(trainer, phases=['preflop'],
                                           samples=samples)
        all_results.append(exp_all)
        preflop_results.append(exp_pf)

    result = {
        "samples": samples,
        "num_seeds": len(seeds),
        "all_phases": {
            "mean": float(np.mean(all_results)),
            "std": float(np.std(all_results)),
            "values": [round(v, 4) for v in all_results],
        },
        "preflop_only": {
            "mean": float(np.mean(preflop_results)),
            "std": float(np.std(preflop_results)),
            "values": [round(v, 4) for v in preflop_results],
        },
    }

    if len(seeds) > 1:
        # 95% confidence interval
        n = len(all_results)
        mean = np.mean(all_results)
        std = np.std(all_results, ddof=1)
        ci_half = 1.96 * std / np.sqrt(n)
        result["all_phases"]["ci_95"] = [
            round(float(mean - ci_half), 4),
            round(float(mean + ci_half), 4),
        ]

    return result


def run_breakdown(trainer: CFRTrainer, phase: str = 'preflop',
                  top_n: int = 15) -> list[dict]:
    """Get top exploitable infosets."""
    raw = exploitability_breakdown(trainer, phase=phase, top_n=top_n)
    return [
        {
            "key": item["key"],
            "regret_magnitude": round(item["regret_magnitude"], 2),
            "entropy": round(item["entropy"], 2),
            "visit_weight": round(item["visit_weight"], 0),
            "strategy": {k: round(v, 4) for k, v in item["strategy"].items()},
        }
        for item in raw
    ]


def run_benchmark(trainer: CFRTrainer, name: str,
                  train_time: float = 0.0) -> dict:
    """Run full benchmark suite and return results dict."""
    print("\n=== Exploitability (500 samples, 5 seeds) ===")
    t0 = time.time()
    exp_500 = run_exploitability(trainer, samples=500,
                                 seeds=[42, 123, 456, 789, 1000])
    exp_time_500 = time.time() - t0
    print(f"  All phases: {exp_500['all_phases']['mean']:.4f} "
          f"± {exp_500['all_phases']['std']:.4f}")
    print(f"  Preflop:    {exp_500['preflop_only']['mean']:.4f} "
          f"± {exp_500['preflop_only']['std']:.4f}")
    print(f"  Time: {exp_time_500:.1f}s")

    print("\n=== Exploitability (2000 samples, 3 seeds) ===")
    t0 = time.time()
    exp_2000 = run_exploitability(trainer, samples=2000,
                                  seeds=[42, 123, 456])
    exp_time_2000 = time.time() - t0
    print(f"  All phases: {exp_2000['all_phases']['mean']:.4f} "
          f"± {exp_2000['all_phases']['std']:.4f}")
    print(f"  Preflop:    {exp_2000['preflop_only']['mean']:.4f} "
          f"± {exp_2000['preflop_only']['std']:.4f}")
    print(f"  Time: {exp_time_2000:.1f}s")

    print("\n=== Strategy Tables ===")
    tables = collect_strategy_tables(trainer)
    # Print preflop summary
    for ht_name, strats in tables["preflop"].items():
        eq7 = strats.get("eq_7", {})
        eq0 = strats.get("eq_0", {})
        print(f"  {ht_name:20s}  EQ7: {_fmt_strat(eq7)}  "
              f"EQ0: {_fmt_strat(eq0)}")

    print("\n=== Top Exploitable Infosets (preflop) ===")
    breakdown_pf = run_breakdown(trainer, 'preflop', top_n=10)
    for item in breakdown_pf[:5]:
        print(f"  {item['key']}: regret={item['regret_magnitude']:.1f}, "
              f"entropy={item['entropy']:.2f}")

    print("\n=== Top Exploitable Infosets (flop) ===")
    breakdown_flop = run_breakdown(trainer, 'flop', top_n=10)
    for item in breakdown_flop[:5]:
        print(f"  {item['key']}: regret={item['regret_magnitude']:.1f}, "
              f"entropy={item['entropy']:.2f}")

    print("\n=== Strategy Audit ===")
    audit = strategy_audit(trainer)
    for category, count in audit["counts"].items():
        print(f"  {category}: {count}")
    if audit["leak_summary"]:
        print("  Top leaks:")
        for leak in audit["leak_summary"][:5]:
            print(f"    [{leak['severity']}] {leak['description']}")

    report = {
        "name": name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "iterations": trainer.iterations,
        "num_nodes": len(trainer.nodes),
        "train_time_s": round(train_time, 1),
        "exploitability_500": exp_500,
        "exploitability_2000": exp_2000,
        "eval_time_500_s": round(exp_time_500, 1),
        "eval_time_2000_s": round(exp_time_2000, 1),
        "breakdown_preflop": breakdown_pf,
        "breakdown_flop": breakdown_flop,
        "strategy_tables": tables,
        "audit": {
            "counts": audit["counts"],
            "leak_summary": audit["leak_summary"],
            "premium_limp": audit["premium_limp"][:10],
            "allin_overuse": audit["allin_overuse"][:10],
        },
    }

    return report


def compare_benchmarks(current: dict, baseline_name: str):
    """Print comparison between current benchmark and a saved baseline."""
    baseline_path = BENCHMARKS_DIR / f"{baseline_name}.json"
    if not baseline_path.exists():
        print(f"Baseline not found: {baseline_path}")
        return

    with open(baseline_path) as f:
        baseline = json.load(f)

    print(f"\n{'='*60}")
    print(f"Comparison: {current['name']} vs {baseline.get('name', baseline_name)}")
    print(f"{'='*60}")

    # Node count
    b_nodes = baseline.get("strategy_stats", {}).get("num_nodes",
              baseline.get("num_nodes", 0))
    c_nodes = current["num_nodes"]
    print(f"  Nodes:         {b_nodes:>8,} → {c_nodes:>8,} "
          f"({c_nodes - b_nodes:+,})")

    # Exploitability
    b_exp = baseline.get("exploitability", {}).get("all_phases_500_samples",
            baseline.get("exploitability_500", {}).get("all_phases", {}).get("mean", 0))
    c_exp = current["exploitability_500"]["all_phases"]["mean"]
    if b_exp > 0:
        pct = (c_exp - b_exp) / b_exp * 100
        print(f"  Exploit (500): {b_exp:>8.4f} → {c_exp:>8.4f} ({pct:+.1f}%)")
    else:
        print(f"  Exploit (500): {b_exp:>8.4f} → {c_exp:>8.4f}")

    # Iterations
    b_iters = baseline.get("training", {}).get("iterations",
              baseline.get("iterations", 0))
    c_iters = current["iterations"]
    print(f"  Iterations:    {b_iters:>8,} → {c_iters:>8,}")


def _fmt_strat(strat: dict) -> str:
    """Format a strategy dict as a compact string."""
    parts = []
    for action, prob in strat.items():
        if prob >= 0.01:
            parts.append(f"{action}={prob:.0%}")
    return ", ".join(parts) if parts else "uniform"


def main():
    parser = argparse.ArgumentParser(description="Run reproducible benchmark")
    parser.add_argument("--train", type=int, default=0,
                        help="Train fresh for N iterations")
    parser.add_argument("--load", type=str, default=None,
                        help="Load strategy from file")
    parser.add_argument("--name", type=str, default=None,
                        help="Benchmark name (default: auto)")
    parser.add_argument("--compare", type=str, default="baseline_1m",
                        help="Compare against this baseline")
    parser.add_argument("--save-strategy", type=str, default=None,
                        help="Save strategy to file after training")
    args = parser.parse_args()

    trainer = CFRTrainer()
    train_time = 0.0

    if args.load:
        filepath = args.load
        if not os.path.isabs(filepath):
            filepath = str(Path(__file__).parent / filepath)
        if not trainer.load(filepath):
            print(f"Failed to load {filepath}")
            sys.exit(1)
    elif args.train > 0:
        print(f"Training {args.train} iterations...")
        t0 = time.time()
        delay = args.train // 4
        trainer.train(args.train, averaging_delay=delay, sampling='external')
        train_time = time.time() - t0
        print(f"Training done in {train_time:.1f}s ({len(trainer.nodes)} nodes)")
    else:
        # Try loading the default strategy files
        strategy_file = str(Path(__file__).parent / "server" / "gto" / "strategy.json")
        baseline_file = str(Path(__file__).parent / "strategy_baseline_1m.json")
        if os.path.exists(strategy_file):
            trainer.load(strategy_file)
        elif os.path.exists(baseline_file):
            trainer.load(baseline_file)
        else:
            print("No strategy found. Use --load or --train.")
            sys.exit(1)

    if args.save_strategy:
        trainer.save(args.save_strategy)

    name = args.name or f"benchmark_{trainer.iterations // 1000}k"
    report = run_benchmark(trainer, name, train_time)

    # Save report
    report_path = BENCHMARKS_DIR / f"{name}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    # Compare
    if args.compare:
        compare_benchmarks(report, args.compare)

    print("\nDone.")


if __name__ == "__main__":
    main()
