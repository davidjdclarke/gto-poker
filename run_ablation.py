#!/usr/bin/env python3
"""
Ablation experiment runner for GTO solver research.

Orchestrates: pin baseline -> apply config change -> train from scratch
-> eval -> behavioral regression -> comparison report.

Usage:
    # Run a single ablation experiment
    python run_ablation.py --name exp_A_2x_schedule \
        --iterations 50000000 --workers 6 \
        --config phase_schedule_mode=0

    # Run with checkpoint evaluation
    python run_ablation.py --name exp_B_old_dampen \
        --iterations 50000000 --workers 6 \
        --config allin_dampen_mode=0 \
        --checkpoint-interval 10000000

    # Eval-only (no training, just evaluate an existing strategy)
    python run_ablation.py --name exp_D_blend --eval-only \
        --strategy server/gto/strategy.json

    # Compare two experiments
    python run_ablation.py --compare exp_A_2x_schedule exp_B_old_dampen

    # List all experiments
    python run_ablation.py --list
"""
import sys
import os
import json
import time
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict


ABLATIONS_DIR = Path("experiments/ablations")
BASELINE_DIR = ABLATIONS_DIR / "baseline_v7"


@dataclass
class AblationConfig:
    name: str
    iterations: int = 50_000_000
    workers: int = 6
    checkpoint_interval: int = 0
    config_overrides: dict = field(default_factory=dict)
    strategy_path: str = None  # For eval-only runs
    eval_only: bool = False


def get_experiment_dir(name: str) -> Path:
    return ABLATIONS_DIR / name


def run_training(config: AblationConfig) -> str:
    """Run training and return path to final strategy."""
    exp_dir = get_experiment_dir(config.name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    strategy_path = str(exp_dir / "strategy.json")

    # Build training command
    cmd = [
        sys.executable, "train_gto.py",
        str(config.iterations),
        "--fresh",
        "--workers", str(config.workers),
    ]

    # Apply config overrides as CLI flags
    overrides = config.config_overrides
    if 'phase_schedule_mode' in overrides:
        mode = overrides['phase_schedule_mode']
        cmd.extend(["--phase-schedule", "2x" if mode == 0 else "3x"])
    if 'allin_dampen_mode' in overrides:
        mode = overrides['allin_dampen_mode']
        cmd.extend(["--allin-dampen", "old" if mode == 0 else "new"])
    if overrides.get('adaptive_averaging', 0) == 1:
        cmd.append("--adaptive-averaging")

    if config.checkpoint_interval > 0:
        ckpt_dir = str(exp_dir / "checkpoints")
        cmd.extend([
            "--checkpoint-interval", str(config.checkpoint_interval),
            "--checkpoint-dir", ckpt_dir,
            "--checkpoint-eval",
        ])

    print(f"\n{'='*60}")
    print(f"  TRAINING: {config.name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # Save config
    with open(exp_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    train_time = time.time() - t0

    if result.returncode != 0:
        print(f"ERROR: Training failed with return code {result.returncode}")
        return None

    # Move strategy.json to experiment dir
    from server.gto.engine import STRATEGY_FILE
    if Path(STRATEGY_FILE).exists():
        import shutil
        shutil.copy2(STRATEGY_FILE, strategy_path)
        print(f"  Strategy copied to {strategy_path}")

    # Save timing
    with open(exp_dir / "timing.json", "w") as f:
        json.dump({"train_time_s": round(train_time, 1)}, f)

    return strategy_path


def run_evaluation(name: str, strategy_path: str,
                   baseline_behavioral: str = None) -> dict:
    """Run full eval harness and behavioral regression."""
    exp_dir = get_experiment_dir(name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    eval_path = str(exp_dir / "eval.json")

    cmd = [
        sys.executable, "run_eval_harness.py",
        "--strategy", strategy_path,
        "--save", eval_path,
        # Run all stages: gauntlet, offtree, bridge, leakage, behavioral
        "--gauntlet", "--offtree", "--bridge", "--leakage", "--behavioral",
    ]
    if baseline_behavioral:
        cmd.extend(["--behavioral-baseline", baseline_behavioral])

    print(f"\n{'='*60}")
    print(f"  EVALUATING: {name}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    eval_time = time.time() - t0

    if result.returncode != 0:
        print(f"ERROR: Evaluation failed with return code {result.returncode}")
        return {}

    # Load and return results
    if Path(eval_path).exists():
        with open(eval_path) as f:
            results = json.load(f)
        results['_eval_time_s'] = round(eval_time, 1)
        # Re-save with timing
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return results
    return {}


def compare_experiments(name_a: str, name_b: str):
    """Compare two experiment results side by side."""
    dir_a = get_experiment_dir(name_a)
    dir_b = get_experiment_dir(name_b)

    eval_a = dir_a / "eval.json"
    eval_b = dir_b / "eval.json"

    if not eval_a.exists() or not eval_b.exists():
        print("ERROR: Both experiments must have eval.json")
        return

    with open(eval_a) as f:
        a = json.load(f)
    with open(eval_b) as f:
        b = json.load(f)

    print(f"\n{'='*60}")
    print(f"  COMPARISON: {name_a} vs {name_b}")
    print(f"{'='*60}")

    # Exploitability
    exp_a = a.get("leakage", {}).get("exploitability", {}).get("mean", "?")
    exp_b = b.get("leakage", {}).get("exploitability", {}).get("mean", "?")
    print(f"\n  Exploitability: {exp_a} vs {exp_b}")

    # Gauntlet
    g_a = a.get("gauntlet", {})
    g_b = b.get("gauntlet", {})
    if g_a and g_b:
        print(f"\n  {'Bot':<20} {'A':>10} {'B':>10} {'Delta':>10}")
        print(f"  {'-'*50}")
        for bot in g_a:
            if bot.startswith('_'):
                continue
            bb_a = g_a[bot].get('bb_per_100', 0) if isinstance(g_a[bot], dict) else 0
            bb_b = g_b.get(bot, {}).get('bb_per_100', 0) if isinstance(g_b.get(bot, {}), dict) else 0
            delta = bb_b - bb_a
            marker = " *" if abs(delta) > 50 else ""
            print(f"  {bot:<20} {bb_a:>+10.1f} {bb_b:>+10.1f} {delta:>+10.1f}{marker}")

        avg_a = g_a.get("_summary", {}).get("avg_bb_per_100", 0)
        avg_b = g_b.get("_summary", {}).get("avg_bb_per_100", 0)
        print(f"  {'AVERAGE':<20} {avg_a:>+10.1f} {avg_b:>+10.1f} {avg_b - avg_a:>+10.1f}")

    # Behavioral regression
    beh_a = a.get("behavioral", {})
    beh_b = b.get("behavioral", {})
    if beh_a and beh_b:
        print(f"\n  {'Family':<25} {'A entropy':>10} {'B entropy':>10} {'A F/C/R':>15} {'B F/C/R':>15}")
        print(f"  {'-'*75}")
        for fam in beh_a:
            ra = beh_a[fam]
            rb = beh_b.get(fam, {})
            ea = ra.get('entropy', 0)
            eb = rb.get('entropy', 0)
            fcr_a = f"{ra.get('fold_pct', 0):.0f}/{ra.get('call_pct', 0):.0f}/{ra.get('raise_pct', 0):.0f}"
            fcr_b = f"{rb.get('fold_pct', 0):.0f}/{rb.get('call_pct', 0):.0f}/{rb.get('raise_pct', 0):.0f}"
            print(f"  {fam:<25} {ea:>10.2f} {eb:>10.2f} {fcr_a:>15} {fcr_b:>15}")

    print()


def list_experiments():
    """List all ablation experiments."""
    if not ABLATIONS_DIR.exists():
        print("No experiments yet.")
        return

    print(f"\n{'='*60}")
    print(f"  ABLATION EXPERIMENTS")
    print(f"{'='*60}")
    print(f"\n  {'Name':<30} {'Status':<15} {'Iters':<12} {'Expl':<10}")
    print(f"  {'-'*67}")

    for d in sorted(ABLATIONS_DIR.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        has_strategy = (d / "strategy.json").exists()
        has_eval = (d / "eval.json").exists()

        status = "complete" if has_eval else "trained" if has_strategy else "partial"
        iters = "?"
        expl = "?"

        if has_eval:
            try:
                with open(d / "eval.json") as f:
                    ev = json.load(f)
                expl = ev.get("leakage", {}).get("exploitability", {}).get("mean", "?")
            except Exception:
                pass

        config_path = d / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                iters = cfg.get("iterations", "?")
                if isinstance(iters, int):
                    iters = f"{iters/1e6:.0f}M"
            except Exception:
                pass

        print(f"  {name:<30} {status:<15} {iters:<12} {expl:<10}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Ablation Experiment Runner")
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--iterations", type=int, default=50_000_000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--config", type=str, nargs="*", default=[],
                        help="Config overrides as key=value pairs")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Strategy path for eval-only runs")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, just evaluate")
    parser.add_argument("--compare", nargs=2, metavar=("EXP_A", "EXP_B"),
                        help="Compare two experiments")
    parser.add_argument("--list", action="store_true",
                        help="List all experiments")
    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    if args.compare:
        compare_experiments(args.compare[0], args.compare[1])
        return

    if not args.name:
        parser.error("--name is required for running experiments")

    # Parse config overrides
    overrides = {}
    for kv in args.config:
        if '=' not in kv:
            print(f"ERROR: Config override must be key=value, got: {kv}")
            sys.exit(1)
        k, v = kv.split('=', 1)
        try:
            overrides[k] = int(v)
        except ValueError:
            try:
                overrides[k] = float(v)
            except ValueError:
                overrides[k] = v

    config = AblationConfig(
        name=args.name,
        iterations=args.iterations,
        workers=args.workers,
        checkpoint_interval=args.checkpoint_interval,
        config_overrides=overrides,
        strategy_path=args.strategy,
        eval_only=args.eval_only,
    )

    # Determine strategy path
    if args.eval_only:
        if not args.strategy:
            # Check if experiment already has a strategy
            exp_strat = get_experiment_dir(args.name) / "strategy.json"
            if exp_strat.exists():
                strategy_path = str(exp_strat)
            else:
                from server.gto.engine import STRATEGY_FILE
                strategy_path = STRATEGY_FILE
        else:
            strategy_path = args.strategy
    else:
        strategy_path = run_training(config)
        if not strategy_path:
            sys.exit(1)

    # Determine baseline behavioral path
    baseline_beh = str(BASELINE_DIR / "behavioral.json")
    if not Path(baseline_beh).exists():
        baseline_beh = None

    # Run evaluation
    results = run_evaluation(config.name, strategy_path,
                             baseline_behavioral=baseline_beh)

    # Save behavioral results separately for future baselines
    if "behavioral" in results:
        beh_path = get_experiment_dir(config.name) / "behavioral.json"
        with open(beh_path, "w") as f:
            json.dump(results["behavioral"], f, indent=2)

    # Compare with baseline if available
    baseline_eval = BASELINE_DIR / "eval.json"
    if baseline_eval.exists():
        print("\n  Comparing with baseline...")
        compare_experiments("baseline_v7", config.name)

    print(f"\n  Experiment complete: {get_experiment_dir(config.name)}/")


if __name__ == "__main__":
    main()
