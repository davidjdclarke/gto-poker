#!/usr/bin/env python3
"""
V10 Phase 0 — Measurement hardening and baseline validation.

Subcommands:
    validate   — Multi-seed gauntlet + dual exploitability settings
    diagnose   — B0 vs v7 causal EV decomposition + bridge pain
    cliff      — Checkpoint coverage analysis (20M→30M transition)
    all        — Run everything

Usage:
    venv/bin/python run_phase0_validation.py validate
    venv/bin/python run_phase0_validation.py diagnose
    venv/bin/python run_phase0_validation.py cliff
    venv/bin/python run_phase0_validation.py all
    venv/bin/python run_phase0_validation.py validate --quick   # 200 hands, 2 seeds
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from server.gto.cfr import CFRTrainer
from server.gto.exploitability import exploitability_multiseed

DEFAULT_B0_STRATEGY = "experiments/best/v9_B0_100M_allbots_positive.json"
DEFAULT_V7_STRATEGY = "experiments/best/v7_67M_reference.json"
DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_HANDS = 5000
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "docs/results"


# ---------------------------------------------------------------------------
# validate — Multi-seed gauntlet + dual exploitability
# ---------------------------------------------------------------------------
def cmd_validate(args):
    """B0 validation suite: multi-seed gauntlet + exploitability."""
    import run_eval_harness as _reh
    from run_eval_harness import run_gauntlet_multiseed

    print("=" * 60)
    print("  V10 PHASE 0: BASELINE VALIDATION")
    print("=" * 60)

    # Disable opponent model for pure GTO evaluation
    _reh._USE_OPPONENT_MODEL = False

    # Load strategy
    trainer = _load_strategy(args.strategy)

    # 1. Multi-seed gauntlet
    print(f"\n--- Multi-seed gauntlet ({args.hands} hands/bot x {len(args.seeds)} seeds) ---")
    gauntlet = run_gauntlet_multiseed(
        trainer, args.hands, args.seeds, args.bb, parallel=True)

    # 2. Standard exploitability (3-seed, 500 samples)
    print("\n--- Exploitability: standard (3-seed, 500 samples) ---")
    t0 = time.time()
    exp_standard = exploitability_multiseed(
        trainer, samples=500, seeds=[42, 123, 456])
    print(f"  Mean: {exp_standard['mean']:.4f} +/- {exp_standard['std']:.4f} "
          f"({time.time() - t0:.1f}s)")
    _print_per_phase(exp_standard)

    # 3. High-fidelity exploitability (5-seed, 1000 samples)
    print("\n--- Exploitability: high-fidelity (5-seed, 1000 samples) ---")
    t0 = time.time()
    exp_hifi = exploitability_multiseed(
        trainer, samples=1000, seeds=[42, 123, 456, 789, 999])
    print(f"  Mean: {exp_hifi['mean']:.4f} +/- {exp_hifi['std']:.4f} "
          f"({time.time() - t0:.1f}s)")
    _print_per_phase(exp_hifi)

    # Save results
    results = {
        "gauntlet": gauntlet,
        "exploitability_standard": exp_standard,
        "exploitability_hifi": exp_hifi,
        "config": {
            "strategy": args.strategy,
            "hands_per_seed": args.hands,
            "seeds": args.seeds,
            "big_blind": args.bb,
        },
    }
    save_path = os.path.join(RESULTS_DIR, "v10_phase0_validate.json")
    _save_json(results, save_path)
    return results


# ---------------------------------------------------------------------------
# diagnose — B0 vs v7 causal diagnostics
# ---------------------------------------------------------------------------
def cmd_diagnose(args):
    """B0 vs v7 causal EV decomposition + bridge pain."""
    from eval_harness.match_engine import GTOAgent, HeadsUpMatch
    from eval_harness.adversaries import get_all_adversaries
    from eval_harness.fast_equity import build_preflop_cache
    from eval_harness.ev_decomposition import (
        decompose_by_street, decompose_by_action_family,
        decompose_by_bucket, callstation_dashboard, format_dashboard,
    )
    from eval_harness.bridge_pain import analyze_bridge_pain, format_pain_map

    print("=" * 60)
    print("  V10 PHASE 0: CAUSAL DIAGNOSTICS (B0 vs v7)")
    print("=" * 60)

    # Load both strategies
    print("\nLoading B0 strategy...")
    trainer_b0 = _load_strategy(args.strategy)
    print("Loading v7 strategy...")
    trainer_v7 = _load_strategy(args.v7_strategy)

    # Build preflop cache
    build_preflop_cache(simulations=200)

    # Select bots
    all_adversaries = get_all_adversaries(trainer_b0)
    bot_names = {a.name for a in all_adversaries}
    if args.bots:
        selected = [b.strip() for b in args.bots.split(",")]
        for b in selected:
            if b not in bot_names:
                print(f"  WARNING: unknown bot '{b}', available: {sorted(bot_names)}")
        target_bots = [b for b in selected if b in bot_names]
    else:
        target_bots = ["CallStationBot", "NitBot", "WeirdSizingBot"]

    print(f"\n  Target bots: {target_bots}")
    print(f"  Hands per matchup: {args.hands}")
    print(f"  Seed: {args.seeds[0]}")

    results = {}
    seed = args.seeds[0]

    for bot_name in target_bots:
        print(f"\n{'='*50}")
        print(f"  {bot_name}")
        print(f"{'='*50}")

        matchup_result = {}
        for label, trainer in [("B0", trainer_b0), ("v7", trainer_v7)]:
            print(f"\n  --- {label} vs {bot_name} ---")

            # Get the right adversary instance for this trainer
            adversaries = get_all_adversaries(trainer)
            opp = next(a for a in adversaries if a.name == bot_name)

            gto = GTOAgent(trainer, name=f"GTO-{label}", simulations=80)  # No OpponentProfile for pure GTO eval
            match = HeadsUpMatch(gto, opp, big_blind=args.bb, seed=seed,
                                 detailed_tracking=True)
            match_result = match.play(args.hands)

            bb100 = match_result.p0_bb_per_100
            print(f"    Overall: {bb100:+.1f} bb/100")

            # EV decomposition
            street_ev = decompose_by_street(match_result.hands, args.bb)
            action_ev = {}
            for phase in ["preflop", "flop", "turn", "river"]:
                action_ev[phase] = decompose_by_action_family(
                    match_result.hands, phase=phase, big_blind=args.bb)
            bucket_ev = decompose_by_bucket(match_result.hands, big_blind=args.bb)

            print(f"    Street EV: ", end="")
            for phase in ["preflop", "flop", "turn", "river"]:
                v = street_ev.get(phase, 0)
                print(f"{phase}={v:+.1f}  ", end="")
            print()

            # CallStation dashboard
            cs_dashboard = None
            if bot_name == "CallStationBot":
                cs_dashboard = callstation_dashboard(match_result.hands, args.bb)
                print(f"\n{format_dashboard(cs_dashboard)}")

            # Bridge pain
            bridge_analysis = None
            if gto.bridge_log:
                bridge_analysis = analyze_bridge_pain(
                    gto.bridge_log, match_result.hands, args.bb)
                print(f"\n{format_pain_map(bridge_analysis)}")

            matchup_result[label] = {
                "bb_per_100": round(bb100, 2),
                "street_ev": {k: round(v, 2) for k, v in street_ev.items()},
                "action_ev": _serialize_action_ev(action_ev),
                "bucket_ev": _serialize_bucket_ev(bucket_ev),
                "callstation_dashboard": cs_dashboard,
                "bridge_pain": _serialize_bridge(bridge_analysis),
            }

        # Compute deltas
        delta = _compute_deltas(matchup_result["B0"], matchup_result["v7"])
        matchup_result["delta"] = delta
        _print_delta_table(bot_name, matchup_result)

        results[bot_name] = matchup_result

    # Save results
    save_path = os.path.join(RESULTS_DIR, "v10_phase0_diagnose.json")
    _save_json(results, save_path)
    return results


# ---------------------------------------------------------------------------
# cliff — Checkpoint coverage analysis
# ---------------------------------------------------------------------------
def cmd_cliff(args):
    """Analyze checkpoint convergence: node coverage, averaging weights."""
    print("=" * 60)
    print("  V10 PHASE 0: CHECKPOINT CLIFF ANALYSIS")
    print("=" * 60)

    # Find checkpoint files
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        print(f"  ERROR: checkpoint directory '{ckpt_dir}' not found")
        return {}

    ckpt_files = sorted(ckpt_dir.glob("ckpt_*.json"))
    if not ckpt_files:
        print("  No checkpoint files found")
        return {}

    print(f"  Found {len(ckpt_files)} checkpoint files")

    # Load existing checkpoint log for exploitability data
    log_path = ckpt_dir / "checkpoint_log.json"
    existing_log = {}
    if log_path.exists():
        with open(log_path) as f:
            for entry in json.load(f):
                existing_log[entry["iterations"]] = entry

    results = []
    for ckpt_file in ckpt_files:
        # Parse iteration count from filename
        fname = ckpt_file.stem  # e.g. "ckpt_10000000"
        try:
            iters = int(fname.split("_")[1])
        except (IndexError, ValueError):
            continue

        print(f"\n  Loading {ckpt_file.name}...", end="", flush=True)
        t0 = time.time()
        trainer = CFRTrainer()
        if not trainer.load(str(ckpt_file)):
            print(" FAILED")
            continue

        elapsed = time.time() - t0
        total_nodes = len(trainer.nodes)
        print(f" {total_nodes:,} nodes ({elapsed:.1f}s)")

        # Classify: B0 (~1.05M nodes) vs 16-action (~2M nodes)
        is_b0 = total_nodes < 1_500_000

        # Coverage metrics
        well_visited = 0
        total_weight = 0.0
        phase_stats = defaultdict(lambda: {"total": 0, "well_visited": 0, "weight": 0.0})

        for key, node in trainer.nodes.items():
            weight = float(node.strategy_sum.sum())
            phase = key.split(":")[0]
            phase_stats[phase]["total"] += 1
            total_weight += weight

            if weight >= 100:
                well_visited += 1
                phase_stats[phase]["well_visited"] += 1
            phase_stats[phase]["weight"] += weight

        coverage_pct = (well_visited / total_nodes * 100) if total_nodes > 0 else 0
        avg_weight = total_weight / total_nodes if total_nodes > 0 else 0

        # Pull exploitability from existing log if available
        exploit = existing_log.get(iters, {}).get("exploitability")
        gauntlet_avg = existing_log.get(iters, {}).get("gauntlet", {}).get("avg_bb_per_100")

        entry = {
            "iterations": iters,
            "total_nodes": total_nodes,
            "is_b0": is_b0,
            "well_visited": well_visited,
            "coverage_pct": round(coverage_pct, 1),
            "avg_strategy_weight": round(avg_weight, 1),
            "exploitability": exploit,
            "gauntlet_avg": gauntlet_avg,
            "per_phase": {
                phase: {
                    "total": s["total"],
                    "well_visited": s["well_visited"],
                    "coverage_pct": round(s["well_visited"] / s["total"] * 100, 1) if s["total"] > 0 else 0,
                    "avg_weight": round(s["weight"] / s["total"], 1) if s["total"] > 0 else 0,
                }
                for phase, s in sorted(phase_stats.items())
            },
        }
        results.append(entry)

    # Print summary table
    results.sort(key=lambda x: x["iterations"])

    print(f"\n{'='*90}")
    print(f"  {'Iters':>10}  {'Type':>5}  {'Nodes':>10}  {'Coverage':>8}  "
          f"{'AvgWgt':>10}  {'Exploit':>8}  {'Gauntlet':>10}")
    print(f"  {'-'*10}  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")

    for r in results:
        typ = "B0" if r["is_b0"] else "16act"
        exploit_str = f"{r['exploitability']:.2f}" if r["exploitability"] is not None else "—"
        gauntlet_str = f"{r['gauntlet_avg']:+.1f}" if r["gauntlet_avg"] is not None else "—"
        print(f"  {r['iterations']:>10,}  {typ:>5}  {r['total_nodes']:>10,}  "
              f"{r['coverage_pct']:>7.1f}%  {r['avg_strategy_weight']:>10.1f}  "
              f"{exploit_str:>8}  {gauntlet_str:>10}")

    # Highlight B0 cliff transition
    b0_entries = [r for r in results if r["is_b0"]]
    if len(b0_entries) >= 2:
        print(f"\n  B0 checkpoint progression ({len(b0_entries)} checkpoints):")
        for i, r in enumerate(b0_entries):
            delta_cov = ""
            if i > 0:
                prev = b0_entries[i - 1]
                dc = r["coverage_pct"] - prev["coverage_pct"]
                dw = r["avg_strategy_weight"] - prev["avg_strategy_weight"]
                delta_cov = f"  Δcov={dc:+.1f}%  Δwgt={dw:+.1f}"
            print(f"    {r['iterations']:>10,}: cov={r['coverage_pct']:.1f}%  "
                  f"wgt={r['avg_strategy_weight']:.1f}{delta_cov}")

    # Save results
    save_path = os.path.join(RESULTS_DIR, "v10_phase0_cliff.json")
    _save_json(results, save_path)
    return results


# ---------------------------------------------------------------------------
# all — Run everything
# ---------------------------------------------------------------------------
def cmd_all(args):
    """Run all Phase 0 subcommands."""
    print("=" * 60)
    print("  V10 PHASE 0: FULL VALIDATION SUITE")
    print("=" * 60)

    t0_all = time.time()
    all_results = {}

    print("\n\n[1/3] VALIDATE")
    all_results["validate"] = cmd_validate(args)

    print("\n\n[2/3] DIAGNOSE")
    all_results["diagnose"] = cmd_diagnose(args)

    print("\n\n[3/3] CLIFF")
    all_results["cliff"] = cmd_cliff(args)

    elapsed = time.time() - t0_all
    print(f"\n\n{'='*60}")
    print(f"  PHASE 0 COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*60}")

    save_path = os.path.join(RESULTS_DIR, "v10_phase0_all.json")
    _save_json(all_results, save_path)
    return all_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_strategy(path: str) -> CFRTrainer:
    """Load a strategy file into a CFRTrainer."""
    t0 = time.time()

    # For parallel workers (gauntlet), set global strategy file
    import server.gto.engine as _engine
    _engine.STRATEGY_FILE = path
    _engine._trainer = None  # force reload

    trainer = CFRTrainer()
    if not trainer.load(path):
        print(f"  ERROR: could not load strategy from {path}")
        sys.exit(1)

    # Auto-detect and set action grid from loaded strategy
    from server.gto.abstraction import detect_action_grid_from_strategy, set_action_grid
    grid = detect_action_grid_from_strategy(trainer)
    set_action_grid(grid)
    print(f"  Loaded: {trainer.iterations:,} iters, "
          f"{len(trainer.nodes):,} nodes, {grid}-action grid ({time.time()-t0:.1f}s)")
    return trainer


def _print_per_phase(exp: dict):
    """Print per-phase exploitability breakdown."""
    for phase, stats in exp.get("per_phase", {}).items():
        print(f"    {phase:<12} {stats['mean']:.4f} +/- {stats['std']:.4f}")


def _serialize_action_ev(action_ev: dict) -> dict:
    """Make action_ev JSON-serializable."""
    out = {}
    for phase, families in action_ev.items():
        out[phase] = {}
        for fam, stats in families.items():
            out[phase][fam] = {
                "total_ev_bb": round(stats["total_ev_bb"], 2),
                "count": stats["count"],
                "avg_ev_bb": round(stats["avg_ev_bb"], 2),
            }
    return out


def _serialize_bucket_ev(bucket_ev: dict) -> dict:
    """Make bucket_ev JSON-serializable."""
    return {
        str(k): {
            "total_ev_bb": round(v["total_ev_bb"], 2),
            "count": v["count"],
            "avg_ev_bb": round(v["avg_ev_bb"], 2),
        }
        for k, v in bucket_ev.items()
    }


def _serialize_bridge(bridge_analysis: dict | None) -> dict | None:
    """Make bridge pain analysis JSON-serializable."""
    if bridge_analysis is None:
        return None
    # Already mostly serializable, just ensure no numpy types
    return json.loads(json.dumps(bridge_analysis, default=str))


def _compute_deltas(b0: dict, v7: dict) -> dict:
    """Compute B0 - v7 deltas for key metrics."""
    delta = {
        "bb_per_100": round(b0["bb_per_100"] - v7["bb_per_100"], 2),
        "street_ev": {},
        "bucket_ev": {},
    }

    # Street EV deltas
    all_phases = set(b0.get("street_ev", {}).keys()) | set(v7.get("street_ev", {}).keys())
    for phase in all_phases:
        b0_val = b0.get("street_ev", {}).get(phase, 0)
        v7_val = v7.get("street_ev", {}).get(phase, 0)
        delta["street_ev"][phase] = round(b0_val - v7_val, 2)

    # Bucket EV deltas
    for eq in range(8):
        eq_str = str(eq)
        b0_val = b0.get("bucket_ev", {}).get(eq_str, {}).get("avg_ev_bb", 0)
        v7_val = v7.get("bucket_ev", {}).get(eq_str, {}).get("avg_ev_bb", 0)
        delta["bucket_ev"][eq_str] = round(b0_val - v7_val, 2)

    return delta


def _print_delta_table(bot_name: str, matchup: dict):
    """Print a formatted B0 vs v7 delta table."""
    b0 = matchup["B0"]
    v7 = matchup["v7"]
    delta = matchup["delta"]

    print(f"\n  === {bot_name}: B0 vs v7 Delta ===")
    print(f"  {'':20s} {'B0':>10} {'v7':>10} {'Delta':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Overall (bb/100)':20s} {b0['bb_per_100']:>+10.1f} "
          f"{v7['bb_per_100']:>+10.1f} {delta['bb_per_100']:>+10.1f}")

    print(f"\n  Street EV (bb/100):")
    for phase in ["preflop", "flop", "turn", "river"]:
        b0_v = b0.get("street_ev", {}).get(phase, 0)
        v7_v = v7.get("street_ev", {}).get(phase, 0)
        d = delta.get("street_ev", {}).get(phase, 0)
        print(f"    {phase:12s} {b0_v:>+10.1f} {v7_v:>+10.1f} {d:>+10.1f}")

    print(f"\n  Bucket EV (avg bb, last street):")
    for eq in range(8):
        eq_str = str(eq)
        b0_v = b0.get("bucket_ev", {}).get(eq_str, {}).get("avg_ev_bb", 0)
        v7_v = v7.get("bucket_ev", {}).get(eq_str, {}).get("avg_ev_bb", 0)
        d = delta.get("bucket_ev", {}).get(eq_str, 0)
        print(f"    EQ{eq}         {b0_v:>+10.2f} {v7_v:>+10.2f} {d:>+10.2f}")


def _save_json(data, path):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n  Results saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="V10 Phase 0 — Measurement hardening and baseline validation")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # Common args added to each subparser
    def add_common(sub):
        sub.add_argument("--hands", type=int, default=DEFAULT_HANDS,
                         help=f"Hands per matchup (default {DEFAULT_HANDS})")
        sub.add_argument("--seeds", type=str, default=None,
                         help="Comma-separated seeds (default 42,123,456)")
        sub.add_argument("--bb", type=int, default=20, help="Big blind size")
        sub.add_argument("--strategy", type=str, default=DEFAULT_B0_STRATEGY,
                         help=f"B0 strategy path (default {DEFAULT_B0_STRATEGY})")
        sub.add_argument("--quick", action="store_true",
                         help="Quick mode: 200 hands, 2 seeds")

    # validate
    p_validate = subparsers.add_parser("validate", help="Multi-seed gauntlet + exploitability")
    add_common(p_validate)

    # diagnose
    p_diagnose = subparsers.add_parser("diagnose", help="B0 vs v7 causal diagnostics")
    add_common(p_diagnose)
    p_diagnose.add_argument("--v7-strategy", type=str, default=DEFAULT_V7_STRATEGY,
                            help=f"v7 strategy path (default {DEFAULT_V7_STRATEGY})")
    p_diagnose.add_argument("--bots", type=str, default=None,
                            help="Comma-separated bot names (default: CallStationBot,NitBot,WeirdSizingBot)")

    # cliff
    p_cliff = subparsers.add_parser("cliff", help="Checkpoint coverage analysis")
    add_common(p_cliff)
    p_cliff.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
                         help=f"Checkpoint directory (default {CHECKPOINT_DIR})")

    # all
    p_all = subparsers.add_parser("all", help="Run all Phase 0 subcommands")
    add_common(p_all)
    p_all.add_argument("--v7-strategy", type=str, default=DEFAULT_V7_STRATEGY,
                       help=f"v7 strategy path (default {DEFAULT_V7_STRATEGY})")
    p_all.add_argument("--bots", type=str, default=None,
                       help="Comma-separated bot names for diagnose")
    p_all.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
                       help=f"Checkpoint directory (default {CHECKPOINT_DIR})")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Process common args
    if args.quick:
        args.hands = 200
        args.seeds = [42, 123]
    elif args.seeds:
        args.seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        args.seeds = DEFAULT_SEEDS

    # Ensure diagnose/all have v7_strategy and bots attrs
    if not hasattr(args, "v7_strategy"):
        args.v7_strategy = DEFAULT_V7_STRATEGY
    if not hasattr(args, "bots"):
        args.bots = None
    if not hasattr(args, "checkpoint_dir"):
        args.checkpoint_dir = CHECKPOINT_DIR

    # Dispatch
    if args.command == "validate":
        cmd_validate(args)
    elif args.command == "diagnose":
        cmd_diagnose(args)
    elif args.command == "cliff":
        cmd_cliff(args)
    elif args.command == "all":
        cmd_all(args)


if __name__ == "__main__":
    main()
