#!/usr/bin/env python3
"""
Evaluation harness for GTO poker solver.

Runs a comprehensive suite of tests to validate that strategy improvements
are real and catch false progress. Produces four reports:

    1. Opponent gauntlet      - bb/100 against 7 exploit bots
    2. Off-tree robustness    - EV loss from action translation mismatches
    3. Live bridge loss       - A/B comparison of mapping schemes
    4. Rare-node EV leakage   - regret hotspots and coverage gaps

Usage:
    venv/bin/python run_eval_harness.py                    # Full suite
    venv/bin/python run_eval_harness.py --gauntlet         # Opponent gauntlet only
    venv/bin/python run_eval_harness.py --offtree          # Off-tree tests only
    venv/bin/python run_eval_harness.py --bridge           # Bridge A/B only
    venv/bin/python run_eval_harness.py --leakage          # Rare-node leakage only
    venv/bin/python run_eval_harness.py --hands 2000       # More hands (slower)
    venv/bin/python run_eval_harness.py --quick            # Fast check (200 hands)
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from server.gto.engine import get_trainer
from server.gto.cfr import CFRTrainer
from server.gto.opponent_model import OpponentProfile
from server.gto.abstraction import (
    Action, ACTION_NAMES, decode_bucket, HandType,
    NUM_BUCKETS, NUM_EQUITY_BUCKETS, NUM_HAND_TYPES,
)
from server.gto.exploitability import (
    exploitability_multiseed, exploitability_breakdown, strategy_audit,
    river_bluff_audit,
)


_USE_OPPONENT_MODEL = True  # Module-level flag, overridden by --no-opponent-model


def _play_one_matchup(opp_index: int, num_hands: int, seed: int,
                      big_blind: int) -> dict:
    """Worker function for parallel gauntlet. Loads trainer in-process."""
    from eval_harness.match_engine import GTOAgent, HeadsUpMatch
    from eval_harness.adversaries import get_all_adversaries
    from eval_harness.fast_equity import build_preflop_cache

    trainer = get_trainer()
    build_preflop_cache(simulations=200)

    opp_profile = OpponentProfile() if _USE_OPPONENT_MODEL else None
    gto = GTOAgent(trainer, name="GTO", simulations=80,
                   opponent_profile=opp_profile)
    adversaries = get_all_adversaries(trainer)
    opp = adversaries[opp_index]

    t0 = time.time()
    match = HeadsUpMatch(gto, opp, big_blind=big_blind, seed=seed)
    match_result = match.play(num_hands)
    elapsed = time.time() - t0

    bb100 = match_result.p0_bb_per_100
    showdown_pct = (match_result.showdown_count / num_hands * 100
                    if num_hands > 0 else 0)

    return {
        "opp_name": opp.name,
        "bb_per_100": round(bb100, 2),
        "showdown_pct": round(showdown_pct, 1),
        "num_hands": num_hands,
        "p0_total_bb": round(match_result.p0_total_won, 2),
        "phases": dict(match_result.phase_counts),
        "elapsed": round(elapsed, 1),
        "lookup_hits": gto.lookup_hits,
        "lookup_misses": gto.lookup_misses,
    }


def run_gauntlet(trainer: CFRTrainer, num_hands: int, seed: int,
                 big_blind: int, parallel: bool = True) -> dict:
    """Run GTO agent against all adversaries and measure bb/100."""
    from eval_harness.match_engine import GTOAgent, HeadsUpMatch
    from eval_harness.adversaries import get_all_adversaries

    print("\n" + "=" * 60)
    print("  OPPONENT GAUNTLET")
    print("=" * 60)

    adversaries = get_all_adversaries(trainer)
    num_opps = len(adversaries)
    results = {}

    if parallel and num_opps > 1:
        workers = min(num_opps, os.cpu_count() or 4)
        print(f"  Running {num_opps} matchups across {workers} workers...")
        t0_all = time.time()

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_play_one_matchup, i, num_hands, seed, big_blind): i
                for i in range(num_opps)
            }
            total_hits = 0
            total_misses = 0
            pbar = tqdm(as_completed(futures), total=num_opps,
                        desc="  Gauntlet", unit="matchup", leave=True)
            for future in pbar:
                r = future.result()
                name = r["opp_name"]
                bb100 = r["bb_per_100"]
                total_hits += r["lookup_hits"]
                total_misses += r["lookup_misses"]

                status = "OK" if bb100 > -5 else "WARN" if bb100 > -15 else "FAIL"
                pbar.set_postfix_str(f"vs {name} {bb100:+.1f} bb/100")
                tqdm.write(f"  [{status:>4}] vs {name:<20} {bb100:>+8.1f} bb/100  "
                           f"({r['showdown_pct']:.0f}% SD, {r['elapsed']:.1f}s)")

                results[name] = {
                    "bb_per_100": r["bb_per_100"],
                    "showdown_pct": r["showdown_pct"],
                    "num_hands": r["num_hands"],
                    "p0_total_bb": r["p0_total_bb"],
                    "phases": r["phases"],
                }

        print(f"  Wall time: {time.time() - t0_all:.1f}s")
    else:
        total_hits = 0
        total_misses = 0

        for opp in tqdm(adversaries, desc="  Gauntlet", unit="matchup", leave=True):
            gto = GTOAgent(trainer, name="GTO", simulations=80,
                           opponent_profile=OpponentProfile())
            t0 = time.time()
            match = HeadsUpMatch(gto, opp, big_blind=big_blind, seed=seed)
            match_result = match.play(num_hands)
            elapsed = time.time() - t0

            bb100 = match_result.p0_bb_per_100
            showdown_pct = (match_result.showdown_count / num_hands * 100
                            if num_hands > 0 else 0)

            status = "OK" if bb100 > -5 else "WARN" if bb100 > -15 else "FAIL"
            tqdm.write(f"  [{status:>4}] vs {opp.name:<20} {bb100:>+8.1f} bb/100  "
                       f"({showdown_pct:.0f}% SD, {elapsed:.1f}s)")

            results[opp.name] = {
                "bb_per_100": round(bb100, 2),
                "showdown_pct": round(showdown_pct, 1),
                "num_hands": num_hands,
                "p0_total_bb": round(match_result.p0_total_won, 2),
                "phases": dict(match_result.phase_counts),
            }

        total_hits = gto.lookup_hits
        total_misses = gto.lookup_misses

    # Summary
    avg_bb = sum(r["bb_per_100"] for r in results.values()) / len(results)
    worst = min(results.items(), key=lambda x: x[1]["bb_per_100"])
    best = max(results.items(), key=lambda x: x[1]["bb_per_100"])

    print(f"\n  Average bb/100: {avg_bb:+.1f}")
    print(f"  Best:  vs {best[0]:<20} {best[1]['bb_per_100']:+.1f}")
    print(f"  Worst: vs {worst[0]:<20} {worst[1]['bb_per_100']:+.1f}")

    # GTO lookup diagnostics
    hit_rate = (total_hits / (total_hits + total_misses) * 100
                if (total_hits + total_misses) > 0 else 0)
    print(f"  Strategy hit rate: {hit_rate:.1f}% "
          f"({total_hits}/{total_hits + total_misses})")

    results["_summary"] = {
        "avg_bb_per_100": round(avg_bb, 2),
        "best_matchup": best[0],
        "worst_matchup": worst[0],
        "strategy_hit_rate": round(hit_rate, 1),
    }
    return results


def run_gauntlet_multiseed(trainer: CFRTrainer, num_hands: int,
                           seeds: list[int], big_blind: int,
                           parallel: bool = True) -> dict:
    """Run gauntlet across multiple seeds and compute per-bot statistics.

    Returns dict with per-bot mean/std/CI and per-seed raw values.
    """
    from eval_harness.adversaries import get_all_adversaries

    print("\n" + "=" * 60)
    print("  MULTI-SEED OPPONENT GAUNTLET")
    print("=" * 60)
    print(f"  {num_hands} hands/bot x {len(seeds)} seeds = "
          f"{num_hands * len(seeds)} total hands/bot")

    adversaries = get_all_adversaries(trainer)
    num_opps = len(adversaries)
    opp_names = [a.name for a in adversaries]

    # Build all (opp_index, seed) tasks
    tasks = [(i, s) for s in seeds for i in range(num_opps)]
    total_tasks = len(tasks)

    # Collect raw results: {opp_name: {seed: matchup_result}}
    raw = defaultdict(dict)

    if parallel and total_tasks > 1:
        workers = min(total_tasks, os.cpu_count() or 4)
        print(f"  Running {total_tasks} matchups across {workers} workers...")
        t0_all = time.time()

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_play_one_matchup, opp_i, num_hands, seed, big_blind): (opp_i, seed)
                for opp_i, seed in tasks
            }
            pbar = tqdm(as_completed(futures), total=total_tasks,
                        desc="  Gauntlet", unit="matchup", leave=True)
            for future in pbar:
                r = future.result()
                name = r["opp_name"]
                _, seed_val = futures[future]
                raw[name][seed_val] = r
                pbar.set_postfix_str(f"vs {name} seed={seed_val} {r['bb_per_100']:+.1f}")

        print(f"  Wall time: {time.time() - t0_all:.1f}s")
    else:
        for opp_i, seed_val in tqdm(tasks, desc="  Gauntlet", unit="matchup"):
            r = _play_one_matchup(opp_i, num_hands, seed_val, big_blind)
            raw[r["opp_name"]][seed_val] = r

    # Compute per-bot statistics
    n_seeds = len(seeds)
    per_bot = {}
    for name in opp_names:
        bb_values = [raw[name][s]["bb_per_100"] for s in seeds if s in raw[name]]
        arr = np.array(bb_values)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        ci = 1.96 * std / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
        per_bot[name] = {
            "mean": round(mean, 2),
            "std": round(std, 2),
            "ci_95": round(ci, 2),
            "ci_low": round(mean - ci, 2),
            "ci_high": round(mean + ci, 2),
            "per_seed": {s: raw[name][s]["bb_per_100"] for s in seeds if s in raw[name]},
            "num_hands_per_seed": num_hands,
        }

    # Summary
    avg_mean = np.mean([b["mean"] for b in per_bot.values()])
    worst = min(per_bot.items(), key=lambda x: x[1]["mean"])
    best = max(per_bot.items(), key=lambda x: x[1]["mean"])

    print(f"\n  {'Bot':<20} {'Mean':>8} {'Std':>8} {'95% CI':>16}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*16}")
    for name in opp_names:
        b = per_bot[name]
        status = "OK" if b["mean"] > -5 else "WARN" if b["mean"] > -15 else "FAIL"
        print(f"  [{status:>4}] {name:<15} {b['mean']:>+8.1f} {b['std']:>8.1f} "
              f"[{b['ci_low']:>+7.1f}, {b['ci_high']:>+7.1f}]")

    print(f"\n  Average: {avg_mean:+.1f} bb/100")
    print(f"  Best:  {best[0]:<20} {best[1]['mean']:+.1f}")
    print(f"  Worst: {worst[0]:<20} {worst[1]['mean']:+.1f}")

    return {
        "per_bot": per_bot,
        "summary": {
            "avg_mean": round(float(avg_mean), 2),
            "best_bot": best[0],
            "worst_bot": worst[0],
            "num_hands_per_seed": num_hands,
            "seeds": seeds,
            "num_seeds": n_seeds,
        },
    }


def run_offtree(trainer: CFRTrainer, num_hands: int, seed: int,
                big_blind: int) -> dict:
    """Run off-tree action stress tests."""
    from eval_harness.offtree_stress import (
        analyze_action_mapping, run_offtree_stress_tests,
    )

    print("\n" + "=" * 60)
    print("  OFF-TREE ACTION STRESS TESTS")
    print("=" * 60)

    # Static mapping analysis
    mapping = analyze_action_mapping(big_blind)
    print(f"\n{mapping['summary']}")

    # Dynamic stress tests
    print(f"\n  Running {num_hands} hands per sizing variant...")
    t0 = time.time()
    pbar = tqdm(total=11, desc="  Off-tree", unit="sizing", leave=True)

    def _offtree_cb(label):
        pbar.set_postfix_str(label)
        pbar.update(1)

    stress = run_offtree_stress_tests(trainer, num_hands, seed, big_blind,
                                      progress_cb=_offtree_cb)
    pbar.close()
    elapsed = time.time() - t0

    print(f"\n{stress['summary']}")
    print(f"  ({elapsed:.1f}s total)")

    return {
        "mapping": {
            "preflop": [
                {"label": e.concrete_label, "mapped": e.mapped_abstract}
                for e in mapping["preflop"]
            ],
            "postflop": [
                {"label": e.concrete_label, "mapped": e.mapped_abstract}
                for e in mapping["postflop"]
            ],
            "collisions": {k: v for k, v in mapping["collision_groups"].items()
                           if len(v) > 1},
        },
        "stress_results": [
            {
                "sizing": r.sizing_label,
                "bb_per_100": round(r.gto_bb_per_100, 2),
                "mapped_to": r.mapped_to,
            }
            for r in stress["results"]
        ],
    }


def run_bridge_ab(trainer: CFRTrainer, num_hands: int, seed: int,
                  big_blind: int) -> dict:
    """Run A/B action translation comparison."""
    from eval_harness.translation_ab import run_translation_ab

    print("\n" + "=" * 60)
    print("  LIVE BRIDGE A/B TEST")
    print("=" * 60)

    t0 = time.time()
    pbar = tqdm(total=12, desc="  Bridge A/B", unit="match", leave=True)

    def _bridge_cb(map_name, opp_name):
        pbar.set_postfix_str(f"{map_name} vs {opp_name}")
        pbar.update(1)

    ab = run_translation_ab(trainer, num_hands, seed, big_blind,
                            progress_cb=_bridge_cb)
    pbar.close()
    elapsed = time.time() - t0

    print(f"\n{ab['summary']}")
    print(f"  ({elapsed:.1f}s total)")

    return {
        "results": [
            {
                "mapping": r.mapping_name,
                "opponent": r.opponent_name,
                "bb_per_100": round(r.bb_per_100, 2),
            }
            for r in ab["results"]
        ],
    }


def run_leakage(trainer: CFRTrainer) -> dict:
    """Analyze rare-node EV leakage and regret hotspots."""
    print("\n" + "=" * 60)
    print("  RARE-NODE EV LEAKAGE REPORT")
    print("=" * 60)

    # Exploitability baseline
    print("\n  Computing exploitability (3-seed)...")
    t0 = time.time()
    exp = exploitability_multiseed(trainer, samples=500,
                                    seeds=[42, 123, 456])
    elapsed = time.time() - t0
    print(f"  Exploitability: {exp['mean']:.2f} +/- {exp['std']:.2f} ({elapsed:.1f}s)")

    # Per-phase breakdown
    print(f"\n  Per-phase exploitability:")
    for phase, stats in exp.get("per_phase", {}).items():
        print(f"    {phase:<12} {stats['mean']:.2f} +/- {stats['std']:.2f}")

    # Regret hotspots per phase
    print(f"\n  TOP REGRET HOTSPOTS:")
    hotspots = {}
    for phase in tqdm(['preflop', 'flop', 'turn', 'river'],
                      desc="  Hotspots", unit="phase", leave=True):
        breakdown = exploitability_breakdown(trainer, phase=phase, top_n=5)
        if breakdown:
            hotspots[phase] = breakdown
            tqdm.write(f"\n  {phase}:")
            for item in breakdown[:3]:
                key = item['key']
                regret = item['regret_magnitude']
                entropy = item['entropy']
                strat = item['strategy']
                top_actions = sorted(strat.items(), key=lambda x: -float(x[1]))[:3]
                strat_str = ", ".join(f"{k}:{float(v):.0%}" for k, v in top_actions)
                tqdm.write(f"    {key:<40} regret={regret:>8.1f}  H={entropy:.2f}  [{strat_str}]")

    # Strategy audit
    print(f"\n  STRATEGY AUDIT:")
    audit = strategy_audit(trainer)
    counts = audit.get("counts", {})
    for anomaly_type, count in counts.items():
        status = "OK" if count == 0 else f"WARN ({count})"
        print(f"    {anomaly_type:<25} {status}")

    if audit.get("leak_summary"):
        print(f"\n  TOP LEAKS:")
        for leak in audit["leak_summary"][:5]:
            print(f"    [{leak['severity']:>6}] {leak['type']}: {leak['description']}")

    # River bluff ratio diagnostic
    print(f"\n  RIVER BLUFF ANALYSIS:")
    bluff = river_bluff_audit(trainer)
    bluff_ratio = bluff["overall_bluff_ratio"]
    diag = bluff["diagnosis"]
    status = "OK" if diag == "HEALTHY" else "WARN"
    print(f"    [{status}] Overall bluff ratio: {bluff_ratio:.1%} ({diag})")
    print(f"    Value weight: {bluff['total_value_weight']:.0f}  "
          f"Bluff weight: {bluff['total_bluff_weight']:.0f}")
    for sizing, stats in bluff["by_sizing"].items():
        if stats["value_weight"] + stats["bluff_weight"] > 0:
            print(f"    {sizing:<16} bluff={stats['bluff_ratio']:.0%}")
    for pos, stats in bluff["by_position"].items():
        if stats["value_weight"] + stats["bluff_weight"] > 0:
            print(f"    {pos:<16} bluff={stats['bluff_ratio']:.0%}")
    if bluff["worst_bluffers"]:
        print(f"    Top over-bluffers:")
        for item in bluff["worst_bluffers"][:5]:
            print(f"      {item['key']}: bet={item['bet_prob']:.0%} "
                  f"EQ{item['eq_bucket']} {item['hand_type']}")

    # Coverage analysis
    total_nodes = len(trainer.nodes)
    low_visit_nodes = sum(
        1 for node in trainer.nodes.values()
        if node.strategy_sum.sum() < 100
    )
    coverage = (total_nodes - low_visit_nodes) / total_nodes * 100 if total_nodes > 0 else 0
    print(f"\n  COVERAGE:")
    print(f"    Total nodes:     {total_nodes:,}")
    print(f"    Well-visited:    {total_nodes - low_visit_nodes:,} ({coverage:.1f}%)")
    print(f"    Low-visit (<100): {low_visit_nodes:,} ({100-coverage:.1f}%)")

    # Per-phase node distribution
    phase_nodes = defaultdict(int)
    for key in trainer.nodes:
        phase = key.split(":")[0]
        phase_nodes[phase] += 1
    print(f"\n    Node distribution:")
    for phase in ['preflop', 'flop', 'turn', 'river']:
        count = phase_nodes.get(phase, 0)
        print(f"      {phase:<12} {count:>8,} ({count/total_nodes*100:.1f}%)")

    return {
        "exploitability": exp,
        "hotspots": {
            phase: [
                {"key": h["key"], "regret": round(h["regret_magnitude"], 2),
                 "entropy": round(h["entropy"], 3)}
                for h in items[:5]
            ]
            for phase, items in hotspots.items()
        },
        "audit_counts": counts,
        "river_bluff": {
            "overall_bluff_ratio": bluff["overall_bluff_ratio"],
            "diagnosis": bluff["diagnosis"],
            "by_sizing": bluff["by_sizing"],
            "by_position": bluff["by_position"],
            "worst_bluffers_count": len(bluff["worst_bluffers"]),
        },
        "coverage": {
            "total_nodes": total_nodes,
            "well_visited": total_nodes - low_visit_nodes,
            "coverage_pct": round(coverage, 1),
        },
        "phase_distribution": dict(phase_nodes),
    }


def main():
    parser = argparse.ArgumentParser(description="GTO Evaluation Harness")
    parser.add_argument("--gauntlet", action="store_true", help="Run opponent gauntlet only")
    parser.add_argument("--offtree", action="store_true", help="Run off-tree tests only")
    parser.add_argument("--bridge", action="store_true", help="Run bridge A/B only")
    parser.add_argument("--leakage", action="store_true", help="Run rare-node leakage only")
    parser.add_argument("--behavioral", action="store_true", help="Run behavioral regression suite")
    parser.add_argument("--behavioral-baseline", type=str, default=None,
                        help="Path to baseline behavioral_regression.json for delta comparison")
    parser.add_argument("--hands", type=int, default=2000, help="Hands per matchup (default 2000)")
    parser.add_argument("--quick", action="store_true", help="Quick check (100 hands)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-seed gauntlet (e.g. 42,123,456)")
    parser.add_argument("--bb", type=int, default=20, help="Big blind size")
    parser.add_argument("--save", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel gauntlet")
    parser.add_argument("--no-opponent-model", action="store_true",
                        help="Disable opponent model adjustments (pure GTO eval)")
    parser.add_argument("--strategy", type=str, default=None, help="Path to strategy JSON file (default: server/gto/strategy.json)")
    args = parser.parse_args()

    if args.quick:
        args.hands = 100

    # Apply opponent model flag
    global _USE_OPPONENT_MODEL
    if args.no_opponent_model:
        _USE_OPPONENT_MODEL = False

    run_all = not (args.gauntlet or args.offtree or args.bridge or args.leakage or args.behavioral)

    # Load strategy
    print("Loading strategy...")
    t0 = time.time()
    if args.strategy:
        import server.gto.engine as _engine
        _engine.STRATEGY_FILE = args.strategy
    trainer = get_trainer()
    print(f"Strategy loaded: {trainer.iterations:,} iterations, "
          f"{len(trainer.nodes):,} nodes ({time.time()-t0:.1f}s)")

    # Warm up preflop bucket cache (169 hands, ~30s one-time cost)
    if run_all or args.gauntlet or args.offtree or args.bridge:
        from eval_harness.fast_equity import build_preflop_cache
        print("Building preflop bucket cache...")
        t0 = time.time()
        build_preflop_cache(simulations=200)
        print(f"Preflop cache ready ({time.time()-t0:.1f}s)")

    all_results = {}

    # Build stage list
    stages = []
    if run_all or args.gauntlet:
        stages.append("gauntlet")
    if run_all or args.offtree:
        stages.append("offtree")
    if run_all or args.bridge:
        stages.append("bridge")
    if run_all or args.leakage:
        stages.append("leakage")
    if run_all or args.behavioral:
        stages.append("behavioral")

    stage_bar = tqdm(stages, desc="Overall", unit="stage", leave=True,
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} stages [{elapsed}<{remaining}]")

    for stage in stage_bar:
        stage_bar.set_postfix_str(stage)
        if stage == "gauntlet":
            if args.seeds:
                seed_list = [int(s.strip()) for s in args.seeds.split(",")]
                all_results["gauntlet"] = run_gauntlet_multiseed(
                    trainer, args.hands, seed_list, args.bb,
                    parallel=not args.no_parallel)
            else:
                all_results["gauntlet"] = run_gauntlet(
                    trainer, args.hands, args.seed, args.bb,
                    parallel=not args.no_parallel)
        elif stage == "offtree":
            all_results["offtree"] = run_offtree(
                trainer, args.hands, args.seed, args.bb)
        elif stage == "bridge":
            all_results["bridge"] = run_bridge_ab(
                trainer, args.hands, args.seed, args.bb)
        elif stage == "leakage":
            all_results["leakage"] = run_leakage(trainer)
        elif stage == "behavioral":
            from eval_harness.behavioral_regression import (
                run_behavioral_regression, print_behavioral_report)
            beh_results = run_behavioral_regression(
                trainer, baseline_path=args.behavioral_baseline)
            print_behavioral_report(beh_results)
            all_results["behavioral"] = beh_results

    stage_bar.close()

    # Final summary
    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)

    if "gauntlet" in all_results:
        gauntlet = all_results["gauntlet"]
        if "summary" in gauntlet and "avg_mean" in gauntlet.get("summary", {}):
            # Multi-seed format
            avg = gauntlet["summary"]["avg_mean"]
            n_seeds = gauntlet["summary"]["num_seeds"]
            status = "PASS" if avg > -5 else "WARN" if avg > -15 else "FAIL"
            print(f"  [{status}] Gauntlet avg: {avg:+.1f} bb/100 ({n_seeds} seeds)")
        else:
            # Single-seed format
            summary = gauntlet.get("_summary", {})
            avg = summary.get("avg_bb_per_100", 0)
            status = "PASS" if avg > -5 else "WARN" if avg > -15 else "FAIL"
            print(f"  [{status}] Gauntlet avg: {avg:+.1f} bb/100")

    if "offtree" in all_results:
        stress = all_results["offtree"].get("stress_results", [])
        if stress:
            worst = min(stress, key=lambda x: x["bb_per_100"])
            status = "PASS" if worst["bb_per_100"] > -10 else "WARN"
            print(f"  [{status}] Off-tree worst: {worst['sizing']} at "
                  f"{worst['bb_per_100']:+.1f} bb/100")

    if "bridge" in all_results:
        br = all_results["bridge"].get("results", [])
        if br:
            by_map = defaultdict(list)
            for r in br:
                by_map[r["mapping"]].append(r["bb_per_100"])
            best_map = max(by_map.items(), key=lambda x: sum(x[1]) / len(x[1]))
            avg = sum(best_map[1]) / len(best_map[1])
            print(f"  [INFO] Best bridge mapping: {best_map[0]} ({avg:+.1f} avg)")

    if "leakage" in all_results:
        exp = all_results["leakage"].get("exploitability", {})
        cov = all_results["leakage"].get("coverage", {})
        print(f"  [INFO] Exploitability: {exp.get('mean', '?')}")
        print(f"  [INFO] Node coverage: {cov.get('coverage_pct', '?')}%")

    if "behavioral" in all_results:
        beh = all_results["behavioral"]
        families_with_nodes = sum(1 for r in beh.values() if r.get('num_nodes', 0) > 0)
        print(f"  [INFO] Behavioral: {families_with_nodes}/6 families audited")

    # Save results
    if args.save:
        save_path = args.save
    else:
        os.makedirs("eval_results", exist_ok=True)
        save_path = f"eval_results/eval_{int(time.time())}.json"

    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {save_path}")


if __name__ == "__main__":
    main()
