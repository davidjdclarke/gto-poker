#!/usr/bin/env python3
"""
Pre-train the GTO strategy offline with live progress visualization.

Usage:
    python train_gto.py [iterations] [sampling]

Examples:
    python train_gto.py                        # 10000 iters, external sampling
    python train_gto.py 1000000                # 1M iters (Cython: ~4 min)
    python train_gto.py 5000000                # 5M iters (Cython: ~20 min)
    python train_gto.py 50000 vanilla          # 50000 iters, vanilla CFR
    python train_gto.py 20000000 --workers 6   # 20M iters, 6 parallel workers
"""
import sys
import time
import json
import numpy as np
import random as rng
from pathlib import Path
from server.gto.cfr import CFRTrainer, HAS_CYTHON, STRATEGY_VERSION
from server.gto.engine import STRATEGY_FILE
from server.gto.abstraction import (
    Action, ACTION_NAMES, NUM_EQUITY_BUCKETS, NUM_HAND_TYPES, NUM_BUCKETS,
    HandType, make_bucket, decode_bucket, get_available_actions, count_raises,
)
from server.gto.exploitability import (
    exploitability_abstracted, exploitability_breakdown, strategy_audit,
)

# Global trainer ref for parallel exploitability workers (set before fork)
_eval_trainer = None


def _eval_br_worker(args):
    """Worker: compute one best-response value (phase × player × seed)."""
    from server.gto.exploitability import best_response_abstracted
    seed, phase, br_player, samples = args
    rng.seed(seed + br_player * 7 + hash(phase))
    np.random.seed(seed + br_player * 7 + abs(hash(phase)) % (2**31))
    return best_response_abstracted(_eval_trainer, br_player=br_player,
                                    phase=phase, samples=samples)

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
MATRIX_FILE = EXPERIMENTS_DIR / "matrix.json"

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def format_num(n):
    """Format large numbers: 1000000 -> '1.0M', 50000 -> '50K'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def train_with_progress(trainer, iterations, averaging_delay, sampling):
    """Train with a tqdm progress bar showing live stats."""
    if not HAS_TQDM:
        print(f"Training {format_num(iterations)} iterations "
              f"(install tqdm for progress bar)...")
        trainer.train(iterations, averaging_delay=averaging_delay,
                      sampling=sampling)
        return

    engine = "Cython" if HAS_CYTHON and sampling == 'external' else "Python"
    t_start = time.time()

    bar = tqdm(
        total=iterations,
        desc=f"CFR+ ({engine})",
        unit="iter",
        unit_scale=True,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}] "
            "{postfix}"
        ),
        dynamic_ncols=True,
        smoothing=0.1,
    )

    last_update = [0]

    def progress_callback(done, total, nodes):
        delta = done - last_update[0]
        last_update[0] = done
        elapsed = time.time() - t_start
        ms_per_iter = (elapsed / done * 1000) if done > 0 else 0
        bar.update(delta)
        bar.set_postfix_str(
            f"nodes={format_num(nodes)} "
            f"ms/it={ms_per_iter:.2f}",
            refresh=True,
        )

    trainer.train(
        iterations,
        averaging_delay=averaging_delay,
        sampling=sampling,
        progress_callback=progress_callback,
        chunk_size=max(1000, iterations // 200),  # ~200 bar updates
    )

    bar.close()
    elapsed = time.time() - t_start
    print(f"\nDone: {len(trainer.nodes):,} nodes in {elapsed:.1f}s "
          f"({elapsed / iterations * 1000:.2f} ms/iter)")


def train_parallel_with_progress(trainer, iterations, averaging_delay,
                                  sampling, num_workers):
    """Parallel training with tqdm progress bar."""
    import multiprocessing
    from server.gto.cfr import _parallel_worker, HAS_CYTHON
    from server.gto import cfr_fast as _cfr_fast

    if not HAS_TQDM:
        print(f"Parallel training with {num_workers} workers "
              f"({multiprocessing.cpu_count()} CPUs available)...")
        trainer.train(iterations, averaging_delay=averaging_delay,
                      sampling=sampling, num_workers=num_workers)
        return

    warmup_frac = 0.005
    warmup_iters = max(10000, int(iterations * warmup_frac))
    parallel_iters = iterations - warmup_iters
    chunk_size = max(1000, warmup_iters // 100)

    # Pre-allocate shared memory
    _cfr_fast.init_pool_shared(2_000_000)
    _cfr_fast.init_progress_counters(num_workers)
    if trainer.nodes:
        trainer._export_nodes_to_cython()

    seed = rng.randint(0, 2**31 - 1)
    t_start = time.time()

    # Phase 1: Warmup with tqdm
    bar = tqdm(
        total=iterations,
        desc=f"CFR+ (Cython x{num_workers})",
        unit="iter",
        unit_scale=True,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}] "
            "{postfix}"
        ),
        dynamic_ncols=True,
        smoothing=0.1,
    )
    bar.set_postfix_str("warmup (single-threaded)")

    done = 0
    while done < warmup_iters:
        batch = min(chunk_size, warmup_iters - done)
        _cfr_fast.train_fast(
            batch,
            start_iter=trainer.iterations + done,
            averaging_delay=averaging_delay,
            seed=seed + done,
        )
        done += batch
        elapsed = time.time() - t_start
        ms_per_iter = (elapsed / done * 1000) if done > 0 else 0
        bar.update(batch)
        bar.set_postfix_str(
            f"warmup nodes={format_num(_cfr_fast.get_node_count())} "
            f"ms/it={ms_per_iter:.2f}",
            refresh=True,
        )

    warmup_nodes = _cfr_fast.get_node_count()

    # Phase 2: Parallel with tqdm
    bar.set_postfix_str(
        f"parallel x{num_workers} nodes={format_num(warmup_nodes)}")

    iters_per_worker = parallel_iters // num_workers
    processes = []
    for w in range(num_workers):
        start = trainer.iterations + warmup_iters + w * iters_per_worker
        w_iters = (iters_per_worker if w < num_workers - 1
                   else parallel_iters - w * iters_per_worker)
        w_seed = seed + warmup_iters + w * 1_000_000

        p = multiprocessing.Process(
            target=_parallel_worker,
            args=(w, w_iters, start, averaging_delay, w_seed))
        processes.append((p, w_iters))
        p.start()

    # Poll workers using shared progress counters
    completed_iters = 0
    while any(p.is_alive() for p, _ in processes):
        time.sleep(0.5)
        counters = _cfr_fast.get_progress_counters(num_workers)
        alive = sum(1 for p, _ in processes if p.is_alive())
        # Sum progress across all workers
        est_done = sum(counters)
        delta = est_done - completed_iters
        if delta > 0:
            bar.update(delta)
            completed_iters = est_done
        elapsed = time.time() - t_start
        total_done = warmup_iters + completed_iters
        ms_per_iter = (elapsed / total_done * 1000) if total_done > 0 else 0
        bar.set_postfix_str(
            f"parallel {alive}/{num_workers} alive "
            f"nodes={format_num(warmup_nodes)} "
            f"ms/it={ms_per_iter:.2f}",
            refresh=True,
        )

    for p, _ in processes:
        p.join()

    # Final update
    final_delta = parallel_iters - completed_iters
    if final_delta > 0:
        bar.update(final_delta)

    _cfr_fast.cleanup_progress_counters()

    bar.set_postfix_str(
        f"importing nodes={format_num(warmup_nodes)}")

    trainer._import_nodes_from_cython()
    trainer.iterations += iterations

    elapsed = time.time() - t_start
    bar.set_postfix_str(
        f"done nodes={format_num(len(trainer.nodes))} "
        f"ms/it={elapsed / iterations * 1000:.2f}")
    bar.close()
    print(f"\nDone: {len(trainer.nodes):,} nodes in {elapsed:.1f}s "
          f"({elapsed / iterations * 1000:.2f} ms/iter)")


def print_strategy_tables(trainer):
    """Print representative strategy tables."""
    print("\n╔══════════════════════════════════════════════════╗")
    print("║       Preflop OOP (acts first, empty history)     ║")
    print("╚══════════════════════════════════════════════════╝")

    for ht in HandType:
        print(f"\n  {ht.name}:")
        for eq in [0, 2, 4, 6, 7]:
            bucket = make_bucket(eq, int(ht))
            strategy = trainer.get_strategy('preflop', bucket, (),
                                            position='oop')
            strat_str = ', '.join(
                f"{ACTION_NAMES.get(Action(a), str(a))}: {p:.0%}"
                for a, p in strategy.items()
                if p >= 0.01
            )
            print(f"    EQ {eq}/{NUM_EQUITY_BUCKETS-1}: {strat_str}")

    # IP acts after OOP — show responses to open raise and limp
    print("\n╔══════════════════════════════════════════════════╗")
    print("║    Preflop IP (vs open raise / vs limp)           ║")
    print("╚══════════════════════════════════════════════════╝")

    vs_raise = (int(Action.OPEN_RAISE),)   # OOP opened
    vs_limp = (int(Action.CHECK_CALL),)    # OOP limped

    for ht in [HandType.PREMIUM_PAIR, HandType.HIGH_PAIR, HandType.BROADWAY,
               HandType.SUITED_ACE, HandType.HIGH_SUITED_CONNECTOR, HandType.TRASH]:
        print(f"\n  {ht.name}:")
        for eq in [0, 3, 5, 7]:
            bucket = make_bucket(eq, int(ht))
            # vs raise
            strat = trainer.get_strategy('preflop', bucket, vs_raise, position='ip')
            s1 = ', '.join(
                f"{ACTION_NAMES.get(Action(a), str(a))}: {p:.0%}"
                for a, p in strat.items() if p >= 0.01
            )
            # vs limp
            strat2 = trainer.get_strategy('preflop', bucket, vs_limp, position='ip')
            s2 = ', '.join(
                f"{ACTION_NAMES.get(Action(a), str(a))}: {p:.0%}"
                for a, p in strat2.items() if p >= 0.01
            )
            print(f"    EQ {eq}/{NUM_EQUITY_BUCKETS-1} vs raise: {s1}")
            print(f"    EQ {eq}/{NUM_EQUITY_BUCKETS-1} vs limp:  {s2}")

    print("\n╔══════════════════════════════════════════════════╗")
    print("║            Flop Strategies (OOP sample)          ║")
    print("╚══════════════════════════════════════════════════╝")

    for eq in [0, 3, 5, 7]:
        for ht in [HandType.PREMIUM_PAIR, HandType.HIGH_SUITED_CONNECTOR,
                    HandType.TRASH]:
            bucket = make_bucket(eq, int(ht))
            strategy = trainer.get_strategy('flop', bucket, (), position='oop')
            strat_str = ', '.join(
                f"{ACTION_NAMES.get(Action(a), str(a))}: {p:.0%}"
                for a, p in strategy.items()
                if p >= 0.01
            )
            print(f"  EQ {eq} {ht.name:25s}: {strat_str}")


def print_exploitability_breakdown(trainer):
    """Print top exploitable infosets."""
    print("\n╔══════════════════════════════════════════════════╗")
    print("║          Most Exploitable Infosets (preflop)      ║")
    print("╚══════════════════════════════════════════════════╝")

    breakdown = exploitability_breakdown(trainer, phase='preflop', top_n=10)
    for item in breakdown:
        print(f"  {item['key']}: regret={item['regret_magnitude']:.2f}, "
              f"entropy={item['entropy']:.2f}, "
              f"visits={item['visit_weight']:.0f}")


def save_experiment(trainer, iterations, sampling, train_time):
    """Evaluate and log training run to the experiment matrix."""
    import multiprocessing
    EXPERIMENTS_DIR.mkdir(exist_ok=True)

    # Evaluate exploitability (parallel: 3 seeds × 4 phases × 2 players)
    import os
    num_eval_workers = min(os.cpu_count() or 4, 8)
    print(f"\n  Evaluating exploitability (3 seeds, {num_eval_workers} workers)...")
    global _eval_trainer
    _eval_trainer = trainer

    seeds = [42, 123, 456]
    all_phases = ['preflop', 'flop', 'turn', 'river']

    # Build all tasks: (seed, phase, br_player, samples)
    tasks = []
    for seed in seeds:
        for phase in all_phases:
            for br_player in [0, 1]:
                tasks.append((seed, phase, br_player, 500))
    # Also preflop-only tasks for exp_pf
    pf_tasks = []
    for seed in seeds:
        for br_player in [0, 1]:
            pf_tasks.append((seed, 'preflop', br_player, 500))

    all_tasks = tasks + pf_tasks  # 24 + 6 = 30 tasks

    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(num_eval_workers) as pool:
        all_results = pool.map(_eval_br_worker, all_tasks)

    # Reassemble: tasks[0..23] are full exploitability, tasks[24..29] are preflop-only
    full_results = all_results[:24]  # 3 seeds × 4 phases × 2 players
    pf_results = all_results[24:]    # 3 seeds × 2 players

    exp_values = []
    for s_idx in range(3):
        seed_total = 0.0
        for p_idx in range(4):
            base = s_idx * 8 + p_idx * 2
            br0 = full_results[base]
            br1 = full_results[base + 1]
            seed_total += (br0 + br1) / 2.0
        exp_values.append(seed_total / 4.0)

    exp_pf_values = []
    for s_idx in range(3):
        base = s_idx * 2
        br0 = pf_results[base]
        br1 = pf_results[base + 1]
        exp_pf_values.append((br0 + br1) / 2.0)

    exp_mean = float(np.mean(exp_values))
    exp_std = float(np.std(exp_values, ddof=1))
    exp_pf_mean = float(np.mean(exp_pf_values))

    print(f"  Exploitability: {exp_mean:.4f} ± {exp_std:.4f}")
    print(f"  Preflop:        {exp_pf_mean:.4f}")

    # Strategy audit
    audit = strategy_audit(trainer)

    # Auto-generate experiment name
    name = f"v{STRATEGY_VERSION}_{format_num(iterations)}_{time.strftime('%Y%m%d_%H%M%S')}"

    # Save strategy copy
    strategy_file = EXPERIMENTS_DIR / f"{name}_strategy.json"
    trainer.save(str(strategy_file))

    # Build result entry
    result = {
        "name": name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "iterations": iterations,
            "sampling": sampling,
            "strategy_version": STRATEGY_VERSION,
            "equity_buckets": NUM_EQUITY_BUCKETS,
            "hand_types": NUM_HAND_TYPES,
            "total_buckets": NUM_BUCKETS,
            "engine": "Cython" if HAS_CYTHON and sampling == 'external' else "Python",
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

    # Append to matrix
    matrix = []
    if MATRIX_FILE.exists():
        with open(MATRIX_FILE) as f:
            matrix = json.load(f)
    matrix.append(result)
    with open(MATRIX_FILE, 'w') as f:
        json.dump(matrix, f, indent=2)

    print(f"  Experiment '{name}' saved to {MATRIX_FILE}")


def apply_strategy_overrides(trainer: CFRTrainer) -> dict:
    """Post-training strategy clamps for known structural anomalies.

    1. Premium limp clamp: PREMIUM/HIGH_PAIR at EQ5+, preflop, no history
       → cap CHECK_CALL at 5%, redistribute to OPEN_RAISE
    2. Strong hand fold clamp: EQ6+ facing bet
       → cap FOLD at 20%, redistribute to CHECK_CALL

    Returns dict with counts of clamped nodes.
    """
    premium_clamped = 0
    fold_clamped = 0

    for key, node in trainer.nodes.items():
        # Parse infoset key: "phase:position:bucket:(history)"
        parts = key.split(':')
        if len(parts) < 4:
            continue
        phase = parts[0]
        bucket = int(parts[2])
        eq_bucket, hand_type = decode_bucket(bucket)

        avg = node.get_average_strategy()
        total = node.strategy_sum.sum()
        if total <= 0:
            continue

        # 1. Premium limp clamp: preflop, no prior bet, premium/high pair, EQ5+
        if (phase == 'preflop' and eq_bucket >= 5
                and hand_type in (int(HandType.PREMIUM_PAIR), int(HandType.HIGH_PAIR))):
            # Find CHECK_CALL and OPEN_RAISE indices in the action list
            history_str = parts[3] if len(parts) > 3 else "()"
            if history_str == "()":
                # Root OOP node — actions are [FOLD, CHECK_CALL, OPEN_RAISE, ALL_IN]
                # or similar; we need to find the indices
                actions = list(range(node.num_actions))
                # CHECK_CALL is typically index 1 in preflop no-bet menu
                # Reconstruct: get_available_actions(has_bet=True, can_raise=True, 'preflop', 0)
                # preflop root for OOP: [FOLD, CHECK_CALL, OPEN_RAISE, ALL_IN]
                cc_prob = avg[1] if node.num_actions > 1 else 0.0
                if cc_prob > 0.05:
                    excess = cc_prob - 0.05
                    node.strategy_sum[1] *= (0.05 / cc_prob) if cc_prob > 0 else 1.0
                    # Redistribute to OPEN_RAISE (index 2)
                    if node.num_actions > 2:
                        or_share = node.strategy_sum[2]
                        node.strategy_sum[2] += excess * total
                    premium_clamped += 1

        # 2. Strong hand fold clamp: EQ6+, facing bet (FOLD is action 0)
        if eq_bucket >= 6 and phase != 'preflop':
            fold_prob = avg[0] if node.num_actions > 0 else 0.0
            if fold_prob > 0.20:
                excess = fold_prob - 0.20
                node.strategy_sum[0] *= (0.20 / fold_prob) if fold_prob > 0 else 1.0
                # Redistribute to CHECK_CALL (index 1)
                if node.num_actions > 1:
                    node.strategy_sum[1] += excess * total
                fold_clamped += 1

    result = {'premium_clamped': premium_clamped, 'fold_clamped': fold_clamped}
    print(f"  Strategy overrides: {premium_clamped} premium limp nodes, "
          f"{fold_clamped} strong-hand fold nodes clamped")
    return result


def main():
    import os

    # Parse args: train_gto.py [iterations] [sampling] [--fresh] [--workers N]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]
    fresh = '--fresh' in flags

    # Parse --workers N and track which positional args to skip
    num_workers = 1
    skip_indices = set()
    for i, flag in enumerate(sys.argv[1:], 1):
        if flag == '--workers' and i < len(sys.argv) - 1:
            num_workers = int(sys.argv[i + 1])
            skip_indices.add(i + 1)  # skip the value after --workers
            break
    if num_workers < 1:
        num_workers = 1

    args = [a for i, a in enumerate(sys.argv[1:], 1)
            if not a.startswith('--') and i not in skip_indices]
    iterations = int(args[0]) if len(args) > 0 else 10000
    sampling = args[1] if len(args) > 1 else 'external'

    if sampling not in ('external', 'vanilla'):
        print(f"Unknown sampling method: {sampling}")
        print("Options: external, vanilla")
        sys.exit(1)

    engine_str = 'Cython' if HAS_CYTHON else 'Python'
    if num_workers > 1:
        engine_str += f' x{num_workers}'

    print(f"┌─────────────────────────────────────────────────┐")
    print(f"│  GTO Solver Training                            │")
    print(f"│  Iterations: {format_num(iterations):>10}                        │")
    print(f"│  Sampling:   {sampling:>10}                        │")
    print(f"│  Buckets:    {NUM_BUCKETS:>10} (8 eq × 15 ht)       │")
    print(f"│  Engine:     {engine_str:>10}                        │")
    print(f"└─────────────────────────────────────────────────┘")

    trainer = CFRTrainer()

    # Load existing if available (continue training)
    if not fresh and trainer.load(STRATEGY_FILE):
        print(f"Continuing from {trainer.iterations:,} iterations...")
    else:
        if fresh:
            print("Starting fresh training (--fresh flag)...")
        else:
            print("Starting fresh training...")

    averaging_delay = iterations // 4

    t_start = time.time()
    if num_workers > 1:
        train_parallel_with_progress(trainer, iterations, averaging_delay,
                                     sampling, num_workers)
    else:
        train_with_progress(trainer, iterations, averaging_delay, sampling)
    train_time = time.time() - t_start

    # Apply post-training strategy overrides before saving
    apply_strategy_overrides(trainer)

    trainer.save(STRATEGY_FILE)

    print_strategy_tables(trainer)
    print_exploitability_breakdown(trainer)

    print(f"\n  Total nodes: {len(trainer.nodes):,}")
    print(f"  Strategy saved to: {STRATEGY_FILE}")

    # Log to experiment matrix
    save_experiment(trainer, iterations, sampling, train_time)


if __name__ == "__main__":
    main()
