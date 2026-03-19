"""
Head-to-head GTO vs GTO match runner with detailed analysis.

Runs two GTO configurations against each other and reports where each
strategy wins/loses: overall EV, by street, by position, by hand strength,
and strategy divergence (infosets where the two strategies disagree most).

Usage:
    venv/bin/python run_h2h.py [options]

    # Default: poly2+refine (P0) vs B0+refine (P1)
    venv/bin/python run_h2h.py

    # Custom configs
    venv/bin/python run_h2h.py \
        --p0-strategy experiments/v11_poly2_100M.json --p0-mapping refine \
        --p1-strategy experiments/best/v9_B0_100M_allbots_positive.json --p1-mapping refine \
        --hands 50000 --seeds 42,123,456
"""
import argparse
import math
import multiprocessing
import sys
from collections import defaultdict
from dataclasses import dataclass, field

from server.gto.cfr import CFRTrainer
from server.gto.abstraction import Action, ACTION_NAMES, detect_action_grid_from_strategy, set_action_grid
from eval_harness.match_engine import HeadsUpMatch, GTOAgent, MatchResult, HandRecord, DecisionRecord

BIG_BLIND = 20
STARTING_CHIPS = 10_000

# ── worker state ────────────────────────────────────────────────────────────
_p0_trainer = None
_p1_trainer = None
_p0_mapping = None
_p1_mapping = None
_p0_name = None
_p1_name = None
_p0_kwargs = {}
_p1_kwargs = {}


def _init_workers(p0_path, p0_map, p0_nm, p1_path, p1_map, p1_nm,
                  p0_kw=None, p1_kw=None):
    global _p0_trainer, _p1_trainer, _p0_mapping, _p1_mapping
    global _p0_name, _p1_name, _p0_kwargs, _p1_kwargs
    _p0_trainer = CFRTrainer(); _p0_trainer.load(p0_path)
    grid = detect_action_grid_from_strategy(_p0_trainer); set_action_grid(grid)
    _p1_trainer = CFRTrainer(); _p1_trainer.load(p1_path)
    _p0_mapping = p0_map; _p1_mapping = p1_map
    _p0_name = p0_nm; _p1_name = p1_nm
    _p0_kwargs = p0_kw or {}; _p1_kwargs = p1_kw or {}


def _run_seed(args):
    seed, hands_per_seed = args
    p0 = GTOAgent(_p0_trainer, name=_p0_name, mapping=_p0_mapping, simulations=80,
                  **_p0_kwargs)
    p1 = GTOAgent(_p1_trainer, name=_p1_name, mapping=_p1_mapping, simulations=80,
                  **_p1_kwargs)
    match = HeadsUpMatch(p0, p1, big_blind=BIG_BLIND,
                         starting_chips=STARTING_CHIPS,
                         seed=seed, detailed_tracking=True)
    return match.play(hands_per_seed)


# ── analysis helpers ─────────────────────────────────────────────────────────

def _p0_position(hand: HandRecord) -> str:
    """Infer P0 position (ip/oop) from preflop infoset key."""
    for d in hand.decisions:
        if d.player == 0 and d.phase == "preflop" and d.infoset_key:
            parts = d.infoset_key.split(":")
            if len(parts) >= 2:
                return parts[1]  # 'ip' or 'oop'
    return "unknown"


def _kl_divergence(p: dict, q: dict, actions: list) -> float:
    """KL(P||Q) in nats.  P = p0 strategy, Q = p1 strategy at same infoset."""
    kl = 0.0
    for a in actions:
        pi = p.get(a, 0.0)
        qi = q.get(a, 0.0)
        if pi > 1e-9:
            qi = max(qi, 1e-9)
            kl += pi * math.log(pi / qi)
    return kl


def _tv_distance(p: dict, q: dict, actions: list) -> float:
    """Total variation distance: 0.5 * sum|P - Q|."""
    return 0.5 * sum(abs(p.get(a, 0.0) - q.get(a, 0.0)) for a in actions)


def analyze(results: list[MatchResult], p1_trainer: CFRTrainer,
            p0_name: str, p1_name: str) -> dict:
    """Aggregate all hand records and compute detailed breakdown."""
    all_hands: list[HandRecord] = []
    for r in results:
        all_hands.extend(r.hands)

    total_hands = len(all_hands)
    if total_hands == 0:
        return {}

    # ── Overall ──────────────────────────────────────────────────────────────
    seed_bbs = [r.p0_bb_per_100 for r in results]
    overall_mean = sum(seed_bbs) / len(seed_bbs)
    if len(seed_bbs) > 1:
        std = math.sqrt(sum((x - overall_mean) ** 2 for x in seed_bbs) / (len(seed_bbs) - 1))
        ci_95 = 1.96 * std / math.sqrt(len(seed_bbs))
    else:
        std = 0.0; ci_95 = 0.0

    # ── By street ────────────────────────────────────────────────────────────
    street_totals = defaultdict(float)
    for h in all_hands:
        for phase, ev in h.street_ev.items():
            street_totals[phase] += ev / BIG_BLIND
    street_bb100 = {ph: (v / total_hands) * 100 for ph, v in street_totals.items()}

    # ── By position ──────────────────────────────────────────────────────────
    pos_net = defaultdict(list)
    for h in all_hands:
        pos = _p0_position(h)
        pos_net[pos].append(h.p0_net / BIG_BLIND)
    position_bb100 = {}
    for pos, nets in pos_net.items():
        position_bb100[pos] = {
            "bb100": (sum(nets) / max(len(nets), 1)) * 100,
            "hands": len(nets),
        }

    # ── By equity bucket (P0's preflop bucket as proxy) ─────────────────────
    bucket_net = defaultdict(list)
    for h in all_hands:
        pf_bucket = h.p0_bucket_per_street.get("preflop", -1)
        if pf_bucket >= 0:
            eq = pf_bucket // 15
            bucket_net[eq].append(h.p0_net / BIG_BLIND)
    eq_labels = ["EQ0 (0–12%)", "EQ1 (12–25%)", "EQ2 (25–37%)", "EQ3 (37–50%)",
                 "EQ4 (50–62%)", "EQ5 (62–75%)", "EQ6 (75–87%)", "EQ7 (87–100%)"]
    equity_bb100 = {}
    for eq in range(8):
        nets = bucket_net.get(eq, [])
        equity_bb100[eq] = {
            "label": eq_labels[eq],
            "bb100": (sum(nets) / max(len(nets), 1)) * 100,
            "hands": len(nets),
        }

    # ── Action frequencies (P0 vs P1 by phase) ───────────────────────────────
    action_counts = {
        "p0": defaultdict(lambda: defaultdict(int)),
        "p1": defaultdict(lambda: defaultdict(int)),
    }
    for h in all_hands:
        for d in h.decisions:
            pkey = "p0" if d.player == 0 else "p1"
            family = _action_family(d.action)
            action_counts[pkey][d.phase][family] += 1
    action_freq = {}
    for pkey in ("p0", "p1"):
        action_freq[pkey] = {}
        for phase, counts in action_counts[pkey].items():
            total = sum(counts.values())
            action_freq[pkey][phase] = {fam: (n / total * 100) for fam, n in counts.items()}

    # ── Showdown stats ───────────────────────────────────────────────────────
    showdowns = [h for h in all_hands if h.went_to_showdown]
    p0_showdown_wins = sum(1 for h in showdowns if h.winner == 0)
    p0_showdown_splits = sum(1 for h in showdowns if h.winner == -1)
    showdown_stats = {
        "rate_pct": len(showdowns) / total_hands * 100,
        "p0_win_pct": p0_showdown_wins / max(len(showdowns), 1) * 100,
        "p0_split_pct": p0_showdown_splits / max(len(showdowns), 1) * 100,
    }

    # ── Strategy divergence ──────────────────────────────────────────────────
    # For each P0 decision with a populated strategy, look up P1's strategy
    # at the same infoset and compute total variation distance.
    divergence_by_key: dict[str, dict] = {}
    for h in all_hands:
        for d in h.decisions:
            if d.player != 0 or not d.infoset_key or not d.strategy or not d.available_actions:
                continue
            key = d.infoset_key
            if key not in divergence_by_key:
                # Query P1 strategy at the same infoset
                parts = key.split(":")
                if len(parts) < 4:
                    continue
                phase, position, bucket_str, history_str = parts[0], parts[1], parts[2], parts[3]
                try:
                    bucket = int(bucket_str)
                    raw_hist = history_str.strip("()").strip()
                    history = tuple(int(x) for x in raw_hist.split(",") if x.strip()) if raw_hist else ()
                except (ValueError, IndexError):
                    continue
                p1_strat = p1_trainer.get_strategy(phase, bucket, history, position)
                p0_strat_norm = d.strategy
                actions = d.available_actions
                tv = _tv_distance(p0_strat_norm, p1_strat, actions)
                kl = _kl_divergence(p0_strat_norm, p1_strat, actions)
                divergence_by_key[key] = {
                    "tv": tv, "kl": kl,
                    "count": 0,
                    "p0_strategy": p0_strat_norm,
                    "p1_strategy": p1_strat,
                    "actions": actions,
                    "phase": phase, "position": position,
                    "eq_bucket": d.eq_bucket,
                }
            divergence_by_key[key]["count"] += 1

    # Top 15 by frequency-weighted TV distance
    top_divergent = sorted(
        divergence_by_key.items(),
        key=lambda x: x[1]["tv"] * x[1]["count"],
        reverse=True,
    )[:15]

    return {
        "total_hands": total_hands,
        "overall": {"mean": overall_mean, "std": std, "ci_95": ci_95, "per_seed": seed_bbs},
        "by_street": street_bb100,
        "by_position": position_bb100,
        "by_equity": equity_bb100,
        "action_freq": action_freq,
        "showdown": showdown_stats,
        "top_divergent": top_divergent,
    }


def _action_family(action: str) -> str:
    if action == "fold": return "fold"
    if action == "check": return "check"
    if action == "call": return "call"
    return "bet/raise"


# ── pretty printer ───────────────────────────────────────────────────────────

def print_report(analysis: dict, p0_name: str, p1_name: str):
    total = analysis["total_hands"]
    ov = analysis["overall"]
    sign = "+" if ov["mean"] >= 0 else ""

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"║  HEAD-TO-HEAD: {p0_name:20s} vs {p1_name:15s}  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  {total:,} hands  |  {len(ov['per_seed'])} seeds")
    print()

    print("  ── Overall ─────────────────────────────────────────────────")
    print(f"  {p0_name}: {sign}{ov['mean']:.1f} bb/100  "
          f"(±{ov['ci_95']:.1f} 95% CI, std={ov['std']:.1f})")
    print(f"  {p1_name}: {-ov['mean']:+.1f} bb/100  (mirror)")
    seed_str = "  Per seed: " + "  ".join(f"{v:+.1f}" for v in ov["per_seed"])
    print(seed_str)

    print()
    print("  ── By Street ───────────────────────────────────────────────")
    street_order = ["preflop", "flop", "turn", "river"]
    running = 0.0
    for ph in street_order:
        val = analysis["by_street"].get(ph, 0.0)
        running += val
        bar = "▲" if val >= 0 else "▼"
        print(f"  {bar} {ph:8s}  {val:+8.1f} bb/100   cumulative: {running:+.1f}")

    print()
    print("  ── By Position ─────────────────────────────────────────────")
    for pos, data in sorted(analysis["by_position"].items()):
        print(f"  {pos.upper():3s}  {data['bb100']:+8.1f} bb/100   ({data['hands']:,} hands)")

    print()
    print("  ── By Equity Bucket (P0 preflop strength) ──────────────────")
    for eq in range(8):
        d = analysis["by_equity"][eq]
        bar_len = int(abs(d["bb100"]) / 20)
        bar = ("█" * bar_len) if d["bb100"] >= 0 else ("░" * bar_len)
        sign_ch = "+" if d["bb100"] >= 0 else "-"
        print(f"  {d['label']:20s}  {d['bb100']:+8.1f} bb/100   {bar}  ({d['hands']:,} hands)")

    print()
    print("  ── Action Frequency Comparison ─────────────────────────────")
    phases = sorted({ph for pkey in ("p0", "p1")
                     for ph in analysis["action_freq"].get(pkey, {})})
    for ph in phases:
        p0f = analysis["action_freq"].get("p0", {}).get(ph, {})
        p1f = analysis["action_freq"].get("p1", {}).get(ph, {})
        all_fams = sorted(set(list(p0f) + list(p1f)))
        print(f"  {ph}:")
        for fam in all_fams:
            p0v = p0f.get(fam, 0.0)
            p1v = p1f.get(fam, 0.0)
            diff = p0v - p1v
            diff_str = f"  [{diff:+.1f}%]" if abs(diff) >= 0.5 else ""
            print(f"    {fam:12s}  {p0_name}: {p0v:5.1f}%   {p1_name}: {p1v:5.1f}%{diff_str}")

    print()
    print("  ── Showdown Stats ───────────────────────────────────────────")
    sd = analysis["showdown"]
    print(f"  Showdown rate: {sd['rate_pct']:.1f}%")
    print(f"  {p0_name} wins at SD: {sd['p0_win_pct']:.1f}%  "
          f"(splits: {sd['p0_split_pct']:.1f}%)")

    print()
    print("  ── Top Strategy Divergence Points ───────────────────────────")
    print("  (Infosets where the two strategies disagree most, weighted by visit frequency)")
    print()
    print(f"  {'Infoset Key':42s}  {'TV':5s}  {'Visits':6s}  "
          f"{'Weighted':8s}  {p0_name[:12]:12s} vs {p1_name[:12]:12s}")
    print("  " + "-" * 110)
    for key, data in analysis["top_divergent"]:
        p0s = data["p0_strategy"]
        p1s = data["p1_strategy"]
        acts = data["actions"]
        # Show the biggest individual action difference
        max_diff_action = max(acts, key=lambda a: abs(p0s.get(a, 0) - p1s.get(a, 0)))
        act_name = ACTION_NAMES.get(max_diff_action, str(max_diff_action))
        p0_pct = p0s.get(max_diff_action, 0) * 100
        p1_pct = p1s.get(max_diff_action, 0) * 100
        weighted = data["tv"] * data["count"]
        short_key = key[:42]
        print(f"  {short_key:42s}  {data['tv']:.3f}  {data['count']:6d}  "
              f"{weighted:8.1f}  "
              f"{act_name}: {p0_pct:.0f}% vs {p1_pct:.0f}%")

    print()
    print("  ── Verdict ──────────────────────────────────────────────────")
    if ov["mean"] > ov["ci_95"]:
        verdict = f"{p0_name} leads significantly ({ov['mean']:+.1f} ± {ov['ci_95']:.1f})"
    elif ov["mean"] < -ov["ci_95"]:
        verdict = f"{p1_name} leads significantly ({-ov['mean']:+.1f} ± {ov['ci_95']:.1f})"
    else:
        verdict = f"Within margin of error — no significant edge at this sample size"
    print(f"  {verdict}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GTO vs GTO head-to-head analysis")
    parser.add_argument("--p0-strategy", default="experiments/v11_poly2_100M.json",
                        help="Strategy file for P0")
    parser.add_argument("--p0-mapping", default="refine",
                        choices=["nearest", "conservative", "confidence_nearest", "refine",
                                 "pseudo_harmonic"],
                        help="Mapping for P0")
    parser.add_argument("--p0-name", default="poly2+refine")
    parser.add_argument("--p1-strategy",
                        default="experiments/best/v9_B0_100M_allbots_positive.json",
                        help="Strategy file for P1")
    parser.add_argument("--p1-mapping", default="refine",
                        choices=["nearest", "conservative", "confidence_nearest", "refine",
                                 "pseudo_harmonic"],
                        help="Mapping for P1")
    parser.add_argument("--p1-name", default="B0+refine")
    parser.add_argument("--hands", type=int, default=50_000,
                        help="Total hands to play (split evenly across seeds)")
    parser.add_argument("--seeds", default="42,123,456",
                        help="Comma-separated seed list")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: num seeds)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    hands_per_seed = args.hands // len(seeds)
    num_workers = args.workers or len(seeds)

    print(f"\n  Loading strategies...")
    print(f"  P0: {args.p0_name}  ({args.p0_strategy})")
    print(f"  P1: {args.p1_name}  ({args.p1_strategy})")
    print(f"  {hands_per_seed:,} hands × {len(seeds)} seeds = {hands_per_seed * len(seeds):,} total")

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(
        processes=num_workers,
        initializer=_init_workers,
        initargs=(args.p0_strategy, args.p0_mapping, args.p0_name,
                  args.p1_strategy, args.p1_mapping, args.p1_name),
    ) as pool:
        tasks = [(seed, hands_per_seed) for seed in seeds]
        print(f"  Running {len(tasks)} seeds in parallel ({num_workers} workers)...\n")
        results = pool.map(_run_seed, tasks)

    # Load P1 trainer in main process for divergence analysis
    print("  Analyzing results...")
    p1_trainer = CFRTrainer()
    p1_trainer.load(args.p1_strategy)
    grid = detect_action_grid_from_strategy(p1_trainer)
    set_action_grid(grid)

    analysis = analyze(results, p1_trainer, args.p0_name, args.p1_name)
    print_report(analysis, args.p0_name, args.p1_name)


if __name__ == "__main__":
    main()
