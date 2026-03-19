#!/usr/bin/env python3
"""EMD-based equity histogram clustering for card abstraction (WS4).

Replaces uniform E[HS²] equity bands with Earth Mover's Distance (EMD)
clustering of equity distribution histograms. Hands with the same mean
equity but different distribution shapes (e.g., flush draw vs top pair)
get separated into different clusters.

Usage:
    venv/bin/python -m server.gto.emd_clustering

Outputs:
    server/gto/emd_centroids.json  — 4 streets × K centroids × N_BINS
    server/gto/emd_preflop_table.json — 169 canonical hands → cluster_id
"""
import json
import random
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

from server.deck import Card, SUITS
from server.gto.equity import hand_equity, make_deck

# Use Cython-accelerated equity when available (~47x faster)
try:
    from eval_harness.eval_fast import hand_equity_fast as _cy_equity
    _HAS_FAST_EVAL = True
except ImportError:
    _HAS_FAST_EVAL = False


def _fast_equity(hole_cards, community, simulations=50):
    """Equity computation using Cython when available."""
    if _HAS_FAST_EVAL:
        return _cy_equity(hole_cards, community, simulations=simulations)
    return hand_equity(hole_cards, community, num_opponents=1,
                       simulations=simulations)

N_BINS = 10       # Histogram bins covering [0, 1) in 10% increments
K_CLUSTERS = 12   # Clusters per street (matching NUM_EQUITY_BUCKETS in EMD mode)
STREETS = ['preflop', 'flop', 'turn', 'river']

_MODULE_DIR = Path(__file__).parent
CENTROIDS_FILE = _MODULE_DIR / 'emd_centroids.json'
PREFLOP_TABLE_FILE = _MODULE_DIR / 'emd_preflop_table.json'


def compute_equity_histogram(hole_cards: list[Card], community: list[Card],
                             n_bins: int = N_BINS, simulations: int = 300,
                             inner_sims: int = 50) -> np.ndarray:
    """Compute equity distribution histogram for a hand-board combination.

    Instead of collapsing MC rollouts to a single E[HS²] number, we bin
    individual equity outcomes into a histogram capturing the full distribution.

    For pre-river: sample future community cards, compute equity at each.
    For river: compute equity vs random opponents (already a distribution).
    """
    hist = np.zeros(n_bins, dtype=np.float64)

    if len(community) >= 5:
        # River: equity distribution comes from opponent hand range.
        # Run MC simulations, each giving a win/loss/tie outcome.
        dead = set((c.rank, c.suit) for c in hole_cards + community)
        remaining = [c for c in make_deck() if (c.rank, c.suit) not in dead]
        n = min(simulations, len(remaining) // 2)
        for _ in range(n):
            random.shuffle(remaining)
            eq = _fast_equity(hole_cards, community, simulations=inner_sims)
            b = min(int(eq * n_bins), n_bins - 1)
            hist[b] += 1
        # River equity is nearly deterministic — just one evaluation
        eq = _fast_equity(hole_cards, community,
                         simulations=max(100, simulations))
        b = min(int(eq * n_bins), n_bins - 1)
        hist[b] = 1.0
    else:
        # Pre-river: sample next community card(s), compute equity at each.
        dead = set((c.rank, c.suit) for c in hole_cards + community)
        remaining = [c for c in make_deck() if (c.rank, c.suit) not in dead]
        num_rollouts = min(simulations, len(remaining))
        sampled_cards = random.sample(remaining, num_rollouts)

        for next_card in sampled_cards:
            future_community = list(community) + [next_card]
            eq = _fast_equity(hole_cards, future_community,
                             simulations=inner_sims)
            b = min(int(eq * n_bins), n_bins - 1)
            hist[b] += 1

    # Normalize to probability distribution
    total = hist.sum()
    if total > 0:
        hist /= total
    else:
        hist[:] = 1.0 / n_bins  # Uniform fallback

    return hist


def emd_1d(h1: np.ndarray, h2: np.ndarray) -> float:
    """Wasserstein-1 (Earth Mover's) distance for 1D histograms on a fixed grid.

    For 1D distributions on regular bins, EMD = sum of |CDF differences|.
    O(N_BINS) per pair, no scipy needed.
    """
    cdf1 = np.cumsum(h1)
    cdf2 = np.cumsum(h2)
    return float(np.sum(np.abs(cdf1 - cdf2))) / len(h1)


def emd_kmeans(histograms: np.ndarray, K: int = K_CLUSTERS,
               max_iters: int = 50, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """K-means clustering with EMD distance.

    Args:
        histograms: (N, N_BINS) array of normalized histograms
        K: number of clusters
        max_iters: maximum iterations

    Returns:
        centroids: (K, N_BINS) cluster centers
        labels: (N,) cluster assignments
    """
    rng = np.random.RandomState(seed)
    N = len(histograms)

    # K-means++ initialization
    centroids = np.zeros((K, histograms.shape[1]))
    idx = rng.randint(N)
    centroids[0] = histograms[idx]

    for k in range(1, K):
        # Compute min distance to existing centroids
        dists = np.array([
            min(emd_1d(histograms[i], centroids[j]) for j in range(k))
            for i in range(N)
        ])
        # Probability proportional to distance²
        probs = dists ** 2
        probs /= probs.sum()
        idx = rng.choice(N, p=probs)
        centroids[k] = histograms[idx]

    labels = np.zeros(N, dtype=np.int32)

    for iteration in range(max_iters):
        # Assignment step
        old_labels = labels.copy()
        for i in range(N):
            best_k = 0
            best_d = emd_1d(histograms[i], centroids[0])
            for k in range(1, K):
                d = emd_1d(histograms[i], centroids[k])
                if d < best_d:
                    best_d = d
                    best_k = k
            labels[i] = best_k

        # Update step (arithmetic mean of assigned histograms)
        for k in range(K):
            mask = labels == k
            if mask.sum() > 0:
                centroids[k] = histograms[mask].mean(axis=0)
                # Re-normalize centroid
                s = centroids[k].sum()
                if s > 0:
                    centroids[k] /= s

        # Convergence check
        if np.array_equal(labels, old_labels):
            print(f"  K-means converged at iteration {iteration + 1}")
            break
    else:
        print(f"  K-means reached max iterations ({max_iters})")

    return centroids, labels


def _sort_clusters_by_equity(centroids: np.ndarray, labels: np.ndarray,
                             n_bins: int = N_BINS) -> tuple[np.ndarray, np.ndarray]:
    """Sort clusters so index 0 = weakest equity, index K-1 = strongest.

    Critical for the Cython training drift model which assumes equity
    cluster indices are ordered weak→strong (±1 drift = get weaker/stronger).
    """
    K = len(centroids)
    # Mean equity for each centroid: weighted average of bin centers
    mean_eq = np.array([
        sum(centroids[k][b] * (b + 0.5) / n_bins for b in range(n_bins))
        for k in range(K)
    ])
    sort_order = np.argsort(mean_eq)
    sorted_centroids = centroids[sort_order]

    # Relabel: build inverse mapping old_index → new_index
    inv_map = np.zeros(K, dtype=np.int32)
    for new_idx, old_idx in enumerate(sort_order):
        inv_map[old_idx] = new_idx
    sorted_labels = inv_map[labels]

    return sorted_centroids, sorted_labels


def _canonical_preflop_hands() -> list[tuple[int, int, bool]]:
    """Generate all 169 canonical preflop hands as (high_rank, low_rank, suited)."""
    hands = []
    for r1 in range(2, 15):
        for r2 in range(2, r1 + 1):
            if r1 == r2:
                hands.append((r1, r2, False))  # Pair
            else:
                hands.append((r1, r2, True))   # Suited
                hands.append((r1, r2, False))   # Offsuit
    return hands


def _make_hand(high: int, low: int, suited: bool) -> list[Card]:
    """Create a concrete hand from canonical representation."""
    if suited:
        return [Card(high, 'c'), Card(low, 'c')]
    else:
        return [Card(high, 'c'), Card(low, 'd')]


def _canonical_key(high: int, low: int, suited: bool) -> str:
    """Canonical hand key matching fast_equity.py format."""
    if high == low:
        return f"{high}_{low}"
    s = 's' if suited else 'o'
    return f"{high}_{low}_{s}"


def generate_preflop_histograms(n_boards: int = 500,
                                n_bins: int = N_BINS) -> dict[str, np.ndarray]:
    """Generate equity histograms for all 169 canonical preflop hands.

    For each hand, sample n_boards random 5-card boards and compute equity
    on each. The equity values are binned into a histogram.
    """
    hands = _canonical_preflop_hands()
    histograms = {}

    print(f"  Generating preflop histograms: {len(hands)} hands × {n_boards} boards")
    t0 = time.time()

    for idx, (high, low, suited) in enumerate(hands):
        hole = _make_hand(high, low, suited)
        key = _canonical_key(high, low, suited)

        hist = np.zeros(n_bins, dtype=np.float64)
        dead = set((c.rank, c.suit) for c in hole)
        remaining = [c for c in make_deck() if (c.rank, c.suit) not in dead]

        for _ in range(n_boards):
            board = random.sample(remaining, 5)
            eq = _fast_equity(hole, board, simulations=50)
            b = min(int(eq * n_bins), n_bins - 1)
            hist[b] += 1

        # Normalize
        hist /= hist.sum()
        histograms[key] = hist

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    {idx + 1}/{len(hands)} hands ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Preflop done: {len(hands)} hands in {elapsed:.0f}s")
    return histograms


def generate_postflop_histograms(street: str, n_samples: int = 15000,
                                 n_bins: int = N_BINS,
                                 sims: int = 200,
                                 inner_sims: int = 50) -> np.ndarray:
    """Generate equity histograms for randomly sampled hand-board combos.

    Args:
        street: 'flop', 'turn', or 'river'
        n_samples: number of random hand-board combos to generate
    """
    n_community = {'flop': 3, 'turn': 4, 'river': 5}[street]
    histograms = np.zeros((n_samples, n_bins), dtype=np.float64)

    print(f"  Generating {street} histograms: {n_samples} samples")
    t0 = time.time()

    for i in range(n_samples):
        deck = make_deck()
        random.shuffle(deck)
        hole = deck[:2]
        board = deck[2:2 + n_community]

        histograms[i] = compute_equity_histogram(
            hole, board, n_bins=n_bins, simulations=sims, inner_sims=inner_sims)

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate
            print(f"    {i + 1}/{n_samples} ({elapsed:.0f}s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"  {street} done: {n_samples} samples in {elapsed:.0f}s")
    return histograms


def build_all_centroids(K: int = K_CLUSTERS, n_bins: int = N_BINS,
                        preflop_boards: int = 500,
                        postflop_samples: int = 15000):
    """Build EMD cluster centroids for all streets.

    This is the main offline pipeline. Run once, save results.
    """
    print("=" * 60)
    print("  EMD Clustering Pipeline")
    print(f"  K={K} clusters, {n_bins} bins per histogram")
    print("=" * 60)

    all_centroids = {}

    # --- Preflop ---
    print("\n[1/4] Preflop")
    preflop_hists = generate_preflop_histograms(n_boards=preflop_boards,
                                                 n_bins=n_bins)
    keys = sorted(preflop_hists.keys())
    hist_array = np.array([preflop_hists[k] for k in keys])

    centroids, labels = emd_kmeans(hist_array, K=K)
    centroids, labels = _sort_clusters_by_equity(centroids, labels, n_bins)
    all_centroids['preflop'] = centroids.tolist()

    # Build preflop lookup table
    preflop_table = {keys[i]: int(labels[i]) for i in range(len(keys))}

    # Print preflop cluster summary
    for k in range(K):
        members = [keys[i] for i in range(len(keys)) if labels[i] == k]
        print(f"  Cluster {k}: {len(members)} hands — {members[:5]}...")

    # --- Postflop streets ---
    for si, street in enumerate(['flop', 'turn', 'river'], 2):
        print(f"\n[{si}/4] {street.capitalize()}")
        hists = generate_postflop_histograms(street, n_samples=postflop_samples,
                                              n_bins=n_bins)
        centroids, labels = emd_kmeans(hists, K=K)
        centroids, labels = _sort_clusters_by_equity(centroids, labels, n_bins)
        all_centroids[street] = centroids.tolist()

        # Print cluster sizes
        for k in range(K):
            count = int((labels == k).sum())
            centroid_str = ', '.join(f'{v:.2f}' for v in centroids[k])
            print(f"  Cluster {k}: {count} samples, centroid=[{centroid_str}]")

    # --- Validation: compare intra-cluster EMD vs uniform bands ---
    print("\n[Validation] Intra-cluster EMD comparison")
    # Use flop histograms for comparison
    flop_hists = generate_postflop_histograms('flop', n_samples=5000, n_bins=n_bins)

    # EMD clustering assignments
    flop_centroids = np.array(all_centroids['flop'])
    emd_total = 0.0
    for i in range(len(flop_hists)):
        dists = [emd_1d(flop_hists[i], flop_centroids[k]) for k in range(K)]
        emd_total += min(dists)
    avg_emd = emd_total / len(flop_hists)

    # Uniform band assignments (current system with K bands)
    uniform_total = 0.0
    for i in range(len(flop_hists)):
        # Current system: mean equity → uniform band
        mean_eq = sum(flop_hists[i][b] * (b + 0.5) / n_bins for b in range(n_bins))
        band = min(int(mean_eq * K), K - 1)
        # "Centroid" for uniform band is a delta at the band center
        uniform_centroid = np.zeros(n_bins)
        center_bin = min(int((band + 0.5) / K * n_bins), n_bins - 1)
        uniform_centroid[center_bin] = 1.0
        uniform_total += emd_1d(flop_hists[i], uniform_centroid)
    avg_uniform = uniform_total / len(flop_hists)

    print(f"  Avg intra-cluster EMD (EMD clusters): {avg_emd:.4f}")
    print(f"  Avg intra-cluster EMD (uniform bands): {avg_uniform:.4f}")
    improvement = (avg_uniform - avg_emd) / avg_uniform * 100
    print(f"  Improvement: {improvement:.1f}%")

    # --- Save ---
    print("\n[Save]")
    with open(CENTROIDS_FILE, 'w') as f:
        json.dump(all_centroids, f, indent=2)
    print(f"  Centroids: {CENTROIDS_FILE}")

    with open(PREFLOP_TABLE_FILE, 'w') as f:
        json.dump(preflop_table, f, indent=2)
    print(f"  Preflop table: {PREFLOP_TABLE_FILE}")

    print(f"\n  Done. {K} clusters × {n_bins} bins × 4 streets")
    return all_centroids, preflop_table


def load_centroids() -> dict[str, np.ndarray]:
    """Load precomputed centroids from JSON."""
    with open(CENTROIDS_FILE) as f:
        data = json.load(f)
    return {street: np.array(cents) for street, cents in data.items()}


def load_preflop_table() -> dict[str, int]:
    """Load preflop hand → cluster_id lookup table."""
    with open(PREFLOP_TABLE_FILE) as f:
        return json.load(f)


def nearest_centroid(hist: np.ndarray, centroids: np.ndarray) -> int:
    """Find the nearest centroid by EMD distance. Returns cluster index."""
    best_k = 0
    best_d = emd_1d(hist, centroids[0])
    for k in range(1, len(centroids)):
        d = emd_1d(hist, centroids[k])
        if d < best_d:
            best_d = d
            best_k = k
    return best_k


def compute_equity_boundaries(centroids: np.ndarray, n_bins: int = N_BINS) -> list[float]:
    """Precompute equity boundaries from sorted centroids for fast lookup.

    Since centroids are sorted by mean equity, we compute the midpoint
    between adjacent centroid means. A raw equity value can then be
    bucketed with a simple comparison chain — no histogram or EMD needed.

    Returns list of K-1 boundary values.
    """
    means = [sum(centroids[k][b] * (b + 0.5) / n_bins for b in range(n_bins))
             for k in range(len(centroids))]
    boundaries = [(means[i] + means[i + 1]) / 2.0 for i in range(len(means) - 1)]
    return boundaries


def fast_emd_bucket(equity: float, boundaries: list[float]) -> int:
    """Fast EMD bucket from raw equity using precomputed boundaries.

    O(K) comparison chain — no MC rollouts or histogram computation.
    """
    for k, boundary in enumerate(boundaries):
        if equity < boundary:
            return k
    return len(boundaries)  # Last cluster


# Cached boundaries per street (lazy-loaded)
_EQUITY_BOUNDARIES = None


def load_equity_boundaries() -> dict[str, list[float]]:
    """Load centroids and compute equity boundaries for fast lookup."""
    global _EQUITY_BOUNDARIES
    if _EQUITY_BOUNDARIES is not None:
        return _EQUITY_BOUNDARIES

    centroids = load_centroids()
    _EQUITY_BOUNDARIES = {}
    for street, cents in centroids.items():
        _EQUITY_BOUNDARIES[street] = compute_equity_boundaries(cents)
    return _EQUITY_BOUNDARIES


if __name__ == '__main__':
    build_all_centroids()
