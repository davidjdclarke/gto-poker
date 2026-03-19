#!/usr/bin/env python3
"""
Train the Embedding CFR model (WS5b).

Generates training data from random hand deals, trains a small MLP to classify
hands into buckets, and saves the model weights + centroid embeddings for use
at eval time with --mapping embedding.

Usage:
    # Full pipeline (generate data + train + save)
    venv/bin/python train_embedding.py \
        --strategy experiments/best/v9_B0_100M_allbots_positive.json \
        --samples 50000 --epochs 30 --workers 6 \
        --output server/gto/embedding_weights.json

    # Save/load cached training data to skip generation on re-runs
    venv/bin/python train_embedding.py \
        --strategy experiments/best/v9_B0_100M_allbots_positive.json \
        --samples 50000 --workers 6 \
        --cache-data server/gto/embedding_train_data.npz \
        --output server/gto/embedding_weights.json

    # Load cached data only (no strategy needed)
    venv/bin/python train_embedding.py \
        --cache-data server/gto/embedding_train_data.npz \
        --load-cache-only \
        --output server/gto/embedding_weights.json
"""
import argparse
import time
import sys
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Train Embedding CFR model (WS5b)")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Path to strategy JSON (for bucket count detection)")
    parser.add_argument("--samples", type=int, default=50000,
                        help="Number of training samples to generate")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Max training epochs (early stopping enabled)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for data generation")
    parser.add_argument("--simulations", type=int, default=50,
                        help="MC simulations per equity computation in data gen")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str,
                        default="server/gto/embedding_weights.json",
                        help="Output path for model weights + centroids")
    parser.add_argument("--cache-data", type=str, default=None,
                        help="Path to cache training data as .npz (save/load)")
    parser.add_argument("--load-cache-only", action="store_true",
                        help="Only load cached data (skip generation even if --strategy given)")
    parser.add_argument("--emd-texture", action="store_true",
                        help="Enable EMD+texture mode (180 buckets)")
    parser.add_argument("--hidden-dim", type=int, default=32,
                        help="Hidden layer dimension")
    parser.add_argument("--embed-dim", type=int, default=16,
                        help="Embedding dimension")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Enable EMD mode if requested
    if args.emd_texture:
        from server.gto.abstraction import enable_emd_mode
        enable_emd_mode()
        import server.gto.equity as _eq
        _eq.EMD_MODE_ENABLED = True

    from server.gto.abstraction import NUM_BUCKETS

    print(f"Embedding CFR Training (WS5b)")
    print(f"  Buckets: {NUM_BUCKETS}")
    print(f"  Samples: {args.samples}")
    print(f"  Epochs:  {args.epochs} (patience={args.patience})")
    print(f"  Workers: {args.workers}")
    print(f"  Output:  {args.output}")
    print()

    # ---------------------------------------------------------------
    # Step 1: Generate or load training data
    # ---------------------------------------------------------------
    X, y = None, None

    if args.cache_data and args.load_cache_only and os.path.exists(args.cache_data):
        print(f"Loading cached training data from {args.cache_data} ...")
        data = np.load(args.cache_data)
        X, y = data["X"], data["y"]
        print(f"  Loaded {len(X)} samples")
    elif args.cache_data and os.path.exists(args.cache_data) and not args.load_cache_only:
        print(f"Loading cached training data from {args.cache_data} ...")
        data = np.load(args.cache_data)
        X, y = data["X"], data["y"]
        print(f"  Loaded {len(X)} samples")
    else:
        from server.gto.embedding_model import generate_training_data

        print(f"Generating {args.samples} training samples ({args.workers} workers, "
              f"{args.simulations} sims) ...")
        t0 = time.time()
        X, y = generate_training_data(
            num_samples=args.samples,
            simulations=args.simulations,
            num_workers=args.workers,
            seed=args.seed,
        )
        elapsed = time.time() - t0
        print(f"  Generated {len(X)} samples in {elapsed:.1f}s")

        # Save cache if requested
        if args.cache_data:
            np.savez_compressed(args.cache_data, X=X, y=y)
            print(f"  Cached to {args.cache_data}")

    # Print bucket distribution summary
    unique, counts = np.unique(y, return_counts=True)
    print(f"  Unique buckets seen: {len(unique)} / {NUM_BUCKETS}")
    print(f"  Bucket counts: min={counts.min()}, max={counts.max()}, "
          f"mean={counts.mean():.1f}, median={np.median(counts):.1f}")
    print()

    # ---------------------------------------------------------------
    # Step 2: Train MLP
    # ---------------------------------------------------------------
    from server.gto.embedding_model import EmbeddingMLP, train_embedding

    model = EmbeddingMLP(
        input_dim=21,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_buckets=NUM_BUCKETS,
    )

    print(f"Training MLP ({model.input_dim} → {model.hidden_dim} → "
          f"{model.embed_dim} → {model.num_buckets}) ...")
    t0 = time.time()
    history = train_embedding(
        X, y, model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Best epoch: {history['best_epoch']}")
    print(f"  Val accuracy: {history['final_val_acc']:.1%}")
    print()

    # ---------------------------------------------------------------
    # Step 3: Compute centroids
    # ---------------------------------------------------------------
    from server.gto.embedding_model import compute_bucket_centroids

    print("Computing bucket centroids ...")
    centroids = compute_bucket_centroids(model, X, y, num_buckets=NUM_BUCKETS)

    # Sanity check: nearby equity buckets should have closer centroids
    print("  Centroid distance sanity check:")
    for eq_b in range(0, min(7, NUM_BUCKETS // 15)):
        b1 = eq_b * 15  # hand_type=0
        b2 = (eq_b + 1) * 15
        d = np.linalg.norm(centroids[b1] - centroids[b2])
        print(f"    eq_bucket {eq_b} ↔ {eq_b+1}: distance = {d:.3f}")
    print()

    # ---------------------------------------------------------------
    # Step 4: Save model + centroids
    # ---------------------------------------------------------------
    model.save_with_centroids(args.output, centroids)
    file_size = os.path.getsize(args.output) / 1024
    print(f"Saved model + centroids to {args.output} ({file_size:.0f} KB)")
    print("Done.")


if __name__ == "__main__":
    main()
