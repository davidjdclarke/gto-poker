"""
Embedding CFR: Continuous infoset representations (WS5b prototype).

Trains a small MLP to map hand features → 16D embedding space, then uses
K-nearest centroid interpolation at eval time to smooth bucket boundaries.

Architecture:
    21 inputs → 32 hidden (ReLU) → 16 embedding (ReLU) → 120 logits (training only)

The embedding layer captures strategy-relevant hand similarity. At eval time,
the classification head is dropped and strategies are interpolated from the
K nearest bucket centroids in embedding space.

No external dependencies beyond numpy. Training uses Adam optimizer with
early stopping — converges in ~20 epochs on 50k samples.
"""
import json
import logging
import random as _rng
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase / hand-type constants (avoid circular import from abstraction)
# ---------------------------------------------------------------------------
PHASES = ("preflop", "flop", "turn", "river")
_PHASE_INDEX = {p: i for i, p in enumerate(PHASES)}
_NUM_HAND_TYPES = 15


# ===================================================================
# Feature extraction
# ===================================================================

def extract_features(hole_cards, community_cards, phase,
                     equity_float=None, texture=None):
    """Build 21D feature vector for the embedding MLP.

    Components:
        [0]       equity_float  — raw equity in [0, 1]
        [1:16]    hand_type     — one-hot (15 dims)
        [16:20]   phase         — one-hot (4 dims)
        [20]      texture       — board texture in [0, 1] (0 for preflop)

    Args:
        hole_cards: list of Card objects
        community_cards: list of Card objects (may be empty for preflop)
        phase: str, one of "preflop", "flop", "turn", "river"
        equity_float: optional precomputed equity (skips MC if provided)
        texture: optional precomputed texture int (0-4)

    Returns:
        np.ndarray of shape (21,)
    """
    features = np.zeros(21, dtype=np.float64)

    # --- equity float ---
    if equity_float is not None:
        features[0] = equity_float
    else:
        from eval_harness.fast_equity import fast_equity_float
        features[0] = fast_equity_float(hole_cards, community_cards)

    # --- hand type one-hot ---
    from server.gto.abstraction import classify_hand_type
    c1, c2 = hole_cards
    ht = int(classify_hand_type(c1.rank, c2.rank, c1.suit == c2.suit))
    features[1 + ht] = 1.0

    # --- phase one-hot ---
    pi = _PHASE_INDEX.get(phase, 0)
    features[16 + pi] = 1.0

    # --- texture ---
    if texture is not None:
        features[20] = texture / 4.0
    elif community_cards:
        from server.gto.board_texture import classify_board_texture
        features[20] = int(classify_board_texture(community_cards)) / 4.0

    return features


# ===================================================================
# MLP model (numpy only)
# ===================================================================

class EmbeddingMLP:
    """3-layer MLP: 21 → 32 (ReLU) → 16 (ReLU) → num_buckets (softmax).

    At inference time only layers 1-2 are used (returns 16D embedding).
    The classification head (layer 3) is used only during training.
    """

    def __init__(self, input_dim=21, hidden_dim=32, embed_dim=16,
                 num_buckets=120):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_buckets = num_buckets

        # Xavier initialization
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(embed_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(embed_dim)
        self.W3 = np.random.randn(num_buckets, embed_dim) * np.sqrt(2.0 / embed_dim)
        self.b3 = np.zeros(num_buckets)

    def forward(self, x):
        """Compute 16D embedding (inference path).

        Args:
            x: np.ndarray, shape (N, 21) or (21,)
        Returns:
            np.ndarray, shape (N, 16) or (16,)
        """
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]
        h1 = np.maximum(0, x @ self.W1.T + self.b1)       # (N, 32)
        h2 = np.maximum(0, h1 @ self.W2.T + self.b2)      # (N, 16)
        return h2[0] if single else h2

    def forward_classify(self, x):
        """Full forward pass returning (embedding, logits, intermediates).

        Used during training for backprop.

        Returns:
            (h1, h2, logits) where h1=(N,32), h2=(N,16), logits=(N,num_buckets)
        """
        h1 = np.maximum(0, x @ self.W1.T + self.b1)       # (N, 32)
        h2 = np.maximum(0, h1 @ self.W2.T + self.b2)      # (N, 16)
        logits = h2 @ self.W3.T + self.b3                  # (N, num_buckets)
        return h1, h2, logits

    def save(self, path):
        """Save model weights and metadata as JSON."""
        data = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
            "num_buckets": self.num_buckets,
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "W3": self.W3.tolist(),
            "b3": self.b3.tolist(),
        }
        Path(path).write_text(json.dumps(data))
        logger.info("Saved embedding model to %s", path)

    @classmethod
    def load(cls, path):
        """Load model weights from JSON.

        Returns:
            dict with 'model' (EmbeddingMLP) and optionally 'centroids' (ndarray)
        """
        data = json.loads(Path(path).read_text())
        model = cls(
            input_dim=data["input_dim"],
            hidden_dim=data["hidden_dim"],
            embed_dim=data["embed_dim"],
            num_buckets=data["num_buckets"],
        )
        model.W1 = np.array(data["W1"])
        model.b1 = np.array(data["b1"])
        model.W2 = np.array(data["W2"])
        model.b2 = np.array(data["b2"])
        model.W3 = np.array(data["W3"])
        model.b3 = np.array(data["b3"])
        result = {"model": model}
        if "centroids" in data:
            result["centroids"] = np.array(data["centroids"])
        return result

    def save_with_centroids(self, path, centroids):
        """Save model weights + centroid array as JSON."""
        data = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
            "num_buckets": self.num_buckets,
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "W3": self.W3.tolist(),
            "b3": self.b3.tolist(),
            "centroids": centroids.tolist(),
        }
        Path(path).write_text(json.dumps(data))
        logger.info("Saved embedding model + centroids to %s", path)


# ===================================================================
# Training data generation
# ===================================================================

def _deal_random_cards(phase, rng):
    """Deal random hole cards + community for a given phase.

    Returns:
        (hole_cards, community_cards) as lists of Card objects
    """
    from server.deck import Card, SUITS
    all_cards = [(r, s) for r in range(2, 15) for s in SUITS]
    rng.shuffle(all_cards)

    hole = [Card(all_cards[0][0], all_cards[0][1]),
            Card(all_cards[1][0], all_cards[1][1])]

    n_community = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}[phase]
    community = [Card(all_cards[2 + i][0], all_cards[2 + i][1])
                 for i in range(n_community)]
    return hole, community


def _generate_worker(args):
    """Worker for parallel data generation. Returns (X_chunk, y_chunk)."""
    chunk_size, simulations, seed = args
    rng = _rng.Random(seed)

    from eval_harness.fast_equity import fast_bucket, build_preflop_cache, _ehs2_fast
    from server.gto.abstraction import classify_hand_type, make_bucket, NUM_EQUITY_BUCKETS

    build_preflop_cache(simulations=200)

    X_list = []
    y_list = []

    for _ in range(chunk_size):
        phase = rng.choice(PHASES)
        hole, community = _deal_random_cards(phase, rng)

        # Compute equity float
        if phase == "preflop":
            ehs2 = _ehs2_fast(hole, [], simulations=simulations)
        else:
            ehs2 = _ehs2_fast(hole, community, simulations=simulations)
        eq_float = ehs2 ** 0.5

        # Compute bucket label (ground truth)
        bucket = fast_bucket(hole, community, simulations=simulations)

        # Extract features using precomputed equity
        features = extract_features(hole, community, phase,
                                    equity_float=eq_float)
        X_list.append(features)
        y_list.append(bucket)

    return np.array(X_list), np.array(y_list, dtype=np.int64)


def generate_training_data(num_samples=50000, simulations=50,
                           num_workers=1, seed=42):
    """Generate (features, bucket_label) training pairs.

    Uses parallel workers and Cython equity for speed.

    Args:
        num_samples: total number of samples to generate
        simulations: MC simulations per equity computation (50 is fine for training)
        num_workers: parallel workers for data generation
        seed: base random seed

    Returns:
        (X, y): X is (N, 21) features, y is (N,) bucket labels
    """
    if num_workers <= 1:
        X, y = _generate_worker((num_samples, simulations, seed))
        return X, y

    from multiprocessing import Pool

    chunk_size = num_samples // num_workers
    remainder = num_samples - chunk_size * num_workers

    args_list = []
    for i in range(num_workers):
        cs = chunk_size + (1 if i < remainder else 0)
        args_list.append((cs, simulations, seed + i))

    with Pool(num_workers) as pool:
        results = pool.map(_generate_worker, args_list)

    X = np.concatenate([r[0] for r in results], axis=0)
    y = np.concatenate([r[1] for r in results], axis=0)
    return X, y


# ===================================================================
# Training (Adam + cross-entropy)
# ===================================================================

def _log_softmax(logits):
    """Numerically stable log-softmax. logits shape (N, C)."""
    max_logits = logits.max(axis=1, keepdims=True)
    shifted = logits - max_logits
    log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return shifted - log_sum_exp


def _softmax(logits):
    """Numerically stable softmax. logits shape (N, C)."""
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


class AdamOptimizer:
    """Adam optimizer for a list of parameter arrays."""

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads, lr_override=None):
        """Update parameters with gradients."""
        self.t += 1
        lr = lr_override if lr_override is not None else self.lr
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


def train_embedding(X, y, model, epochs=30, lr=0.001, batch_size=256,
                    val_fraction=0.1, patience=5, warmup_epochs=3,
                    verbose=True):
    """Train embedding MLP via cross-entropy classification with Adam.

    Args:
        X: (N, 21) feature array
        y: (N,) bucket labels
        model: EmbeddingMLP instance
        epochs: maximum training epochs
        lr: learning rate for Adam
        batch_size: mini-batch size
        val_fraction: fraction of data for validation (early stopping)
        patience: early stopping patience (epochs without improvement)
        warmup_epochs: linear LR warmup period
        verbose: print progress

    Returns:
        dict with training history (losses, best_epoch, final_accuracy)
    """
    N = len(X)
    n_val = max(1, int(N * val_fraction))
    n_train = N - n_val

    # Shuffle and split
    idx = np.random.permutation(N)
    X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
    X_val, y_val = X[idx[n_train:]], y[idx[n_train:]]

    params = [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3]
    optimizer = AdamOptimizer(params, lr=lr)

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    try:
        from tqdm import trange
        epoch_iter = trange(epochs, desc="Training", disable=not verbose)
    except ImportError:
        epoch_iter = range(epochs)

    for epoch in epoch_iter:
        # Learning rate warmup
        if epoch < warmup_epochs:
            current_lr = lr * (epoch + 1) / warmup_epochs
        else:
            current_lr = lr

        # Shuffle training data
        perm = np.random.permutation(n_train)
        X_train = X_train[perm]
        y_train = y_train[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            xb = X_train[start:end]
            yb = y_train[start:end]
            bs = end - start

            # Forward
            h1, h2, logits = model.forward_classify(xb)
            log_probs = _log_softmax(logits)
            loss = -log_probs[np.arange(bs), yb].mean()
            epoch_loss += loss * bs

            # Backward
            probs = _softmax(logits)
            d_logits = probs.copy()
            d_logits[np.arange(bs), yb] -= 1.0
            d_logits /= bs  # (bs, num_buckets)

            # Layer 3 gradients
            dW3 = d_logits.T @ h2          # (num_buckets, 16)
            db3 = d_logits.sum(axis=0)     # (num_buckets,)

            # Backprop through layer 3
            d_h2 = d_logits @ model.W3     # (bs, 16)
            d_h2 *= (h2 > 0)               # ReLU derivative

            # Layer 2 gradients
            dW2 = d_h2.T @ h1              # (16, 32)
            db2 = d_h2.sum(axis=0)         # (16,)

            # Backprop through layer 2
            d_h1 = d_h2 @ model.W2         # (bs, 32)
            d_h1 *= (h1 > 0)               # ReLU derivative

            # Layer 1 gradients
            dW1 = d_h1.T @ xb              # (32, 21)
            db1 = d_h1.sum(axis=0)         # (32,)

            optimizer.step([dW1, db1, dW2, db2, dW3, db3],
                           lr_override=current_lr)
            n_batches += 1

        train_loss = epoch_loss / n_train

        # Validation
        _, _, val_logits = model.forward_classify(X_val)
        val_log_probs = _log_softmax(val_logits)
        val_loss = -val_log_probs[np.arange(n_val), y_val].mean()
        val_preds = val_logits.argmax(axis=1)
        val_acc = (val_preds == y_val).mean()

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        if hasattr(epoch_iter, 'set_postfix'):
            epoch_iter.set_postfix(
                loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                acc=f"{val_acc:.1%}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = [p.copy() for p in params]
        elif epoch - best_epoch >= patience:
            if verbose:
                logger.info("Early stopping at epoch %d (best: %d)", epoch, best_epoch)
            break

    # Restore best weights
    if best_weights is not None:
        for p, bw in zip(params, best_weights):
            p[:] = bw

    history["best_epoch"] = best_epoch
    history["final_val_acc"] = float(history["val_acc"][best_epoch])
    return history


# ===================================================================
# Centroid computation
# ===================================================================

def compute_bucket_centroids(model, X, y, num_buckets=120):
    """Compute mean embedding for each bucket.

    Args:
        model: trained EmbeddingMLP
        X: (N, 21) feature array
        y: (N,) bucket labels
        num_buckets: total number of buckets

    Returns:
        centroids: (num_buckets, embed_dim) array
    """
    embeddings = model.forward(X)  # (N, embed_dim)
    centroids = np.zeros((num_buckets, model.embed_dim), dtype=np.float64)
    counts = np.zeros(num_buckets, dtype=np.int64)

    for b in range(num_buckets):
        mask = y == b
        count = mask.sum()
        if count > 0:
            centroids[b] = embeddings[mask].mean(axis=0)
            counts[b] = count

    # For buckets with no samples, use the nearest populated centroid
    empty = counts == 0
    if empty.any():
        populated = ~empty
        pop_idx = np.where(populated)[0]
        for b in np.where(empty)[0]:
            # Find nearest populated bucket by index distance
            dists = np.abs(pop_idx - b)
            nearest = pop_idx[dists.argmin()]
            centroids[b] = centroids[nearest]
        logger.info("Filled %d empty bucket centroids from nearest neighbors",
                     empty.sum())

    return centroids


# ===================================================================
# Eval-time strategy interpolation
# ===================================================================

def embedding_strategy(model, centroids, trainer, features, phase,
                       history, position, K=3, texture=0):
    """Look up strategy by interpolating K nearest bucket centroids.

    Args:
        model: trained EmbeddingMLP
        centroids: (num_buckets, 16) centroid array
        trainer: CFRTrainer with loaded strategy
        features: (21,) feature vector from extract_features()
        phase: game phase string
        history: abstract history tuple
        position: 'ip' or 'oop'
        K: number of nearest centroids to blend
        texture: board texture int for EMD mode

    Returns:
        dict[int, float]: blended action probability distribution
    """
    embedding = model.forward(features)  # (16,)

    # Compute distances to all centroids — vectorized
    dists = np.linalg.norm(centroids - embedding, axis=1)  # (num_buckets,)

    # Find K nearest (O(N) partial sort)
    K = min(K, len(centroids))
    top_k_idx = np.argpartition(dists, K)[:K]
    top_k_dists = dists[top_k_idx]

    # Inverse-distance weights (handle zero distance: use that bucket exactly)
    if top_k_dists.min() < 1e-10:
        # Exact match — use the zero-distance bucket
        exact_idx = top_k_idx[top_k_dists.argmin()]
        return trainer.get_strategy(phase, int(exact_idx), history, position)

    weights = 1.0 / top_k_dists
    weights /= weights.sum()

    # Look up and blend strategies
    blended = {}
    for i, (bucket_idx, w) in enumerate(zip(top_k_idx, weights)):
        strat = trainer.get_strategy(phase, int(bucket_idx), history, position)
        for action, prob in strat.items():
            blended[action] = blended.get(action, 0.0) + w * prob

    # Renormalize (strategies should already sum to ~1 but be safe)
    total = sum(blended.values())
    if total > 1e-10:
        blended = {a: v / total for a, v in blended.items()}

    return blended
