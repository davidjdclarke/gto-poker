"""
Counterfactual Regret Minimization (CFR+) with Monte Carlo sampling.

Implements CFR+ (Tammelin 2014) with external sampling MCCFR
(Lanctot et al. 2009) for abstracted multi-street poker.

Key features:
- Multi-street traversal: preflop → flop → turn → river connected
  so that flop buckets are conditioned on preflop action.
- 2D hand bucketing: (equity, hand_type) preserves strategic distinctions.
- Phase-aware action abstraction: preflop uses open/3bet/4bet; postflop
  uses pot-fraction bets.
- CFR+ regret-matching+ with linear weighted averaging [Tammelin 2014].
- Alternating updates for stability.

References:
  [1] Zinkevich et al. "Regret Minimization in Games with Incomplete
      Information." NIPS 2007.
  [2] Lanctot et al. "Monte Carlo Sampling for Regret Minimization
      in Extensive Games." NIPS 2009.
  [3] Tammelin. "Solving Large Imperfect Information Games Using CFR+." 2014.
"""
import json
import os
import random
import numpy as np
from pathlib import Path
from server.gto.abstraction import (
    Action, InfoSet, NUM_BUCKETS, NUM_EQUITY_BUCKETS, NUM_HAND_TYPES,
    get_available_actions, count_raises, make_bucket, decode_bucket,
    get_position,
)

# Try to import Cython-accelerated training
try:
    from server.gto import cfr_fast as _cfr_fast
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

STRATEGY_VERSION = 6  # v6: 2/3pot, overbet, donk bets, all-in audit

def _parallel_worker(worker_id, n_iters, start_iter, avg_delay, seed,
                     phase_schedule_mode=1, allin_dampen_mode=1,
                     adaptive_averaging=0, regret_discount=1.0,
                     weight_schedule_mode=0, weight_schedule_param=1.0,
                     action_grid_size=0, solver_mode=0,
                     vr_mccfr=0, zhang_alpha=1.5, zhang_c=1.0,
                     vr_mccfr_warmup=0, emd_mode=0):
    """Worker function for parallel CFR training.

    Runs in a forked child process. The node_pool is in shared mmap memory,
    so regret/strategy updates are visible to all processes (benign races).
    node_index is inherited read-only via COW. No new nodes are created.
    """
    _cfr_fast.set_no_create_mode(True)
    _cfr_fast.set_worker_id(worker_id)
    _cfr_fast.train_fast(n_iters, start_iter=start_iter,
                         averaging_delay=avg_delay, seed=seed,
                         phase_schedule_mode=phase_schedule_mode,
                         allin_dampen_mode=allin_dampen_mode,
                         adaptive_averaging=adaptive_averaging,
                         regret_discount=regret_discount,
                         weight_schedule_mode=weight_schedule_mode,
                         weight_schedule_param=weight_schedule_param,
                         action_grid_size=action_grid_size,
                         solver_mode=solver_mode,
                         vr_mccfr=vr_mccfr,
                         zhang_alpha=zhang_alpha,
                         zhang_c=zhang_c,
                         vr_mccfr_warmup=vr_mccfr_warmup,
                         emd_mode=emd_mode)

PHASES = ['preflop', 'flop', 'turn', 'river']

# Phase schedules (configurable for ablation)
PHASE_SCHEDULE_2X = [0, 1, 1, 2, 2, 3]        # original 2x flop/turn
PHASE_SCHEDULE_3X = [0, 1, 1, 1, 2, 2, 2, 3]  # current default 3x flop/turn

# Default: 3x schedule. Matches the Cython hot path.
PHASE_SCHEDULE = PHASE_SCHEDULE_3X


class CFRNode:
    """A node in the CFR game tree, representing an information set."""

    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions, dtype=np.float64)
        self.strategy_sum = np.zeros(num_actions, dtype=np.float64)

    def get_strategy(self) -> np.ndarray:
        """
        Compute current strategy via regret-matching+.
        Positive regrets are normalized into a probability distribution.
        Since CFR+ floors regrets, regret_sum is already non-negative.
        """
        normalizing_sum = self.regret_sum.sum()

        if normalizing_sum > 0:
            return self.regret_sum / normalizing_sum
        else:
            return np.ones(self.num_actions) / self.num_actions

    def update_regrets(self, regrets: np.ndarray, discount: float = 1.0):
        """
        Update cumulative regrets with optional DCFR discounting + CFR+ flooring.
        R^{+,T}(I,a) = max(discount * R^{+,T-1}(I,a) + r^T(I,a), 0)
        discount=1.0 gives standard CFR+. discount<1.0 gives DCFR.
        """
        if discount < 1.0:
            self.regret_sum *= discount
        self.regret_sum += regrets
        np.maximum(self.regret_sum, 0, out=self.regret_sum)

    def accumulate_strategy(self, strategy: np.ndarray, weight: float):
        """
        Accumulate weighted strategy for averaging.
        CFR+ uses w^T = max(T - d, 0) weighting.
        [Tammelin 2014, Algorithm 1 line 37]
        """
        self.strategy_sum += weight * strategy

    def get_average_strategy(self) -> np.ndarray:
        """
        Return the average strategy over all iterations.
        This converges to Nash Equilibrium.
        """
        normalizing_sum = self.strategy_sum.sum()
        if normalizing_sum > 0:
            return self.strategy_sum / normalizing_sum
        return np.ones(self.num_actions) / self.num_actions

    def to_dict(self) -> dict:
        return {
            'regret_sum': self.regret_sum.tolist(),
            'strategy_sum': self.strategy_sum.tolist(),
            'num_actions': self.num_actions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CFRNode':
        node = cls(d['num_actions'])
        node.regret_sum = np.array(d['regret_sum'], dtype=np.float64)
        node.strategy_sum = np.array(d['strategy_sum'], dtype=np.float64)
        return node


class CFRTrainer:
    """
    Trains poker strategies using CFR+ with Monte Carlo sampling.

    The game is modeled as a multi-street two-player zero-sum game with:
    - Connected streets: preflop → flop → turn → river
    - 2D hand buckets: (equity_bucket, hand_type)
    - Phase-aware action abstraction
    - Utility computed from cumulative pot investment

    Based on:
      [1] Zinkevich et al. (2007) - Counterfactual Regret Minimization
      [2] Lanctot et al. (2009) - External Sampling MCCFR
      [3] Tammelin (2014) - CFR+ with regret-matching+ and linear averaging
    """

    def __init__(self):
        self.nodes: dict[str, CFRNode] = {}
        self.iterations = 0

    def train(self, num_iterations: int = 10000,
              averaging_delay: int = 0,
              phase_schedule_mode: int = 1,
              allin_dampen_mode: int = 1,
              adaptive_averaging: int = 0,
              regret_discount: float = 1.0,
              weight_schedule_mode: int = 0,
              weight_schedule_param: float = 1.0,
              sampling: str = 'external',
              progress_callback=None,
              chunk_size: int = 5000,
              num_workers: int = 1,
              action_grid_size: int = 0,
              solver_mode: int = 0,
              vr_mccfr: int = 0,
              zhang_alpha: float = 1.5,
              zhang_c: float = 1.0,
              vr_mccfr_warmup: int = 0,
              emd_mode: int = 0):
        """
        Run CFR+ training with per-street traversal and correlated buckets.

        Each iteration trains one street at a time. Cross-street connection
        is maintained through correlated bucket sampling: hand type stays
        fixed across streets while equity drifts, so the solver learns
        that a hand's structural type persists through the game.

        At street boundaries, continuation value is approximated by
        sampling the next street with correlated buckets.

        Uses Cython-accelerated inner loop when available (50x+ faster).

        Args:
            progress_callback: Optional callable(iterations_done, total, nodes)
                               called periodically for progress reporting.
            chunk_size: Iterations per progress update (Cython only).
            num_workers: Number of parallel workers (>1 uses shared memory).
            action_grid_size: 0=use current, 13=v6/B0, 16=v9 expanded.
        """
        if HAS_CYTHON and sampling == 'external':
            if num_workers > 1:
                self._train_cython_parallel(num_iterations, averaging_delay,
                                            num_workers=num_workers,
                                            progress_callback=progress_callback,
                                            chunk_size=chunk_size,
                                            phase_schedule_mode=phase_schedule_mode,
                                            allin_dampen_mode=allin_dampen_mode,
                                            adaptive_averaging=adaptive_averaging,
                                            regret_discount=regret_discount,
                                            weight_schedule_mode=weight_schedule_mode,
                                            weight_schedule_param=weight_schedule_param,
                                            action_grid_size=action_grid_size,
                                            solver_mode=solver_mode,
                                            vr_mccfr=vr_mccfr,
                                            zhang_alpha=zhang_alpha,
                                            zhang_c=zhang_c,
                                            vr_mccfr_warmup=vr_mccfr_warmup,
                                            emd_mode=emd_mode)
            else:
                self._train_cython(num_iterations, averaging_delay,
                                   progress_callback=progress_callback,
                                   chunk_size=chunk_size,
                                   phase_schedule_mode=phase_schedule_mode,
                                   allin_dampen_mode=allin_dampen_mode,
                                   adaptive_averaging=adaptive_averaging,
                                   regret_discount=regret_discount,
                                   weight_schedule_mode=weight_schedule_mode,
                                   weight_schedule_param=weight_schedule_param,
                                   action_grid_size=action_grid_size,
                                   solver_mode=solver_mode,
                                   vr_mccfr=vr_mccfr,
                                   zhang_alpha=zhang_alpha,
                                   zhang_c=zhang_c,
                                   vr_mccfr_warmup=vr_mccfr_warmup,
                                   emd_mode=emd_mode)
            return

        # Python fallback: store config for _cfr_single_street access
        self._allin_dampen_mode = allin_dampen_mode
        self._regret_discount = regret_discount
        self._weight_schedule_mode = weight_schedule_mode
        self._weight_schedule_param = weight_schedule_param
        schedule = PHASE_SCHEDULE_3X if phase_schedule_mode == 1 else PHASE_SCHEDULE_2X
        start_iter = self.iterations

        for i in range(num_iterations):
            t = start_iter + i + 1
            if weight_schedule_mode == 3:
                # Scheduled DCFR
                alpha = 1.5
                gamma = weight_schedule_param if weight_schedule_param != 1.0 else 2.0
                weight = (t / (t + 1.0)) ** gamma
                self._regret_discount = (t ** alpha) / (t ** alpha + 1.0)
            else:
                weight = max(t - averaging_delay, 0)

            # Train each street with correlated buckets
            for phase_idx in schedule:
                phase = PHASES[phase_idx]
                buckets_p0 = self._sample_street_buckets()
                buckets_p1 = self._sample_street_buckets()
                b0 = buckets_p0[phase]
                b1 = buckets_p1[phase]

                if sampling == 'external':
                    for traverser in [0, 1]:
                        self._cfr_single_street(
                            phase, b0, b1, (), traverser, weight,
                            buckets_p0, buckets_p1)
                else:  # vanilla
                    self._cfr_single_street_vanilla(
                        phase, b0, b1, (), 1.0, 1.0, weight,
                        buckets_p0, buckets_p1)

            self.iterations += 1

            if (i + 1) % 1000 == 0:
                print(f"CFR+ iteration {i + 1}/{num_iterations}")

    def _train_cython(self, num_iterations: int, averaging_delay: int,
                      progress_callback=None, chunk_size: int = 5000,
                      phase_schedule_mode: int = 1,
                      allin_dampen_mode: int = 1,
                      adaptive_averaging: int = 0,
                      regret_discount: float = 1.0,
                      weight_schedule_mode: int = 0,
                      weight_schedule_param: float = 1.0,
                      action_grid_size: int = 0,
                      solver_mode: int = 0,
                      vr_mccfr: int = 0,
                      zhang_alpha: float = 1.5,
                      zhang_c: float = 1.0,
                      vr_mccfr_warmup: int = 0,
                      emd_mode: int = 0):
        """Run training using Cython-accelerated CFR.

        Args:
            progress_callback: Optional callable(iterations_done, total, nodes)
                               called after each chunk for progress reporting.
            chunk_size: Number of iterations per chunk (controls update frequency).
            phase_schedule_mode: 0=2x, 1=3x (default).
            allin_dampen_mode: 0=old, 1=new (default).
            regret_discount: DCFR discount factor (1.0=standard CFR+, <1.0=DCFR).
            weight_schedule_mode: 0=linear, 1=exponential, 2=polynomial.
            weight_schedule_param: parameter for weight schedule (base/power).
            action_grid_size: 0=use current, 13=v6/B0, 16=v9 expanded.
            vr_mccfr_warmup: iters of standard CFR+ before VR-MCCFR activation.
        """
        # Set action grid before training
        if action_grid_size > 0:
            _cfr_fast.set_action_grid_size(action_grid_size)

        # Initialize Cython node pool
        _cfr_fast.init_pool()

        # Export existing nodes to Cython if we have any (continued training)
        if self.nodes:
            self._export_nodes_to_cython()

        seed = random.randint(0, 2**31 - 1)
        done = 0

        while done < num_iterations:
            batch = min(chunk_size, num_iterations - done)
            _cfr_fast.train_fast(
                batch,
                start_iter=self.iterations + done,
                averaging_delay=averaging_delay,
                seed=seed + done,
                phase_schedule_mode=phase_schedule_mode,
                allin_dampen_mode=allin_dampen_mode,
                adaptive_averaging=adaptive_averaging,
                regret_discount=regret_discount,
                weight_schedule_mode=weight_schedule_mode,
                weight_schedule_param=weight_schedule_param,
                action_grid_size=action_grid_size,
                solver_mode=solver_mode,
                vr_mccfr=vr_mccfr,
                zhang_alpha=zhang_alpha,
                zhang_c=zhang_c,
                vr_mccfr_warmup=vr_mccfr_warmup,
                emd_mode=emd_mode,
            )
            done += batch

            if progress_callback:
                progress_callback(done, num_iterations, _cfr_fast.get_node_count())

        # Import nodes back from Cython
        self._import_nodes_from_cython()
        self.iterations += num_iterations

    def _train_cython_parallel(self, num_iterations: int, averaging_delay: int,
                               num_workers: int = 4, warmup_frac: float = 0.005,
                               progress_callback=None, chunk_size: int = 5000,
                               phase_schedule_mode: int = 1,
                               allin_dampen_mode: int = 1,
                               adaptive_averaging: int = 0,
                               regret_discount: float = 1.0,
                               weight_schedule_mode: int = 0,
                               weight_schedule_param: float = 1.0,
                               action_grid_size: int = 0,
                               solver_mode: int = 0,
                               vr_mccfr: int = 0,
                               zhang_alpha: float = 1.5,
                               zhang_c: float = 1.0,
                               vr_mccfr_warmup: int = 0,
                               emd_mode: int = 0):
        """Parallel CFR+ using shared memory and multiprocessing.

        Phase 1 (warmup): Single-threaded training to discover all game tree
        nodes. This populates the shared node pool and index.

        Phase 2 (parallel): Fork workers that update the same shared memory
        node pool with benign races (lock-free). Each worker runs its own
        iteration range with a unique seed. Workers don't create new nodes.

        The benign-race approach works because CFR+ is robust to noisy
        regret updates — Facebook Research (2019) showed lock-free parallel
        MCCFR converges nearly as fast as sequential.
        """
        import multiprocessing
        import time as _time

        warmup_iters = max(10000, int(num_iterations * warmup_frac))
        parallel_iters = num_iterations - warmup_iters

        # Set action grid before training
        if action_grid_size > 0:
            _cfr_fast.set_action_grid_size(action_grid_size)

        # Pre-allocate shared memory pool (2M nodes, ~430 MB)
        _cfr_fast.init_pool_shared(10_000_000)

        # Export existing nodes if continuing training
        if self.nodes:
            self._export_nodes_to_cython()

        seed = random.randint(0, 2**31 - 1)

        # Phase 1: Warmup (single-threaded) — discover nodes
        print(f"  Phase 1: Warmup ({warmup_iters:,} iters, single-threaded)...")
        t0 = _time.time()
        done = 0
        while done < warmup_iters:
            batch = min(chunk_size, warmup_iters - done)
            _cfr_fast.train_fast(
                batch,
                start_iter=self.iterations + done,
                averaging_delay=averaging_delay,
                seed=seed + done,
                phase_schedule_mode=phase_schedule_mode,
                allin_dampen_mode=allin_dampen_mode,
                adaptive_averaging=adaptive_averaging,
                regret_discount=regret_discount,
                weight_schedule_mode=weight_schedule_mode,
                weight_schedule_param=weight_schedule_param,
                action_grid_size=action_grid_size,
                solver_mode=solver_mode,
                vr_mccfr=vr_mccfr,
                zhang_alpha=zhang_alpha,
                zhang_c=zhang_c,
                vr_mccfr_warmup=vr_mccfr_warmup,
                emd_mode=emd_mode,
            )
            done += batch
            if progress_callback:
                progress_callback(done, num_iterations,
                                  _cfr_fast.get_node_count())
        warmup_nodes = _cfr_fast.get_node_count()
        t1 = _time.time()
        print(f"  Warmup done: {warmup_nodes:,} nodes in {t1-t0:.1f}s")

        # Phase 2: Parallel training with shared memory
        iters_per_worker = parallel_iters // num_workers
        print(f"  Phase 2: Parallel ({parallel_iters:,} iters, "
              f"{num_workers} workers, {iters_per_worker:,} each)...")
        t2 = _time.time()

        processes = []
        for w in range(num_workers):
            start = self.iterations + warmup_iters + w * iters_per_worker
            w_iters = (iters_per_worker if w < num_workers - 1
                       else parallel_iters - w * iters_per_worker)
            w_seed = seed + warmup_iters + w * 1_000_000

            p = multiprocessing.Process(
                target=_parallel_worker,
                args=(w, w_iters, start, averaging_delay, w_seed,
                      phase_schedule_mode, allin_dampen_mode,
                      adaptive_averaging, regret_discount,
                      weight_schedule_mode, weight_schedule_param,
                      action_grid_size, solver_mode,
                      vr_mccfr, zhang_alpha, zhang_c,
                      vr_mccfr_warmup, emd_mode))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        t3 = _time.time()
        print(f"  Parallel done: {num_workers} workers in {t3-t2:.1f}s")

        # Import results from shared memory back to Python nodes
        self._import_nodes_from_cython()
        self.iterations += num_iterations

    def _export_nodes_to_cython(self):
        """Export Python nodes to Cython's flat array format."""
        # We need to re-encode string keys as int64 keys
        # This maps the same infosets using the Cython key encoding
        phase_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
        pos_map = {'oop': 0, 'ip': 1, '': 0}

        from server.gto.abstraction import EMD_MODE

        export_dict = {}
        for str_key, node in self.nodes.items():
            # Parse the string key:
            #   standard: phase:position:bucket:history
            #   EMD mode: phase:position:bucket:texture:history
            parts = str_key.split(':')
            if EMD_MODE and len(parts) == 5:
                phase_str, pos_str, bucket_str, texture_str, hist_str = parts
                texture = int(texture_str)
            elif len(parts) == 4:
                phase_str, pos_str, bucket_str, hist_str = parts
                texture = 0
            elif len(parts) == 3:
                phase_str, bucket_str, hist_str = parts
                pos_str = ''
                texture = 0
            else:
                continue

            phase = phase_map.get(phase_str, 0)
            position = pos_map.get(pos_str, 0)
            bucket = int(bucket_str)
            history = [int(c) for c in hist_str] if hist_str else []

            # Encode as int64 key (must match cfr_fast.pyx make_key)
            int_key = phase
            int_key = (int_key << 1) | position
            int_key = (int_key << 8) | bucket
            if EMD_MODE:
                int_key = (int_key << 2) | (texture & 3)
            for a in history:
                int_key = (int_key << 4) | a
            int_key = (int_key << 4) | len(history)

            export_dict[int_key] = {
                'num_actions': node.num_actions,
                'regret_sum': node.regret_sum.tolist(),
                'strategy_sum': node.strategy_sum.tolist(),
            }

        _cfr_fast.import_nodes(export_dict)
        print(f"Exported {len(export_dict)} nodes to Cython")

    def _import_nodes_from_cython(self):
        """Import nodes from Cython's flat arrays back to Python CFRNodes."""
        phase_names = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}
        pos_names = {0: 'oop', 1: 'ip'}

        cython_nodes = _cfr_fast.get_all_nodes()
        self.nodes = {}

        for int_key, data in cython_nodes.items():
            # Decode int64 key back to string key
            str_key = self._decode_int_key(int_key)
            if str_key is None:
                continue

            node = CFRNode(data['num_actions'])
            node.regret_sum = np.array(data['regret_sum'], dtype=np.float64)
            node.strategy_sum = np.array(data['strategy_sum'], dtype=np.float64)
            self.nodes[str_key] = node

        _cfr_fast.cleanup_pool()

    @staticmethod
    def _decode_int_key(int_key: int) -> str | None:
        """Decode a Cython int64 key back to the string format."""
        phase_names = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}
        pos_names = {0: 'oop', 1: 'ip'}

        # Extract history length (bottom 4 bits)
        hlen = int_key & 0xF
        int_key >>= 4

        # Extract history (hlen × 4 bits, reading from bottom)
        history = []
        for _ in range(hlen):
            history.append(int_key & 0xF)
            int_key >>= 4
        history.reverse()

        # Extract texture (2 bits) if EMD mode
        from server.gto.abstraction import EMD_MODE
        texture = 0
        if EMD_MODE:
            texture = int_key & 0x3
            int_key >>= 2

        # Extract bucket (8 bits)
        bucket = int_key & 0xFF
        int_key >>= 8

        # Extract position (1 bit)
        position = int_key & 0x1
        int_key >>= 1

        # Extract phase (2 bits)
        phase = int_key & 0x3

        phase_str = phase_names.get(phase, 'preflop')
        pos_str = pos_names.get(position, 'oop')
        hist_str = ''.join(str(a) for a in history)

        if EMD_MODE:
            return f"{phase_str}:{pos_str}:{bucket}:{texture}:{hist_str}"
        return f"{phase_str}:{pos_str}:{bucket}:{hist_str}"

    def _sample_street_buckets(self) -> dict[str, int]:
        """
        Sample a player's bucket for each street.

        Hand type is fixed (structural property of hole cards).
        Equity bucket can shift across streets as community cards
        change hand strength, modeled as a random walk.
        """
        hand_type = random.randint(0, NUM_HAND_TYPES - 1)
        preflop_eq = random.randint(0, NUM_EQUITY_BUCKETS - 1)

        buckets = {}
        eq = preflop_eq
        for phase in PHASES:
            buckets[phase] = make_bucket(eq, hand_type)
            # Equity can drift between streets (community cards change strength)
            # Random walk with bounds - models equity realization
            drift = random.choice([-1, 0, 0, 1])  # Slight drift
            eq = max(0, min(NUM_EQUITY_BUCKETS - 1, eq + drift))

        return buckets

    def _cfr_single_street(self, phase: str, bucket_p0: int, bucket_p1: int,
                            history: tuple, traverser: int, weight: float,
                            buckets_p0: dict = None,
                            buckets_p1: dict = None) -> float:
        """
        External sampling MCCFR for a single street with continuation value.

        Trains one street at a time. When the street completes (check-check
        or bet-call), computes a continuation value by recursing into the
        next street's trained strategy (if available), creating implicit
        cross-street coupling.

        Returns utility for the traversing player.
        """
        plays = len(history)
        acting_player = plays % 2

        # Fold terminal
        terminal = self._check_terminal(history, bucket_p0, bucket_p1, phase)
        if terminal is not None:
            return terminal if traverser == 0 else -terminal

        # Street completion -> continuation or showdown
        if self._is_street_complete(history):
            return self._continuation_value(
                phase, bucket_p0, bucket_p1, history, traverser,
                buckets_p0, buckets_p1)

        # Depth limit
        if plays >= 8:
            return self._continuation_value(
                phase, bucket_p0, bucket_p1, history, traverser,
                buckets_p0, buckets_p1)

        # Info set for acting player
        bucket = bucket_p0 if acting_player == 0 else bucket_p1
        position = get_position(acting_player, phase)
        has_bet = self._has_bet_to_call(history, phase)
        raise_count = count_raises(history, phase)
        can_raise = raise_count < 4

        actions = get_available_actions(has_bet, can_raise, phase, raise_count,
                                       history_len=len(history),
                                       eq_bucket=decode_bucket(bucket)[0])
        info_set = InfoSet(bucket, phase, history, position=position)
        key = info_set.key

        if key not in self.nodes:
            self.nodes[key] = CFRNode(len(actions))
        node = self.nodes[key]

        strategy = node.get_strategy()

        if acting_player != traverser:
            # Opponent: sample one action
            action_idx = np.random.choice(len(actions), p=strategy)
            new_history = history + (int(actions[action_idx]),)
            return self._cfr_single_street(
                phase, bucket_p0, bucket_p1, new_history,
                traverser, weight, buckets_p0, buckets_p1)

        # Traverser: explore all actions
        action_utilities = np.zeros(len(actions))
        for i, action in enumerate(actions):
            new_history = history + (int(action),)
            action_utilities[i] = self._cfr_single_street(
                phase, bucket_p0, bucket_p1, new_history,
                traverser, weight, buckets_p0, buckets_p1)

        node_utility = strategy @ action_utilities
        regrets = action_utilities - node_utility

        # All-in dampening (configurable for ablation)
        dampen_mode = getattr(self, '_allin_dampen_mode', 1)
        for i, action in enumerate(actions):
            if dampen_mode == 1:
                if action == Action.ALL_IN and raise_count == 0:
                    regrets[i] *= 0.5
            else:
                if action == Action.ALL_IN and raise_count < 2:
                    regrets[i] *= 0.7

        discount = getattr(self, '_regret_discount', 1.0)
        node.update_regrets(regrets, discount=discount)
        node.accumulate_strategy(strategy, weight)

        return node_utility

    def _cfr_single_street_vanilla(self, phase: str,
                                    bucket_p0: int, bucket_p1: int,
                                    history: tuple,
                                    p0: float, p1: float, weight: float,
                                    buckets_p0: dict = None,
                                    buckets_p1: dict = None) -> float:
        """Vanilla CFR+ for a single street. Returns utility for P0."""
        plays = len(history)
        acting_player = plays % 2

        terminal = self._check_terminal(history, bucket_p0, bucket_p1, phase)
        if terminal is not None:
            return terminal

        if self._is_street_complete(history):
            return self._continuation_value_p0(
                phase, bucket_p0, bucket_p1, history, buckets_p0, buckets_p1)

        if plays >= 8:
            return self._continuation_value_p0(
                phase, bucket_p0, bucket_p1, history, buckets_p0, buckets_p1)

        bucket = bucket_p0 if acting_player == 0 else bucket_p1
        position = get_position(acting_player, phase)
        has_bet = self._has_bet_to_call(history, phase)
        raise_count = count_raises(history, phase)
        can_raise = raise_count < 4

        actions = get_available_actions(has_bet, can_raise, phase, raise_count,
                                       history_len=len(history),
                                       eq_bucket=decode_bucket(bucket)[0])
        info_set = InfoSet(bucket, phase, history, position=position)
        key = info_set.key

        if key not in self.nodes:
            self.nodes[key] = CFRNode(len(actions))
        node = self.nodes[key]

        strategy = node.get_strategy()
        reach = p0 if acting_player == 0 else p1
        node.accumulate_strategy(strategy, reach * weight)

        action_utilities = np.zeros(len(actions))
        for i, action in enumerate(actions):
            new_history = history + (int(action),)
            if acting_player == 0:
                action_utilities[i] = self._cfr_single_street_vanilla(
                    phase, bucket_p0, bucket_p1, new_history,
                    p0 * strategy[i], p1, weight, buckets_p0, buckets_p1)
            else:
                action_utilities[i] = self._cfr_single_street_vanilla(
                    phase, bucket_p0, bucket_p1, new_history,
                    p0, p1 * strategy[i], weight, buckets_p0, buckets_p1)

        node_utility = strategy @ action_utilities
        opponent_reach = p1 if acting_player == 0 else p0
        sign = 1.0 if acting_player == 0 else -1.0
        discount = getattr(self, '_regret_discount', 1.0)
        node.update_regrets(sign * opponent_reach * (action_utilities - node_utility),
                            discount=discount)

        return node_utility

    def _continuation_value(self, phase: str, bucket_p0: int, bucket_p1: int,
                            history: tuple, traverser: int,
                            buckets_p0: dict = None,
                            buckets_p1: dict = None) -> float:
        """
        Compute continuation value at a street boundary.

        Uses bucket-comparison showdown utility. Each street is trained
        independently with its own bucket samples, so cross-street
        coupling comes from correlated bucket sampling (shared hand type,
        drifting equity) rather than recursive policy evaluation.
        """
        # Use next-street buckets if available for better equity comparison
        if buckets_p0 and buckets_p1:
            phase_idx = PHASES.index(phase)
            if phase_idx < len(PHASES) - 1:
                next_phase = PHASES[phase_idx + 1]
                next_b0 = buckets_p0[next_phase]
                next_b1 = buckets_p1[next_phase]
                value = self._showdown_utility(next_b0, next_b1, history, phase)
                return value if traverser == 0 else -value

        value = self._showdown_utility(bucket_p0, bucket_p1, history, phase)
        return value if traverser == 0 else -value

    def _continuation_value_p0(self, phase: str, bucket_p0: int, bucket_p1: int,
                                history: tuple,
                                buckets_p0: dict = None,
                                buckets_p1: dict = None) -> float:
        """Continuation value from P0's perspective (for vanilla CFR)."""
        if buckets_p0 and buckets_p1:
            phase_idx = PHASES.index(phase)
            if phase_idx < len(PHASES) - 1:
                next_phase = PHASES[phase_idx + 1]
                next_b0 = buckets_p0[next_phase]
                next_b1 = buckets_p1[next_phase]
                return self._showdown_utility(next_b0, next_b1, history, phase)

        return self._showdown_utility(bucket_p0, bucket_p1, history, phase)

    def _evaluate_street(self, phase: str, bucket_p0: int, bucket_p1: int,
                         history: tuple,
                         buckets_p0: dict = None,
                         buckets_p1: dict = None) -> float:
        """
        Sampled frozen policy evaluation: traverse a street by sampling
        one action per node from the current average strategy (no regret
        updates). Returns value for player 0.

        Uses Monte Carlo sampling (like external sampling MCCFR) rather
        than full tree traversal to keep per-iteration cost low. Each
        call traces one sampled path through the next street.
        """
        # Fold terminal
        terminal = self._check_terminal(history, bucket_p0, bucket_p1, phase)
        if terminal is not None:
            return terminal

        # Street complete -> recurse to next street or showdown
        if self._is_street_complete(history):
            phase_idx = PHASES.index(phase)
            if phase_idx < len(PHASES) - 1 and buckets_p0 and buckets_p1:
                next_phase = PHASES[phase_idx + 1]
                next_b0 = buckets_p0[next_phase]
                next_b1 = buckets_p1[next_phase]
                return self._evaluate_street(
                    next_phase, next_b0, next_b1, (),
                    buckets_p0, buckets_p1)
            return self._showdown_utility(bucket_p0, bucket_p1, history, phase)

        # Depth limit
        if len(history) >= 8:
            return self._showdown_utility(bucket_p0, bucket_p1, history, phase)

        plays = len(history)
        acting_player = plays % 2
        bucket = bucket_p0 if acting_player == 0 else bucket_p1
        position = get_position(acting_player, phase)
        has_bet = self._has_bet_to_call(history, phase)
        raise_count = count_raises(history, phase)
        can_raise = raise_count < 4

        actions = get_available_actions(has_bet, can_raise, phase, raise_count,
                                       history_len=len(history),
                                       eq_bucket=decode_bucket(bucket)[0])
        info_set = InfoSet(bucket, phase, history, position=position)
        key = info_set.key

        # Use average strategy (frozen — no updates)
        if key in self.nodes:
            sigma = self.nodes[key].get_average_strategy()
        else:
            sigma = np.ones(len(actions)) / len(actions)

        # Sample one action from strategy (Monte Carlo evaluation)
        action_idx = np.random.choice(len(actions), p=sigma)
        new_history = history + (int(actions[action_idx]),)
        return self._evaluate_street(
            phase, bucket_p0, bucket_p1, new_history,
            buckets_p0, buckets_p1)

    def _check_terminal(self, history: tuple,
                        bucket_p0: int, bucket_p1: int,
                        phase: str = 'preflop') -> float | None:
        """
        Check if history is a terminal state due to a fold.
        Returns utility for player 0, or None if non-terminal.

        Street completion (check-check, bet-call) is handled separately
        by _is_street_complete() in the traversal methods.
        """
        if len(history) < 1:
            return None

        if history[-1] == int(Action.FOLD):
            folder = (len(history) - 1) % 2
            inv0, inv1 = self._player_investments(history, phase)
            if folder == 0:
                return -inv0
            else:
                return inv1

        return None

    def _has_bet_to_call(self, history: tuple, phase: str = '') -> bool:
        """Check if there's an outstanding bet to call.

        At preflop with empty history, the big blind IS a bet to call
        (SB must call, fold, or raise). Postflop empty history = check.
        """
        if not history:
            # Preflop: BB is a forced bet — first actor faces a bet
            return phase == 'preflop'
        last = history[-1]
        return last in (int(Action.BET_QUARTER_POT), int(Action.BET_THIRD_POT),
                        int(Action.BET_HALF_POT),
                        int(Action.BET_TWO_THIRDS_POT), int(Action.BET_THREE_QUARTER_POT),
                        int(Action.BET_POT),
                        int(Action.BET_OVERBET), int(Action.BET_DOUBLE_POT),
                        int(Action.ALL_IN),
                        int(Action.OPEN_RAISE), int(Action.THREE_BET),
                        int(Action.FOUR_BET),
                        int(Action.DONK_SMALL), int(Action.DONK_MEDIUM))

    def _player_investments(self, history: tuple, phase: str = 'preflop') -> tuple[float, float]:
        """
        Track how much each player has invested into the pot.
        Handles both preflop (open/3bet/4bet) and postflop (pot-fraction) actions.

        Preflop: P0 posts SB (0.5), P1 posts BB (1.0).
        The BB is an outstanding bet of 0.5 that P0 must call/raise/fold.
        Postflop: both start at 0 with no outstanding bet.
        """
        if phase == 'preflop':
            inv = [0.5, 1.0]   # SB=0.5, BB=1.0
            outstanding = 0.5  # P0 must call 0.5 more to match BB
        else:
            # Postflop: normalize pot to 1.0 (each player invested 0.5)
            # Without this, all pot-fraction bets are 0 and nothing is learned
            inv = [0.5, 0.5]
            outstanding = 0.0

        for idx, action in enumerate(history):
            player = idx % 2
            if action == int(Action.FOLD):
                break
            elif action == int(Action.CHECK_CALL):
                inv[player] += outstanding
                outstanding = 0.0
            elif action in (int(Action.BET_QUARTER_POT), int(Action.BET_THIRD_POT),
                            int(Action.BET_HALF_POT),
                            int(Action.BET_TWO_THIRDS_POT), int(Action.BET_THREE_QUARTER_POT),
                            int(Action.BET_POT),
                            int(Action.BET_OVERBET), int(Action.BET_DOUBLE_POT),
                            int(Action.ALL_IN),
                            int(Action.OPEN_RAISE), int(Action.THREE_BET),
                            int(Action.FOUR_BET),
                            int(Action.DONK_SMALL), int(Action.DONK_MEDIUM)):
                inv[player] += outstanding
                outstanding = 0.0
                total_pot = inv[0] + inv[1]

                if action == int(Action.BET_QUARTER_POT):
                    bet = total_pot * 0.25
                elif action == int(Action.BET_THIRD_POT):
                    bet = total_pot * 0.33
                elif action == int(Action.BET_HALF_POT):
                    bet = total_pot * 0.5
                elif action == int(Action.BET_TWO_THIRDS_POT):
                    bet = total_pot * 0.67
                elif action == int(Action.BET_THREE_QUARTER_POT):
                    bet = total_pot * 0.75
                elif action == int(Action.BET_POT):
                    bet = total_pot
                elif action == int(Action.BET_OVERBET):
                    bet = total_pot * 1.25
                elif action == int(Action.BET_DOUBLE_POT):
                    bet = total_pot * 2.0
                elif action == int(Action.OPEN_RAISE):
                    bet = total_pot * 0.75  # ~2.5bb relative to blinds
                elif action == int(Action.THREE_BET):
                    bet = total_pot * 1.5  # ~3x the open
                elif action == int(Action.FOUR_BET):
                    bet = total_pot * 2.0  # ~2.2x the 3bet
                elif action == int(Action.DONK_SMALL):
                    bet = total_pot * 0.25
                elif action == int(Action.DONK_MEDIUM):
                    bet = total_pot * 0.5
                else:  # ALL_IN
                    bet = total_pot * 3.0
                inv[player] += bet
                outstanding = bet

        return inv[0], inv[1]

    def _showdown_utility(self, bucket_p0: int, bucket_p1: int,
                          history: tuple, phase: str = 'preflop') -> float:
        """
        Showdown value for player 0 based on per-player investment.
        Uses equity bucket for comparison (higher equity bucket wins).
        """
        inv0, inv1 = self._player_investments(history, phase)
        # Compare equity buckets (first dimension of 2D bucket)
        eq0 = bucket_p0 // NUM_HAND_TYPES
        eq1 = bucket_p1 // NUM_HAND_TYPES
        if eq0 > eq1:
            return inv1
        elif eq0 < eq1:
            return -inv0
        return 0.0

    def _is_street_complete(self, history: tuple) -> bool:
        """Check if the current street's action is complete."""
        if len(history) < 2:
            return False
        last_two = history[-2:]
        is_check_check = (
            last_two == (int(Action.CHECK_CALL), int(Action.CHECK_CALL))
        )
        is_bet_call = (
            last_two[-1] == int(Action.CHECK_CALL) and
            last_two[-2] in (int(Action.BET_QUARTER_POT), int(Action.BET_THIRD_POT),
                             int(Action.BET_HALF_POT),
                             int(Action.BET_TWO_THIRDS_POT), int(Action.BET_THREE_QUARTER_POT),
                             int(Action.BET_POT),
                             int(Action.BET_OVERBET), int(Action.BET_DOUBLE_POT),
                             int(Action.ALL_IN),
                             int(Action.OPEN_RAISE), int(Action.THREE_BET),
                             int(Action.FOUR_BET),
                             int(Action.DONK_SMALL), int(Action.DONK_MEDIUM))
        )
        return is_check_check or is_bet_call

    def get_strategy(self, phase: str, bucket: int,
                     history: tuple, position: str = '') -> dict[int, float]:
        """Look up the trained strategy for an information set."""
        has_bet = self._has_bet_to_call(history, phase)
        raise_count = count_raises(history, phase)
        can_raise = raise_count < 4

        actions = get_available_actions(has_bet, can_raise, phase, raise_count,
                                       history_len=len(history),
                                       eq_bucket=decode_bucket(bucket)[0])
        info_set = InfoSet(bucket, phase, history, position=position)
        key = info_set.key

        if key in self.nodes:
            node = self.nodes[key]
            avg_strategy = node.get_average_strategy()
            n = min(len(actions), len(avg_strategy))
            return {int(actions[i]): float(avg_strategy[i])
                    for i in range(n)}

        uniform = 1.0 / len(actions)
        return {int(a): uniform for a in actions}

    def save(self, filepath: str):
        """Save trained strategy to file."""
        data = {
            'version': STRATEGY_VERSION,
            'iterations': self.iterations,
            'nodes': {k: v.to_dict() for k, v in self.nodes.items()},
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"Saved {len(self.nodes)} nodes after {self.iterations} iterations")

    def load(self, filepath: str) -> bool:
        """Load trained strategy from file. Returns True if loaded."""
        if not os.path.exists(filepath):
            return False
        with open(filepath, 'r') as f:
            data = json.load(f)

        version = data.get('version', 1)
        if version < STRATEGY_VERSION:
            print(f"Strategy v{version} is outdated (need v{STRATEGY_VERSION}). "
                  "Retraining required.")
            return False

        self.iterations = data['iterations']
        self.nodes = {k: CFRNode.from_dict(v) for k, v in data['nodes'].items()}
        print(f"Loaded {len(self.nodes)} nodes ({self.iterations} iterations)")
        return True
