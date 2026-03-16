# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: nonecheck=False, initializedcheck=False
"""
Cython-accelerated CFR+ inner loop.

Speeds up the hot path: recursive tree traversal, regret matching,
strategy computation, and terminal/investment calculations.

All node data is stored in flat C arrays indexed by integer IDs,
eliminating Python object overhead in the inner loop.
"""
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
from libc.math cimport fabs

cdef extern from "sys/mman.h":
    void* mmap(void* addr, size_t length, int prot, int flags,
               int fd, long offset) nogil
    int munmap(void* addr, size_t length) nogil
    cdef int PROT_READ
    cdef int PROT_WRITE
    cdef int MAP_SHARED
    cdef int MAP_ANONYMOUS
    cdef void* MAP_FAILED

np.import_array()

# --- Action constants (must match abstraction.py) ---
cdef int ACT_FOLD = 0
cdef int ACT_CHECK_CALL = 1
cdef int ACT_BET_THIRD = 2
cdef int ACT_BET_HALF = 3
cdef int ACT_BET_POT = 4
cdef int ACT_ALL_IN = 5
cdef int ACT_OPEN_RAISE = 6
cdef int ACT_THREE_BET = 7
cdef int ACT_FOUR_BET = 8
# v6 additions
cdef int ACT_BET_TWO_THIRDS = 9
cdef int ACT_BET_OVERBET = 10
cdef int ACT_DONK_SMALL = 11
cdef int ACT_DONK_MEDIUM = 12
# v9 additions
cdef int ACT_BET_QUARTER = 13
cdef int ACT_BET_THREE_QUARTER = 14
cdef int ACT_BET_DOUBLE_POT = 15
# v12 selective addition (river only via add_selective_action / set_selective_action)
cdef int ACT_BET_TRIPLE_POT = 16

cdef int MAX_HISTORY = 12


# --- Fast random (xoshiro128+ — better quality than libc rand) ---
cdef unsigned long long _rng_state[4]

cdef inline void rng_seed(unsigned long long seed) noexcept nogil:
    """Seed xoshiro256+ using SplitMix64 to fill state."""
    cdef unsigned long long z
    cdef int i
    for i in range(4):
        seed = seed + <unsigned long long>0x9E3779B97F4A7C15
        z = seed
        z = (z ^ (z >> 30)) * <unsigned long long>0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * <unsigned long long>0x94D049BB133111EB
        _rng_state[i] = z ^ (z >> 31)

cdef inline unsigned long long _rng_next() noexcept nogil:
    """xoshiro256+ step."""
    cdef unsigned long long result = _rng_state[0] + _rng_state[3]
    cdef unsigned long long t = _rng_state[1] << 17
    _rng_state[2] ^= _rng_state[0]
    _rng_state[3] ^= _rng_state[1]
    _rng_state[1] ^= _rng_state[2]
    _rng_state[0] ^= _rng_state[3]
    _rng_state[2] ^= t
    # rotl(_rng_state[3], 45)
    _rng_state[3] = (_rng_state[3] << 45) | (_rng_state[3] >> 19)
    return result

cdef inline double rand_double() noexcept nogil:
    return <double>(_rng_next() >> 11) * (1.0 / <double>(1ULL << 53))

cdef inline int rand_int(int n) noexcept nogil:
    return <int>(_rng_next() % <unsigned long long>n)


# --- Terminal / street checks (pure C) ---

cdef inline bint is_raise_action(int action) noexcept nogil:
    return (action == ACT_BET_QUARTER or action == ACT_BET_THIRD or
            action == ACT_BET_HALF or
            action == ACT_BET_TWO_THIRDS or action == ACT_BET_THREE_QUARTER or
            action == ACT_BET_POT or
            action == ACT_BET_OVERBET or action == ACT_BET_DOUBLE_POT or
            action == ACT_BET_TRIPLE_POT or
            action == ACT_ALL_IN or
            action == ACT_OPEN_RAISE or action == ACT_THREE_BET or
            action == ACT_FOUR_BET or
            action == ACT_DONK_SMALL or action == ACT_DONK_MEDIUM)


cdef inline bint has_bet_to_call(int* history, int hlen, bint is_preflop=0) noexcept nogil:
    if hlen == 0:
        # Preflop: BB is a forced bet — first actor faces a bet
        return is_preflop
    return is_raise_action(history[hlen - 1])


cdef int count_raises(int* history, int hlen) noexcept nogil:
    cdef int count = 0
    cdef int i
    for i in range(hlen):
        if is_raise_action(history[i]):
            count += 1
    return count


cdef bint is_street_complete(int* history, int hlen) noexcept nogil:
    cdef int last, prev
    if hlen < 2:
        return 0
    last = history[hlen - 1]
    prev = history[hlen - 2]
    if prev == ACT_CHECK_CALL and last == ACT_CHECK_CALL:
        return 1
    if last == ACT_CHECK_CALL and is_raise_action(prev):
        return 1
    return 0


cdef bint is_fold_terminal(int* history, int hlen) noexcept nogil:
    if hlen < 1:
        return 0
    return history[hlen - 1] == ACT_FOLD


cdef void player_investments(int* history, int hlen,
                              double* inv0, double* inv1,
                              bint is_preflop=1) noexcept nogil:
    cdef double outstanding
    cdef double total_pot, bet
    cdef int i, action, player

    if is_preflop:
        inv0[0] = 0.5   # SB
        inv1[0] = 1.0   # BB
        outstanding = 0.5  # P0 must call 0.5 to match BB
    else:
        # Postflop: normalize pot to 1.0 (each player invested 0.5)
        # Without this, all pot-fraction bets are 0 and nothing is learned
        inv0[0] = 0.5
        inv1[0] = 0.5
        outstanding = 0.0

    for i in range(hlen):
        player = i % 2
        action = history[i]

        if action == ACT_FOLD:
            break
        elif action == ACT_CHECK_CALL:
            if player == 0:
                inv0[0] += outstanding
            else:
                inv1[0] += outstanding
            outstanding = 0.0
        elif is_raise_action(action):
            if player == 0:
                inv0[0] += outstanding
            else:
                inv1[0] += outstanding
            outstanding = 0.0
            total_pot = inv0[0] + inv1[0]

            if action == ACT_BET_QUARTER:
                bet = total_pot * 0.25
            elif action == ACT_BET_THIRD:
                bet = total_pot * 0.33
            elif action == ACT_BET_HALF:
                bet = total_pot * 0.5
            elif action == ACT_BET_TWO_THIRDS:
                bet = total_pot * 0.67
            elif action == ACT_BET_THREE_QUARTER:
                bet = total_pot * 0.75
            elif action == ACT_BET_POT:
                bet = total_pot * 1.0
            elif action == ACT_BET_OVERBET:
                bet = total_pot * 1.25
            elif action == ACT_BET_DOUBLE_POT:
                bet = total_pot * 2.0
            elif action == ACT_BET_TRIPLE_POT:
                bet = total_pot * 3.0
            elif action == ACT_OPEN_RAISE:
                bet = total_pot * 0.75
            elif action == ACT_THREE_BET:
                bet = total_pot * 1.5
            elif action == ACT_FOUR_BET:
                bet = total_pot * 2.0
            elif action == ACT_DONK_SMALL:
                bet = total_pot * 0.25
            elif action == ACT_DONK_MEDIUM:
                bet = total_pot * 0.5
            else:  # ALL_IN
                bet = total_pot * 3.0

            if player == 0:
                inv0[0] += bet
            else:
                inv1[0] += bet
            outstanding = bet


cdef double showdown_utility(int bucket_p0, int bucket_p1,
                              int* history, int hlen,
                              int num_hand_types,
                              bint is_preflop=1) noexcept nogil:
    cdef double inv0, inv1
    cdef int eq0, eq1
    player_investments(history, hlen, &inv0, &inv1, is_preflop)
    eq0 = bucket_p0 // num_hand_types
    eq1 = bucket_p1 // num_hand_types
    if eq0 > eq1:
        return inv1
    elif eq0 < eq1:
        return -inv0
    return 0.0


cdef double check_terminal_value(int* history, int hlen,
                                  int bucket_p0, int bucket_p1,
                                  bint is_preflop=1) noexcept nogil:
    cdef int folder
    cdef double inv0, inv1
    if hlen < 1:
        return -999999.0
    if history[hlen - 1] != ACT_FOLD:
        return -999999.0
    folder = (hlen - 1) % 2
    player_investments(history, hlen, &inv0, &inv1, is_preflop)
    if folder == 0:
        return -inv0
    else:
        return inv1


# --- Available actions (pure C) ---

cdef int get_preflop_actions(bint has_bet, bint can_raise, int raise_count,
                              int* out_actions) noexcept nogil:
    cdef int n = 0
    if not has_bet:
        # BB checked to (after limps, no raise)
        out_actions[n] = ACT_CHECK_CALL; n += 1
        if can_raise:
            out_actions[n] = ACT_OPEN_RAISE; n += 1
            out_actions[n] = ACT_ALL_IN; n += 1
    elif raise_count == 0:
        # Facing the big blind only — no one has raised yet
        # SB/limpers: can fold, call (limp), open raise, or jam
        out_actions[n] = ACT_FOLD; n += 1
        out_actions[n] = ACT_CHECK_CALL; n += 1
        if can_raise:
            out_actions[n] = ACT_OPEN_RAISE; n += 1
            out_actions[n] = ACT_ALL_IN; n += 1
    else:
        # Facing a real raise
        out_actions[n] = ACT_FOLD; n += 1
        out_actions[n] = ACT_CHECK_CALL; n += 1
        if can_raise:
            if raise_count <= 1:
                out_actions[n] = ACT_THREE_BET; n += 1
                out_actions[n] = ACT_ALL_IN; n += 1
            elif raise_count == 2:
                out_actions[n] = ACT_FOUR_BET; n += 1
                out_actions[n] = ACT_ALL_IN; n += 1
            else:
                out_actions[n] = ACT_ALL_IN; n += 1
    return n


cdef int get_postflop_actions(bint has_bet, bint can_raise,
                               bint is_donk_eligible,
                               int phase_int,
                               int* out_actions) noexcept nogil:
    cdef int n = 0
    cdef int context = 1 if has_bet else 0
    cdef unsigned int sel_mask
    cdef int a
    if has_bet:
        out_actions[n] = ACT_FOLD; n += 1
        out_actions[n] = ACT_CHECK_CALL; n += 1
    else:
        out_actions[n] = ACT_CHECK_CALL; n += 1
    if can_raise:
        if is_donk_eligible:
            # Flop/turn OOP lead: donk sizes + standard big sizes
            out_actions[n] = ACT_DONK_SMALL; n += 1
            out_actions[n] = ACT_DONK_MEDIUM; n += 1
            out_actions[n] = ACT_BET_TWO_THIRDS; n += 1
            out_actions[n] = ACT_BET_POT; n += 1
            out_actions[n] = ACT_ALL_IN; n += 1
        elif _action_grid_size >= 16:
            # 16-action grid: full sizing menu (v9 expansion)
            out_actions[n] = ACT_BET_QUARTER; n += 1
            out_actions[n] = ACT_BET_THIRD; n += 1
            out_actions[n] = ACT_BET_HALF; n += 1
            out_actions[n] = ACT_BET_TWO_THIRDS; n += 1
            out_actions[n] = ACT_BET_THREE_QUARTER; n += 1
            out_actions[n] = ACT_BET_POT; n += 1
            out_actions[n] = ACT_BET_OVERBET; n += 1
            out_actions[n] = ACT_BET_DOUBLE_POT; n += 1
            out_actions[n] = ACT_ALL_IN; n += 1
        else:
            # 13-action grid (v6/B0): standard postflop sizing
            out_actions[n] = ACT_BET_THIRD; n += 1
            out_actions[n] = ACT_BET_HALF; n += 1
            out_actions[n] = ACT_BET_TWO_THIRDS; n += 1
            out_actions[n] = ACT_BET_POT; n += 1
            out_actions[n] = ACT_BET_OVERBET; n += 1
            out_actions[n] = ACT_ALL_IN; n += 1

        # Append selective actions from bitmask (v11 D1)
        sel_mask = _selective_actions[phase_int * 2 + context]
        if sel_mask != 0:
            for a in range(17):
                if (sel_mask >> a) & 1:
                    # Only add if not already in the action list
                    if not _action_in_list(out_actions, n, a):
                        out_actions[n] = a; n += 1
    return n


cdef inline bint _action_in_list(int* actions, int count, int action) noexcept nogil:
    cdef int i
    for i in range(count):
        if actions[i] == action:
            return True
    return False


cdef int get_actions(bint has_bet, bint can_raise, bint is_preflop,
                      int raise_count, int hlen, int phase, int bucket,
                      int* out_actions) noexcept nogil:
    cdef bint is_donk_eligible
    if is_preflop:
        return get_preflop_actions(has_bet, can_raise, raise_count, out_actions)
    else:
        is_donk_eligible = ((phase == PHASE_FLOP or phase == PHASE_TURN)
                             and hlen == 0 and not has_bet)
        return get_postflop_actions(has_bet, can_raise, is_donk_eligible,
                                     phase, out_actions)


# --- Node operations using flat arrays ---

cdef struct NodeData:
    int num_actions
    int first_visit_iter   # iteration when this node was first visited (-1 = unvisited)
    double regret_sum[16]
    double strategy_sum[16]


cdef void node_get_strategy(NodeData* node, double* out_strategy) noexcept nogil:
    cdef double total = 0.0
    cdef int i
    cdef int n = node.num_actions
    cdef double uniform

    for i in range(n):
        total += node.regret_sum[i]

    if total > 0:
        for i in range(n):
            out_strategy[i] = node.regret_sum[i] / total
    else:
        uniform = 1.0 / <double>n
        for i in range(n):
            out_strategy[i] = uniform


cdef void node_update_regrets(NodeData* node, double* regrets,
                               double discount) noexcept nogil:
    cdef int i
    for i in range(node.num_actions):
        if discount < 1.0:
            node.regret_sum[i] *= discount
        node.regret_sum[i] += regrets[i]
        if node.regret_sum[i] < 0:
            node.regret_sum[i] = 0.0


cdef void node_accumulate_strategy(NodeData* node, double* strategy,
                                     double weight) noexcept nogil:
    cdef int i
    for i in range(node.num_actions):
        node.strategy_sum[i] += weight * strategy[i]


cdef void node_get_average_strategy(NodeData* node, double* out) noexcept nogil:
    cdef double total = 0.0
    cdef int i
    cdef double uniform
    for i in range(node.num_actions):
        total += node.strategy_sum[i]
    if total > 0:
        for i in range(node.num_actions):
            out[i] = node.strategy_sum[i] / total
    else:
        uniform = 1.0 / <double>node.num_actions
        for i in range(node.num_actions):
            out[i] = uniform


cdef int sample_action(double* probs, int n) noexcept nogil:
    cdef double r = rand_double()
    cdef double cumsum = 0.0
    cdef int i
    for i in range(n - 1):
        cumsum += probs[i]
        if r < cumsum:
            return i
    return n - 1


# --- Key building ---

cdef int PHASE_PREFLOP = 0
cdef int PHASE_FLOP = 1
cdef int PHASE_TURN = 2
cdef int PHASE_RIVER = 3

cdef inline long long make_key(int phase, int position, int bucket,
                                int* history, int hlen) noexcept nogil:
    """
    Encode infoset key as 64-bit int.
    Layout: phase(2) | position(1) | bucket(8) | history(hlen*4) | hlen(4)
    """
    cdef long long key = 0
    cdef int i
    key = phase
    key = (key << 1) | position
    key = (key << 8) | bucket
    for i in range(hlen):
        key = (key << 4) | history[i]
    key = (key << 4) | hlen
    return key


# --- Node storage ---

cdef int MAX_NODES = 500000
cdef NodeData* node_pool = NULL
cdef int node_count = 0
cdef bint _shared_mode = False
cdef bint _no_create_mode = False
cdef size_t _shared_pool_size = 0

node_index = {}

# --- Shared progress counters (one long long per worker, mmap'd) ---
cdef long long* _progress_counters = NULL
cdef size_t _progress_counters_size = 0
cdef int _worker_id = -1

# --- Action grid config (shared source of truth with abstraction.py) ---
cdef int _action_grid_size = 16    # 13=v6/B0 grid, 16=v9 expanded grid

# --- Selective action config (v11 D1) ---
# Bitmask of extra actions to add per phase+context.
# Indexed as: _selective_actions[phase * 2 + context]
#   phase: 0=preflop, 1=flop, 2=turn, 3=river
#   context: 0=no_bet, 1=facing_bet
# Each entry is a bitmask where bit i means action i is added.
cdef unsigned int _selective_actions[8]  # 4 phases * 2 contexts


# --- Ablation config flags (set by train_fast) ---
cdef int _allin_dampen_mode = 1    # 0=old (rc<2, 0.7x), 1=new (rc==0, 0.5x)
cdef int _phase_schedule_mode = 1  # 0=2x (6-step), 1=3x (8-step)
cdef int _adaptive_averaging = 0   # 0=global (default), 1=per-node adaptive
cdef int _current_iter = 0         # current iteration (set by train_fast per iteration)
cdef int _global_averaging_delay = 0  # global averaging delay (set by train_fast)

# --- DCFR / weight schedule config (set by train_fast) ---
cdef double _regret_discount = 1.0     # 1.0=CFR+, <1.0=DCFR (e.g. 0.995)
cdef int _weight_schedule_mode = 0     # 0=linear, 1=exponential, 2=polynomial
cdef double _weight_schedule_param = 1.0  # base for exp, power for polynomial


def init_progress_counters(int num_workers):
    """Allocate shared-memory progress counters (call before fork)."""
    global _progress_counters, _progress_counters_size
    _progress_counters_size = num_workers * sizeof(long long)
    if _progress_counters != NULL:
        munmap(_progress_counters, _progress_counters_size)
    _progress_counters = <long long*>mmap(
        NULL, _progress_counters_size,
        PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_ANONYMOUS,
        -1, 0)
    if _progress_counters == MAP_FAILED:
        _progress_counters = NULL
        raise MemoryError("Failed to mmap progress counters")
    memset(_progress_counters, 0, _progress_counters_size)


def cleanup_progress_counters():
    """Free shared progress counters."""
    global _progress_counters, _progress_counters_size
    if _progress_counters != NULL:
        munmap(_progress_counters, _progress_counters_size)
        _progress_counters = NULL
        _progress_counters_size = 0


def set_worker_id(int wid):
    """Set the worker ID for this process (call in child after fork)."""
    global _worker_id
    _worker_id = wid


def get_progress_counters(int num_workers):
    """Read all worker progress counters (call from main process)."""
    cdef int i
    if _progress_counters == NULL:
        return [0] * num_workers
    return [_progress_counters[i] for i in range(num_workers)]


cdef int get_or_create_node(long long key, int num_actions) noexcept:
    global node_count
    cdef int idx
    cdef int i

    if key in node_index:
        idx = node_index[key]
        # If action menu changed (e.g. version upgrade), reinitialize the node
        if node_pool[idx].num_actions != num_actions:
            node_pool[idx].num_actions = num_actions
            for i in range(16):
                node_pool[idx].regret_sum[i] = 0.0
                node_pool[idx].strategy_sum[i] = 0.0
        return idx

    # No-create mode or shared pool full: skip unknown nodes
    if _no_create_mode:
        return -1
    if _shared_mode and node_count >= MAX_NODES:
        return -1

    idx = node_count
    if idx >= MAX_NODES:
        _grow_pool()

    node_pool[idx].num_actions = num_actions
    node_pool[idx].first_visit_iter = -1
    for i in range(16):
        node_pool[idx].regret_sum[i] = 0.0
        node_pool[idx].strategy_sum[i] = 0.0

    node_index[key] = idx
    node_count += 1
    return idx


cdef void _grow_pool():
    global node_pool, MAX_NODES
    cdef int new_max = MAX_NODES * 2
    cdef NodeData* new_pool = <NodeData*>malloc(new_max * sizeof(NodeData))
    if new_pool == NULL:
        raise MemoryError("Failed to allocate node pool")
    memcpy(new_pool, node_pool, MAX_NODES * sizeof(NodeData))
    free(node_pool)
    node_pool = new_pool
    MAX_NODES = new_max


def init_pool():
    """Initialize the node pool. Call before training."""
    global node_pool, node_count, node_index, MAX_NODES, _shared_mode, _shared_pool_size
    cdef int i
    MAX_NODES = 500000
    if node_pool != NULL:
        if _shared_mode:
            munmap(node_pool, _shared_pool_size)
            _shared_mode = False
        else:
            free(node_pool)
    node_pool = <NodeData*>malloc(MAX_NODES * sizeof(NodeData))
    if node_pool == NULL:
        raise MemoryError("Failed to allocate node pool")
    node_count = 0
    node_index = {}
    _shared_pool_size = 0
    # Clear selective actions to prevent cross-run contamination
    for i in range(8):
        _selective_actions[i] = 0


def init_pool_shared(int max_nodes=2000000):
    """Initialize node pool in anonymous shared memory (for multiprocessing).

    After fork(), child processes share the same physical memory for the
    node pool. Writes (regret/strategy updates) are immediately visible
    across all processes — no synchronization needed (benign races).

    The pool is fixed-size (cannot grow), so pre-allocate enough nodes.
    """
    global node_pool, node_count, node_index, MAX_NODES
    global _shared_mode, _shared_pool_size, _no_create_mode

    MAX_NODES = max_nodes
    _shared_pool_size = MAX_NODES * sizeof(NodeData)

    if node_pool != NULL:
        if _shared_mode:
            munmap(node_pool, _shared_pool_size)
        else:
            free(node_pool)

    node_pool = <NodeData*>mmap(
        NULL, _shared_pool_size,
        PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_ANONYMOUS,
        -1, 0)
    if node_pool == MAP_FAILED:
        node_pool = NULL
        raise MemoryError("Failed to mmap shared node pool")

    memset(node_pool, 0, _shared_pool_size)
    node_count = 0
    node_index = {}
    _shared_mode = True
    _no_create_mode = False
    # Clear selective actions to prevent cross-run contamination
    cdef int i
    for i in range(8):
        _selective_actions[i] = 0


def cleanup_pool():
    """Free the node pool."""
    global node_pool, node_count, node_index, _shared_mode, _shared_pool_size
    if node_pool != NULL:
        if _shared_mode:
            munmap(node_pool, _shared_pool_size)
            _shared_mode = False
            _shared_pool_size = 0
        else:
            free(node_pool)
        node_pool = NULL
    node_count = 0
    node_index = {}


def set_no_create_mode(bint enabled=True):
    """Enable/disable no-create mode. When enabled, unknown nodes return -1
    instead of being created. Used by worker processes in parallel training."""
    global _no_create_mode
    _no_create_mode = enabled


def set_action_grid_size(int size):
    """Set the action grid size for Cython training (13 or 16).

    Must match abstraction.py's set_action_grid(). Call before training
    or importing nodes to ensure Python and Cython use the same grid.
    """
    global _action_grid_size
    if size not in (13, 16):
        raise ValueError(f"Invalid action grid size: {size}. Must be 13 or 16.")
    _action_grid_size = size


def get_action_grid_size():
    """Return the current Cython action grid size."""
    return _action_grid_size


def set_selective_action(int phase, int context, int action_id, bint enabled=True):
    """Add or remove a selective action for a specific phase+context.

    Mirrors abstraction.py's add_selective_action() for Cython parity.

    Args:
        phase: 0=preflop, 1=flop, 2=turn, 3=river
        context: 0=no_bet, 1=facing_bet
        action_id: Action enum int (0-15)
        enabled: True to add, False to remove
    """
    cdef int idx = phase * 2 + context
    if idx < 0 or idx >= 8:
        raise ValueError(f"Invalid phase={phase} context={context}")
    if action_id < 0 or action_id > 16:
        raise ValueError(f"Invalid action_id={action_id}")
    if enabled:
        _selective_actions[idx] |= (1 << action_id)
    else:
        _selective_actions[idx] &= ~(1 << action_id)


def clear_selective_actions():
    """Remove all selective action additions."""
    cdef int i
    for i in range(8):
        _selective_actions[i] = 0


def get_selective_actions_mask(int phase, int context):
    """Return the selective action bitmask for debugging."""
    cdef int idx = phase * 2 + context
    if idx < 0 or idx >= 8:
        return 0
    return _selective_actions[idx]


def sync_selective_actions_from_python():
    """Sync selective actions from abstraction.py's _SELECTIVE_ACTIONS dict.

    Call this before training to ensure Cython and Python menus match.
    """
    from server.gto.abstraction import _SELECTIVE_ACTIONS
    cdef int i
    # Clear first
    for i in range(8):
        _selective_actions[i] = 0
    # Phase name to int
    phase_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
    context_map = {'no_bet': 0, 'facing_bet': 1}
    for (phase_name, ctx_name), actions in _SELECTIVE_ACTIONS.items():
        phase_int = phase_map.get(phase_name, -1)
        ctx_int = context_map.get(ctx_name, -1)
        if phase_int < 0 or ctx_int < 0:
            continue
        idx = phase_int * 2 + ctx_int
        for action in actions:
            aid = int(action)
            if 0 <= aid <= 16:
                _selective_actions[idx] |= (1 << aid)


# --- Phase constants ---
cdef int NUM_HAND_TYPES_C = 15
cdef int NUM_EQUITY_BUCKETS_C = 8
cdef int NUM_BUCKETS_C = 120


# --- Core CFR traversal (external sampling) ---

cdef double cfr_single_street(int phase, int bucket_p0, int bucket_p1,
                                int* history, int hlen,
                                int traverser, double weight,
                                int* buckets_p0, int* buckets_p1) noexcept:
    """External sampling MCCFR for a single street."""
    cdef int acting_player = hlen % 2
    cdef double terminal_val
    cdef int position
    cdef int bucket
    cdef bint hb, cr_bool, is_pf
    cdef int rc
    cdef int actions[16]
    cdef int num_actions
    cdef long long key
    cdef int node_idx
    cdef NodeData* node
    cdef double strategy[16]
    cdef int new_history[12]
    cdef int i
    cdef int action_idx
    cdef double action_utilities[16]
    cdef double node_utility
    cdef double regrets[16]
    cdef double node_weight
    cdef int node_delay

    # Fold terminal
    terminal_val = check_terminal_value(history, hlen, bucket_p0, bucket_p1, phase == PHASE_PREFLOP)
    if terminal_val > -999998.0:
        return terminal_val if traverser == 0 else -terminal_val

    # Street complete → continuation
    if is_street_complete(history, hlen):
        return continuation_value(phase, bucket_p0, bucket_p1,
                                   history, hlen, traverser,
                                   buckets_p0, buckets_p1)

    # Depth limit
    if hlen >= 8:
        return continuation_value(phase, bucket_p0, bucket_p1,
                                   history, hlen, traverser,
                                   buckets_p0, buckets_p1)

    # Info set
    bucket = bucket_p0 if acting_player == 0 else bucket_p1
    position = 1 if acting_player == 1 else 0
    is_pf = (phase == PHASE_PREFLOP)
    hb = has_bet_to_call(history, hlen, is_pf)
    rc = count_raises(history, hlen)
    cr_bool = rc < 4

    num_actions = get_actions(hb, cr_bool, is_pf, rc, hlen, phase, bucket, actions)
    key = make_key(phase, position, bucket, history, hlen)
    node_idx = get_or_create_node(key, num_actions)

    # Track first visit and compute per-node weight if adaptive averaging
    node_weight = weight
    if node_idx >= 0:
        if node_pool[node_idx].first_visit_iter < 0:
            node_pool[node_idx].first_visit_iter = _current_iter
        if _adaptive_averaging == 1:
            # Per-node delay: max(global_delay, first_visit + warmup)
            node_delay = node_pool[node_idx].first_visit_iter + 1000
            if node_delay < _global_averaging_delay:
                node_delay = _global_averaging_delay
            node_weight = <double>(_current_iter - node_delay) if _current_iter > node_delay else 0.0

    # Get strategy (uniform if node unknown)
    if node_idx >= 0:
        node_get_strategy(&node_pool[node_idx], strategy)
    else:
        for i in range(num_actions):
            strategy[i] = 1.0 / <double>num_actions

    for i in range(hlen):
        new_history[i] = history[i]

    if acting_player != traverser:
        # Opponent: sample one action
        action_idx = sample_action(strategy, num_actions)
        new_history[hlen] = actions[action_idx]
        return cfr_single_street(phase, bucket_p0, bucket_p1,
                                  new_history, hlen + 1,
                                  traverser, weight,
                                  buckets_p0, buckets_p1)

    # Traverser: explore all actions
    node_utility = 0.0
    for i in range(num_actions):
        new_history[hlen] = actions[i]
        action_utilities[i] = cfr_single_street(
            phase, bucket_p0, bucket_p1,
            new_history, hlen + 1,
            traverser, weight,
            buckets_p0, buckets_p1)
        node_utility += strategy[i] * action_utilities[i]

    # Update node (skip if unknown)
    if node_idx >= 0:
        for i in range(num_actions):
            regrets[i] = action_utilities[i] - node_utility
            # All-in dampening (configurable for ablation)
            if _allin_dampen_mode == 1:
                # New: cold jams only (rc==0), stronger factor
                if actions[i] == ACT_ALL_IN and rc == 0:
                    regrets[i] *= 0.5
            else:
                # Old: broader dampening (rc<2), weaker factor
                if actions[i] == ACT_ALL_IN and rc < 2:
                    regrets[i] *= 0.7

        # Re-derive pointer: recursion may have triggered _grow_pool()
        node = &node_pool[node_idx]
        node_update_regrets(node, regrets, _regret_discount)
        node_accumulate_strategy(node, strategy, node_weight)

    return node_utility


cdef double evaluate_street(int phase, int bucket_p0, int bucket_p1,
                              int* history, int hlen,
                              int* buckets_p0, int* buckets_p1) noexcept:
    """Sampled frozen policy evaluation. Returns value for P0."""
    cdef double terminal_val
    cdef int next_phase, next_b0, next_b1
    cdef int empty_hist[1]
    cdef int acting_player
    cdef int bucket, position
    cdef bint hb, cr_bool, is_pf
    cdef int rc
    cdef int actions[16]
    cdef int num_actions
    cdef long long key
    cdef double sigma[16]
    cdef int node_idx
    cdef double uniform
    cdef int action_idx
    cdef int new_history[12]
    cdef int i

    terminal_val = check_terminal_value(history, hlen, bucket_p0, bucket_p1, phase == PHASE_PREFLOP)
    if terminal_val > -999998.0:
        return terminal_val

    if is_street_complete(history, hlen):
        if phase < PHASE_RIVER and buckets_p0 != NULL and buckets_p1 != NULL:
            next_phase = phase + 1
            next_b0 = buckets_p0[next_phase]
            next_b1 = buckets_p1[next_phase]
            return evaluate_street(next_phase, next_b0, next_b1,
                                    empty_hist, 0, buckets_p0, buckets_p1)
        return showdown_utility(bucket_p0, bucket_p1, history, hlen,
                                 NUM_HAND_TYPES_C, phase == PHASE_PREFLOP)

    if hlen >= 8:
        return showdown_utility(bucket_p0, bucket_p1, history, hlen,
                                 NUM_HAND_TYPES_C, phase == PHASE_PREFLOP)

    acting_player = hlen % 2
    bucket = bucket_p0 if acting_player == 0 else bucket_p1
    position = 1 if acting_player == 1 else 0
    is_pf = (phase == PHASE_PREFLOP)
    hb = has_bet_to_call(history, hlen, is_pf)
    rc = count_raises(history, hlen)
    cr_bool = rc < 4

    num_actions = get_actions(hb, cr_bool, is_pf, rc, hlen, phase, bucket, actions)
    key = make_key(phase, position, bucket, history, hlen)

    if key in node_index:
        node_idx = node_index[key]
        node_get_average_strategy(&node_pool[node_idx], sigma)
    else:
        uniform = 1.0 / <double>num_actions
        for i in range(num_actions):
            sigma[i] = uniform

    action_idx = sample_action(sigma, num_actions)
    for i in range(hlen):
        new_history[i] = history[i]
    new_history[hlen] = actions[action_idx]

    return evaluate_street(phase, bucket_p0, bucket_p1,
                            new_history, hlen + 1,
                            buckets_p0, buckets_p1)


cdef double continuation_value(int phase, int bucket_p0, int bucket_p1,
                                int* history, int hlen, int traverser,
                                int* buckets_p0, int* buckets_p1) noexcept:
    """Compute continuation value at street boundary using bucket comparison."""
    cdef double value
    cdef int next_phase, next_b0, next_b1

    # Use next-street buckets for better equity comparison
    if buckets_p0 != NULL and buckets_p1 != NULL:
        if phase < PHASE_RIVER:
            next_phase = phase + 1
            next_b0 = buckets_p0[next_phase]
            next_b1 = buckets_p1[next_phase]
            value = showdown_utility(next_b0, next_b1, history, hlen,
                                      NUM_HAND_TYPES_C, phase == PHASE_PREFLOP)
            return value if traverser == 0 else -value

    value = showdown_utility(bucket_p0, bucket_p1, history, hlen,
                              NUM_HAND_TYPES_C)
    return value if traverser == 0 else -value


# --- Bucket sampling (C-level) ---

cdef void sample_street_buckets(int* out_buckets) noexcept nogil:
    """Sample buckets for all 4 streets."""
    cdef int hand_type = rand_int(NUM_HAND_TYPES_C)
    cdef int eq = rand_int(NUM_EQUITY_BUCKETS_C)
    cdef int drift_r
    cdef int i

    for i in range(4):
        out_buckets[i] = eq * NUM_HAND_TYPES_C + hand_type
        drift_r = rand_int(4)
        if drift_r == 0:
            eq = eq - 1 if eq > 0 else 0
        elif drift_r == 3:
            eq = eq + 1 if eq < NUM_EQUITY_BUCKETS_C - 1 else NUM_EQUITY_BUCKETS_C - 1


# --- Python-callable training function ---

def train_fast(int num_iterations, int start_iter=0, int averaging_delay=0,
               unsigned int seed=42, int phase_schedule_mode=1,
               int allin_dampen_mode=1, int adaptive_averaging=0,
               double regret_discount=1.0, int weight_schedule_mode=0,
               double weight_schedule_param=1.0, int action_grid_size=0):
    """
    Run CFR+ training entirely in Cython.

    Uses weighted phase schedule: flop/turn get 2x or 3x iterations to reduce
    their higher exploitability vs preflop/river.

    Args:
        phase_schedule_mode: 0=2x (6-step), 1=3x (8-step, default)
        allin_dampen_mode: 0=old (rc<2, 0.7x), 1=new (rc==0, 0.5x, default)
        adaptive_averaging: 0=global delay (default), 1=per-node adaptive delay
        regret_discount: DCFR discount (1.0=CFR+, <1.0=DCFR)
        weight_schedule_mode: 0=linear, 1=exponential, 2=polynomial
        weight_schedule_param: parameter for weight schedule
        action_grid_size: 0=use current setting, 13=v6/B0 grid, 16=v9 expanded

    Returns total iterations completed.
    """
    global _allin_dampen_mode, _phase_schedule_mode, _adaptive_averaging
    global _current_iter, _global_averaging_delay
    global _regret_discount, _weight_schedule_mode, _weight_schedule_param
    global _action_grid_size
    _allin_dampen_mode = allin_dampen_mode
    _phase_schedule_mode = phase_schedule_mode
    _adaptive_averaging = adaptive_averaging
    _global_averaging_delay = averaging_delay
    _regret_discount = regret_discount
    _weight_schedule_mode = weight_schedule_mode
    _weight_schedule_param = weight_schedule_param
    if action_grid_size > 0:
        if action_grid_size not in (13, 16):
            raise ValueError(f"Invalid action_grid_size: {action_grid_size}")
        _action_grid_size = action_grid_size

    rng_seed(<unsigned long long>seed)

    cdef int buckets_p0[4]
    cdef int buckets_p1[4]
    cdef int empty_history[1]
    cdef int t, phase, traverser
    cdef double weight
    cdef int b0, b1
    cdef int i, s

    # Phase schedule: configurable for ablation
    cdef int phase_schedule[8]
    cdef int schedule_len
    if phase_schedule_mode == 0:
        # 2x schedule (original)
        schedule_len = 6
        phase_schedule[0] = PHASE_PREFLOP
        phase_schedule[1] = PHASE_FLOP
        phase_schedule[2] = PHASE_FLOP
        phase_schedule[3] = PHASE_TURN
        phase_schedule[4] = PHASE_TURN
        phase_schedule[5] = PHASE_RIVER
    else:
        # 3x schedule (current default)
        schedule_len = 8
        phase_schedule[0] = PHASE_PREFLOP
        phase_schedule[1] = PHASE_FLOP
        phase_schedule[2] = PHASE_FLOP
        phase_schedule[3] = PHASE_FLOP
        phase_schedule[4] = PHASE_TURN
        phase_schedule[5] = PHASE_TURN
        phase_schedule[6] = PHASE_TURN
        phase_schedule[7] = PHASE_RIVER

    # For scheduled DCFR (mode 3), we need per-iteration discount
    cdef double alpha_dcfr = 1.5   # positive regret discount exponent
    cdef double gamma_dcfr = _weight_schedule_param if _weight_schedule_mode == 3 else 2.0
    cdef double t_ratio

    for i in range(num_iterations):
        t = start_iter + i + 1
        _current_iter = t
        # Compute strategy accumulation weight based on schedule mode
        if t <= averaging_delay:
            weight = 0.0
        elif _weight_schedule_mode == 1:
            # Exponential: param^(t - delay)
            weight = _weight_schedule_param ** <double>(t - averaging_delay)
        elif _weight_schedule_mode == 2:
            # Polynomial: (t - delay)^param
            weight = (<double>(t - averaging_delay)) ** _weight_schedule_param
        elif _weight_schedule_mode == 3:
            # Scheduled DCFR: weight = (t/(t+1))^gamma, discount = t^alpha/(t^alpha+1)
            # Based on Brown & Sandholm "Dynamic DCFR" + Hyperparameter Schedules
            t_ratio = <double>t / (<double>t + 1.0)
            weight = t_ratio ** gamma_dcfr
            # Time-varying regret discount: t^alpha / (t^alpha + 1)
            # Approaches 1.0 as t grows (less discounting over time)
            _regret_discount = (<double>t ** alpha_dcfr) / (<double>t ** alpha_dcfr + 1.0)
        else:
            # Linear (default): t - delay
            weight = <double>(t - averaging_delay)

        for s in range(schedule_len):
            phase = phase_schedule[s]
            sample_street_buckets(buckets_p0)
            sample_street_buckets(buckets_p1)

            b0 = buckets_p0[phase]
            b1 = buckets_p1[phase]

            for traverser in range(2):
                cfr_single_street(phase, b0, b1,
                                   empty_history, 0,
                                   traverser, weight,
                                   buckets_p0, buckets_p1)

        # Update shared progress counter every 1000 iterations
        if _progress_counters != NULL and _worker_id >= 0 and (i & 1023) == 0:
            _progress_counters[_worker_id] = i + 1

    # Final update
    if _progress_counters != NULL and _worker_id >= 0:
        _progress_counters[_worker_id] = num_iterations

    return start_iter + num_iterations


def get_node_count():
    """Return current number of nodes."""
    return node_count


def get_all_nodes():
    """Export all nodes back to Python dicts for saving/compatibility."""
    cdef int j
    result = {}

    for py_key, idx in node_index.items():
        result[py_key] = {
            'num_actions': node_pool[idx].num_actions,
            'first_visit_iter': node_pool[idx].first_visit_iter,
            'regret_sum': [node_pool[idx].regret_sum[j]
                           for j in range(node_pool[idx].num_actions)],
            'strategy_sum': [node_pool[idx].strategy_sum[j]
                             for j in range(node_pool[idx].num_actions)],
        }

    return result


def import_nodes(nodes_dict):
    """Import nodes from a Python dict."""
    global node_count
    cdef int idx, i, n

    for py_key, data in nodes_dict.items():
        n = data['num_actions']
        idx = node_count
        if idx >= MAX_NODES:
            _grow_pool()

        node_pool[idx].num_actions = n
        node_pool[idx].first_visit_iter = data.get('first_visit_iter', -1)
        regrets = data['regret_sum']
        strats = data['strategy_sum']
        for i in range(n):
            node_pool[idx].regret_sum[i] = regrets[i]
            node_pool[idx].strategy_sum[i] = strats[i]

        node_index[py_key] = idx
        node_count += 1


def get_strategy_fast(int phase, int bucket, tuple history, str position=''):
    """Look up trained strategy. Returns dict {action: prob}."""
    cdef int hlen = len(history)
    cdef int c_history[12]
    cdef int i
    cdef int pos
    cdef bint hb, cr_bool, is_pf
    cdef int rc
    cdef int actions[16]
    cdef int num_actions
    cdef long long key
    cdef double avg_strat[16]
    cdef int node_idx
    cdef double uniform

    for i in range(hlen):
        c_history[i] = history[i]

    pos = 1 if position == 'ip' else 0
    is_pf = (phase == PHASE_PREFLOP)
    hb = has_bet_to_call(c_history, hlen, is_pf)
    rc = count_raises(c_history, hlen)
    cr_bool = rc < 4

    num_actions = get_actions(hb, cr_bool, is_pf, rc, hlen, phase, bucket, actions)
    key = make_key(phase, pos, bucket, c_history, hlen)

    if key in node_index:
        node_idx = node_index[key]
        node_get_average_strategy(&node_pool[node_idx], avg_strat)
    else:
        uniform = 1.0 / <double>num_actions
        for i in range(num_actions):
            avg_strat[i] = uniform

    result = {}
    for i in range(num_actions):
        result[actions[i]] = avg_strat[i]
    return result
