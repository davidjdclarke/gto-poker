"""
GTO Engine - bridges the CFR strategy with the actual poker game.

Translates real game states into abstracted information sets,
looks up the CFR strategy, and converts back to concrete actions.

v5: position-aware, debug mode, runtime fallbacks, latency profiling.
"""
import asyncio
import logging
import os
import random
import time
from pathlib import Path
from server.deck import Card
from server.player import Player
from server.gto.equity import hand_equity, hand_strength_bucket
from server.gto.abstraction import (
    Action, NUM_BUCKETS, ACTION_NAMES, get_position,
    concrete_to_abstract_history,
)
from server.gto.cfr import CFRTrainer, STRATEGY_VERSION

logger = logging.getLogger(__name__)

# Default strategy file. NOTE: strategy.json may be a 16-action grid model.
# For evaluation, use --strategy to specify the correct file explicitly.
# Recommended: experiments/best/v9_B0_100M_allbots_positive.json (13-action B0)
STRATEGY_FILE = str(Path(__file__).parent / "strategy.json")

# Global trainer instance
_trainer: CFRTrainer | None = None
_training_lock = asyncio.Lock()

# Debug mode: when True, gto_decide returns full diagnostic info
_debug_mode = False


def set_debug_mode(enabled: bool = True):
    """Enable/disable debug mode for detailed decision diagnostics."""
    global _debug_mode
    _debug_mode = enabled


def get_trainer() -> CFRTrainer:
    """Get the global CFR trainer, loading or training if needed."""
    global _trainer
    if _trainer is None:
        _trainer = CFRTrainer()
        if not _trainer.load(STRATEGY_FILE):
            logger.warning("No saved strategy found. Training CFR+...")
            _trainer.train(num_iterations=5000, sampling='external')
            _trainer.save(STRATEGY_FILE)
            logger.info("Training complete.")
        # Auto-detect and set action grid from loaded strategy
        from server.gto.abstraction import detect_action_grid_from_strategy, set_action_grid
        grid = detect_action_grid_from_strategy(_trainer)
        set_action_grid(grid)
    return _trainer


async def get_trainer_async() -> CFRTrainer:
    """Async version - ensures only one training happens at a time."""
    global _trainer
    async with _training_lock:
        if _trainer is None:
            _trainer = CFRTrainer()
            if not _trainer.load(STRATEGY_FILE):
                logger.warning("No saved strategy found. Training CFR+...")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: _trainer.train(5000, sampling='external'))
                _trainer.save(STRATEGY_FILE)
                logger.info("Training complete.")
    return _trainer


def reload_strategy(filepath: str = None) -> bool:
    """Reload strategy from disk. Returns True on success."""
    global _trainer
    filepath = filepath or STRATEGY_FILE
    new_trainer = CFRTrainer()
    if new_trainer.load(filepath):
        _trainer = new_trainer
        logger.info(f"Reloaded strategy: {new_trainer.iterations} iterations, "
                    f"{len(new_trainer.nodes)} nodes")
        return True
    logger.error(f"Failed to reload strategy from {filepath}")
    return False


class GTODecision:
    """Represents a GTO-based decision."""

    def __init__(self, action: str, amount: int = 0, strategy_info: dict = None):
        self.action = action   # "fold", "check", "call", "raise"
        self.amount = amount
        self.strategy_info = strategy_info or {}


def gto_decide(player: Player, community_cards: list[Card], pot: int,
               current_bet: int, min_raise: int, big_blind: int,
               num_opponents: int = 1, betting_history: list[str] = None,
               is_in_position: bool = True) -> GTODecision:
    """
    Make a GTO-informed decision.

    Uses the CFR strategy table, with the hand bucketed by 2D
    (equity via E[HS^2], hand_type).

    Args:
        is_in_position: Whether this player is in position (acts last postflop).
                        In heads-up: BB is in position, SB is out of position.
    """
    t_start = time.perf_counter()

    try:
        trainer = get_trainer()
    except Exception as e:
        logger.error(f"Failed to get trainer: {e}")
        return _emergency_fallback(current_bet, player, {
            "error": str(e), "fallback": "emergency"
        })

    t_trainer = time.perf_counter()

    # Compute 2D hand strength bucket
    try:
        bucket = hand_strength_bucket(
            player.hole_cards, community_cards,
            num_opponents=num_opponents,
            simulations=200,
            use_ehs2=True
        )
    except Exception as e:
        logger.error(f"Bucket computation failed: {e}")
        bucket = 0  # Default to lowest bucket

    t_bucket = time.perf_counter()

    # Determine game phase
    phase = _get_phase(community_cards)

    # Convert betting history to abstract history (with error handling)
    abstract_history = _abstract_history(betting_history or [], phase)

    # Position encoding: in the CFR model, P0 ('oop') acts at even-length
    # history positions, P1 ('ip') at odd-length.
    position = 'oop' if len(abstract_history) % 2 == 0 else 'ip'

    # Sanity-check against caller's explicit position knowledge.
    # The history-derived value is authoritative when history is complete;
    # a mismatch usually means the caller passed a truncated history.
    caller_position = 'ip' if is_in_position else 'oop'
    if position != caller_position:
        logger.warning(
            f"Position mismatch: history-derived={position!r}, "
            f"caller-provided={caller_position!r} (is_in_position={is_in_position}). "
            f"history_len={len(abstract_history)}, phase={phase}. "
            f"Using history-derived position. Check that betting_history covers the full street."
        )

    # Look up strategy
    strategy = trainer.get_strategy(phase, bucket, abstract_history,
                                    position=position)

    t_lookup = time.perf_counter()

    # Sample action from strategy distribution
    actions = list(strategy.keys())
    probs = list(strategy.values())

    # Normalize (safety)
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    else:
        probs = [1.0 / len(actions)] * len(actions)

    chosen_action = random.choices(actions, weights=probs, k=1)[0]

    t_sample = time.perf_counter()

    # Build strategy info
    strategy_info = {
        "bucket": bucket,
        "phase": phase,
        "position": position,
        "position_caller": caller_position,
        "strategy": {ACTION_NAMES.get(Action(a), str(a)): f"{p:.1%}"
                     for a, p in strategy.items()},
        "chosen": ACTION_NAMES.get(Action(chosen_action), str(chosen_action)),
    }

    # Debug mode: include full diagnostics
    if _debug_mode:
        strategy_info["debug"] = {
            "abstract_history": list(abstract_history),
            "abstract_history_names": [
                ACTION_NAMES.get(Action(a), str(a)) for a in abstract_history
            ],
            "raw_distribution": {str(a): round(p, 4)
                                 for a, p in zip(actions, probs)},
            "strategy_version": STRATEGY_VERSION,
            "trainer_iterations": trainer.iterations,
            "trainer_nodes": len(trainer.nodes),
            "latency_ms": {
                "trainer_load": round((t_trainer - t_start) * 1000, 2),
                "bucket_compute": round((t_bucket - t_trainer) * 1000, 2),
                "strategy_lookup": round((t_lookup - t_bucket) * 1000, 2),
                "sampling": round((t_sample - t_lookup) * 1000, 2),
                "total": round((t_sample - t_start) * 1000, 2),
            },
        }

    # Convert abstract action to concrete game action
    return _to_concrete_action(
        Action(chosen_action), player, pot, current_bet, min_raise, big_blind,
        strategy_info=strategy_info
    )


def _get_phase(community_cards: list[Card]) -> str:
    n = len(community_cards)
    if n == 0:
        return "preflop"
    elif n == 3:
        return "flop"
    elif n == 4:
        return "turn"
    else:
        return "river"


def _abstract_history(history: list[str], phase: str) -> tuple:
    """Convert concrete action history to abstract action tuple.

    Delegates to the canonical implementation in abstraction.py so that
    engine.py and match_engine.py are guaranteed to use identical mappings.
    """
    return concrete_to_abstract_history(history, phase)


def _to_concrete_action(abstract_action: Action, player: Player,
                         pot: int, current_bet: int, min_raise: int,
                         big_blind: int, strategy_info: dict) -> GTODecision:
    """Convert an abstract CFR action to a concrete game action."""
    to_call = current_bet - player.current_bet

    if abstract_action == Action.FOLD:
        if to_call == 0:
            return GTODecision("check", 0, strategy_info)
        return GTODecision("fold", 0, strategy_info)

    elif abstract_action == Action.CHECK_CALL:
        if to_call == 0:
            return GTODecision("check", 0, strategy_info)
        return GTODecision("call", 0, strategy_info)

    elif abstract_action == Action.OPEN_RAISE:
        if player.chips <= to_call:
            return GTODecision("call", 0, strategy_info)
        raise_amount = max(min_raise, int(big_blind * 2.5))
        total = current_bet + raise_amount
        total = min(total, player.current_bet + player.chips)
        if total <= current_bet:
            return _fallback_action(to_call, strategy_info)
        return GTODecision("raise", total, strategy_info)

    elif abstract_action == Action.THREE_BET:
        if player.chips <= to_call:
            return GTODecision("call", 0, strategy_info)
        raise_amount = max(min_raise, current_bet * 3)
        total = raise_amount
        total = min(total, player.current_bet + player.chips)
        if total <= current_bet:
            return _fallback_action(to_call, strategy_info)
        return GTODecision("raise", total, strategy_info)

    elif abstract_action == Action.FOUR_BET:
        if player.chips <= to_call:
            return GTODecision("call", 0, strategy_info)
        raise_amount = max(min_raise, int(current_bet * 2.2))
        total = raise_amount
        total = min(total, player.current_bet + player.chips)
        if total <= current_bet:
            return _fallback_action(to_call, strategy_info)
        return GTODecision("raise", total, strategy_info)

    elif abstract_action == Action.BET_QUARTER_POT:
        if player.chips <= to_call:
            return GTODecision("call", 0, strategy_info)
        raise_amount = max(min_raise, pot // 4)
        total = current_bet + raise_amount
        total = min(total, player.current_bet + player.chips)
        if total <= current_bet:
            return _fallback_action(to_call, strategy_info)
        return GTODecision("raise", total, strategy_info)

    elif abstract_action == Action.BET_THIRD_POT:
        if player.chips <= to_call:
            return GTODecision("call", 0, strategy_info)
        raise_amount = max(min_raise, pot // 3)
        total = current_bet + raise_amount
        total = min(total, player.current_bet + player.chips)
        if total <= current_bet:
            return _fallback_action(to_call, strategy_info)
        return GTODecision("raise", total, strategy_info)

    elif abstract_action == Action.BET_HALF_POT:
        if player.chips <= to_call:
            return GTODecision("call", 0, strategy_info)
        raise_amount = max(min_raise, pot // 2)
        total = current_bet + raise_amount
        total = min(total, player.current_bet + player.chips)
        if total <= current_bet:
            return _fallback_action(to_call, strategy_info)
        return GTODecision("raise", total, strategy_info)

    elif abstract_action == Action.BET_TWO_THIRDS_POT:
        if player.chips <= to_call:
            return GTODecision("call", 0, strategy_info)
        raise_amount = max(min_raise, int(pot * 2 / 3))
        total = current_bet + raise_amount
        total = min(total, player.current_bet + player.chips)
        if total <= current_bet:
            return _fallback_action(to_call, strategy_info)
        return GTODecision("raise", total, strategy_info)

    elif abstract_action == Action.BET_THREE_QUARTER_POT:
        if player.chips <= to_call:
            return GTODecision("call", 0, strategy_info)
        raise_amount = max(min_raise, int(pot * 3 / 4))
        total = current_bet + raise_amount
        total = min(total, player.current_bet + player.chips)
        if total <= current_bet:
            return _fallback_action(to_call, strategy_info)
        return GTODecision("raise", total, strategy_info)

    elif abstract_action == Action.BET_POT:
        if player.chips <= to_call:
            return GTODecision("call", 0, strategy_info)
        raise_amount = max(min_raise, pot)
        total = current_bet + raise_amount
        total = min(total, player.current_bet + player.chips)
        if total <= current_bet:
            return _fallback_action(to_call, strategy_info)
        return GTODecision("raise", total, strategy_info)

    elif abstract_action == Action.BET_OVERBET:
        if player.chips <= to_call:
            return GTODecision("call", 0, strategy_info)
        raise_amount = max(min_raise, int(pot * 1.25))
        total = current_bet + raise_amount
        total = min(total, player.current_bet + player.chips)
        if total <= current_bet:
            return _fallback_action(to_call, strategy_info)
        return GTODecision("raise", total, strategy_info)

    elif abstract_action == Action.BET_DOUBLE_POT:
        if player.chips <= to_call:
            return GTODecision("call", 0, strategy_info)
        raise_amount = max(min_raise, pot * 2)
        total = current_bet + raise_amount
        total = min(total, player.current_bet + player.chips)
        if total <= current_bet:
            return _fallback_action(to_call, strategy_info)
        return GTODecision("raise", total, strategy_info)

    elif abstract_action == Action.BET_TRIPLE_POT:
        if player.chips <= to_call:
            return GTODecision("call", 0, strategy_info)
        raise_amount = max(min_raise, pot * 3)
        total = current_bet + raise_amount
        total = min(total, player.current_bet + player.chips)
        if total <= current_bet:
            return _fallback_action(to_call, strategy_info)
        return GTODecision("raise", total, strategy_info)

    elif abstract_action == Action.DONK_SMALL:
        if player.chips <= to_call:
            return GTODecision("call", 0, strategy_info)
        raise_amount = max(min_raise, pot // 4)
        total = current_bet + raise_amount
        total = min(total, player.current_bet + player.chips)
        if total <= current_bet:
            return _fallback_action(to_call, strategy_info)
        return GTODecision("raise", total, strategy_info)

    elif abstract_action == Action.DONK_MEDIUM:
        if player.chips <= to_call:
            return GTODecision("call", 0, strategy_info)
        raise_amount = max(min_raise, pot // 2)
        total = current_bet + raise_amount
        total = min(total, player.current_bet + player.chips)
        if total <= current_bet:
            return _fallback_action(to_call, strategy_info)
        return GTODecision("raise", total, strategy_info)

    elif abstract_action == Action.ALL_IN:
        total = player.current_bet + player.chips
        if total <= current_bet:
            return GTODecision("call", 0, strategy_info)
        return GTODecision("raise", total, strategy_info)

    return _fallback_action(to_call, strategy_info)


def _fallback_action(to_call: int, strategy_info: dict) -> GTODecision:
    """Fallback when a raise isn't possible."""
    if to_call == 0:
        return GTODecision("check", 0, strategy_info)
    return GTODecision("call", 0, strategy_info)


def _emergency_fallback(current_bet: int, player: Player,
                         strategy_info: dict) -> GTODecision:
    """Emergency fallback when the trainer is unavailable."""
    to_call = current_bet - player.current_bet
    logger.error("Using emergency fallback — no trained strategy available")
    return _fallback_action(to_call, strategy_info)
