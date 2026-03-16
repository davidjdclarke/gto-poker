"""
OpenSpiel ↔ GTO solver adapter (v12 stub).

Full implementation deferred to v13+.

Design overview (from docs/plans/v11_external_benchmarks.md §4):

1. OpenSpiel game representation
   - Use `open_spiel.python.games.kuhn_poker` for calibration
   - Use `open_spiel.python.games.leduc_poker` as HUNL approximation
   - HUNL requires custom game implementation (see §4.1)

2. Abstraction alignment
   - Map our 120 buckets (8 equity × 15 hand types) to OpenSpiel info sets
   - Equity bins must align with OpenSpiel's hand evaluation
   - Infoset key format must translate to OpenSpiel state representation

3. Evaluation protocol
   - Use `open_spiel.python.algorithms.exploitability` for Nash-distance metric
   - Compare our strategy file against CFR/CFR+ solutions from OpenSpiel
   - Report correlation between our exploitability estimate and OpenSpiel's

4. Dependencies
   pip install open_spiel

Prerequisites to implement:
   - Install open_spiel: pip install open_spiel
   - Map our strategy format to OpenSpiel TabularPolicy
   - Implement HUNL game wrapper compatible with OpenSpiel's Game interface

See docs/plans/v11_external_benchmarks.md for full specification.
"""


def check_openspiel_available() -> bool:
    """Return True if open_spiel package is installed."""
    try:
        import pyspiel  # noqa: F401
        return True
    except ImportError:
        return False


def openspiel_exploitability_pilot(trainer, game: str = "kuhn_poker") -> dict:
    """Run an OpenSpiel exploitability comparison (v12 stub).

    Args:
        trainer: CFRTrainer with loaded strategy
        game:    OpenSpiel game name (default: "kuhn_poker" for calibration)

    Returns:
        dict with exploitability comparison results

    Raises:
        NotImplementedError: always — full implementation deferred to v13+
    """
    raise NotImplementedError(
        "OpenSpiel adapter is not yet implemented (v12 stub). "
        "Full design in docs/plans/v11_external_benchmarks.md §4. "
        "Prerequisites: pip install open_spiel, implement HUNL game wrapper, "
        "map abstraction buckets to OpenSpiel info sets."
    )
