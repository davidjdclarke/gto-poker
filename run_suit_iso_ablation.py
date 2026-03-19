#!/usr/bin/env python3
"""Ablation: measure suit iso effect on B0 exploitability (no retraining).

Runs exploitability_multiseed on the B0 strategy with and without suit
isomorphism canonicalization in hand_strength_bucket().
"""
import sys
import time
import numpy as np

from server.gto.cfr import CFRTrainer
from server.gto.exploitability import exploitability_multiseed
from server.gto import equity as equity_mod

B0_STRATEGY = "experiments/best/v9_B0_100M_allbots_positive.json"
SEEDS = [42, 123, 456]


def run_exploitability(label, suit_iso_enabled):
    equity_mod.SUIT_ISO_ENABLED = suit_iso_enabled
    trainer = CFRTrainer()
    if not trainer.load(B0_STRATEGY):
        print(f"ERROR: Cannot load {B0_STRATEGY}")
        sys.exit(1)
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  suit_iso={suit_iso_enabled}, strategy={B0_STRATEGY}")
    print(f"{'='*50}")
    t0 = time.time()
    result = exploitability_multiseed(trainer, samples=500, seeds=SEEDS)
    elapsed = time.time() - t0
    ci_half = (result['ci_95_high'] - result['ci_95_low']) / 2
    print(f"  Exploitability: {result['mean']:.4f} ± {ci_half:.4f}")
    print(f"  Per-phase: {result['per_phase']}")
    print(f"  Time: {elapsed:.1f}s")
    result['ci_half'] = ci_half
    return result


def main():
    r_off = run_exploitability("B0 — suit iso OFF (original)", False)
    r_on = run_exploitability("B0 — suit iso ON (WS3)", True)

    print(f"\n{'='*50}")
    print(f"  SUIT ISO ABLATION SUMMARY")
    print(f"{'='*50}")
    print(f"  OFF: {r_off['mean']:.4f} ± {r_off['ci_half']:.4f}")
    print(f"  ON:  {r_on['mean']:.4f} ± {r_on['ci_half']:.4f}")
    delta = r_on['mean'] - r_off['mean']
    print(f"  Delta (ON - OFF): {delta:+.4f}")
    if abs(delta) < max(r_off['ci_half'], r_on['ci_half']):
        print(f"  Verdict: WITHIN noise — suit iso has no significant effect on exploitability")
    elif delta > 0:
        print(f"  Verdict: suit iso INCREASES exploitability (NEGATIVE)")
    else:
        print(f"  Verdict: suit iso DECREASES exploitability (POSITIVE)")


if __name__ == "__main__":
    main()
