#!/bin/bash
# Run remaining ablation experiments sequentially
# Exp A, B from main repo; Exp E from worktree (different abstraction)
set -e

VENV="venv/bin/python"
MAIN_DIR="/home/djclarke/drinking/poker"
WORKTREE_DIR="/home/djclarke/drinking/poker/.claude/worktrees/agent-a588b571"

echo "===== Exp A: Revert 3x schedule to 2x ====="
cd "$MAIN_DIR"
$VENV run_ablation.py --name exp_A_2x_schedule \
    --iterations 50000000 --workers 6 \
    --config phase_schedule_mode=0

echo ""
echo "===== Exp B: Revert new dampening to old ====="
cd "$MAIN_DIR"
$VENV run_ablation.py --name exp_B_old_dampen \
    --iterations 50000000 --workers 6 \
    --config allin_dampen_mode=0

echo ""
echo "===== Exp E: EQ0 split (9 equity buckets) ====="
cd "$WORKTREE_DIR"
$VENV run_ablation.py --name exp_E_eq0_split \
    --iterations 50000000 --workers 6

echo ""
echo "===== ALL EXPERIMENTS COMPLETE ====="
cd "$MAIN_DIR"
$VENV run_ablation.py --list
