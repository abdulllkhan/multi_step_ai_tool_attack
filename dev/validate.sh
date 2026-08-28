#!/usr/bin/env bash
# Pre-submit local validation gate. Run this before pushing attack.py to Kaggle.
# Usage: bash dev/validate.sh [real_eval_budget_s]   (default 200)
#
# Gate:
#   1. Official SDK static validation (syntax/imports/structure).
#   2. Deterministic regression (GPU-free): machinery + scoring under AllowAll.
#   3. Real-model eval on the GB10: attack vs real gpt-oss + OptimalGuardrail
#      (this mirrors the gpt_oss_public leaderboard cell). Compare the SUMMARY
#      score against the last submitted version — only push if it improved.
set -e
cd "$(dirname "$0")/.."
BUDGET="${1:-200}"

echo "=================================================================="
echo "[1/3] Official SDK static validation"
echo "=================================================================="
.venv/bin/aicomp validate redteam attack.py 2>&1 | tail -3

echo
echo "=================================================================="
echo "[2/3] Deterministic regression (GPU-free, AllowAll bookend)"
echo "=================================================================="
.venv/bin/python dev/harness.py --guardrail allow --budget-s 15 2>&1 | tail -2

echo
echo "=================================================================="
echo "[3/3] Real-model eval: gpt-oss + OptimalGuardrail (public proxy, ${BUDGET}s)"
echo "=================================================================="
.venv/bin/python dev/eval_real.py --model gpt_oss --guardrail optimal --budget-s "$BUDGET" 2>&1 \
  | grep -aE "SUMMARY|PREDICATES|WIN"

echo
echo "GATE DONE. Push only if the real-model SUMMARY score beats the last submitted version."
