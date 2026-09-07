#!/usr/bin/env bash
# Quick fitness check: run attack.py against the real target and print the SDK score.
# NOTE: needs the GPU free for gpt-oss (~26GB) — stop the Qwen server first if it's up:
#   systemctl --user stop qwen-server.service
set -euo pipefail
cd "$(dirname "$0")/.."
BUDGET="${1:-150}"
.venv/bin/python dev/eval_real.py --model "${MODEL:-gpt_oss}" --guardrail "${GUARDRAIL:-optimal}" --budget-s "$BUDGET"
