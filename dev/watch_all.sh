#!/usr/bin/env bash
cd /home/abdulllkhan/Documents/multi_step_tool_attack
K=".venv/bin/kaggle"; COMP="ai-agent-security-multi-step-tool-attacks"; V10_REF="55832820"
echo "watch_all start $(date +%F_%H:%M:%S) — poll v10($V10_REF) score + gemma-3-4b license, 20min ticks, ~5h heartbeat"
for i in $(seq 1 15); do
  if .venv/bin/python -c "from huggingface_hub import HfApi; HfApi().auth_check('google/gemma-3-4b-it')" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] gemma-3-4b DOWNLOAD ACCESS granted -> wake to farm real gemma target"; exit 10
  fi
  LINE=$($K competitions submissions -c "$COMP" 2>/dev/null | grep "$V10_REF")
  if echo "$LINE" | grep -q "COMPLETE"; then
    SCORE=$(echo "$LINE" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    echo "[$(date +%H:%M:%S)] V10 SCORED: publicScore=$SCORE"; echo "$LINE"; exit 11
  fi
  echo "[$(date +%H:%M:%S)] tick $i/15 — v10 pending, gemma-3-4b still gated"
  sleep 1200
done
echo "[$(date +%H:%M:%S)] 5h heartbeat — nothing changed"; exit 12
