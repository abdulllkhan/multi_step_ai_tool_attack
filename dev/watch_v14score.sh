#!/usr/bin/env bash
cd /home/abdulllkhan/Documents/multi_step_tool_attack
K=".venv/bin/kaggle"; COMP="ai-agent-security-multi-step-tool-attacks"; REF="55874450"
for i in $(seq 1 15); do
  LINE=$($K competitions submissions -c "$COMP" 2>/dev/null | grep "$REF")
  if echo "$LINE" | grep -q "COMPLETE"; then
    SC=$(echo "$LINE" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    echo "[$(date +%H:%M:%S)] V14 SCORED: $SC (v10 baseline=57.205)"; exit 11
  fi
  echo "[$(date +%H:%M:%S)] tick $i v14 pending"; sleep 1200
done
echo "5h heartbeat, v14 pending"; exit 12
