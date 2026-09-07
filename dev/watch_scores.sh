#!/usr/bin/env bash
cd /home/abdulllkhan/Documents/multi_step_tool_attack
K=".venv/bin/kaggle"; COMP="ai-agent-security-multi-step-tool-attacks"
V10="55832820"; V11="55834790"
echo "score-watcher start $(date +%F_%H:%M:%S) — poll v10($V10) & v11($V11), 20min ticks, ~5h heartbeat"
prev_done=0
for i in $(seq 1 15); do
  SUBS=$($K competitions submissions -c "$COMP" 2>/dev/null)
  L10=$(echo "$SUBS" | grep "$V10"); L11=$(echo "$SUBS" | grep "$V11")
  done=0
  echo "$L10" | grep -q "COMPLETE" && done=$((done+1))
  echo "$L11" | grep -q "COMPLETE" && done=$((done+1))
  if [ "$done" -gt "$prev_done" ]; then
    S10=$(echo "$L10" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    S11=$(echo "$L11" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    echo "[$(date +%H:%M:%S)] SCORE UPDATE -> v10=${S10:-pending} v11=${S11:-pending} (base v9=47.565)"
    exit 11
  fi
  echo "[$(date +%H:%M:%S)] tick $i/15 — v10/v11 pending ($done/2 complete)"
  sleep 1200
done
echo "[$(date +%H:%M:%S)] 5h heartbeat — scores still pending"; exit 12
