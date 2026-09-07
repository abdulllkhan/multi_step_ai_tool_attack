#!/usr/bin/env bash
cd /home/abdulllkhan/Documents/multi_step_tool_attack
K=".venv/bin/kaggle"; COMP="ai-agent-security-multi-step-tool-attacks"
V10_REF="55832820"
echo "long-watcher start $(date +%F_%H:%M:%S) — poll v10($V10_REF) score + HF_TOKEN, 20min ticks, ~5h heartbeat"
for i in $(seq 1 15); do
  # 1) HF_TOKEN arrived?
  if [ -f .env ] && grep -qiE '^(HF_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACE_TOKEN)=..' .env; then
    echo "[$(date +%H:%M:%S)] HF_TOKEN DETECTED in .env -> waking to farm gemma"; exit 10
  fi
  # 2) v10 public score landed?
  SUBS=$($K competitions submissions -c "$COMP" 2>/dev/null)
  LINE=$(echo "$SUBS" | grep "$V10_REF")
  if echo "$LINE" | grep -q "COMPLETE"; then
    SCORE=$(echo "$LINE" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    echo "[$(date +%H:%M:%S)] V10 SCORED: publicScore=$SCORE"; echo "$LINE"; exit 11
  fi
  echo "[$(date +%H:%M:%S)] tick $i/15 — v10 still pending, no HF_TOKEN"
  sleep 1200
done
echo "[$(date +%H:%M:%S)] 5h heartbeat — nothing changed"; exit 12
