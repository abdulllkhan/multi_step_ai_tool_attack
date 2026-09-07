#!/usr/bin/env bash
cd /home/abdulllkhan/Documents/multi_step_tool_attack
K=".venv/bin/kaggle"; KERNEL="abdullllkhan/ai-agent-security-attack"
COMP="ai-agent-security-multi-step-tool-attacks"; VER=11
MSG="v11 adaptive per-cell: fn(gpt-oss)+gecho JSON-echo(gemma). Unlocks gemma cell 0->firing (local N=22/120s, gecho fire=1.00); gpt-oss unchanged from v10"
echo "v11 watcher start $(date +%H:%M:%S)"
for i in $(seq 1 45); do
  ST=$($K kernels status "$KERNEL" 2>/dev/null)
  echo "[$(date +%H:%M:%S)] $ST"
  if echo "$ST" | grep -q "COMPLETE"; then
    echo ">> COMPLETE -> submit v$VER"
    $K competitions submit -c "$COMP" -k "$KERNEL" -v "$VER" -f submission.csv -m "$MSG" 2>&1
    echo ">> SUBMIT_ATTEMPTED"; sleep 8
    $K competitions submissions -c "$COMP" 2>&1 | head -5
    exit 0
  fi
  if echo "$ST" | grep -qiE "error|cancelAcknowledged|fail"; then echo ">> FAILED, not submitting"; exit 2; fi
  sleep 60
done
echo ">> TIMEOUT"; exit 3
