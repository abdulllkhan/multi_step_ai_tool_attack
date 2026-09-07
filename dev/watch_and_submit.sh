#!/usr/bin/env bash
cd /home/abdulllkhan/Documents/multi_step_tool_attack
K=".venv/bin/kaggle"
KERNEL="abdullllkhan/ai-agent-security-attack"
COMP="ai-agent-security-multi-step-tool-attacks"
VER=10
MSG="v10 fn-syntax EXFIL: 2.1x throughput vs v9 (local 4.0 vs 8.3s/cand, fire=1.00; end-to-end 5.13 vs 2.43)"
echo "watcher start $(date +%H:%M:%S) — polling kernel $KERNEL for COMPLETE, then submit v$VER"
for i in $(seq 1 45); do
  ST=$($K kernels status "$KERNEL" 2>/dev/null)
  echo "[$(date +%H:%M:%S)] $ST"
  if echo "$ST" | grep -q "COMPLETE"; then
    echo ">> kernel COMPLETE -> submitting v$VER"
    $K competitions submit -c "$COMP" -k "$KERNEL" -v "$VER" -f submission.csv -m "$MSG" 2>&1
    echo ">> SUBMIT_ATTEMPTED"
    sleep 8
    echo "=== current submissions ==="
    $K competitions submissions -c "$COMP" 2>&1 | head -6
    exit 0
  fi
  if echo "$ST" | grep -qiE "error|cancel|fail"; then
    echo ">> kernel FAILED/CANCELLED -> NOT submitting"; exit 2
  fi
  sleep 60
done
echo ">> TIMEOUT waiting for kernel COMPLETE"; exit 3
