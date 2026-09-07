#!/usr/bin/env bash
cd /home/abdulllkhan/Documents/multi_step_tool_attack
K=".venv/bin/kaggle"; KERNEL="abdullllkhan/ai-agent-security-attack"
COMP="ai-agent-security-multi-step-tool-attacks"; VER=14
MSG="v14 THROUGHPUT-ADAPTIVE: probe fn+nothink per cell, lock fastest-firing. Board runs gpt-oss GGUF (locks fn) + gemma-4-26B GGUF (locks nothink, ~+18% gemma throughput). Both cells fire; per-cell optimum vs v10 fn-both."
echo "v14 watcher start $(date +%H:%M:%S)"
for i in $(seq 1 50); do
  ST=$($K kernels status "$KERNEL" 2>/dev/null)
  echo "[$(date +%H:%M:%S)] $ST"
  if echo "$ST" | grep -q "COMPLETE"; then
    echo ">> COMPLETE -> submit v$VER"
    $K competitions submit -c "$COMP" -k "$KERNEL" -v "$VER" -f submission.csv -m "$MSG" 2>&1
    echo ">> SUBMIT_ATTEMPTED"; sleep 8
    $K competitions submissions -c "$COMP" 2>&1 | head -5
    exit 0
  fi
  echo "$ST" | grep -qiE "error|cancelAcknowledged|fail" && { echo ">> FAILED"; exit 2; }
  sleep 60
done
echo ">> TIMEOUT"; exit 3
