#!/usr/bin/env bash
cd /home/abdulllkhan/Documents/multi_step_tool_attack
K=".venv/bin/kaggle"; KERNEL="abdullllkhan/ai-agent-security-attack"
COMP="ai-agent-security-multi-step-tool-attacks"; VER=15
MSG="v15 OUTPUT-DETECTOR: 2 fast fn probes -> detect model from output (gemma-4 emits thought/channel marker, gpt-oss clean) -> fn(gpt-oss)/nothink(gemma-4). Fixes v14's slow-probe budget starvation (47.26). Gets +18% gemma throughput safely; gpt-oss unchanged (N45)."
echo "v15 watcher start $(date +%H:%M:%S)"
for i in $(seq 1 50); do
  ST=$($K kernels status "$KERNEL" 2>/dev/null)
  echo "[$(date +%H:%M:%S)] $ST"
  if echo "$ST" | grep -q "COMPLETE"; then
    echo ">> COMPLETE -> submit v$VER"
    $K competitions submit -c "$COMP" -k "$KERNEL" -v "$VER" -f submission.csv -m "$MSG" 2>&1
    echo ">> SUBMIT_ATTEMPTED"; sleep 8
    $K competitions submissions -c "$COMP" 2>&1 | head -4
    exit 0
  fi
  echo "$ST" | grep -qiE "error|cancelAcknowledged|fail" && { echo ">> FAILED"; exit 2; }
  sleep 60
done
echo ">> TIMEOUT"; exit 3
