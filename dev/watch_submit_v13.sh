#!/usr/bin/env bash
cd /home/abdulllkhan/Documents/multi_step_tool_attack
K=".venv/bin/kaggle"; KERNEL="abdullllkhan/ai-agent-security-attack"
COMP="ai-agent-security-multi-step-tool-attacks"; VER=13
MSG="v13: fn(gpt-oss) + refined 9-style gemma portfolio. Adds benign/jailbreak framings (telemetry/pipeline/content-type) to dodge gemma REFUSAL; drops gcopy. Parallel refusal-dodge test vs v12 anti-fence. gpt-oss==v10(57.2)"
echo "v13 watcher start $(date +%H:%M:%S)"
for i in $(seq 1 45); do
  ST=$($K kernels status "$KERNEL" 2>/dev/null)
  echo "[$(date +%H:%M:%S)] $ST"
  if echo "$ST" | grep -q "COMPLETE"; then
    echo ">> COMPLETE -> submit v$VER"
    $K competitions submit -c "$COMP" -k "$KERNEL" -v "$VER" -f submission.csv -m "$MSG" 2>&1
    echo ">> SUBMIT_ATTEMPTED"; sleep 8
    $K competitions submissions -c "$COMP" 2>&1 | head -6
    exit 0
  fi
  echo "$ST" | grep -qiE "error|cancelAcknowledged|fail" && { echo ">> FAILED"; exit 2; }
  sleep 60
done
echo ">> TIMEOUT"; exit 3
