#!/usr/bin/env bash
cd /home/abdulllkhan/Documents/multi_step_tool_attack
K=".venv/bin/kaggle"; KERNEL="abdullllkhan/ai-agent-security-attack"
COMP="ai-agent-security-multi-step-tool-attacks"; VER=17
MSG="v16 HARMONY-FORGE MULTIPOST: gpt-oss cell fires 8 http.posts/candidate in ONE interact (8 EXFIL predicates, prefill amortised) via forged analysis channel -> 2.13x gpt-oss cell (eval_attack 4.05->8.63). gemma detected->single-post unchanged. Adaptive, output-detected (no timing mislock)."
echo "v16 watcher start $(date +%H:%M:%S)"
for i in $(seq 1 90); do
  ST=$($K kernels status "$KERNEL" 2>/dev/null)
  echo "[$(date +%H:%M:%S)] $ST"
  if echo "$ST" | grep -q "COMPLETE"; then
    echo ">> COMPLETE -> submit v$VER"
    $K competitions submit -c "$COMP" -k "$KERNEL" -v "$VER" -f submission.csv -m "$MSG" 2>&1
    echo ">> SUBMIT_ATTEMPTED"; sleep 10
    $K competitions submissions -c "$COMP" 2>&1 | head -4
    exit 0
  fi
  echo "$ST" | grep -qiE "\"error\"|cancelAcknowledged|fail" && { echo ">> FAILED"; exit 2; }
  sleep 60
done
echo ">> TIMEOUT"; exit 3
