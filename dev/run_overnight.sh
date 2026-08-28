#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Overnight, unattended attack-config optimization for the Kaggle competition.
#
#   1. stop the Qwen server        (frees the GB10 for the target model)
#   2. run the variant sweep       (dev/overnight_optimize.py, scored vs gpt-oss-20b)
#   3. bake the winning config      into attack.py (on a branch)
#   4. rebuild submission.ipynb     (staged — NOT submitted)
#   5. restart the Qwen server      (back for interactive use in the morning)
#   6. write a summary you review over coffee
#
# Nothing is submitted to Kaggle. Runs detached; survives logout (linger is on).
#
# Usage:
#   MODEL=gpt_oss GUARDRAIL=optimal BUDGET_S=150 DEADLINE_MIN=420 \
#     nohup bash dev/run_overnight.sh > logs_overnight.txt 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
PY="$REPO/.venv/bin/python"

MODEL="${MODEL:-gpt_oss}"
GUARDRAIL="${GUARDRAIL:-optimal}"
BUDGET_S="${BUDGET_S:-150}"
DEADLINE_MIN="${DEADLINE_MIN:-420}"
VARIANTS="${VARIANTS:-dev/variants.json}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$REPO/dev/overnight-$STAMP.log"
BRANCH="agent/overnight-$STAMP"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== overnight optimize start ($STAMP) — model=$MODEL guardrail=$GUARDRAIL budget=${BUDGET_S}s deadline=${DEADLINE_MIN}min ==="

# 1. Work on a dedicated branch; stash anything dirty.
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { log "not a git repo"; exit 1; }
if ! git diff --quiet || ! git diff --cached --quiet; then
  log "stashing dirty tree"; git stash push -u -m "overnight-$STAMP" >>"$LOG" 2>&1
fi
git switch -c "$BRANCH" >>"$LOG" 2>&1 || git switch "$BRANCH" >>"$LOG" 2>&1
log "branch: $BRANCH"

# 2. Free the GPU: stop the Qwen server (best-effort).
if systemctl --user is-active qwen-server.service >/dev/null 2>&1; then
  systemctl --user stop qwen-server.service && log "stopped qwen-server (GPU freed)"
  QWEN_WAS_UP=1
else
  QWEN_WAS_UP=0; log "qwen-server was not running"
fi

restart_qwen(){ [ "$QWEN_WAS_UP" = "1" ] && systemctl --user start qwen-server.service && log "restarted qwen-server"; }
trap restart_qwen EXIT   # always bring Qwen back, even on error/kill

# 3. Run the sweep (loads target model once, scores every variant).
log "running sweep..."
"$PY" dev/overnight_optimize.py \
    --model "$MODEL" --guardrail "$GUARDRAIL" \
    --budget-s "$BUDGET_S" --deadline-min "$DEADLINE_MIN" \
    --variants "$VARIANTS" \
    --out dev/best_config.json --results dev/optimize_results.jsonl >>"$LOG" 2>&1
log "sweep finished (rc=$?)"

# 4. Bake the winning config into attack.py's defaults, then verify it still parses.
if [ -f dev/best_config.json ]; then
  "$PY" dev/apply_best_config.py dev/best_config.json attack.py >>"$LOG" 2>&1 \
    && log "baked best config into attack.py" \
    || log "WARN: could not bake best config (attack.py left unchanged)"
  "$PY" -c "import ast,sys; ast.parse(open('attack.py').read()); print('attack.py parses OK')" >>"$LOG" 2>&1 \
    || { log "ERROR: attack.py no longer parses — reverting"; git checkout -- attack.py; }
else
  log "no best_config.json produced — attack.py unchanged"
fi

# 5. Rebuild the submission notebook from attack.py (staged only).
if [ -f notebook/build_notebook.py ]; then
  "$PY" notebook/build_notebook.py >>"$LOG" 2>&1 && log "rebuilt notebook/submission.ipynb (staged, NOT submitted)"
fi

# 6. Commit the staged result on the branch; write a summary.
git add -A >>"$LOG" 2>&1
git commit -m "overnight optimize $STAMP: best config baked, notebook rebuilt (staged)" >>"$LOG" 2>&1 || log "nothing to commit"

SUMMARY="$REPO/dev/OVERNIGHT_SUMMARY-$STAMP.md"
{
  echo "# Overnight optimize — $STAMP"
  echo; echo "- Branch: \`$BRANCH\`  (nothing submitted to Kaggle)"
  echo "- Target: $MODEL x $GUARDRAIL, ${BUDGET_S}s/variant"
  echo; echo "## Best config"
  [ -f dev/best_config.json ] && sed 's/^/    /' dev/best_config.json || echo "    (none)"
  echo; echo "## Ranked variants"
  if [ -f dev/optimize_results.jsonl ]; then
    "$PY" - <<'PYEOF'
import json,glob
rows=[json.loads(l) for l in open("dev/optimize_results.jsonl") if l.strip()]
rows=[r for r in rows if "score" in r]
for r in sorted(rows,key=lambda x:-x["score"]):
    print(f"    {r['score']:8.3f}  n={r.get('n_findings','?'):>5}  {r['name']}")
PYEOF
  fi
  echo; echo "## To submit (when you've reviewed):"
  echo "    KAGGLE_USERNAME=<you> KAGGLE_KEY=\$(grep -oP '(?<=KAGGLE_API_KEY=).*' .env) bash dev/push_to_kaggle.sh"
} > "$SUMMARY"
log "wrote $SUMMARY"
log "=== DONE — review $SUMMARY. Branch $BRANCH holds the staged notebook. Nothing was submitted. ==="
