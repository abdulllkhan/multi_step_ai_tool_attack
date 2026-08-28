#!/usr/bin/env bash
# Push the submission notebook to Kaggle as a competition kernel, then it runs on
# Kaggle's GPU and (after it completes) can be submitted to the leaderboard.
#
# Usage:  KAGGLE_KEY=$(grep -oP '(?<=KAGGLE_API_KEY=).*' .env) \
#         KAGGLE_USERNAME=<your-username> bash dev/push_to_kaggle.sh
#
# Requires: your Kaggle username + key (the CLI needs BOTH), and that you have
# accepted the competition rules on the website first.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${KAGGLE_USERNAME:?set KAGGLE_USERNAME=<your kaggle username>}"
: "${KAGGLE_KEY:?set KAGGLE_KEY=<your kaggle api key>}"
export KAGGLE_USERNAME KAGGLE_KEY

KAGGLE=".venv/bin/kaggle"
COMP="ai-agent-security-multi-step-tool-attacks"

echo ">> Rebuilding notebook from attack.py ..."
.venv/bin/python notebook/build_notebook.py

echo ">> Writing kernel metadata for user '$KAGGLE_USERNAME' ..."
sed "s/__KAGGLE_USERNAME__/${KAGGLE_USERNAME}/" notebook/kernel-metadata.json \
  > notebook/kernel-metadata.local.json
mv notebook/kernel-metadata.local.json notebook/kernel-metadata.json

echo ">> Verifying auth ..."
"$KAGGLE" competitions list -s "$COMP" >/dev/null && echo "   auth OK"

echo ">> Pushing kernel (queues a GPU run on Kaggle) ..."
"$KAGGLE" kernels push -p notebook

echo
echo ">> Pushed. Track the run with:"
echo "   $KAGGLE kernels status ${KAGGLE_USERNAME}/ai-agent-security-attack"
echo ">> When the run COMPLETES, submit it from the notebook's 'Output' -> 'Submit',"
echo "   or (if enabled for this comp):"
echo "   $KAGGLE competitions submit -c $COMP -f submission.csv -m 'go-explore v1'"
