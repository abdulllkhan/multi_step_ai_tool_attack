#!/usr/bin/env bash
cd /home/abdulllkhan/Documents/multi_step_tool_attack
echo "=== pre-downloading google/gemma-3-4b-it (cached HF token) ==="
.venv/bin/python -c "from huggingface_hub import snapshot_download; p=snapshot_download('google/gemma-3-4b-it'); print('downloaded to', p)" 2>&1 | tail -5
echo "=== running gemma lab (fn + multihop) ==="
.venv/bin/python dev/lab_model.py --model gemma --k 6 --score-budget 200 2>&1
