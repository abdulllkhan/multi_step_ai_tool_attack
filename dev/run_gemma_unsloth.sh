#!/usr/bin/env bash
cd /home/abdulllkhan/Documents/multi_step_tool_attack
export GEMMA_MODEL_ID="unsloth/gemma-3-4b-it"
echo "=== downloading unsloth/gemma-3-4b-it (ungated mirror, ~8.6GB) $(date +%H:%M:%S) ==="
.venv/bin/python -c "
from huggingface_hub import snapshot_download
p=snapshot_download('unsloth/gemma-3-4b-it', allow_patterns=['*.safetensors','*.json','*.model','*.txt','*.jinja','tokenizer*'])
print('downloaded to', p)
" 2>&1 | tail -3
echo "=== running gemma lab (real target via mirror; fn + multihop) $(date +%H:%M:%S) ==="
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
GEMMA_MODEL_ID="unsloth/gemma-3-4b-it" .venv/bin/python dev/lab_model.py --model gemma --k 6 --score-budget 150 2>&1
echo "=== gemma lab done $(date +%H:%M:%S) ==="
