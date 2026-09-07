#!/usr/bin/env bash
cd /home/abdulllkhan/Documents/multi_step_tool_attack
echo "=== downloading google/gemma-4-26B-A4B-it (~52GB, cached token) $(date +%H:%M:%S) ==="
.venv/bin/python -c "
from huggingface_hub import snapshot_download
p=snapshot_download('google/gemma-4-26B-A4B-it', allow_patterns=['*.safetensors','*.json','*.model','*.txt','tokenizer*'])
print('downloaded to', p)
" 2>&1 | tail -4
echo "=== running gemma_4 lab (fn + multihop) $(date +%H:%M:%S) ==="
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
.venv/bin/python dev/lab_model.py --model gemma_4 --k 6 --score-budget 200 2>&1
echo "=== gemma4 lab done $(date +%H:%M:%S) ==="
