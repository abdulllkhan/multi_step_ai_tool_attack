#!/usr/bin/env bash
cd /home/abdulllkhan/Documents/multi_step_tool_attack
echo "################ ADAPTIVE on GPT-OSS (must lock fn, no regression) ################"
.venv/bin/python dev/lab_model.py --model gpt_oss --k 4 --score-budget 200 2>&1 | grep -aE "PHASE|env=|score=|RANK|proj|ERR"
echo; echo "################ ADAPTIVE on GEMMA (must lock gecho, fire) ################"
GEMMA_MODEL_ID="unsloth/gemma-3-4b-it" .venv/bin/python dev/lab_model.py --model gemma --k 4 --score-budget 200 2>&1 | grep -aE "PHASE|env=|score=|RANK|proj|ERR"
echo "################ ADAPTIVE TEST DONE ################"
