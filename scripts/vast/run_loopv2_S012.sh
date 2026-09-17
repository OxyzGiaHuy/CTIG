#!/bin/bash
cd /workspace/ctig17
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)
python scripts/run_loop_v2.py --config configs/vast_a100.yaml --set prompts_path=data/prompts_simple.json \
  --id S012 --model sdxl_base --prompt-source culture_trip --run-name loopv2_S012
echo "LOOPV2_DONE $(date +%T)"
