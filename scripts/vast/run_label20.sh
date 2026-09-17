#!/bin/bash
cd /workspace/ctig17 && git pull -q && git log --oneline -1
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)
echo "=== label20 $(date +%T): 20 prompt phủ 8 nhóm, bare + system cùng seed, 2 model nền"
python scripts/run_walkthrough.py --config configs/vast_a100.yaml --set prompts_path=data/prompts_simple.json \
  --ids S001,S002,S003,S004,S005,S006,S007,S009,S010,S011,S012,S013,S014,S019,S020,S021,S022,S023,S038,S041 \
  --models "sdxl_base#bare,sdxl_base,realvis_xl#bare,realvis_xl" --run-name label20
echo "LABEL20_DONE $(date +%T)"
