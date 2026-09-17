#!/bin/bash
cd /workspace/ctig17
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)
CFG=configs/vast_a100.yaml
COMMON="--config $CFG --set prompts_path=data/prompts_simple.json --set t2i.render=bare --set multigen.n_candidates=1 --set agents.max_revisions=0 --ids S001 --models sdxl_base,realvis_xl,flux_dev"
echo "=== NHÁNH A: prompt gốc $(date +%T)"
python scripts/run_walkthrough.py $COMMON --prompt-source original --run-name S001_armA
echo "=== NHÁNH B: prompt Culture-TRIP $(date +%T)"
python scripts/run_walkthrough.py $COMMON --prompt-source culture_trip --run-name S001_armB
echo "S001_3MODELS_DONE $(date +%T)"
