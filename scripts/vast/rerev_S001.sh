#!/bin/bash
cd /workspace/ctig17
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)
C="--config configs/vast_a100.yaml --set prompts_path=data/prompts_simple.json --set t2i.render=bare --set multigen.n_candidates=1 --set multigen.adaptive.enabled=false --set multigen.cpu_offload=true --set agents.max_revisions=0 --ids S001 --models sdxl_base,realvis_xl,flux_dev"
python scripts/run_walkthrough.py $C --prompt-source original     --run-name S001_armA
python scripts/run_walkthrough.py $C --prompt-source culture_trip --run-name S001_armB
/venv/main/bin/python scripts/arm_grid.py "A goc=/workspace/runs/S001_armA" "B CultureTRIP=/workspace/runs/S001_armB" -o /workspace/S001_armAB.png --ids S001
echo "REREV_DONE $(date +%T)"
