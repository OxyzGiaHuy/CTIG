#!/bin/bash
cd /workspace/ctig17 && git pull -q && git log --oneline -1
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)
CFG=configs/vast_a100.yaml
echo "=== SMOKE SD3.5 + RealVis (S001,S012,S021; KB auto) $(date +%T)"
python scripts/run_walkthrough.py --config $CFG --set prompts_path=data/prompts_simple.json \
  --ids S001,S012,S021 --models "sd35_medium#bare,sd35_medium,realvis_xl#bare,realvis_xl,realvis_xl+ref" --run-name v18_smoke
echo "SMOKE1_DONE $(date +%T)"
echo "=== SMOKE FLUX (S001,S012) $(date +%T)"
python scripts/run_walkthrough.py --config $CFG --set prompts_path=data/prompts_simple.json \
  --ids S001,S012 --models "flux_dev#bare,flux_dev,flux_dev+ref" --run-name v18_smoke
echo "SMOKE2_DONE $(date +%T)"
