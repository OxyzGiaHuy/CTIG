#!/bin/bash
while pgrep -f "run_walkthroug[h].py" >/dev/null; do sleep 10; done
cd /workspace/ctig17 && git pull -q && git log --oneline -1
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)
CFG=configs/vast_a100.yaml
echo "=== SMOKE 7B v3 (validate_kb + salience + inpaint + context focus) $(date +%T)"
python scripts/run_walkthrough.py --config $CFG --set prompts_path=data/prompts_simple.json \
  --ids S001,S012,S021 --models "realvis_xl#bare,realvis_xl,realvis_xl+ref,sd35_medium#bare,sd35_medium,flux_dev#bare,flux_dev" --run-name v18_smoke7b3
echo "SMOKE7B3_DONE $(date +%T)"
