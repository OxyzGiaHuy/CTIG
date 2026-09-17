#!/bin/bash
PID=$1
while kill -0 "$PID" 2>/dev/null; do sleep 15; done
mkdir -p /workspace/runs/_cache/kb_auto_7b_v1 && mv /workspace/runs/_cache/kb_auto/*.json /workspace/runs/_cache/kb_auto_7b_v1/ 2>/dev/null
cd /workspace/ctig17 && git pull -q && git log --oneline -1
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)
CFG=configs/vast_a100.yaml
echo "=== SMOKE 7B v2 (salience, inpaint fix) $(date +%T)"
python scripts/run_walkthrough.py --config $CFG --set prompts_path=data/prompts_simple.json \
  --ids S001,S012,S021 --models "realvis_xl#bare,realvis_xl,realvis_xl+ref,sd35_medium#bare,sd35_medium,flux_dev#bare,flux_dev" --run-name v18_smoke7b2
echo "SMOKE7B2_DONE $(date +%T)"
