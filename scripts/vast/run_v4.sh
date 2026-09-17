#!/bin/bash
cd /workspace/ctig17 && git pull -q && git log --oneline -1
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)
CFG=configs/vast_a100.yaml
echo "=== v18 smoke v4 (KB 5-8 thuộc tính + kiểm ảnh thật + bản tay có kiểm) $(date +%T)"
python scripts/run_walkthrough.py --config $CFG --set prompts_path=data/prompts_simple.json \
  --ids S001,S012,S021 --models "realvis_xl#bare,realvis_xl,realvis_xl+ref,sd35_medium#bare,sd35_medium,flux_dev#bare,flux_dev" --run-name v18_smoke7b4
echo "V4_DONE $(date +%T)"
python /workspace/compare_pairs.py /workspace/runs/v18_smoke7b4 /workspace/runs/v18_smoke7b4/bare_vs_system.html
echo "V4_PAGE_DONE $(date +%T)"
