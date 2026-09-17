#!/bin/bash
cd /workspace/ctig17 && git pull -q && git log --oneline -1
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)
CFG=configs/vast_a100.yaml
echo "=== v19.2 $(date +%T): hiệu chỉnh ngưỡng từng thuộc tính trên ảnh thật, bỏ thuộc tính không kiểm được"
python scripts/run_walkthrough.py --config $CFG --set prompts_path=data/prompts_simple.json \
  --ids S001,S012 --models "sdxl_base#bare,sdxl_base,realvis_xl#bare,realvis_xl,realvis_xl+ref,sd35_medium#bare,sd35_medium,sd35_medium+init" --run-name v192
echo "V192_DONE $(date +%T)"
python /workspace/compare_pairs.py /workspace/runs/v192 /workspace/runs/v192/bare_vs_system.html
echo "V192_PAGE_DONE $(date +%T)"
