#!/bin/bash
cd /workspace/ctig17 && git pull -q && git log --oneline -1
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)
CFG=configs/vast_a100.yaml
# dựng lại chỉ mục kho cũ sau khi khử trùng (đường lùi), rồi chạy với ảnh ref theo prompt của nhóm
python -m ctig.stages.refindex build --root /workspace/refs --out /workspace/runs/_cache/ref_index.npz 2>&1 | tail -2
mkdir -p /workspace/runs/_cache/kb_auto_7b_v4 && mv /workspace/runs/_cache/kb_auto/*.json /workspace/runs/_cache/kb_auto_7b_v4/ 2>/dev/null
echo "=== v18 v5: ref theo prompt + VQA phát biểu + KB kiểm ảnh thật $(date +%T)"
python scripts/run_walkthrough.py --config $CFG --set prompts_path=data/prompts_simple.json \
  --ids S001,S012,S021,S031 --models "realvis_xl#bare,realvis_xl,realvis_xl+ref,sd35_medium#bare,sd35_medium,flux_dev#bare,flux_dev,flux_dev+ref" --run-name v18_v5
echo "V5_DONE $(date +%T)"
python /workspace/compare_pairs.py /workspace/runs/v18_v5 /workspace/runs/v18_v5/bare_vs_system.html
echo "V5_PAGE_DONE $(date +%T)"
