#!/bin/bash
cd /workspace/ctig17 && git pull -q && git log --oneline -1
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('facebook/sam-vit-base', allow_patterns=['*.json','*.txt','*.safetensors']))" 2>&1 | tail -1
mkdir -p /workspace/runs/_cache/kb_auto_v18 && mv /workspace/runs/_cache/kb_auto/*.json /workspace/runs/_cache/kb_auto_v18/ 2>/dev/null
df -h /workspace | tail -1
CFG=configs/vast_a100.yaml
echo "=== v19 $(date +%T): KB kiểu tay + kiểm ảnh thật, Reviewer v1.9, định vị 4 nấc + SAM, ImageRAG scale 0.5 + caption, +init cho SD3.5"
python scripts/run_walkthrough.py --config $CFG --set prompts_path=data/prompts_simple.json \
  --ids S001,S012,S021,S031 --run-name v19 --report
echo "V19_DONE $(date +%T)"
python /workspace/compare_pairs.py /workspace/runs/v19 /workspace/runs/v19/bare_vs_system.html
echo "V19_PAGE_DONE $(date +%T)"
