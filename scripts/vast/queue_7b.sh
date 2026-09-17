#!/bin/bash
# chờ smoke 3B xong -> xoá cache Qwen 3B và bản KB tự sinh 3B -> smoke lại với Qwen 7B (KB hai bước, inpaint, rewrite lần đầu)
PID=$1
while kill -0 "$PID" 2>/dev/null; do sleep 15; done
rm -rf /workspace/.hf_home/hub/models--Qwen--Qwen2.5-VL-3B-Instruct
mkdir -p /workspace/runs/_cache/kb_auto_3b && mv /workspace/runs/_cache/kb_auto/*.json /workspace/runs/_cache/kb_auto_3b/ 2>/dev/null
rm -rf /workspace/runs/_cache/llm   # cache LLM của 3B không dùng lại được cho 7B
df -h /workspace | tail -1
cd /workspace/ctig17 && git pull -q && git log --oneline -1
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)
CFG=configs/vast_a100.yaml
echo "=== SMOKE 7B (S001,S012,S021) $(date +%T)"
python scripts/run_walkthrough.py --config $CFG --set prompts_path=data/prompts_simple.json \
  --ids S001,S012,S021 --models "realvis_xl#bare,realvis_xl,realvis_xl+ref,sd35_medium#bare,sd35_medium,flux_dev#bare,flux_dev" --run-name v18_smoke7b
echo "SMOKE7B_DONE $(date +%T)"
