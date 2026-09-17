#!/bin/bash
# chờ smoke SD3.5 xong VÀ FLUX tải xong, rồi smoke FLUX (bare / system / +ref) trên S001, S012
PID=$1
while kill -0 "$PID" 2>/dev/null; do sleep 15; done
while ! grep -q "^flux " /workspace/logs/prefetch_flux.log 2>/dev/null; do sleep 20; done
cd /workspace/ctig17 && git pull -q && git log --oneline -1
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)
python scripts/run_walkthrough.py --config configs/vast_a100.yaml --set prompts_path=data/prompts_simple.json \
  --ids S001,S012 --models "flux_dev#bare,flux_dev,flux_dev+ref" --run-name v18_smoke
echo FLUX_SMOKE_DONE
