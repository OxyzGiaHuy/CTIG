#!/bin/bash
# chờ run hiện tại (PID truyền vào) xong rồi chạy bộ complex
PID=$1
while kill -0 "$PID" 2>/dev/null; do sleep 15; done
cd /workspace/ctig17 && git pull -q
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)
python scripts/run_walkthrough.py --config configs/vast_a100.yaml --set prompts_path=data/prompts_complex.json \
  --ids C002,C003,C008,C037 --run-name v17_complex --report
