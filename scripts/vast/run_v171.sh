#!/bin/bash
cd /workspace/ctig17 && git pull -q
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)
python scripts/run_walkthrough.py --config configs/vast_a100.yaml --ids p001,p012,p031,p050 --run-name v17 --report
python scripts/run_walkthrough.py --config configs/vast_a100.yaml --set prompts_path=data/prompts_complex.json --ids C002,C003,C008,C037 --run-name v17_complex
python /workspace/compare_pairs.py /workspace/runs/v17 /workspace/runs/v17/bare_vs_system.html
python /workspace/compare_pairs.py /workspace/runs/v17_complex /workspace/runs/v17_complex/bare_vs_system.html
