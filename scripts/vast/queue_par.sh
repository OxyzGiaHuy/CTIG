#!/bin/bash
# chờ lượt v171 xong, rồi chạy 2 luồng song song theo prompt; dựng trang so sánh sau khi cả hai xong
PID=$1
while kill -0 "$PID" 2>/dev/null; do sleep 15; done
cd /workspace/ctig17 && git pull -q && git log --oneline -1
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)
CFG=configs/vast_a100.yaml
( python scripts/run_walkthrough.py --config $CFG --ids p001,p012 --run-name v17 > /workspace/logs/v172_A.log 2>&1
  python scripts/run_walkthrough.py --config $CFG --set prompts_path=data/prompts_complex.json --ids C002,C003 --run-name v17_complex >> /workspace/logs/v172_A.log 2>&1 ) &
PA=$!
( python scripts/run_walkthrough.py --config $CFG --ids p031,p050 --run-name v17 > /workspace/logs/v172_B.log 2>&1
  python scripts/run_walkthrough.py --config $CFG --set prompts_path=data/prompts_complex.json --ids C008,C037 --run-name v17_complex >> /workspace/logs/v172_B.log 2>&1 ) &
PB=$!
wait $PA $PB
python -c "from ctig.progress_report import build; from ctig.config import Config; build('/workspace/runs/v17', '/workspace/runs/v17/progress_report.html', cfg=Config.load('$CFG'), log=print)"
python /workspace/compare_pairs.py /workspace/runs/v17 /workspace/runs/v17/bare_vs_system.html
python /workspace/compare_pairs.py /workspace/runs/v17_complex /workspace/runs/v17_complex/bare_vs_system.html
echo QUEUE_DONE
