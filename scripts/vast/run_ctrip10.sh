#!/bin/bash
cd /workspace/ctig17
/workspace/venv_ctrip/bin/python scripts/culture_trip_prompts.py \
  --repo /workspace/baselines/Culture-TRIP --prompts data/prompts_simple.json \
  --ids S001,S002,S003,S004,S005,S006,S007,S009,S010,S011 --model llama3:8b
echo "CTRIP10_DONE $(date +%T)"
