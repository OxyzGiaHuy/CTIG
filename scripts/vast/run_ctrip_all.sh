#!/bin/bash
cd /workspace/ctig17
/workspace/venv_ctrip/bin/python scripts/culture_trip_prompts.py \
  --repo /workspace/baselines/Culture-TRIP \
  --prompts data/prompts_simple.json --prompts data/prompts_complex.json --model llama3:8b
echo "CTRIP_ALL_DONE $(date +%T)"
/venv/main/bin/python scripts/culture_trip_review.py -o /workspace/ct_review.html --md /workspace/ct_review.md
echo "CTRIP_REVIEW_DONE $(date +%T)"
