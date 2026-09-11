# Thí nghiệm

Mỗi giả thuyết một thư mục, protocol viết TRƯỚC khi chạy và commit trước (quy tắc autoresearch: khoá protocol).

```
experiments/H1-review-loop/protocol.md   what / why / prediction / config / metric
experiments/H1-review-loop/analysis.md   sau khi chạy: số đo, đúng dự đoán không, học được gì
```

Chạy một thí nghiệm trên tập dev:

```bash
python scripts/run_experiment.py H1 --config configs/kaggle_t4x2.yaml --set review.max_iters=0 --tag baseline
python scripts/run_experiment.py H1 --config configs/kaggle_t4x2.yaml --set review.max_iters=2 --tag review2
```

Script chạy `ctig.cli batch --ids <dev10>` với `--run-name <H>-<tag>`, rồi thêm một dòng vào
`research/research-state.yaml → experiments.trajectory` với metric đọc từ `summary.json`.
