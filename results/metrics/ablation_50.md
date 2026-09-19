# Ablation — 50 prompt, cùng seed, cùng I0 (evaluator Qwen2.5-VL-7B, mù; contracts_v2; CI bootstrap theo prompt ×2000)

Arms: I0 = refined prompt (Culture-TRIP); refs-only = I0 prompt + IP-Adapter (2 ảnh `selected/`), không agent; keep-only = refs-only + câu Keep từ Preservation Card (một lời gọi C, không O/R); SAVIER = keep-only + O quan sát I0 + R sinh ≤3 action sửa + bỏ drop_phrases.

## SDXL

| arm | VQAScore ↑ | VCFS ↑ | CCR ↓ | NRG >0 / =0 / <0 |
|---|---:|---:|---:|---|
| A (prompt gốc) | 0.722 [0.62, 0.81] | 52.9 [44.2, 61.6] | 20.8 [13.7, 28.7] | – |
| I0 (+ refined prompt) | 0.612 [0.51, 0.71] | 55.6 [46.8, 64.7] | 15.3 [9.7, 21.3] | – |
| + refined prompt + reference (refs-only) | 0.713 [0.60, 0.81] | 72.3 [65.6, 78.9] | 14.2 [8.2, 21.0] | 29/6/3 |
| + keep-only | 0.722 [0.62, 0.82] | 70.7 [62.7, 77.8] | 14.2 [8.3, 21.0] | 27/8/3 |
| **+ SAVIER (ours)** | 0.725 [0.62, 0.82] | 67.9 [60.1, 75.2] | 16.8 [10.2, 24.2] | 24/12/2 |

Hiệu số ghép đôi theo prompt (SAVIER − arm):

| so với | ΔVQA | ΔVCFS | ΔCCR | SAVIER hơn / bằng / kém (VCFS) |
|---|---:|---:|---:|---|
| refs-only | +0.012 [-0.03, +0.07] | -4.4 [-9.9, +0.9] | +2.7 [-0.7, +6.7] | 4 / 35 / 11 |
| keep-only | +0.004 [-0.05, +0.06] | -2.8 [-8.6, +2.4] | +2.7 [-0.7, +6.3] | 6 / 34 / 10 |
| I0 | +0.113 [-0.01, +0.23] | +12.3 [+4.5, +20.1] | +1.5 [-3.8, +7.3] | 24 / 19 / 7 |

## FLUX.1-dev

| arm | VQAScore ↑ | VCFS ↑ | CCR ↓ | NRG >0 / =0 / <0 |
|---|---:|---:|---:|---|
| A (prompt gốc) | 0.669 [0.58, 0.76] | 44.8 [35.2, 54.4] | 13.3 [7.3, 19.7] | – |
| I0 (+ refined prompt) | 0.696 [0.60, 0.79] | 59.1 [50.5, 67.1] | 15.0 [8.7, 22.7] | – |
| + refined prompt + reference (refs-only) | 0.713 [0.61, 0.80] | 65.3 [55.7, 74.2] | 16.8 [10.7, 23.8] | 19/13/8 |
| + keep-only | 0.711 [0.61, 0.80] | 66.1 [57.1, 74.5] | 17.2 [11.3, 23.3] | 22/13/5 |
| **+ SAVIER (ours)** | 0.767 [0.68, 0.85] | 66.0 [57.8, 73.5] | 13.7 [7.7, 20.0] | 19/16/5 |

Hiệu số ghép đôi theo prompt (SAVIER − arm):

| so với | ΔVQA | ΔVCFS | ΔCCR | SAVIER hơn / bằng / kém (VCFS) |
|---|---:|---:|---:|---|
| refs-only | +0.054 [-0.02, +0.13] | +0.7 [-5.6, +7.0] | -3.2 [-8.8, +2.2] | 14 / 21 / 15 |
| keep-only | +0.056 [-0.03, +0.13] | -0.2 [-6.2, +6.0] | -3.5 [-8.8, +1.5] | 13 / 22 / 15 |
| I0 | +0.071 [-0.02, +0.16] | +6.9 [-0.3, +14.6] | -1.3 [-7.0, +4.7] | 18 / 22 / 10 |
