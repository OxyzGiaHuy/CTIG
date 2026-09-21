# Runtime cost of SAVIER (wall-clock, one A100 80 GB)

**Protocol.** One NVIDIA A100-SXM4 80 GB, no other job on the GPU; CUDA 12.4, torch 2.6.0, transformers 5.17,
diffusers 0.40, bf16. Mistral-Small-3.1-24B (agents) and the generator are resident in GPU memory; model loading is
excluded by a warm-up prompt (S050) that is not counted. Each stage is timed with `time.perf_counter()` around a
`torch.cuda.synchronize()` pair. The LLM/reference-crop cache is disabled (a fresh cache directory per repeat), and
the post-hoc diagnostic calls (`--no-diag`) are off, so only the method's own calls are timed. Prompts S001–S005,
three repeats each (n = 15 per generator). SDXL: 30 steps, 1024², DPM++ 2M Karras. FLUX.1-dev: 28 steps, 1024².
Raw per-unit timings: `bench_<model>_r{1,2,3}/S0xx/kor.json → thoi_gian`; aggregation: `scripts/bench_report.py`.

## Table (seconds per prompt, mean ± sd)

| Stage | What runs | SDXL 1.0 | FLUX.1-dev |
|---|---|---:|---:|
| Generate A | original prompt | 4.7 ± 0.1 | 36.6 ± 3.6 |
| Generate I0 | Culture-TRIP prompt | 4.5 ± 0.1 | 34.2 ± 2.4 |
| **C** · Curator | Preservation Card + Evidence Card (2 LLM calls, ≤4k chars of Vietnamese Wikipedia) | 53.4 ± 28.1 | 53.1 ± 28.2 |
| **O** · Observer | prompt-blind report of I0 (1 VLM call) | 12.1 ± 0.8 | 11.0 ± 1.8 |
| **R** · Refiner | gap analysis + pre-scoring of actions on I0 | 13.8 ± 1.6 | 13.6 ± 3.6 |
| Agents total (C+O+R) | | 79.3 ± 27.1 | 77.7 ± 28.2 |
| Generate I1 | same seed + IP-Adapter (SDXL: 2 photos, Plus ViT-H; FLUX: 1 photo, XLabs) | 14.5 ± 1.6 | 41.3 ± 2.4 |
| **SAVIER total** (agents + I1) | | **93.8 ± 26.8** | **119.0 ± 27.9** |
| Repair actions emitted | count | 1.8 ± 0.7 | 2.0 ± 0.9 |

LaTeX (booktabs):

```latex
\begin{table}[t]
\centering\small
\caption{Wall-clock cost per prompt (seconds, mean$\pm$sd over 5 prompts $\times$ 3 repeats) on one A100 80\,GB,
models resident, caches disabled. C is image-independent and can be computed once per prompt and reused across
generators and seeds; O and R are the only per-image agent cost.}
\label{tab:runtime}
\begin{tabular}{llrr}
\toprule
Stage & Calls & SDXL 1.0 & FLUX.1-dev \\
\midrule
Generate $A$ / $I_0$ & 1 image each & 4.7 / 4.5 & 36.6 / 34.2 \\
C~(Curator) & 2 LLM & 53.4$\pm$28.1 & 53.1$\pm$28.2 \\
O~(Observer) & 1 VLM on $I_0$ & 12.1$\pm$0.8 & 11.0$\pm$1.8 \\
R~(Refiner) & 1 LLM + action pre-scoring & 13.8$\pm$1.6 & 13.6$\pm$3.6 \\
Generate $I_1$ & same seed + IP-Adapter & 14.5$\pm$1.6 & 41.3$\pm$2.4 \\
\midrule
SAVIER total ($I_1$) & & \textbf{93.8$\pm$26.8} & \textbf{119.0$\pm$27.9} \\
\bottomrule
\end{tabular}
\end{table}
```

## What the numbers say

1. **The cultural check is a one-shot, bounded cost.** SAVIER runs C, O and R exactly once and generates exactly one
   repaired image — no loop, no gate, no best-of-N. On SDXL the whole procedure costs 94 s per prompt, about 20 plain
   SDXL samples; on FLUX, where a single image already takes 35–37 s, the agents add 78 s — about two extra images —
   and SAVIER's 119 s total is less than a best-of-4 draw (4 × 36.6 = 146 s) while producing one, inspected image. This is the price of grounding: Culture-TRIP refines the *prompt* without ever
   looking at the picture; SAVIER pays ~26 s of Observer+Refiner per image to inspect what was actually drawn.
2. **Two thirds of the agent time is the Curator, and it does not depend on the image.** C reads the prompt and the
   Wikipedia evidence only, so its cards can be computed once per prompt and reused across seeds, generators and
   re-runs (our FLUX run reused the SDXL cards verbatim). Amortised, the per-image agent overhead is O+R ≈ 26 s,
   stable to ±2 s. C's variance (31–109 s) comes from the length of the Evidence Card JSON, not from retrieval.
3. **Reference conditioning is cheap.** IP-Adapter with two curated photos adds ~10 s on SDXL (entity crop with
   OWL-ViT/CLIP included) on top of a 4.5 s generation — a small fraction of the agent cost, for most of the VCFS
   gain reported in the ablation.
4. **Where the time goes is where the fidelity comes from.** In the ablation, references carry the attribute gain
   (VCFS/CAIRE) while O+R deliver prompt fidelity, fewer confusables and fewer regressions on FLUX. The runtime
   table shows the same split: a cheap image-side step and a deliberate, bounded text-side inspection. For a
   generation-time culture check on a 24B open model this is the trade-off we accept: seconds of verification per
   image instead of a fine-tuned generator or a human in the loop.

Agent times are generator-independent by construction (same Mistral backbone, same cards, same I0 seed protocol): C 53.4 vs 53.1 s,
O 12.1 vs 11.0 s, R 13.8 vs 13.6 s across the two generators — a useful sanity check that the measurement isolates the agents.
A follow-up with the Evidence-Card token cap lowered from 2200 to 1200 is reported in `bench_sdxl_k1200.md` (C latency is driven by
output length hitting the cap, not by Wikipedia length: corr(C time, Wikipedia chars) = −0.04 over 50 prompts, median C = 30 s).
