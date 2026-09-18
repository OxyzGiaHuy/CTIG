# SAVIER

**Source-Anchored Visual Inspection and Evidence-Guided Repair for Vietnamese Cultural Text-to-Image Generation**

SAVIER is a one-pass, multi-agent *prompt-repair* module that sits behind any text-to-image generator. A first
image is generated from the prompt; three role-specialised agents sharing one MLLM backbone then (i) anchor
what must be preserved and what counts as culturally correct to the **original prompt and hand-chosen
Wikipedia pages**, (ii) **inspect the image without seeing the prompt**, and (iii) turn the discrepancies into a
short, positive repair prompt. The generator is run once more with the same seed and curated reference
photographs. Nothing is edited in pixel space; the output is a repaired prompt and the image it produces.

> Contracts (cultural evidence) are drafted offline from Wikipedia-grounded descriptions using an LLM and
> subsequently inspected and corrected by a Vietnamese annotator. The three agents are role-specialised
> agents sharing the same MLLM backbone. Curated reference images are used during the repair generation.

## Contributions

1. **VN-Culture-50 prompts + source-anchored visual contracts.** 50 Vietnamese cultural prompts
   (`data/prompts_simple.json`, bilingual, 6 categories) and, for each, a human-verified *visual contract*
   (`data/contracts_v2.json`): 2–5 camera-observable identity attributes with importance weights and a
   Wikipedia source each, plus 2–3 *confusables* (the look-alike objects generators actually drift to:
   qipao for áo dài, cart for shoulder-pole vendor, bánh tét rolls for bánh chưng, …). Contracts are frozen
   before generation and are used **only for evaluation**; the agents never read them at inference time.
2. **The SAVIER repair protocol.** Curator → Observer → Refiner, one repair, no acceptance loop, every
   inter-agent message a structured JSON that is machine-validated (verbatim quotes, no negation, no
   framing changes, actions already satisfied in I0 are discarded). We report the method on SDXL 1.0 and
   FLUX.1-dev with a fixed-seed, one-image-per-arm protocol and independent metrics.

## Method

```
P0 (original prompt) ──► G ──► A                              (arm A: prompt-only baseline)
P_ct = P0 + Culture-TRIP expansion ──► G ──► I0               (arm B: prompt-refinement baseline)

I0 ──► O  Prompt-Blind Image Observer ──► visual report
P0 + curated Wikipedia ──► C  Source-Aware Prompt Curator ──► Preservation Card + Evidence Card
report + cards ──► R  Discrepancy-Guided Prompt Refiner ──► ≤3 positive repair actions, phrases to drop
P1 = P0 (verbatim) + trimmed expansion + "Keep clearly visible: …" + actions
P1 + reference photos ──► G (same seed) ──► I1                (arm C: SAVIER)
```

| agent | modality | sees | produces |
|---|---|---|---|
| **C** Source-Aware Prompt Curator | text | `prompt_vi`, `prompt_en`, hand-chosen Wikipedia pages (≤4k chars, cut around entity nouns) | *Preservation Card* (everything the prompt asks for, no citation needed) and *Evidence Card* (≤3 identity cues, ≤2 conditional cues, ≤2 positive disambiguators; every cue carries a **verbatim** Vietnamese quote and URL) |
| **O** Prompt-Blind Image Observer | vision | the image **only** | exhaustive report in 10 fixed sections; never names a culture, never guesses outside the frame |
| **R** Discrepancy-Guided Prompt Refiner | text | P0, expansion text, both cards, the report | gap analysis (`missing_prompt_explicit`, `missing_cultural_identity`, `contradictions`, `already_satisfied`, `uncertain_no_repair`), ≤3 repair actions, `drop_phrases` |
| G | generator | P1 (+ 1–2 reference photos) | I1 |

All three agents run on **Mistral-Small-3.1-24B-Instruct** with different system prompts (`savier/llm.py`,
`savier/agents.py`). Generators (`savier/generators.py`): SDXL 1.0 base (DPM++ 2M Karras, 30 steps,
CFG 5, compel for prompts > 77 CLIP tokens, IP-Adapter Plus ViT-H with two photos, scale 0.5) and
FLUX.1-dev (28 steps, guidance 3.5, T5 `max_sequence_length=512`, XLabs IP-Adapter with one photo, scale
0.6). **No negative prompt is used in any arm.** Same seed for A, I0 and I1.

### Machine checks on agent outputs

Each check encodes a failure that was observed in development, not a hypothetical one.

| where | check | observed failure it blocks |
|---|---|---|
| C | `quote_vi` must occur verbatim in a Vietnamese source **and** share a content word with `cue_vi` | a verbatim quote about boiling broth attached to a cue "stone bowl" |
| C | ≤4k characters of source text in total, two separate calls | 12k-character pages truncated the JSON reply |
| R | actions containing negation or framing words are dropped | diffusion models render the negated noun; framing changes destroyed correct images |
| R | each action is scored on I0 first; actions already ≥ 8/10 are removed and replaced by identity cues | "a high collar" occupied a repair slot on an image that already had one |
| R | `drop_phrases` must occur verbatim in the expansion and never in P0 | the Culture-TRIP expansion pushed áo dài prompts toward qipao on both generators |
| P1 | the Keep clause lists the Preservation Card's supporting objects | reference conditioning dropped the "school gate" and the "tray" the prompt asked for |

If R produces no valid action the module is a **no-op** and I1 = I0; the no-op rate is reported.

## Evaluation protocol

Metrics are computed by `scripts/eval_metrics.py` with an evaluator that is **not** the agents' model
(Qwen2.5-VL-7B-Instruct), on images whose filenames are hashed and whose (image, statement) order is
shuffled with a fixed seed, so the evaluator cannot tell I0 from I1. Contracts are frozen before generation
(`data/contracts_v2.json`, commit `e1c9da0`). Confidence intervals are bootstrapped **by prompt** (2000
resamples).

Let `C` be the contract's `required` items (`scoring = required`), `w_i` their `importance`, and
`y_i(I) ∈ {0, 0.5, 1}` the evaluator's judgement (YES / PARTIAL / NO) for item `i` on image `I`.

**VCFS ↑ — Visual Contract Fulfillment Score**
```
VCFS(I) = 100 · Σ_{i∈C} w_i y_i(I) / Σ_{i∈C} w_i
```
`conditional` items are reported separately; prompt-specific details (e.g. "white") are left to VQAScore.

**CCR ↓ — Cultural Confusion Rate**, over confusable cues `F` with severities `v_j` (all 1 in `contracts_v2`)
```
CCR(I) = 100 · Σ_{j∈F} v_j z_j(I) / Σ_{j∈F} v_j        (N/A when a prompt has no confusables)
```

**NRG ↑ — Net Repair Gain.** With `D0 = {i : y_i(I0) < 1}` (missing on I0) and `S0 = {i : y_i(I0) = 1}`:
```
FR  = Σ_{i∈D0} w_i · max(0, y_i(I1) − y_i(I0)) / Σ_{i∈D0} w_i
RR  = Σ_{i∈S0} w_i · max(0, y_i(I0) − y_i(I1)) / Σ_{i∈S0} w_i
NRG = 100 · (FR − RR)                                    (N/A when D0 = ∅; the correct no-op rate is reported instead)
```

**VQAScore ↑** (Lin et al. 2024) against the **original** English prompt P0, never against P1:
`P(Yes | "Does this figure show "P0"? Please answer yes or no.")`. The reference implementation uses
`clip-flant5-xxl`; `--vqa-backend t2v` calls it through `t2v_metrics`, `--vqa-backend qwen` computes the same
quantity with Qwen2.5-VL-7B. The backbone is printed in every table — numbers from different backbones are
not comparable.

**CAIRE ↑** (Yayavaram et al., EACL 2026): cultural relevance to the label *Vietnam* on a 1–5 scale, run with
the authors' code and knowledge base (`--target_list "Vietnam, China, Japan, South Korea, Thailand, …"`);
merged with `--caire-csv`.

We report I0, I1 and Δ = I1 − I0 per generator, plus paired counts (NRG > 0 / = 0 / < 0 / N/A). Human
pairwise preference is the primary evidence and is collected separately.

### Results — 50 prompts, one seed (S001–S050)

Independent evaluator Qwen2.5-VL-7B (blind to arm and order); contract `contracts_v2.json` frozen before
generation; 95% bootstrap CIs by prompt (2000 resamples). Full tables and every raw evaluator answer are in
`results/metrics_50/`.

| generator | arm | VQAScore¹ ↑ | VCFS ↑ | CCR ↓ | NRG ↑ |
|---|---|---:|---:|---:|---:|
| SDXL 1.0 | A · prompt only | 0.720 | 52.9 | 20.8 | – |
| SDXL 1.0 | I0 · Culture-TRIP | 0.610 | 55.6 | 15.3 | – |
| SDXL 1.0 | **I1 · SAVIER** | **0.740** | **68.9** | 16.2 | **41.2** [26.8, 55.4] |
| SDXL 1.0 | Δ (I1 − I0) | **+0.12** [+0.01, +0.25] | **+13.3** [+5.4, +21.4] | +0.8 [−4.5, +5.8] | |
| FLUX.1-dev | A · prompt only | 0.670 | 44.8 | 13.3 | – |
| FLUX.1-dev | I0 · Culture-TRIP | 0.700 | 59.1 | 15.0 | – |
| FLUX.1-dev | **I1 · SAVIER** | **0.770** | **66.3** | **13.0** | **21.1** [5.5, 36.2] |
| FLUX.1-dev | Δ (I1 − I0) | +0.07 [−0.01, +0.16] | **+7.3** [+0.4, +13.7] | −2.0 [−7.7, +3.7] | |

Paired outcome of I1 vs I0 (NRG): SDXL 24 > 0 · 13 = 0 · **1 < 0** · 12 N/A (I0 already complete);
FLUX 19 > 0 · 16 = 0 · **5 < 0** · 10 N/A. No-op rate (Refiner found nothing to repair): SDXL 2/50, FLUX 5/50.

¹ VQAScore here uses a Qwen2.5-VL-7B backbone (same question and P(Yes) formula as the paper, different
model); it is comparable across arms in this table, not with numbers published for `clip-flant5-xxl`.
CAIRE (Vietnam / China relevance, 1–5) and human pairwise preference are reported separately when available.

## Repository layout

```
savier/
  llm.py          shared Mistral-Small-3.1 backbone, JSON extraction
  wiki.py         per-page Wikipedia extracts, passage cutting
  agents.py       C · O · R, machine checks, P1 construction
  generators.py   SDXLGen, FluxGen (no negative prompt, optional IP-Adapter)
  pipeline.py     arms A / I0 / I1, transcript, contact sheet
scripts/
  run_savier.py   end-to-end run
  fetch_wiki.py   build data/wiki_curated from the contract URLs
  eval_metrics.py VCFS · CCR · NRG · VQAScore (+ CAIRE merge)
  grid_models.py  multi-generator comparison grid
data/
  prompts_simple.json      50 prompts (vi / en, category, entities)
  contracts_v2.json        frozen visual contracts (evaluation only)
  culture_trip/            Culture-TRIP refined prompts (P_ct) for the 50 prompts
  wiki_curated/            curated Wikipedia extracts used by the Curator
```

## Running

```bash
pip install -r requirements.txt
python scripts/fetch_wiki.py --ids S001,S002,S003 -o data/wiki_curated/smoke.json
python scripts/run_savier.py --ids S001,S002,S003 --backend sdxl --refs-dir /path/to/reference_images \
       --wiki data/wiki_curated/smoke.json --out runs/sdxl
python scripts/run_savier.py --ids S001,S002,S003 --backend flux --refs-dir /path/to/reference_images \
       --wiki data/wiki_curated/smoke.json --out runs/flux
python scripts/eval_metrics.py --run SDXL=runs/sdxl --run FLUX.1-dev=runs/flux -o runs/metrics
python scripts/grid_models.py --run SDXL=runs/sdxl --run FLUX.1-dev=runs/flux -o runs/grid.png --cell 1024
```

`--refs-dir` expects `selected/<prompt_id>/` (photos used for conditioning) and `candidates/<prompt_id>/`
(held-out photos used only by the evaluator). Mistral-Small-3.1-24B (~48 GB bf16) and SDXL fit together on
one 80 GB GPU; FLUX.1-dev runs with model CPU offload.

## Data and licensing notes

- Prompts, contracts, Culture-TRIP outputs and Wikipedia extracts are released here. Wikipedia text is
  CC BY-SA 4.0.
- The reference photographs used for conditioning and held-out evaluation are third-party web images and
  are **not** redistributed; the pipeline runs without them (no reference conditioning).
- Culture-TRIP (NAACL 2025) is used as the prompt-refinement baseline; its repository is not included.

## Citation

```
@misc{savier2026,
  title  = {SAVIER: Source-Anchored Visual Inspection and Evidence-Guided Repair for Vietnamese Cultural Text-to-Image Generation},
  author = {Thai Gia Huy},
  year   = {2026}
}
```
