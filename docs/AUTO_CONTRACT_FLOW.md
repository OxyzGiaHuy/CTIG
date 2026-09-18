# Automatic bilingual Visual Contract flow

This module implements a reviewable scaffold for:

```text
Vietnamese/English prompt
        ↓
Qwen3 entity linker
        ↓
Vietnamese Wikipedia retrieval + English supplement
        ↓
Qwen3 contract extractor
        ↓
Observability filter
        ↓
Candidate contract
        ↓
PhoGPT cross-check or mandatory human review
```

It does **not** overwrite `data/contracts_v2.json`. Automatic output is written to `runs/contract_candidates/`
and remains a candidate until a Vietnamese reviewer approves it.

## Why Vietnamese evidence is mandatory

For every `identity` or `canonical_cue` attribute, the default pipeline requires a verbatim quotation from
a Vietnamese Wikipedia passage. English Wikipedia is retrieved as supplementary context. The JSON and the
review page preserve:

- Vietnamese and English entity names;
- Vietnamese and English descriptions;
- Vietnamese and English camera-visible evidence;
- source language, title, URL and verbatim quotation;
- attributes rejected for missing evidence or being non-visual.

Prompt-specific attributes may cite `PROMPT`, but the quote must occur verbatim in the Vietnamese or English
prompt. An English-only cultural attribute is rejected unless `--allow-english-only` is explicitly supplied.

## Run S001-S003

Human verification only, recommended for the first run:

```bash
python3 scripts/extract_visual_contracts.py \
  --ids S001,S002,S003 \
  --extractor-model Qwen/Qwen3-8B
```

Add a PhoGPT cross-check:

```bash
python3 scripts/extract_visual_contracts.py \
  --ids S001,S002,S003 \
  --extractor-model Qwen/Qwen3-8B \
  --checker-backend hf_text \
  --checker-model vinai/PhoGPT-4B-Chat
```

Loading both models simultaneously needs enough VRAM. If memory is limited, run without the checker and use
the generated `.review.md` files. Cross-checking never auto-merges an attribute, even when PhoGPT accepts it.

## Acceptance rules

An attribute is dropped when any of the following is true:

1. its source quotation is not an exact normalized substring of the retrieved passage;
2. an identity/canonical cue lacks Vietnamese evidence;
3. it describes history, origin, meaning, symbolism, taste, smell, fame, or heritage status;
4. the observability stage judges that it cannot be checked from one still image;
5. its ID is empty or duplicated.

Hidden details are downgraded to `check_if_visible/conditional`. The code records all drops rather than
silently deleting them.

## Implementation boundary

`ctig/contracts/wikipedia.py` performs cached bilingual retrieval. `ctig/contracts/pipeline.py` owns the
entity-link, extraction, grounding, observability, and review protocol. `ctig/llm/hf_text.py` provides one
text-only Hugging Face backend for Qwen3 and PhoGPT.

This scaffold has offline unit tests with fake models and sources. A successful offline test means the control
flow and guards work; it does not establish that Qwen3 or PhoGPT produces culturally correct contracts. That
requires running the three pilot prompts and comparing the candidates with human-verified S001-S003 contracts.
