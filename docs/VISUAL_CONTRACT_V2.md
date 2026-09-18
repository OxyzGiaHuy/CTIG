# Visual Contract v2 — implementation and paper protocol

## What this artifact is

`data/contracts_v2.json` contains one contract for each prompt S001–S050.  The cultural facts and
Wikipedia links come from the separately reviewed `ctig_contracts_verified.json`; the v2 builder adds
operational metadata for generation and evaluation.  It is deliberately compatible with VietRepair's
existing `required` and `confusables` keys.

This is a **curated offline knowledge base**, not an automatically extracted inference-time artifact.
The defensible paper wording is:

> We construct the visual contracts offline with LLM assistance, then manually verify observable
> attributes and confusable categories against linked Wikipedia pages. At inference time, a retrieval
> component selects the contract by prompt ID.

Do not claim that the submitted implementation automatically extracts these contracts from arbitrary
prompts.  Do not claim that a Wikipedia citation proves an attribute is sufficient for machine
recognition; that sufficiency remains a design judgement.

## Contract semantics

Each prompt has exactly one `main_entity`, one or more positive requirements, and one or more
`confusables`.

The three independent requirement fields must not be collapsed:

| field | values | meaning |
|---|---|---|
| `requirement_type` | `identity`, `prompt_action`, `prompt_context` | why the fact belongs in the contract |
| `visibility` | `must_be_visible`, `check_if_visible`, `optional` | whether this camera view can be required to show it |
| `scoring` | `required`, `conditional`, `excluded` | how the evaluator is allowed to use it |

`importance` is 3 for core identity evidence, 2 for supporting/context evidence, and 1 for facts kept
only for provenance. The diagnostic contract score is an importance-weighted mean over visible,
non-excluded facts. An `excluded` fact must never lower an image score or trigger a repair.

Examples of corrected evaluation behavior:

- The concealed underwater control system in water puppetry is factual but not observable, so S010
  retains it as `optional/excluded`.
- Pork knuckle is a possible bún bò Huế ingredient, not a universal requirement, so S027 marks it
  `optional/excluded`.
- The inner support of a worn nón quai thao is scored only if visible.
- The green rice inside S029's closed leaf parcel is scored only if the camera exposes it.

## Agent use

Use the v2 file explicitly in the experiment:

```bash
python3 scripts/run_arms.py \
  --config configs/vast_arms.yaml \
  --contracts data/contracts_v2.json \
  --ids S001,S002 \
  --reps 2 \
  --run-name visual-contract-v2
```

The Observer receives neutral part labels such as `roof`, `wrapper`, or `playing hand`, rather than the
correct cultural value. The Critic receives required and conditional evidence separately. Optional
facts are withheld from both Observer attention and Critic repair. Required evidence is ordered by
importance, while confusables remain a whole-object forced choice.

`generation_guidance.positive_cues` is a compact list of high-priority evidence.  Negative cues contain
only short confusable names; full confusable descriptions are for reasoning, not for diffusion negative
prompts, because they may themselves mention desired target features.

## Rebuild and validation

From the repository root:

```bash
python3 scripts/build_visual_contracts.py \
  --source /home/tghuy/ctig_contracts_verified.json
python3 scripts/validate_contracts.py
```

The validator enforces exactly S001–S050, prompt alignment, unique IDs, Wikipedia sources, valid
visibility/scoring combinations, non-empty generation cues, and valid links from confusables back to
required evidence. The expected summary is:

```text
{"contracts": 50, "required": 158, "confusables": 115, "required_scored": 150, "conditional": 6, "excluded": 2}
validation: PASS (0 errors, 0 warnings)
```

The JSON Schema is `data/contract_schema.json`. It is for external tooling and supplementary material;
the dependency-free validator is the canonical preflight check in this repository.

## Paper boundary

The contract score is useful for diagnosing and choosing a repair target. It is not valid as the sole
headline result because the same contract controls both repair and measurement. The main comparison
must remain blinded human pairwise evaluation (B versus M, plus T/S/R ablations). Report contract
coverage as a secondary diagnostic only.
