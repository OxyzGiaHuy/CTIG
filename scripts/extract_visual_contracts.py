#!/usr/bin/env python3
"""Build reviewable candidate contracts from Vietnamese/English prompts.

Examples:
  python3 scripts/extract_visual_contracts.py --ids S001,S002,S003
  python3 scripts/extract_visual_contracts.py --ids S001 --checker-model vinai/PhoGPT-4B-Chat

Nothing is merged into data/contracts_v2.json automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctig.contracts import BilingualWikipediaRetriever, ContractExtractionPipeline  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent


def _backend(kind: str, model: str, device: str, dtype: str):
    if kind == "hf_text":
        from ctig.llm.hf_text import HFTextBackend

        return HFTextBackend(model, device=device, dtype=dtype, temperature=0.0)
    if kind == "anthropic":
        from ctig.llm.anthropic_backend import AnthropicBackend

        return AnthropicBackend(model)
    raise ValueError(f"unsupported backend: {kind}")


def _review_markdown(candidate: dict) -> str:
    ent = candidate["main_entity"]
    out = [
        f"# {candidate['prompt_id']} · {ent['name_vi']} / {ent['name_en']}\n\n",
        f"**Trạng thái:** `{candidate['status']}`\n\n",
        f"> {candidate['prompt_text_vi']}  \n> {candidate['prompt_text_en']}\n\n",
        "## Nguồn tiếng Việt cần kiểm tra trước\n\n",
    ]
    sources = {x["source_id"]: x for x in candidate["sources"]}
    for source in candidate["sources"]:
        if source["lang"] == "vi":
            out.append(f"- [{source['source_id']}] [{source['title']}]({source['url']})\n")
    out.append("\n## Thuộc tính được giữ\n\n")
    for item in candidate["required"]:
        out.append(
            f"### `{item['id']}` — {item['label_vi']}\n\n"
            f"- VI: {item['description_vi']}\n"
            f"- EN: {item['description_en']}\n"
            f"- Nhìn thấy: {item['visual_evidence_vi']}\n"
            f"- Loại: `{item['requirement_type']}` · `{item['visibility']}` · importance {item['importance']}\n"
        )
        for citation in item.get("citations", []):
            source = sources.get(citation["source_id"])
            label = source["title"] if source else "PROMPT"
            out.append(f"- Nguồn `{citation['source_id']}` ({label}): “{citation['quote']}”\n")
        out.append("\n")
    out.append("## Bị loại tự động\n\n")
    for item in candidate.get("dropped", []):
        out.append(f"- `{item.get('id', '?')}`: `{item.get('reason', '?')}`\n")
    out.append("\n## Human approval\n\n- [ ] Tên thực thể đúng\n- [ ] Trích dẫn tiếng Việt đúng nguyên văn\n"
               "- [ ] Mọi required đều quan sát được\n- [ ] Prompt-specific không bị biến thành định nghĩa phổ quát\n"
               "- [ ] Confusable thực sự gần về thị giác\n")
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=str(ROOT / "data" / "prompts_simple.json"))
    parser.add_argument("--ids", default="S001,S002,S003")
    parser.add_argument("--extractor-backend", choices=["hf_text", "anthropic"], default="hf_text")
    parser.add_argument("--extractor-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--checker-backend", choices=["none", "hf_text", "anthropic"], default="none")
    parser.add_argument("--checker-model", default="vinai/PhoGPT-4B-Chat")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--wiki-cache", default=str(ROOT / "runs_cache" / "contract_wikipedia"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "contract_candidates"))
    parser.add_argument("--allow-english-only", action="store_true",
                        help="allow identity cues without Vietnamese evidence (not recommended)")
    args = parser.parse_args()

    prompts = {x["id"]: x for x in json.loads(Path(args.prompts).read_text(encoding="utf-8"))}
    ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    missing = [x for x in ids if x not in prompts]
    if missing:
        raise SystemExit(f"prompt IDs not found: {missing}")

    print(f"Loading extractor: {args.extractor_model}", flush=True)
    extractor = _backend(args.extractor_backend, args.extractor_model, args.device, args.dtype)
    verifier = None
    if args.checker_backend != "none":
        print(f"Loading checker: {args.checker_model}", flush=True)
        verifier = _backend(args.checker_backend, args.checker_model, args.device, args.dtype)
    retriever = BilingualWikipediaRetriever(args.wiki_cache)
    pipeline = ContractExtractionPipeline(
        extractor, retriever, verifier=verifier,
        require_vi_evidence=not args.allow_english_only,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    for pid in ids:
        prompt = prompts[pid]
        print(f"[{pid}] extracting {prompt['text_vi']}", flush=True)
        try:
            result = pipeline.run(pid, prompt["text_vi"], prompt["text_en"])
        except Exception as exc:  # keep other prompts running
            failed += 1
            print(f"[{pid}] ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            continue
        json_path = output_dir / f"{pid}.candidate.json"
        md_path = output_dir / f"{pid}.review.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(_review_markdown(result), encoding="utf-8")
        print(f"[{pid}] {result['status']} · kept={len(result['required'])} "
              f"dropped={len(result['dropped'])} -> {json_path}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
