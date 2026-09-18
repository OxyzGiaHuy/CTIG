"""Run SAVIER (arms A · B=I0 · C=I1) on a list of prompts.

    python scripts/run_savier.py --ids S001,S002,S003 --backend sdxl --refs-dir /path/to/reference_images \
        --out runs/savier_sdxl
    python scripts/run_savier.py --ids S001,S002,S003 --backend flux --refs-dir ... --out runs/savier_flux

`--refs-dir` must contain `selected/<prompt_id>/` (conditioning photos). Omit it to run without reference
conditioning. The Wikipedia pages are read from `data/wiki_curated/*.json` (see scripts/fetch_wiki.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from savier.llm import MistralVL  # noqa: E402
from savier.pipeline import load_prompts, run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--backend", choices=["sdxl", "flux"], default="sdxl")
    ap.add_argument("--prompts", default=str(ROOT / "data/prompts_simple.json"))
    ap.add_argument("--culture-trip", default=str(ROOT / "data/culture_trip"))
    ap.add_argument("--wiki", default=str(ROOT / "data/wiki_curated/S001_S010.json"))
    ap.add_argument("--contracts", default=str(ROOT / "data/contracts_v2.json"), help="only used as keyword anchors for cutting Wikipedia passages")
    ap.add_argument("--refs-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--llm", default="mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--text-only-arm", action="store_true", help="also generate I1 without reference conditioning")
    a = ap.parse_args()

    prompts = load_prompts(a.prompts)
    ids = [x.strip() for x in a.ids.split(",") if x.strip() in prompts]
    wiki = json.loads(Path(a.wiki).read_text(encoding="utf-8"))
    contracts = json.loads(Path(a.contracts).read_text(encoding="utf-8")) if Path(a.contracts).exists() else None
    if a.backend == "flux":
        from savier.generators import FluxGen
        gen = FluxGen(device=a.device, offload=True)
    else:
        from savier.generators import SDXLGen
        gen = SDXLGen(device=a.device)
    llm = MistralVL(a.llm, device=a.device)
    run(llm, gen, prompts, ids, wiki, a.culture_trip, a.out, refs_dir=a.refs_dir, seed=a.seed, contracts=contracts, text_only_arm=a.text_only_arm)
    print("SAVIER_DONE")


if __name__ == "__main__":
    main()
