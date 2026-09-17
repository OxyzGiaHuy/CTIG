"""Chạy thử Agentic Loop v2 (khung T2I-Copilot, không dùng KB) cho một prompt.

    python scripts/run_loop_v2.py --config configs/vast_a100.yaml --id S012 --model sdxl_base \
        [--prompt-source culture_trip] [--threshold 7.5] [--rounds 3] [--out runs/loopv2]

Ba agent ở `ctig/agents/copilot.py`. Script này chỉ lo phần nối dây: dựng Session để lấy ảnh tham chiếu và
bộ sinh, gọi interpret -> sinh ảnh đầu -> run_loop, rồi ghi JSON và lưới ảnh từng vòng.

KHÔNG đọc must_have/must_not. Bảng kiểm duy nhất là ảnh thật của chính prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctig.agents import copilot  # noqa: E402
from ctig.config import Config, set_dotted  # noqa: E402
from ctig.pipeline import load_prompts  # noqa: E402
from ctig.session import Session  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def external_prompt(source: str, pid: str) -> tuple[str, str]:
    """(câu tinh chỉnh, tư liệu thô) từ data/<source>/<pid>.json."""
    f = ROOT / "data" / source / f"{pid}.json"
    if not f.exists():
        return "", ""
    d = json.loads(f.read_text(encoding="utf-8"))
    raw = " ".join(s.get("out_raw", "") for s in (d.get("per_step") or []))
    return " ".join(str(d.get("refined_prompt", "")).split()), raw[:2000]


def grid(images: list[tuple[str, str]], out_png: Path, cell: int = 420, log=print) -> None:
    from PIL import Image, ImageDraw

    from scripts.overview_grid import _font

    ok = [(lab, p) for lab, p in images if p and Path(p).exists()]
    if not ok:
        return
    W, H = len(ok) * (cell + 8), cell + 46
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    f = _font(15)
    for i, (lab, p) in enumerate(ok):
        x = i * (cell + 8)
        q = Image.open(p).convert("RGB")
        q.thumbnail((cell, cell))
        im.paste(q, (x, 34))
        d.text((x + 4, 8), lab[:52], font=f, fill=(20, 20, 20))
    im.save(out_png)
    log(f"{out_png} · {len(ok)} ảnh")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--model", default="sdxl_base")
    ap.add_argument("--prompt-source", default="culture_trip")
    ap.add_argument("--threshold", type=float, default=copilot.DEFAULT_THRESHOLD)
    ap.add_argument("--rounds", type=int, default=copilot.DEFAULT_MAX_ROUNDS)
    ap.add_argument("--run-name", default="loopv2")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    log = lambda *x: print(*x, flush=True)  # noqa: E731

    ov: dict = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        set_dotted(ov, k, v)
    set_dotted(ov, "t2i.render", "bare")
    set_dotted(ov, "multigen.adaptive.enabled", "false")
    set_dotted(ov, "multigen.n_candidates", "1")
    cfg = Config.load(a.config, ov)
    prompt = {p.id: p for p in load_prompts(cfg.prompts_path)}[a.id]
    run_dir = Path(cfg.runs_dir) / a.run_name
    s = Session(cfg, prompt, run_dir=run_dir, log=log)

    # --- grounding chỉ để lấy ảnh tham chiếu và GenSpec; spec KHÔNG được dùng làm bảng kiểm
    s.grounding()
    refined, evidence = external_prompt(a.prompt_source, a.id) if a.prompt_source != "original" else ("", "")
    if refined:
        s.set_prompt_en(refined)
        log(f"[prompt] dùng câu từ data/{a.prompt_source}/: {len(refined.split())} từ")
    gen, _ = s.genspec()
    refs = s.prompt_refs(include_candidates=False)[:3]
    log(f"[refs] {len(refs)} ảnh thật của prompt: {[Path(r).name for r in refs]}")

    a_in = copilot.interpret(s.agent, prompt.text_vi, prompt.text_en, refined, evidence, log=log)

    from ctig.stages import multigen as mg

    out_dir = s.out_dir
    base_terms = list(gen.prompt_terms)

    base_neg = list(gen.negative_terms)

    def make(positive: str, negative: list[str], n: int) -> str | None:
        # positive vào prompt, negative vào negative_prompt. KHÔNG bao giờ dán nhận xét thô vào prompt.
        g = replace(gen, prompt_terms=base_terms + ([positive] if positive else []),
                    negative_terms=base_neg + [x for x in (negative or []) if x not in base_neg],
                    iteration=n, ip_adapter_image=(refs or None), ip_adapter_scale=cfg.multigen.ref_scale)
        key = a.model if not refs else (a.model if "+ref" in a.model else a.model + "+ref")
        r = mg.run(g, s.spec()[0], s.kb, [key], cfg.multigen, out_dir / f"iter{n}", clip=s.clip, itm=None,
                   t2i_cfg=cfg.t2i, prompt_en=s.analysis()[0].prompt_en, log=log,
                   ref_images=refs, force_refs=bool(refs))
        for run_rec in r.runs:
            if run_rec.output and run_rec.output.candidates:
                return run_rec.output.candidates[0].path
        return None

    first = make("", [], 0)
    if not first:
        raise SystemExit("không sinh được ảnh đầu")
    log(f"[gen] ảnh đầu {first}")

    crop = lambda p: __import__("ctig.agents.describe", fromlist=["x"]).subject_crop(  # noqa: E731
        s.agent, p, [a_in.get("entity_en") or "object", "person"])
    out = copilot.run_loop(s.agent, a_in, first, make, refs=refs, crop=crop,
                           threshold=a.threshold, max_rounds=a.rounds, log=log)

    res = {"prompt_id": a.id, "model": a.model, "prompt_source": a.prompt_source,
           "prompt_used": " ".join(base_terms), "report": a_in, "refs": refs,
           "threshold": a.threshold, **out}
    (out_dir / "loop_v2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    imgs = [("vòng 0 · %.1f" % out["rounds"][0]["eval"]["overall"] if out["rounds"] and "eval" in out["rounds"][0]
             else "vòng 0", first)]
    imgs = [(f"vòng 0 (mốc)", first)] + [(f"vòng {r['n']} · {r.get('eval', {}).get('overall', 0):.1f}/10",
                                          r.get("image")) for r in out["rounds"]]
    grid(imgs, out_dir / "loop_v2.png", log=log)
    log(f"[xong] {out['stop']} · ảnh cuối {out['final']}")
    log(f"[xong] {out_dir / 'loop_v2.json'}")


if __name__ == "__main__":
    main()
