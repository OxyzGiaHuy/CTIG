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
    ap.add_argument("--park-vlm", action="store_true",
                    help="gửi bộ chấm xuống RAM trong lúc sinh ảnh. Cần trên card 48 GB: giữ cả Mistral 24B "
                         "(48 GB) lẫn SDXL (đỉnh 24,7 GB) là 57,6 GB. Mất ~3 giây mỗi chiều qua PCIe, rẻ hơn "
                         "nạp lại từ đĩa và rẻ hơn cái giá chất lượng của việc lượng tử hoá.")
    ap.add_argument("--seed-mode", default="fixed", choices=("fixed", "vary"),
                    help="fixed (mặc định): mọi vòng dùng chung seed, khác biệt giữa các vòng chỉ do prompt. "
                         "vary: như bản cũ, mỗi vòng một seed -> các vòng là mẫu độc lập, không so được với nhau.")
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
    # Bộ chấm Mistral-Small-3.1-24B chiếm ~48 GB; giữ thêm pipeline thường trú là tràn card 80 GB.
    # Vòng sửa chạy một prompt một lần nên cũng chẳng tiết kiệm được gì.
    set_dotted(ov, "multigen.keep_loaded", "0")
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
    applied: list[str] = []          # câu mô tả CỘNG DỒN qua các vòng, xem copilot.merge_positive

    def make(positive: str, negative: list[str], n: int) -> str | None:
        # positive vào prompt, negative vào negative_prompt. KHÔNG bao giờ dán nhận xét thô vào prompt.
        #
        # Vòng 0 KHÔNG dùng ảnh tham chiếu. Ảnh mốc phải giống hệt ảnh nhánh B (cùng prompt, cùng seed,
        # cùng model trần) thì hiệu số B->C mới đo đúng công của vòng sửa. Bản trước cho cả vòng 0 chạy
        # '+ref', nên C xuất phát từ một ảnh khác hẳn B và hiệu số trộn lẫn công của IP-Adapter.
        # Ảnh tham chiếu là một HÀNH ĐỘNG SỬA, chỉ vào cuộc từ vòng 1.
        use_refs = refs if n > 0 else []
        nonlocal applied
        applied = copilot.merge_positive(applied, positive, log=log)
        # Seed hiệu dụng ở bộ sinh là gen.seed + 1000*iteration (ctig/stages/generation.py:356). Ở chế độ
        # 'fixed' ta bù lại phần 1000*n để MỌI VÒNG dùng chung một seed: khi đó khác biệt giữa hai vòng chỉ
        # đến từ câu prompt, không từ nhiễu. Chế độ 'vary' giữ như cũ, và khi đó bốn ảnh của bốn vòng là bốn
        # mẫu ĐỘC LẬP — so sánh giữa các vòng không có nghĩa, và vòng sửa khó hơn best-of-N ở chỗ nào.
        seed = (cfg.seed - 1000 * n) if a.seed_mode == "fixed" else cfg.seed
        # Bộ chấm xuống RAM trước khi nạp SDXL: hai thứ không bao giờ cần cùng lúc, mà giữ cả hai trên GPU
        # là 57,6 GB (đo được) — quá chỗ của mọi card 48 GB. Gọi lại tự động ở lần chấm sau.
        if a.park_vlm and s.park_vlm():
            log(f"  [vram] bộ chấm xuống RAM, còn trống {__import__('ctig.models.loader', fromlist=['x']).free_gb(cfg.multigen.device)} GB")
        g = replace(gen, prompt_terms=base_terms + applied, seed=seed,
                    negative_terms=base_neg + [x for x in (negative or []) if x not in base_neg],
                    iteration=n, ip_adapter_image=(use_refs or None), ip_adapter_scale=cfg.multigen.ref_scale)
        key = a.model if not use_refs else (a.model if "+ref" in a.model else a.model + "+ref")
        r = mg.run(g, s.spec()[0], s.kb, [key], cfg.multigen, out_dir / f"iter{n}", clip=s.clip, itm=None,
                   t2i_cfg=cfg.t2i, prompt_en=s.analysis()[0].prompt_en, log=log,
                   ref_images=use_refs, force_refs=bool(use_refs))
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

    res = {"prompt_id": a.id, "model": a.model, "prompt_source": a.prompt_source, "seed_mode": a.seed_mode,
           "applied_positive": applied,
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
