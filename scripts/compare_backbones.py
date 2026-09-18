"""So các model nền T2I trên cùng prompt, cùng seed, KHÔNG agent nào.

    python scripts/compare_backbones.py --config configs/vast_arms.yaml \
        --ids S001,S003,S012,S020,S031,S040 --models sdxl_base,realvis_xl

Câu hỏi của script này hẹp: **chọn model nền cho bảng chính**. Nó cố ý không gọi agent, không chấm
contract, không dùng Mistral — vì trộn hai câu hỏi ("nền nào tốt hơn" và "agent có ích không") vào một
lô là cách chắc chắn nhất để không trả lời được câu nào.

Mỗi ô của lưới = (prompt, model, có/không ảnh tham chiếu), đúng MỘT ảnh, cùng seed cho mọi ô.

Vì sao chỉ còn hai ứng viên:
  - FLUX.1-dev bị loại bằng lý lẽ, không cần chạy: nó là DiT distilled và KHÔNG nhận negative prompt,
    mà Refiner của phương pháp sinh ra negative_terms như một nửa đầu ra. Dùng FLUX là bỏ mất nửa cơ chế.
  - SDXL base và RealVisXL V4.0 cùng họ sdxl, đều có negative, đều chạy được IP-Adapter Plus (ViT-H).
  - Các khoá có LoRA (sdxl_aodai, realvis_aodai) không vào đây: LoRA áo dài chỉ áp cho một thực thể nên
    không so được trên cả bộ prompt, và nó là biến khác chứ không phải model nền.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctig.config import Config, set_dotted  # noqa: E402
from ctig.evaluation import ref_split  # noqa: E402
from ctig.pipeline import load_prompts  # noqa: E402
from ctig.session import Session  # noqa: E402
from scripts.run_loop_v2 import external_prompt  # noqa: E402


def _font(size, bold=False):
    from PIL import ImageFont
    n = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    p = f"/usr/share/fonts/truetype/dejavu/{n}"
    return ImageFont.truetype(p, size) if Path(p).exists() else ImageFont.load_default()


def ve(hang: list[tuple[str, dict]], cot: list[str], out_png: Path, cell: int = 300):
    """Lưới: mỗi HÀNG một prompt, mỗi CỘT một (model, có ref hay không)."""
    from PIL import Image, ImageDraw

    gut, head, pad = 84, 30, 5
    cw, ch = cell + pad, cell + pad
    W, H = gut + len(cot) * cw + pad, head + len(hang) * ch + pad
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    for i, c in enumerate(cot):
        d.text((gut + i * cw + 3, 8), c, font=_font(14, True), fill=(20, 20, 20))
    for r, (pid, o) in enumerate(hang):
        y = head + r * ch
        d.text((4, y + 6), pid, font=_font(13), fill=(20, 20, 20))
        for i, c in enumerate(cot):
            p = o.get(c)
            x = gut + i * cw
            if not p or not Path(p).exists():
                d.text((x + 6, y + 6), "—", font=_font(13), fill=(150, 150, 150))
                continue
            t = Image.open(p).convert("RGB")
            t.thumbnail((cell, cell))
            im.paste(t, (x + (cell - t.width) // 2, y))
        d.line([(0, y - 2), (W, y - 2)], fill=(225, 225, 225))
    im.save(out_png)
    print(f"-> {out_png} · {len(hang)}×{len(cot)} · {out_png.stat().st_size // 1024} KB", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--models", default="sdxl_base,realvis_xl")
    ap.add_argument("--prompt-source", default="culture_trip")
    ap.add_argument("--run-name", default="backbones")
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--no-ref", action="store_true", help="chỉ chạy cột không ảnh tham chiếu")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)

    ov: dict = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        set_dotted(ov, k, v)
    set_dotted(ov, "t2i.render", "bare")
    set_dotted(ov, "multigen.adaptive.enabled", "false")
    set_dotted(ov, "multigen.n_candidates", "1")
    # Giữ cả hai nền thường trú: đã đo, nạp lại từ đĩa cho TỪNG prompt biến 38 s/ảnh thành 72 s.
    set_dotted(ov, "multigen.keep_loaded", "2")
    cfg = Config.load(a.config, ov)

    models = [m.strip() for m in a.models.split(",")]
    allp = {p.id: p for p in load_prompts(cfg.prompts_path)}
    ids = [i.strip() for i in a.ids.split(",") if i.strip() in allp]
    run_dir = Path(cfg.runs_dir) / a.run_name
    cot = [m for m in models] + ([] if a.no_ref else [f"{m}+ref" for m in models])
    print(f"{len(ids)} prompt × {len(cot)} cột = {len(ids) * len(cot)} ảnh · seed {a.seed}", flush=True)

    from ctig.stages import multigen as mg

    hang, so_lieu = [], []
    for pid in ids:
        out_dir = run_dir / pid
        s = Session(cfg, allp[pid], run_dir=out_dir, log=lambda *x: None)
        s.skip_grounding()
        s.spec()
        refined, _ = external_prompt(a.prompt_source, pid) if a.prompt_source != "original" else ("", "")
        if refined:
            s.set_prompt_en(refined)
        gen, _ = s.genspec()

        refs, _ = ref_split(cfg.retrieval.ref_dir, pid, 5)
        refs = refs[:cfg.multigen.ref_images]

        o = {}
        for c in cot:
            dung_ref = c.endswith("+ref")
            key = c
            if dung_ref and not refs:
                print(f"  [{pid}] {c}: không có ảnh thật -> bỏ ô", flush=True)
                continue
            rf = refs if dung_ref else []
            g = replace(gen, seed=a.seed, iteration=0,
                        ip_adapter_image=(rf or None), ip_adapter_scale=cfg.multigen.ref_scale)
            r = mg.run(g, s.spec()[0], s.kb, [key], cfg.multigen, out_dir / c.replace("+", "_"),
                       clip=s.clip, itm=None, t2i_cfg=cfg.t2i,
                       prompt_en=s.analysis()[0].prompt_en, log=lambda *x: None,
                       ref_images=rf, force_refs=bool(rf))
            p = next((rr.output.candidates[0].path for rr in r.runs
                      if rr.output and rr.output.candidates), None)
            o[c] = p
            print(f"  [{pid}] {c}: {'ok' if p else 'HỎNG'}", flush=True)
        hang.append((pid, o))
        so_lieu.append({"prompt_id": pid, "seed": a.seed, "refs": refs, "anh": o})
        ve(hang, cot, run_dir / "backbones.png")        # vẽ lại sau mỗi prompt, xem được ngay
        (run_dir / "backbones.json").write_text(
            json.dumps({"models": models, "cot": cot, "seed": a.seed, "don_vi": so_lieu},
                       ensure_ascii=False, indent=1), encoding="utf-8")

    print("\nBACKBONES_DONE", flush=True)


if __name__ == "__main__":
    main()
