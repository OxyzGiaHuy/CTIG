"""Ablation C3: vòng sửa có hơn "sinh N ảnh rồi chọn cái tốt nhất" không?

    python scripts/bestofn_control.py --config configs/vast_arms.yaml --ids S001,S002,S003 [-n 4] \
        [--set llm.backend=mistral_vl] [--run-name bestof4]

Đây là câu phản biện sẽ hỏi đầu tiên. Vòng sửa hiện SINH LẠI TOÀN BỘ ảnh mỗi vòng với seed mới, nên bốn ảnh
của bốn vòng là bốn mẫu độc lập chứ không phải bốn bước tinh chỉnh. Nếu lấy đúng prompt của nhánh B, sinh N
ảnh bằng N seed khác nhau, rồi chấm bằng CHÍNH bộ chấm đó và giữ ảnh điểm cao nhất, mà kết quả ngang với
vòng sửa, thì toàn bộ phần "agentic" chỉ là bốc thăm nhiều lần.

Giữ mọi thứ giống nhánh C trừ đúng một điều: không có lời phê, không sửa prompt, chỉ đổi seed.
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
from scripts.run_loop_v2 import external_prompt, grid  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def one(cfg: Config, prompt, run_dir: Path, n: int, model: str, source: str, log, shared=None) -> dict:
    s = Session(cfg, prompt, run_dir=run_dir, log=log)
    # dùng chung MỘT agent: mỗi Session mới tự nạp thêm Mistral 45 GB
    if shared is not None and shared[0] is not None:
        s._agent = shared[0]
    # prompt_refs() đọc thẳng từ đĩa theo prompt_id nên không cần grounding, mà nó tốn 22-191 s mỗi prompt
    s.skip_grounding()
    s.spec()
    refined, evidence = external_prompt(source, prompt.id) if source != "original" else ("", "")
    if refined:
        s.set_prompt_en(refined)
    gen, _ = s.genspec()
    refs = s.prompt_refs(include_candidates=False)[:3]
    a_in = copilot.interpret(s.agent, prompt.text_vi, prompt.text_en, refined, evidence, log=log)

    from ctig.stages import multigen as mg

    crop = lambda p: __import__("ctig.agents.describe", fromlist=["x"]).subject_crop(  # noqa: E731
        s.agent, p, [a_in.get("entity_en") or "object", "person"])

    out = []
    for k in range(n):
        # seed khác nhau là KHÁC BIỆT DUY NHẤT so với vòng 0 của nhánh C; prompt giữ nguyên, không ảnh tham chiếu
        g = replace(gen, seed=cfg.seed + k, iteration=k)
        r = mg.run(g, s.spec()[0], s.kb, [model], cfg.multigen, s.out_dir / f"s{k}", clip=s.clip, itm=None,
                   t2i_cfg=cfg.t2i, prompt_en=s.analysis()[0].prompt_en, log=log, ref_images=[], force_refs=False)
        path = next((rr.output.candidates[0].path for rr in r.runs if rr.output and rr.output.candidates), None)
        if not path:
            log(f"[{prompt.id}] seed {cfg.seed + k}: không sinh được ảnh")
            continue
        ev = copilot.evaluate(s.agent, path, a_in, refs, crop(path), log=log)
        out.append({"seed": cfg.seed + k, "image": path, "eval": ev.to_dict()})

    if not out:
        return {"prompt_id": prompt.id, "samples": []}
    if shared is not None:
        shared[0] = s.agent
    best = max(out, key=lambda x: x["eval"]["overall"])
    best_c = max(out, key=lambda x: x["eval"]["axes"]["culture"])
    res = {"prompt_id": prompt.id, "model": model, "prompt_source": source, "n": n, "refs": refs,
           "samples": out, "best_overall": best["image"], "best_culture": best_c["image"],
           "scores_overall": [round(x["eval"]["overall"], 2) for x in out],
           "scores_culture": [round(x["eval"]["axes"]["culture"], 2) for x in out]}
    (s.out_dir / "bestofn.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    grid([(f"seed {x['seed']} · {x['eval']['overall']:.1f}", x["image"]) for x in out],
         s.out_dir / "bestofn.png", log=log)
    log(f"[{prompt.id}] tổng {res['scores_overall']} · văn hoá {res['scores_culture']}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("-n", type=int, default=4, help="số mẫu, nên bằng số ảnh vòng sửa giữ lại (1 mốc + 3 vòng)")
    ap.add_argument("--model", default="sdxl_base")
    ap.add_argument("--prompt-source", default="culture_trip")
    ap.add_argument("--run-name", default="bestofn")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)

    ov: dict = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        set_dotted(ov, k, v)
    set_dotted(ov, "t2i.render", "bare")
    set_dotted(ov, "multigen.adaptive.enabled", "false")
    set_dotted(ov, "multigen.n_candidates", "1")
    set_dotted(ov, "multigen.keep_loaded", "0")     # bộ chấm 24B đã chiếm ~48 GB
    cfg = Config.load(a.config, ov)
    prompts = {p.id: p for p in load_prompts(cfg.prompts_path)}
    run_dir = Path(cfg.runs_dir) / a.run_name
    log = lambda *x: print(*x, flush=True)  # noqa: E731

    shared = [None]
    for pid in [x.strip() for x in a.ids.split(",") if x.strip()]:
        if pid not in prompts:
            log(f"[{pid}] không có trong {cfg.prompts_path}")
            continue
        try:
            one(cfg, prompts[pid], run_dir, a.n, a.model, a.prompt_source, log, shared)
        except Exception as exc:  # noqa: BLE001
            log(f"[{pid}] LỖI {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
