"""Chạy bốn nhánh B / T / S / M của VietRepair trên cùng một ảnh nháp.

    python scripts/run_arms.py --config configs/vast_arms.yaml --ids S001,S002 --reps 2 --run-name arms

Mỗi (prompt, lần lặp) sinh 5 ảnh: ảnh nháp I0 ở seed s0, rồi bốn nhánh ĐỀU sinh ở seed s1.
Giữ nhiễu cố định giữa bốn nhánh là điều kiện để so được với nhau; chỉ CÂU PROMPT khác nhau.

| nhánh | thấy gì | trả lời câu hỏi nào |
|---|---|---|
| B | không agent, đúng prompt Culture-TRIP | mốc dưới |
| T | Refiner có contract nhưng KHÔNG nhìn ảnh | prompt dài thêm có phải là lý do không? |
| S | một VLM nhìn ảnh rồi tự viết mệnh đề sửa | phân vai có ích, hay chỉ cần nhìn ảnh? |
| M | Observer -> Critic -> Refiner -> Critic duyệt | đủ sơ đồ |

M > B nói phản hồi có ích · M > T nói NHÌN ẢNH có ích · M > S nói PHÂN VAI có ích.
Thiếu S thì sơ đồ ba agent chỉ là trang trí, và đó là lỗ hổng lớn nhất của thiết kế cũ.

Ghi lại TOÀN BỘ M1/M2/M3 và bước duyệt vào `trace.json` để đưa vào supplementary: người đọc phải kiểm
được rằng bốn agent thật sự trao đổi chứ không phải sơ đồ vẽ cho đẹp.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctig.agents import vietrepair as vr  # noqa: E402
from ctig.config import Config, set_dotted  # noqa: E402
from ctig.pipeline import load_prompts  # noqa: E402
from ctig.session import Session  # noqa: E402
from scripts.run_loop_v2 import external_prompt, grid  # noqa: E402

ARMS = ("B", "T", "S", "M")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids", default=None, help="mặc định: mọi prompt có trong data/contracts.json")
    ap.add_argument("--reps", type=int, default=2, help="số cặp seed mỗi prompt")
    ap.add_argument("--model", default="sdxl_base")
    ap.add_argument("--prompt-source", default="culture_trip")
    ap.add_argument("--run-name", default="arms")
    ap.add_argument("--contracts", default=None)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)

    ov: dict = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        set_dotted(ov, k, v)
    set_dotted(ov, "t2i.render", "bare")
    set_dotted(ov, "multigen.adaptive.enabled", "false")
    set_dotted(ov, "multigen.n_candidates", "1")
    set_dotted(ov, "multigen.keep_loaded", "0")   # bộ chấm 24B đã ~48 GB, không giữ thêm pipeline
    cfg = Config.load(a.config, ov)

    contracts = vr.load_contracts(a.contracts)
    if not contracts:
        raise SystemExit("không đọc được data/contracts.json")
    allp = {p.id: p for p in load_prompts(cfg.prompts_path)}
    ids = [i.strip() for i in a.ids.split(",")] if a.ids else [i for i in sorted(contracts) if i in allp]
    run_dir = Path(cfg.runs_dir) / a.run_name
    log = lambda *x: print(*x, flush=True)  # noqa: E731
    log(f"{len(ids)} prompt × {a.reps} lần lặp × 5 ảnh = {len(ids) * a.reps * 5} ảnh")

    from ctig.stages import multigen as mg

    shared = [None]
    for pid in ids:
        if pid not in contracts:
            log(f"[{pid}] không có contract -> bỏ"); continue
        contract = contracts[pid]
        for rep in range(a.reps):
            s0, s1 = 1000 + rep * 77, 5000 + rep * 77      # cặp seed cố định, lặp lại được
            tag = f"{pid}_r{rep}"
            log(f"\n========== {tag}  (nháp seed {s0}, bốn nhánh seed {s1}) ==========")
            try:
                out_dir = run_dir / tag
                s = Session(cfg, allp[pid], run_dir=out_dir, log=lambda *x: None)
                if shared[0] is not None:
                    s._agent = shared[0]
                s.skip_grounding()
                s.spec()
                refined, _ = external_prompt(a.prompt_source, pid) if a.prompt_source != "original" else ("", "")
                if refined:
                    s.set_prompt_en(refined)
                gen, _ = s.genspec()
                base_prompt = " ".join(gen.prompt_terms)
                base_neg = list(gen.negative_terms)

                def sinh(prompt_terms, neg, seed, sub):
                    g = replace(gen, prompt_terms=prompt_terms, negative_terms=neg, seed=seed, iteration=0)
                    r = mg.run(g, s.spec()[0], s.kb, [a.model], cfg.multigen, out_dir / sub, clip=s.clip,
                               itm=None, t2i_cfg=cfg.t2i, prompt_en=s.analysis()[0].prompt_en,
                               log=lambda *x: None, ref_images=[], force_refs=False)
                    for rr in r.runs:
                        if rr.output and rr.output.candidates:
                            return rr.output.candidates[0].path
                    return None

                # --- ảnh nháp I0: ba agent nhìn cái này
                i0 = sinh(list(gen.prompt_terms), base_neg, s0, "draft")
                if not i0:
                    log(f"[{tag}] không sinh được ảnh nháp -> bỏ"); continue
                shared[0] = s.agent

                # --- ba nhánh có can thiệp; mỗi nhánh trả (mệnh đề, negative, trace)
                traces = {}
                log("  --- M: Observer -> Critic -> Refiner -> duyệt ---")
                tm = vr.run_multi(s.agent, i0, base_prompt, contract, pid, log)
                traces["M"] = tm.to_dict()
                log("  --- S: một VLM tự viết ---")
                ms = vr.single_agent(s.agent, i0, base_prompt, contract, log)
                traces["S"] = ms
                log("  --- T: có contract, KHÔNG nhìn ảnh ---")
                mt = vr.text_only(s.agent, base_prompt, contract, log)
                traces["T"] = mt

                de_xuat = {"B": ("", []),
                           "T": (mt.get("repair_clause", ""), mt.get("negative_terms") or []),
                           "S": (ms.get("repair_clause", ""), ms.get("negative_terms") or []),
                           "M": (tm.repair_clause, tm.negative_terms)}

                # --- bốn nhánh sinh ở CÙNG seed s1, chỉ khác câu prompt
                anh, ghi_chu = {}, {}
                for arm in ARMS:
                    clause, neg = de_xuat[arm]
                    full, note = vr.append_repair(base_prompt, clause)
                    ghi_chu[arm] = note
                    terms = [full] if clause else list(gen.prompt_terms)
                    anh[arm] = sinh(terms, base_neg + [x for x in neg if x not in base_neg], s1, arm)
                    log(f"  [{arm}] {'no-op, dùng prompt gốc' if not clause else clause[:64]}"
                        + (f"  ({note})" if note else ""))

                res = {"prompt_id": pid, "rep": rep, "seed_draft": s0, "seed_arms": s1,
                       "base_prompt": base_prompt, "draft": i0, "images": anh,
                       "proposals": {k: {"clause": v[0], "negative": v[1]} for k, v in de_xuat.items()},
                       "notes": ghi_chu, "traces": traces,
                       "noop_M": tm.noop, "ly_do_noop_M": tm.ly_do_noop}
                (out_dir / "arms.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
                grid([("nháp I0", i0)] + [(arm, anh[arm]) for arm in ARMS if anh.get(arm)],
                     out_dir / "arms.png", log=lambda *x: None)
                log(f"  -> {out_dir / 'arms.json'}" + ("  [M no-op: " + tm.ly_do_noop + "]" if tm.noop else ""))
            except Exception as exc:  # noqa: BLE001
                import traceback
                log(f"[{tag}] LỖI {type(exc).__name__}: {exc}")
                traceback.print_exc()

    log("\nARMS_DONE")


if __name__ == "__main__":
    main()
