"""Vòng lặp rà-sửa kiểu T2I-Copilot rút gọn, chạy trên 1-3 prompt để soi từng bước.

    python scripts/run_copilot.py --config configs/vast_arms.yaml --ids S001 --max-vong 4

Ba nhánh, TẤT CẢ cùng một seed s1 và cùng prompt Culture-TRIP làm gốc:

| nhánh | là gì | trả lời câu hỏi |
|---|---|---|
| B | một ảnh, prompt gốc, không agent | mốc dưới |
| L | vòng lặp: rà từng mục contract -> sửa -> sinh lại, giữ nguyên seed | vòng lặp có ích không |
| K | bốc thăm N seed khác nhau, prompt gốc, chấm bằng CHÍNH bộ rà của L, lấy điểm cao nhất | **L có hơn "sinh nhiều rồi chọn" không** |

N của nhánh K đặt đúng bằng số ảnh nhánh L đã sinh, nên hai nhánh tiêu cùng ngân sách. K là nhánh
quan trọng nhất và là nhánh dễ bị bỏ quên nhất: đo ngày 2026-09-17, best-of-4 bốc thăm THẮNG vòng lặp
cũ ở 2/3 prompt khi cùng ngân sách. Thiếu K thì mọi cải thiện của L đều có thể chỉ là sinh thêm ảnh.

Vòng 0 của L DÙNG LẠI đúng ảnh nhánh B, nên hiệu số L - B là hiệu số của riêng phần sửa prompt, không
lẫn khác biệt seed.

Điểm ở đây tính TỪ contract mà vòng lặp lại tối ưu thẳng vào nó, nên KHÔNG được báo cáo như kết quả;
nó chỉ để lái vòng lặp, để dừng sớm, và để chấm nhánh K cho công bằng.
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids", required=True, help="1-3 prompt, ví dụ S001 hoặc S001,S012")
    ap.add_argument("--model", default="sdxl_base")
    ap.add_argument("--prompt-source", default="culture_trip")
    ap.add_argument("--run-name", default="copilot")
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--max-vong", type=int, default=4)
    ap.add_argument("--kien-nhan", type=int, default=3)
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
    set_dotted(ov, "multigen.keep_loaded", "0")
    cfg = Config.load(a.config, ov)

    contracts = vr.load_contracts(a.contracts)
    allp = {p.id: p for p in load_prompts(cfg.prompts_path)}
    ids = [i.strip() for i in a.ids.split(",")]
    run_dir = Path(cfg.runs_dir) / a.run_name
    log = lambda *x: print(*x, flush=True)  # noqa: E731

    from ctig.stages import multigen as mg

    shared = [None]
    for pid in ids:
        if pid not in contracts:
            log(f"[{pid}] không có contract -> bỏ"); continue
        if pid not in allp:
            log(f"[{pid}] không có trong bộ prompt -> bỏ"); continue
        contract = contracts[pid]
        log(f"\n================= {pid} · {contract.get('entity_vi', '')} =================")
        out_dir = run_dir / pid
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

        dem = [0]

        def sinh(prompt_terms, neg, sub, seed=None):
            dem[0] += 1
            g = replace(gen, prompt_terms=prompt_terms, negative_terms=neg,
                        seed=a.seed if seed is None else seed, iteration=0)
            r = mg.run(g, s.spec()[0], s.kb, [a.model], cfg.multigen, out_dir / sub, clip=s.clip,
                       itm=None, t2i_cfg=cfg.t2i, prompt_en=s.analysis()[0].prompt_en,
                       log=lambda *x: None, ref_images=[], force_refs=False)
            for rr in r.runs:
                if rr.output and rr.output.candidates:
                    return rr.output.candidates[0].path
            return None

        # --- nhánh B: một ảnh, prompt gốc. Cũng chính là vòng 0 của L.
        anh_b = sinh(list(gen.prompt_terms), base_neg, "B")
        if not anh_b:
            log(f"[{pid}] không sinh được ảnh nền -> bỏ"); continue
        shared[0] = s.agent
        log(f"  [B] {anh_b}")

        # --- nhánh L: vòng lặp, giữ nguyên seed
        L = vr.run_loop(s.agent, lambda p, n, sub: sinh(p, n, f"L_{sub}"), anh_b, base_prompt,
                        contract, allp[pid].text_en, pid, a.max_vong, a.kien_nhan, log)
        n_gen = sum(1 for h in L["lich_su"] if h["vong"] > 0) + 1

        # --- nhánh K: bốc thăm n_gen seed, prompt gốc, chấm bằng chính bộ rà
        log(f"  --- K: bốc thăm {n_gen} seed, cùng ngân sách với L ---")
        K = [{"seed": a.seed, "anh": anh_b, "diem": L["lich_su"][0]["diem"]}]
        for i in range(1, n_gen):
            sd = a.seed + i
            p = sinh(list(gen.prompt_terms), base_neg, f"K_seed{sd}", seed=sd)
            if p:
                K.append({"seed": sd, "anh": p, "diem": vr.kiem_tung_muc(s.agent, p, contract, log)["diem"]})
        bestK = max(K, key=lambda x: x["diem"])

        res = {"prompt_id": pid, "seed": a.seed, "base_prompt": base_prompt,
               "B": {"anh": anh_b, "diem": L["lich_su"][0]["diem"]},
               "L": L, "K": {"ung_vien": K, "chon": bestK},
               "so_anh_sinh": dem[0], "so_required": L["so_required"]}
        (out_dir / "copilot.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

        o = [("B nền", anh_b)] + [(f"L vòng{h['vong']} ({h['diem']}/{L['so_required']})", h["anh"])
                                  for h in L["lich_su"] if h["vong"] > 0] \
            + [(f"K seed{k['seed']} ({k['diem']})", k["anh"]) for k in K[1:]]
        grid(o, out_dir / "copilot.png", log=lambda *x: None)
        log(f"\n  KẾT QUẢ {pid}:  B {L['lich_su'][0]['diem']}/{L['so_required']}"
            f"  ·  L {L['diem']}/{L['so_required']} (vòng {L['vong_chon']}, {L['ly_do_dung']})"
            f"  ·  K {bestK['diem']}/{L['so_required']} (seed {bestK['seed']})"
            f"  ·  {dem[0]} ảnh sinh\n  -> {out_dir / 'copilot.json'}")

    print("\nCOPILOT_DONE", flush=True)


if __name__ == "__main__":
    main()
