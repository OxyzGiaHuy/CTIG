"""Sáu nhánh của thí nghiệm chính. Mỗi nhánh đúng MỘT ảnh.

    python scripts/run_arms.py --config configs/vast_arms.yaml --ids S001 --reps 1
    python scripts/run_arms.py --config configs/vast_arms.yaml --reps 2 --run-name full

| nhánh | thấy I0 | contract | ảnh thật | là gì |
|---|---:|---:|---:|---|
| B | | | | prompt Culture-TRIP để nguyên. Ảnh của B CHÍNH LÀ I0 mà ba agent nhìn |
| P | | ✓ | | Refiner đọc contract, KHÔNG bao giờ nhìn ảnh |
| R | | | ✓ | IP-Adapter trên ảnh thật đã chọn tay, không agent nào |
| S | ✓ | ✓ | ✓ | một VLM nhìn ảnh rồi tự viết mệnh đề sửa |
| M | ✓ | ✓ | ✓ | Observer -> Critic -> Refiner |
| K | | | | sinh lại ngẫu nhiên: cùng prompt B, SEED KHÁC |

Kết luận đọc được:  M>B cả hệ có ích · M>R agent hơn được ảnh thật · M>S phân vai có ích ·
P>B contract dạng chữ đã đủ · M>K hơn được chuyện sinh lại nhiều lần.
M không hơn S -> bỏ claim multi-agent. M không hơn R -> phần lớn cải thiện là do ảnh thật.

Seed: I0 và mọi ảnh sửa dùng CHUNG seed `s1`; chỉ K dùng seed khác. Không nhánh nào được lấy best-of-N.
`so_lan_sinh` đếm từng lần gọi generator cho mỗi nhánh và được ghi vào arms.json, để chênh lệch chi phí
là số liệu chứ không phải chuyện tranh cãi về sau. Nhánh no-op KHÔNG sinh thêm lần nào.

Điểm contract ghi dưới khoá `chan_doan` và CHỈ để chẩn đoán, đo tương quan với nhãn người. Không nhánh
nào được chọn ảnh bằng điểm đó — agent tối ưu thẳng vào chính nó.
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
from ctig.evaluation import ref_split  # noqa: E402
from ctig.pipeline import load_prompts  # noqa: E402
from ctig.session import Session  # noqa: E402
from scripts.run_loop_v2 import external_prompt, grid  # noqa: E402

ARMS = ("B", "P", "R", "S", "M", "K")
#: nhánh nào sinh KÈM ảnh thật qua IP-Adapter
CO_REF = {"R", "S", "M"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids", default=None, help="mặc định: mọi prompt có contract; nên truyền 1-3 khi thử")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--k-seeds", type=int, default=2)
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
    set_dotted(ov, "multigen.keep_loaded", "0")   # bộ chấm 24B đã ~45 GB, không giữ thêm pipeline
    cfg = Config.load(a.config, ov)

    arms = [x.strip() for x in a.arms.split(",") if x.strip() in ARMS]
    contracts = vr.load_contracts(a.contracts)
    if not contracts:
        raise SystemExit("không đọc được contract")
    allp = {p.id: p for p in load_prompts(cfg.prompts_path)}
    ids = [i.strip() for i in a.ids.split(",")] if a.ids else [i for i in sorted(contracts) if i in allp]
    run_dir = Path(cfg.runs_dir) / a.run_name
    log = lambda *x: print(*x, flush=True)  # noqa: E731
    log(f"{len(ids)} prompt × {a.reps} lần lặp × nhánh {arms} (K={a.k_seeds} seed)")

    from ctig.stages import multigen as mg

    shared = [None]          # MỘT agent dùng chung: Session mới cho mỗi prompt sẽ nạp bản 24B thứ hai và OOM
    for pid in ids:
        if pid not in contracts or pid not in allp:
            log(f"[{pid}] thiếu contract hoặc thiếu prompt -> bỏ"); continue
        contract = contracts[pid]
        for rep in range(a.reps):
            s1 = 5000 + rep * 77
            tag = f"{pid}_r{rep}"
            log(f"\n========== {tag} · {contract.get('entity_vi', '')} · seed {s1} ==========")
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

                # selected/ để điều kiện IP-Adapter; candidates/ cất riêng để chấm, rời nhau theo băm
                refs, _ = ref_split(cfg.retrieval.ref_dir, pid, 5)
                refs = refs[:cfg.multigen.ref_images]

                dem = {k: 0 for k in ARMS}
                nhanh_dang_chay = ["B"]

                def sinh(prompt_terms, neg, sub, dung_ref=False, seed=None):
                    dem[nhanh_dang_chay[0]] += 1
                    rf = refs if (dung_ref and refs) else []
                    g = replace(gen, prompt_terms=prompt_terms, negative_terms=base_neg + [
                        x for x in (neg or []) if x not in base_neg],
                        seed=s1 if seed is None else seed, iteration=0,
                        ip_adapter_image=(rf or None), ip_adapter_scale=cfg.multigen.ref_scale)
                    r = mg.run(g, s.spec()[0], s.kb, [a.model + "+ref" if rf else a.model], cfg.multigen,
                               out_dir / sub, clip=s.clip, itm=None, t2i_cfg=cfg.t2i,
                               prompt_en=s.analysis()[0].prompt_en, log=lambda *x: None,
                               ref_images=rf, force_refs=bool(rf))
                    return next((rr.output.candidates[0].path for rr in r.runs
                                 if rr.output and rr.output.candidates), None)

                # ---- B: prompt gốc, không ref. Ảnh này CŨNG là I0 cho S và M.
                i0 = sinh(list(gen.prompt_terms), [], "B")
                if not i0:
                    log(f"[{tag}] không sinh được ảnh nền -> bỏ"); continue
                shared[0] = s.agent
                anh = {"B": i0}
                ket: dict = {}
                log(f"  [B] {i0}")

                # ---- R: ảnh thật, không agent
                if "R" in arms and refs:
                    nhanh_dang_chay[0] = "R"
                    anh["R"] = sinh(list(gen.prompt_terms), [], "R", True)
                    log(f"  [R] IP-Adapter, {len(refs)} ảnh thật, không agent")
                elif "R" in arms:
                    log("  [R] không có ảnh thật cho prompt này -> bỏ nhánh R")

                # ---- P / S / M: một lượt sửa prompt
                for arm in [x for x in ("P", "S", "M") if x in arms]:
                    nhanh_dang_chay[0] = arm
                    log(f"  --- {arm} ---")
                    r = vr.run_once(s.agent, sinh, i0, base_prompt, contract, allp[pid].text_en,
                                    pid, arm=arm, dung_ref=(arm in CO_REF), log=log)
                    anh[arm] = r["anh"]
                    ket[arm] = r

                # ---- K: sinh lại ngẫu nhiên, seed khác, cùng prompt B
                K = []
                if "K" in arms:
                    nhanh_dang_chay[0] = "K"
                    for i in range(a.k_seeds):
                        sk = s1 + 1000 + i
                        pk = sinh(list(gen.prompt_terms), [], f"K_seed{sk}", False, sk)
                        if pk:
                            K.append({"seed": sk, "anh": pk})
                    if K:
                        anh["K"] = K[0]["anh"]          # K là control, KHÔNG được chọn best-of-N

                # ---- chẩn đoán: chấm contract mọi ảnh. CHỈ để debug và đo tương quan với nhãn người.
                chan_doan = {}
                for k, pth in anh.items():
                    if pth:
                        chan_doan[k] = vr.kiem_tung_muc(s.agent, pth, contract, lambda *x: None)

                res = {"prompt_id": pid, "rep": rep, "seed": s1, "model": a.model,
                       "base_prompt": base_prompt, "refs": refs,
                       "images": anh, "K_ung_vien": K,
                       "clause": {k: v["clause"] for k, v in ket.items()},
                       "negative": {k: v["negative"] for k, v in ket.items()},
                       "noop": {k: v["noop"] for k, v in ket.items()},
                       "ly_do_noop": {k: v["ly_do_noop"] for k, v in ket.items()},
                       "traces": {k: v["trace"] for k, v in ket.items()},
                       "so_lan_sinh": dem,
                       "chan_doan": {k: {"diem": v.get("diem"), "bang": v.get("bang")}
                                     for k, v in chan_doan.items()}}
                (out_dir / "arms.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
                grid([(f"{k}{' no-op' if ket.get(k, {}).get('noop') else ''}", anh[k])
                      for k in ARMS if anh.get(k)], out_dir / "arms.png", log=lambda *x: None)
                log(f"  sinh: {dict((k, v) for k, v in dem.items() if v)} · no-op: "
                    + (", ".join(k for k, v in ket.items() if v["noop"]) or "không")
                    + f"\n  -> {out_dir / 'arms.json'}")
            except Exception as exc:  # noqa: BLE001
                import traceback
                log(f"[{tag}] LỖI {type(exc).__name__}: {exc}")
                traceback.print_exc()

    log("\nARMS_DONE")


if __name__ == "__main__":
    main()
