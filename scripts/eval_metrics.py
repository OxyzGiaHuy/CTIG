"""Metric độc lập cho flow SAVIER trên các lô đã sinh: VQAScore · VCFS · CCR · NRG (CAIRE ghép từ CSV riêng).

    python scripts/eval_metrics.py --run SDXL=/workspace/runs/kor_20260918_1349 \
        --run FLUX.1-dev=/workspace/runs/kor_flux_20260918_1530 -o /workspace/runs/metrics_10

Giao thức (đúng theo yêu cầu, và mỗi điều đều kiểm được trong metrics.json):
  - Contract ĐÓNG BĂNG: đọc `data/contracts_v2.json` (commit e1c9da0, trước mọi lô kor*), KHÔNG dùng card K
    sinh lúc chạy. w_i = `importance`; chỉ `scoring == "required"` vào VCFS; `conditional` báo riêng.
  - Evaluator KHÔNG phải model đã làm Observer/Refiner (Mistral-24B): dùng Qwen2.5-VL-7B-Instruct.
  - Evaluator không biết ảnh là I0 hay I1: mỗi lời gọi một ảnh + một câu; thứ tự (ảnh, câu) xáo trộn
    bằng seed cố định; tên tệp không lộ nhánh (chép về tên băm trước khi hỏi).
  - VQAScore chấm theo P0 tiếng Anh GỐC, không theo P1. Model chính chủ clip-flant5 (t2v_metrics); bản
    -xl vì đĩa; -xxl là bản trong bài gốc — ghi rõ trong bảng.
  - CI bootstrap THEO PROMPT (n = số prompt), 2000 lần.
  - NRG = N/A khi I0 không có mục nào thiếu; báo correct no-op riêng.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np

Y = {"YES": 1.0, "PARTIAL": 0.5, "NO": 0.0}


class DanhGia:
    """Qwen2.5-VL-7B: một ảnh + một câu -> YES / PARTIAL / NO. Không thấy prompt, không thấy nhánh."""

    def __init__(self, model_id: str, device: str = "cuda:0"):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.calls = 0

    def hoi(self, image: str, statement: str, kieu: str) -> tuple[float, str]:
        from PIL import Image
        if kieu == "required":
            q = (f'Statement about the main object: "{statement}"\n'
                 "Is this clearly shown in the photograph? Answer with exactly one word: YES (clearly shown), "
                 "PARTIAL (unclear or only partly shown), or NO (not shown or contradicted).")
        else:
            q = (f'Look-alike description: "{statement}"\n'
                 "Does the photograph show THIS look-alike appearance? Answer with exactly one word: YES (clearly present), "
                 "PARTIAL (seems present but unsure), or NO (not present).")
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.open(image).convert("RGB")}, {"type": "text", "text": q}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[Image.open(image).convert("RGB")], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=4, do_sample=False)
        ans = self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().upper()
        self.calls += 1
        for k in ("PARTIAL", "YES", "NO"):
            if k in ans:
                return Y[k], ans
        return 0.5, ans          # không đọc được -> không chắc, không thiên về bên nào


def vqa_yes(ev: "DanhGia", image: str, prompt_en: str) -> float:
    """VQAScore (Lin et al. 2024) = P(Yes | 'Does this figure show "<prompt>"?'), tính từ logits token đầu.

    Backbone ở đây là Qwen2.5-VL-7B, KHÔNG phải clip-flant5-xxl của bài gốc: t2v_metrics 3.0 kéo theo chuỗi
    phụ thuộc (LLaVA-OV, torch 2.5.1, transformers 4.49) và vẫn lỗi trong forward T5 trên máy này sau
    bốn lần vá. Công thức giữ nguyên; phải ghi rõ backbone trong bảng.
    """
    from PIL import Image
    q = f'Does this figure show "{prompt_en.strip()}"? Please answer yes or no.'
    msgs = [{"role": "user", "content": [{"type": "image", "image": Image.open(image).convert("RGB")}, {"type": "text", "text": q}]}]
    text = ev.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = ev.proc(text=[text], images=[Image.open(image).convert("RGB")], return_tensors="pt").to(ev.model.device)
    with ev.torch.no_grad():
        logits = ev.model(**inputs).logits[0, -1]
    tok = ev.proc.tokenizer
    ids_yes = {tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes", " yes")}
    ids_no = {tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no", " no")}
    pr = ev.torch.softmax(logits.float(), dim=-1)
    py, pn = float(sum(pr[i] for i in ids_yes)), float(sum(pr[i] for i in ids_no))
    return py / (py + pn) if (py + pn) > 0 else 0.5


def _bootstrap(vals: list[float], n=2000, seed=0):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if len(v) == 0:
        return None
    rng = np.random.default_rng(seed)
    m = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return {"mean": round(float(v.mean()), 2), "lo": round(float(np.percentile(m, 2.5)), 2),
            "hi": round(float(np.percentile(m, 97.5)), 2), "n": int(len(v))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="<nhãn>=<thư mục lô>")
    ap.add_argument("--contracts", default="data/contracts_v2.json")
    ap.add_argument("--prompts", default="data/prompts_simple.json")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--evaluator", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vqa-model", default="clip-flant5-xl")
    ap.add_argument("--vqa-backend", choices=["t2v", "qwen"], default="qwen",
                    help="qwen = P(Yes) bằng chính evaluator Qwen2.5-VL (mặc định, chạy được); t2v = t2v_metrics clip-flant5")
    ap.add_argument("--caire-csv", default=None, help="combined_outputs.csv của CAIRE (nếu đã chạy), ghép theo đường dẫn ảnh")
    ap.add_argument("--vqa-json", default=None, help="vqa.json từ scripts/vqascore_only.py (venv riêng); có thì không chạy t2v trong tiến trình này")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = lambda *x: print(*x, flush=True)  # noqa: E731

    contracts = json.loads(Path(a.contracts).read_text(encoding="utf-8"))
    prompts = {p["id"]: p for p in json.loads(Path(a.prompts).read_text(encoding="utf-8"))}

    # ---- gom ảnh: (run, pid, nhánh) -> đường dẫn; chép sang tên băm để evaluator không đọc được nhánh từ tên tệp
    mu = out / "_blind"; mu.mkdir(exist_ok=True)
    anh, don_vi = {}, []
    for spec in a.run:
        nhan, _, d = spec.partition("="); rd = Path(d)
        j = json.loads((rd / "savier.json").read_text(encoding="utf-8"))
        for u in j["units"]:
            pid = u["prompt_id"]
            for nh in ("A", "I0", "I1"):
                p = u["images"].get(nh)
                if not p:
                    continue
                src = Path(p) if Path(p).exists() else rd / pid / f"{nh}.png"
                if not src.exists():
                    continue
                h = hashlib.sha1(f"{nhan}|{pid}|{nh}".encode()).hexdigest()[:12]
                dst = mu / f"{h}.png"
                if not dst.exists():
                    shutil.copy(src, dst)
                anh[(nhan, pid, nh)] = str(dst)
            don_vi.append({"run": nhan, "pid": pid, "P0": u["P0"], "noop": not bool(u.get("actions"))})
    log(f"{len(anh)} ảnh · {len(don_vi)} đơn vị")

    # ---- câu hỏi: required (must_be_visible) + conditional + confusables, từ contract ĐÓNG BĂNG
    cau = []
    for (nhan, pid, nh), img in anh.items():
        c = contracts.get(pid, {})
        for r in c.get("required", []):
            sc = r.get("scoring", "required")
            if sc == "excluded":
                continue
            cau.append({"run": nhan, "pid": pid, "nhanh": nh, "img": img, "kieu": "required", "scoring": sc,
                        "id": r["id"], "w": float(r.get("importance", 3)), "text": r.get("visual_evidence") or r["description"]})
        for x in c.get("confusables", []):
            cau.append({"run": nhan, "pid": pid, "nhanh": nh, "img": img, "kieu": "confusable", "scoring": "confusable",
                        "id": x["id"], "w": 1.0, "text": x.get("description") or x.get("name_en", "")})
    random.Random(a.seed).shuffle(cau)         # xáo thứ tự (ảnh, câu); evaluator không biết I0/I1
    log(f"{len(cau)} câu hỏi cho evaluator {a.evaluator}")

    ev = DanhGia(a.evaluator)
    t0 = time.time()
    for k, q in enumerate(cau):
        q["y"], q["raw"] = ev.hoi(q["img"], q["text"], q["kieu"])
        if k % 50 == 0:
            log(f"  {k}/{len(cau)} · {time.time() - t0:.0f}s")
    (out / "raw_evaluator.json").write_text(json.dumps(cau, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- VQAScore theo P0 gốc
    vqa = {}
    if a.vqa_json and Path(a.vqa_json).exists():
        sc = json.loads(Path(a.vqa_json).read_text(encoding="utf-8"))["scores"]
        for key, img in anh.items():
            if Path(img).stem in sc:
                vqa[key] = sc[Path(img).stem]
        log(f"VQAScore đọc từ {a.vqa_json}: {len(vqa)} ảnh")
    if not vqa and a.vqa_backend == "qwen":
        for key, img in anh.items():
            vqa[key] = vqa_yes(ev, img, prompts[key[1]]["text_en"])
        a.vqa_model = "Qwen2.5-VL-7B P(Yes) (không phải clip-flant5)"
        log(f"VQAScore (Qwen2.5-VL P(Yes)) xong {len(vqa)} ảnh")
    try:
        if vqa: raise RuntimeError("đã có VQA")
        import t2v_metrics
        del ev.model; ev.torch.cuda.empty_cache()
        vq = t2v_metrics.VQAScore(model=a.vqa_model)
        for (nhan, pid, nh), img in anh.items():
            vqa[(nhan, pid, nh)] = float(vq(images=[img], texts=[prompts[pid]["text_en"]]).item())
        log(f"VQAScore ({a.vqa_model}) xong {len(vqa)} ảnh")
    except Exception as exc:  # noqa: BLE001
        log(f"VQAScore KHÔNG chạy được: {type(exc).__name__}: {exc}")

    # ---- CAIRE (nếu có CSV): điểm 1-5 cho nhãn Vietnam
    caire = {}
    if a.caire_csv and Path(a.caire_csv).exists():
        import csv
        for row in csv.DictReader(open(a.caire_csv, encoding="utf-8")):
            p = row.get("image_path") or row.get("image") or ""
            col = next((c for c in row if c.lower().startswith("vietnam")), None)
            if col:
                caire[Path(p).stem] = float(row[col])
        for key, img in anh.items():
            if Path(img).stem in caire:
                caire[key] = caire.pop(Path(img).stem)

    # ---- tính chỉ số theo (run, pid, nhánh)
    def diem(nhan, pid, nh):
        rows = [q for q in cau if (q["run"], q["pid"], q["nhanh"]) == (nhan, pid, nh)]
        req = [q for q in rows if q["kieu"] == "required" and q["scoring"] == "required"]
        cond = [q for q in rows if q["kieu"] == "required" and q["scoring"] == "conditional"]
        conf = [q for q in rows if q["kieu"] == "confusable"]
        vcfs = 100 * sum(q["w"] * q["y"] for q in req) / sum(q["w"] for q in req) if req else None
        vc_cond = 100 * sum(q["w"] * q["y"] for q in cond) / sum(q["w"] for q in cond) if cond else None
        ccr = 100 * sum(q["w"] * q["y"] for q in conf) / sum(q["w"] for q in conf) if conf else None
        return {"VCFS": vcfs, "VCFS_cond": vc_cond, "CCR": ccr, "VQA": vqa.get((nhan, pid, nh)), "CAIRE": caire.get((nhan, pid, nh)),
                "y": {q["id"]: (q["y"], q["w"]) for q in req}}

    ket = {"cau_hinh": {"contracts": a.contracts, "evaluator": a.evaluator, "vqa_model": a.vqa_model, "seed": a.seed,
                        "so_loi_goi_evaluator": ev.calls}, "units": []}
    for u in don_vi:
        nhan, pid = u["run"], u["pid"]
        d = {nh: diem(nhan, pid, nh) for nh in ("A", "I0", "I1") if (nhan, pid, nh) in anh}
        # NRG
        y0, y1 = d.get("I0", {}).get("y", {}), d.get("I1", {}).get("y", {})
        D0 = {i for i, (y, w) in y0.items() if y < 1.0}; S0 = {i for i, (y, w) in y0.items() if y >= 1.0}
        FR = sum(y0[i][1] * max(0.0, y1.get(i, (0, 0))[0] - y0[i][0]) for i in D0) / sum(y0[i][1] for i in D0) if D0 else None
        RR = sum(y0[i][1] * max(0.0, y0[i][0] - y1.get(i, (0, 0))[0]) for i in S0) / sum(y0[i][1] for i in S0) if S0 else 0.0
        NRG = None if FR is None else 100 * (FR - RR)
        u.update({"diem": {nh: {k: v for k, v in dd.items() if k != "y"} for nh, dd in d.items()},
                  "D0": sorted(D0), "S0": sorted(S0), "FR": FR, "RR": RR, "NRG": NRG,
                  "correct_noop": (not D0) and u["noop"]})
        ket["units"].append(u)

    # ---- bảng theo run
    md = ["# Metric độc lập trên các lô SAVIER\n", f"Evaluator: `{a.evaluator}` (không phải Mistral) · VQAScore: `{a.vqa_model}` theo P0 gốc · "
          f"contract đóng băng `{a.contracts}` · CI bootstrap theo prompt, 2000 lần.\n"]
    ket["tong_hop"] = {}
    for nhan in dict.fromkeys(u["run"] for u in don_vi):
        us = [u for u in don_vi if u["run"] == nhan]
        md.append(f"\n## {nhan} — {len(us)} prompt\n\n| | CAIRE ↑ | VQAScore ↑ | VCFS ↑ | CCR ↓ | NRG ↑ |\n|---|---:|---:|---:|---:|---:|\n")
        th = {}
        for nh in ("A", "I0", "I1"):
            th[nh] = {m: _bootstrap([u["diem"].get(nh, {}).get(m) for u in us]) for m in ("CAIRE", "VQA", "VCFS", "CCR")}
            nrg = _bootstrap([u["NRG"] for u in us]) if nh == "I1" else None
            f = lambda b: "–" if b is None else f"{b['mean']:.1f} [{b['lo']:.1f}, {b['hi']:.1f}]"  # noqa: E731
            fv = lambda b: "–" if b is None else f"{b['mean']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"  # noqa: E731
            md.append(f"| {nh} | {f(th[nh]['CAIRE'])} | {fv(th[nh]['VQA'])} | {f(th[nh]['VCFS'])} | {f(th[nh]['CCR'])} | {f(nrg) if nh == 'I1' else '–'} |\n")
        dl = {m: _bootstrap([(u["diem"].get("I1", {}).get(m) or 0) - (u["diem"].get("I0", {}).get(m) or 0)
                             for u in us if u["diem"].get("I1", {}).get(m) is not None and u["diem"].get("I0", {}).get(m) is not None])
              for m in ("CAIRE", "VQA", "VCFS", "CCR")}
        md.append(f"| **Δ (I1−I0)** | {f(dl['CAIRE'])} | {fv(dl['VQA'])} | {f(dl['VCFS'])} | {f(dl['CCR'])} | – |\n")
        n = len(us); nrgs = [u["NRG"] for u in us]
        md.append(f"\n| I1 so với I0 | tỷ lệ |\n|---|---:|\n| NRG > 0 | {sum(1 for x in nrgs if x is not None and x > 0)}/{n} |\n"
                  f"| NRG = 0 | {sum(1 for x in nrgs if x is not None and x == 0)}/{n} |\n| NRG < 0 | {sum(1 for x in nrgs if x is not None and x < 0)}/{n} |\n"
                  f"| NRG = N/A (I0 không thiếu gì) | {sum(1 for x in nrgs if x is None)}/{n} |\n| correct no-op | {sum(1 for u in us if u['correct_noop'])}/{n} |\n")
        md.append("\n| prompt | VCFS I0→I1 | CCR I0→I1 | VQA I0→I1 | NRG | D0 (thiếu ở I0) |\n|---|---|---|---|---:|---|\n")
        for u in us:
            g = lambda nh, m, fmt="{:.0f}": (fmt.format(u["diem"].get(nh, {}).get(m)) if u["diem"].get(nh, {}).get(m) is not None else "–")  # noqa: E731
            md.append(f"| {u['pid']} | {g('I0','VCFS')}→{g('I1','VCFS')} | {g('I0','CCR')}→{g('I1','CCR')} | {g('I0','VQA','{:.3f}')}→{g('I1','VQA','{:.3f}')} | "
                      f"{'N/A' if u['NRG'] is None else f'{u[chr(78)+chr(82)+chr(71)]:.0f}'} | {', '.join(u['D0'])[:60]} |\n")
        ket["tong_hop"][nhan] = {"theo_nhanh": th, "delta": dl}
    (out / "metrics.json").write_text(json.dumps(ket, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (out / "metrics.md").write_text("".join(md), encoding="utf-8")
    log("".join(md)); log(f"-> {out / 'metrics.md'}"); log("METRICS_DONE")


if __name__ == "__main__":
    main()
