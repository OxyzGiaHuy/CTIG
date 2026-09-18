"""Metric bổ sung theo chuỗi Diagnose -> Repair -> Preserve, trên dữ liệu đã có của eval_metrics.

    python scripts/eval_extra_metrics.py --metrics runs/metrics_50 --run SDXL=<dir> --run FLUX.1-dev=<dir> [--rrr]

  CALR  Collateral Attribute Loss Rate = |S0 \\ S1| / |S0|, S0 = mục contract đúng ở I0 (y=1), S1 = đúng ở I1.
  SIR   Safe Improvement Rate = 1[VCFS(I1) > VCFS(I0) + δ  và  VQA(I1) >= VQA(I0) - ε]; δ=5 điểm, ε=0.02 CHỐT TRƯỚC.
        Báo kèm Improved / Unchanged / Degraded theo VCFS với cùng δ.
  RE    Repair Efficiency = max(0, VCFS(I1)-VCFS(I0)) / số ảnh sinh thêm (= 1 ở SAVIER).
  GDF1* Gap Diagnosis F1 — bản PROXY: D_p lấy từ evaluator độc lập (mục y(I0) < 1), D̂_p là mục contract mà gap
        analysis của R nhắc tới (khớp từ nội dung giữa missing_*/contradictions và description mục). Không có
        nhãn người nên chỉ là proxy; ghi rõ.
  RRR   Repair Realization Rate (cần GPU, --rrr): mỗi action R yêu cầu được hỏi thẳng trên I0 và I1 bằng
        Qwen2.5-VL (0-10); realized = I1 >= 8 và I0 <= 4. RRR_p = realized / requested.
Macro-average theo prompt, CI bootstrap theo prompt.
"""
from __future__ import annotations

import argparse, json, random, re
from pathlib import Path

DELTA, EPS = 5.0, 0.02   # chốt trước khi tính


def _norm(s): return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _boot(vals, n=2000, seed=0):
    v = [x for x in vals if x is not None]
    if not v: return None
    rng = random.Random(seed); m = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(n))
    return {"mean": round(sum(v) / len(v), 2), "lo": round(m[int(0.025 * n)], 2), "hi": round(m[int(0.975 * n)], 2), "n": len(v)}


def fmt(b, d=1): return "–" if b is None else f"{b['mean']:.{d}f} [{b['lo']:.{d}f}, {b['hi']:.{d}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True); ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--contracts", default="data/contracts_v2.json"); ap.add_argument("--rrr", action="store_true")
    ap.add_argument("--evaluator", default="Qwen/Qwen2.5-VL-7B-Instruct")
    a = ap.parse_args()
    M = Path(a.metrics); raw = json.loads((M / "raw_evaluator.json").read_text(encoding="utf-8"))
    met = json.loads((M / "metrics.json").read_text(encoding="utf-8"))
    contracts = json.loads(Path(a.contracts).read_text(encoding="utf-8"))
    runs = {}
    for spec in a.run:
        nh, _, d = spec.partition("="); j = json.loads((Path(d) / "kor.json").read_text(encoding="utf-8"))
        runs[nh] = (Path(d), {u["prompt_id"]: u for u in j["don_vi"]})

    # y_i theo (run, pid, nhánh, id) cho required
    Y = {}
    for q in raw:
        if q["kieu"] == "required" and q["scoring"] == "required":
            Y[(q["run"], q["pid"], q["nhanh"], q["id"])] = q["y"]
    out, md = {}, ["# Metric bổ sung — Diagnose → Repair → Preserve\n",
                   f"CALR · SIR (δ={DELTA} điểm VCFS, ε={EPS} VQAScore, chốt trước) · RE · GDF1 (proxy, D_p từ evaluator độc lập) · RRR (Qwen, nếu chạy)\n"]
    for nh, (rd, units) in runs.items():
        rows = []
        for u in met["don_vi"]:
            if u["run"] != nh: continue
            pid = u["pid"]; c = contracts.get(pid, {}); kor = units.get(pid, {})
            ids = [r["id"] for r in c.get("required", []) if r.get("scoring", "required") == "required"]
            y0 = {i: Y.get((nh, pid, "I0", i)) for i in ids}; y1 = {i: Y.get((nh, pid, "I1", i)) for i in ids}
            S0 = {i for i, y in y0.items() if y is not None and y >= 1.0}
            lost = {i for i in S0 if (y1.get(i) or 0) < 1.0}
            calr = 100 * len(lost) / len(S0) if S0 else None
            v0 = u["diem"].get("I0", {}).get("VCFS"); v1 = u["diem"].get("I1", {}).get("VCFS")
            q0 = u["diem"].get("I0", {}).get("VQA"); q1 = u["diem"].get("I1", {}).get("VQA")
            sir = None if None in (v0, v1, q0, q1) else int(v1 > v0 + DELTA and q1 >= q0 - EPS)
            cls = None if None in (v0, v1) else ("improved" if v1 > v0 + DELTA else "degraded" if v1 < v0 - DELTA else "unchanged")
            re_ = None if None in (v0, v1) else max(0.0, v1 - v0) / 1.0
            # GDF1 proxy
            Dp = {i for i, y in y0.items() if y is not None and y < 1.0}
            gap = kor.get("gap", {}); texts = " ".join(str(x) for k in ("missing_prompt_explicit", "missing_cultural_identity", "contradictions") for x in (gap.get(k) or []))
            tw = {w for w in _norm(texts).split() if len(w) > 3}
            Dhat = set()
            for r in c.get("required", []):
                if r.get("scoring", "required") != "required": continue
                dw = {w for w in _norm(r.get("visual_evidence") or r["description"]).split() if len(w) > 3}
                if len(dw & tw) >= 2: Dhat.add(r["id"])
            tp = len(Dp & Dhat); P = tp / len(Dhat) if Dhat else None; R = tp / len(Dp) if Dp else None
            gdf1 = None if (P is None or R is None) else (0.0 if tp == 0 else 200 * P * R / (P + R))
            rows.append({"pid": pid, "CALR": calr, "SIR": sir, "cls": cls, "RE": re_, "GDF1": gdf1, "P": P, "R": R,
                         "S0": sorted(S0), "lost": sorted(lost), "Dp": sorted(Dp), "Dhat": sorted(Dhat), "actions": kor.get("actions") or [],
                         "img_I0": kor.get("images", {}).get("B (I0)"), "img_I1": kor.get("images", {}).get("C (I1)")})
        out[nh] = rows
        n = len(rows); cl = [r["cls"] for r in rows]
        md.append(f"\n## {nh} — {n} prompt\n\n| metric | giá trị |\n|---|---|\n"
                  f"| CALR ↓ (%) | {fmt(_boot([r['CALR'] for r in rows]))} · N/A {sum(1 for r in rows if r['CALR'] is None)} (I0 không có mục đúng) |\n"
                  f"| SIR ↑ | {sum(r['SIR'] or 0 for r in rows)}/{sum(1 for r in rows if r['SIR'] is not None)} = {100*sum(r['SIR'] or 0 for r in rows)/max(1,sum(1 for r in rows if r['SIR'] is not None)):.0f}% |\n"
                  f"| Improved / Unchanged / Degraded (VCFS, δ={DELTA}) | {cl.count('improved')} / {cl.count('unchanged')} / {cl.count('degraded')} |\n"
                  f"| RE ↑ (điểm VCFS / ảnh sinh thêm) | {fmt(_boot([r['RE'] for r in rows]))} |\n"
                  f"| GDF1* ↑ (proxy) | {fmt(_boot([r['GDF1'] for r in rows]))} · P {fmt(_boot([None if r['P'] is None else 100*r['P'] for r in rows]))} · R {fmt(_boot([None if r['R'] is None else 100*r['R'] for r in rows]))} |\n")
    if a.rrr:
        import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
        from eval_metrics import DanhGia
        ev = DanhGia(a.evaluator)
        from PIL import Image
        def score(img, st):
            q = (f'Statement: "{st}"\nHow fully does this photograph match that statement? Answer with exactly one word: YES (clearly), PARTIAL (partly/unsure), or NO (not shown).')
            msgs = [{"role": "user", "content": [{"type": "image", "image": Image.open(img).convert("RGB")}, {"type": "text", "text": q}]}]
            t = ev.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = ev.proc(text=[t], images=[Image.open(img).convert("RGB")], return_tensors="pt").to(ev.model.device)
            with ev.torch.no_grad(): o = ev.model.generate(**inp, max_new_tokens=4, do_sample=False)
            ans = ev.proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].upper()
            return 1.0 if "YES" in ans and "PARTIAL" not in ans else 0.5 if "PARTIAL" in ans else 0.0
        for nh, rows in out.items():
            rd = runs[nh][0]
            for r in rows:
                def loc(p):
                    if not p: return None
                    parts = Path(p).parts; i = len(parts) - 1 - parts[::-1].index(rd.name) if rd.name in parts else None
                    return str(rd.joinpath(*parts[i + 1:])) if i is not None else p
                i0, i1 = loc(r["img_I0"]), loc(r["img_I1"])
                if not r["actions"] or not i0 or not i1 or i0 == i1: r["RRR"] = None; continue
                real = 0; det = []
                for act in r["actions"]:
                    s0, s1 = score(i0, act), score(i1, act); ok = s1 >= 1.0 and s0 <= 0.0; real += ok; det.append((act, s0, s1))
                r["RRR"] = 100 * real / len(r["actions"]); r["RRR_detail"] = det
            md.append(f"\n{nh} · RRR ↑ (%) : {fmt(_boot([r.get('RRR') for r in rows]))} · N/A {sum(1 for r in rows if r.get('RRR') is None)}\n")
    (M / "extra_metrics.json").write_text(json.dumps({"delta": DELTA, "eps": EPS, "runs": out}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (M / "extra_metrics.md").write_text("".join(md), encoding="utf-8"); print("".join(md)); print("EXTRA_DONE")


if __name__ == "__main__":
    main()
