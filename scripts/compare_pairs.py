"""python scripts/compare_pairs.py <run_dir> <out.html>  -  trang so sánh bare vs system cho một run (v1.7)."""
"""Trang so sánh bare vs system: mỗi prompt x mỗi model nền, ảnh tốt nhất của hai nhánh đặt cạnh nhau, kèm điểm và lời Reviewer."""
import json, sys, glob, os, base64, io, html
from PIL import Image
run = sys.argv[1]; out = sys.argv[2]
def b64(p, side=300):
    im = Image.open(p).convert("RGB"); im.thumbnail((side, side))
    b = io.BytesIO(); im.save(b, "JPEG", quality=85); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
def base_of(k): return k.split("+")[0].split("#")[0].split("@")[0]
def is_bare(k): return "#bare" in k
E = html.escape
css = """<style>body{font-family:system-ui;font-size:13px;margin:16px;color:#222}h2{margin:24px 0 6px}h3{margin:14px 0 4px;color:#444}
table{border-collapse:collapse}td,th{border-bottom:1px solid #ddd;padding:6px 10px;vertical-align:top;text-align:left}th{background:#f4f3ef}
.ok{color:#168052}.bad{color:#c4302b}.muted{color:#777}.small{font-size:11px}.pair{display:flex;gap:14px;flex-wrap:wrap}
.card{width:320px}.card img{width:300px;display:block;border:3px solid #ccc}.card.sys img{border-color:#168052}.card.bare img{border-color:#c4302b}
.delta{font-weight:600}</style>"""
parts = [css, "<h1>CTIG v1.7 · Bare vs System</h1><p class='muted'>Mỗi hàng: cùng model nền, cùng seed. <b style='color:#c4302b'>Đỏ</b> = bare (prompt tiếng Anh dịch thẳng + negative chung, không KB / LoRA / ảnh tham chiếu, 6 ảnh). <b style='color:#168052'>Xanh</b> = system (Grounding + best-of-N thích nghi + ảnh tham chiếu/LoRA khi có). Ảnh hiện là ảnh TỐT NHẤT của mỗi nhánh theo điểm Reviewer (VLM mô tả rồi khớp must_have/must_not), hoà thì theo hạng ensemble. Điểm Reviewer: +1 = đủ thuộc tính, âm = có must_not hoặc bị loại.</p>"]
summary = []
for mg in sorted(glob.glob(f"{run}/*/multigen.json")):
    pid = mg.split("/")[-2]; d = json.load(open(mg))
    crp = f"{run}/{pid}/step_candidate_review.json"; cr = json.load(open(crp))["value"] if os.path.exists(crp) else None
    ver = {v["path"]: v for v in cr["filter"]["verdicts"]} if cr else {}
    sp = json.load(open(f"{run}/{pid}/step_spec.json"))["value"]
    prompt_en = d.get("prompt_en", "")
    parts.append(f"<h2>{E(pid)}</h2><div><b>{E(prompt_en)}</b></div><div class='small muted'>thực thể: {E(', '.join(e['name_vi'] for e in sp['entities']))}</div>")
    groups = {}
    for r in d["runs"]:
        if not r.get("output"): continue
        for c in r["output"]["candidates"]:
            groups.setdefault((base_of(r["model_key"]), "bare" if is_bare(r["model_key"]) else "system"), {})[c["path"]] = (c, r["model_key"])
    def score(c):
        v = ver.get(c["path"]); return ((v["score"] if v and v["keep"] else -1.0) if v else -0.5, c.get("ensemble") or 0)
    for b in dict.fromkeys(base_of(r["model_key"]) for r in d["runs"]):
        if (b, "bare") not in groups or (b, "system") not in groups: continue
        parts.append(f"<h3>{E(b)}</h3><div class='pair'>")
        vals = {}
        for lab in ("bare", "system"):
            cs = list(groups[(b, lab)].values())
            c, key = max(cs, key=lambda ck: score(ck[0]))
            v = ver.get(c["path"])
            n = len(cs); rev_mean = sum(ver[x["path"]]["score"] for x, _ in cs if x["path"] in ver) / max(1, len([1 for x, _ in cs if x["path"] in ver]))
            itm_mean = sum(x.get("itm_attrs") or 0 for x, _ in cs) / n; attr_mean = sum(x.get("attr_contrast") or 0 for x, _ in cs) / n
            vals[lab] = (rev_mean, itm_mean, attr_mean)
            verdict = ""
            if v:
                verdict = (f"<div class='{'ok' if v['keep'] and not v['matched_must_not'] else 'bad'}'>Reviewer {v['score']:+.2f} · {'giữ' if v['keep'] else 'loại'}</div>"
                           + (f"<div class='small ok'>✓ {E('; '.join(a[:40] for a in v['matched_must_have'][:3]))}</div>" if v.get("matched_must_have") else "")
                           + (f"<div class='small muted'>thiếu: {E('; '.join(a[:40] for a in v['missing_must_have'][:3]))}</div>" if v["missing_must_have"] else "")
                           + (f"<div class='small bad'>must_not: {E('; '.join(a[:40] for a in v['matched_must_not'][:2]))}</div>" if v["matched_must_not"] else ""))
            else:
                verdict = "<div class='muted small'>Reviewer chưa chấm ảnh này</div>"
            parts.append(f"<div class='card {lab}'><img src='{b64(c['path'])}'><div><b>{lab}</b> · {E(key)} · {n} ảnh</div>"
                         f"<div class='small'>ảnh này: CLIP attr {c.get('attr_contrast') or 0:.2f} · ITM attr {c.get('itm_attrs') or 0:.2f} · PickScore {c.get('aesthetic') or 0:.2f}</div>"
                         f"<div class='small muted'>TB nhánh: Reviewer {rev_mean:+.2f} · ITM {itm_mean:.2f} · CLIP attr {attr_mean:.2f}</div>{verdict}</div>")
        dr, di, da = (vals["system"][i] - vals["bare"][i] for i in range(3))
        cls = "ok" if (dr > 0 or di > 0.05) else "bad"
        parts.append(f"<div class='card'><div class='delta {cls}'>Δ system − bare</div><div>Reviewer TB: {dr:+.2f}</div><div>ITM attr TB: {di:+.2f}</div><div>CLIP attr TB: {da:+.2f}</div></div></div>")
        summary.append((pid, b, dr, di, da))
    if cr and cr.get("final_path") and os.path.exists(cr["final_path"]):
        parts.append(f"<div class='pair'><div class='card sys'><img src='{b64(cr['final_path'])}'><div><b>Ảnh cuối của hệ thống</b> · {E(cr.get('final_source',''))} · {len(cr.get('iterations',[]))} vòng loop</div><div class='small muted'>{E(cr.get('stop_reason',''))}</div></div></div>")
rows = "".join(f"<tr><td>{E(p)}</td><td>{E(b)}</td><td class='{'ok' if dr>0 else 'bad'}'>{dr:+.2f}</td><td class='{'ok' if di>0 else 'bad'}'>{di:+.2f}</td><td class='{'ok' if da>0 else 'bad'}'>{da:+.2f}</td></tr>" for p, b, dr, di, da in summary)
wins = sum(1 for _, _, dr, di, _ in summary if dr > 0 or di > 0.05)
parts.insert(2, f"<h2>Tổng hợp Δ (system − bare)</h2><div>system hơn bare ở <b>{wins}/{len(summary)}</b> cặp (Reviewer TB tăng hoặc ITM attr tăng &gt; 0,05)</div><table><tr><th>prompt</th><th>model nền</th><th>Δ Reviewer</th><th>Δ ITM attr</th><th>Δ CLIP attr</th></tr>{rows}</table>")
open(out, "w").write("<!doctype html><meta charset='utf-8'><title>Bare vs System</title>" + "".join(parts))
print(out, os.path.getsize(out) // 1024, "KB", f"{wins}/{len(summary)}")
