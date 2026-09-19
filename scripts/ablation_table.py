"""Bảng ablation 50 prompt từ các thư mục metric: I0 · refs-only · keep-only · SAVIER, kèm hiệu số ghép đôi SAVIER − refs-only.

    python scripts/ablation_table.py --dir docs/report_assets/metrics_final -o results/metrics/ablation_50.md
"""
import argparse, json, random
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("-o", required=True); a = ap.parse_args()
D = Path(a.dir); rng = random.Random(0)
def load(n): return {u["pid"]: u for u in json.loads((D / n / "metrics.json").read_text(encoding="utf-8"))["don_vi"]}
def ci(xs, B=2000):
    m = sum(xs) / len(xs); bs = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(B)); return m, bs[int(0.025 * B)], bs[int(0.975 * B) - 1]
def nrg(us): 
    v = [u["NRG"] for u in us.values() if u.get("NRG") is not None]; return f"{sum(1 for x in v if x > 0)}/{sum(1 for x in v if x == 0)}/{sum(1 for x in v if x < 0)}"
out = ["# Ablation — 50 prompt, cùng seed, cùng I0 (evaluator Qwen2.5-VL-7B, mù; contracts_v2; CI bootstrap theo prompt ×2000)", "",
       "Arms: I0 = refined prompt (Culture-TRIP); refs-only = I0 prompt + IP-Adapter (2 ảnh `selected/`), không agent; keep-only = refs-only + câu Keep từ Preservation Card (một lời gọi C, không O/R); SAVIER = keep-only + O quan sát I0 + R sinh ≤3 action sửa + bỏ drop_phrases.", ""]
for model, main, ro, ko in (("SDXL", "metrics_final_sdxl", "metrics_abl_sdxl_refsonly", "metrics_abl_sdxl_keeponly"), ("FLUX.1-dev", "metrics_final_flux", "metrics_abl_flux_refsonly", "metrics_abl_flux_keeponly")):
    um, ur, uk = load(main), load(ro), load(ko); pids = sorted(um)
    out += [f"## {model}", "", "| arm | VQAScore ↑ | VCFS ↑ | CCR ↓ | NRG >0 / =0 / <0 |", "|---|---:|---:|---:|---|"]
    rows = [("A (prompt gốc)", um, "A"), ("I0 (+ refined prompt)", um, "I0"), ("+ refined prompt + reference (refs-only)", ur, "I1"), ("+ keep-only", uk, "I1"), ("**+ SAVIER (ours)**", um, "I1")]
    for name, us, arm in rows:
        f = lambda k: ci([us[p]["diem"][arm][k] for p in pids])
        vq, vc, cc = f("VQA"), f("VCFS"), f("CCR")
        out.append(f"| {name} | {vq[0]:.3f} [{vq[1]:.2f}, {vq[2]:.2f}] | {vc[0]:.1f} [{vc[1]:.1f}, {vc[2]:.1f}] | {cc[0]:.1f} [{cc[1]:.1f}, {cc[2]:.1f}] | {nrg(us) if arm == 'I1' else '–'} |")
    out += ["", "Hiệu số ghép đôi theo prompt (SAVIER − arm):", "", "| so với | ΔVQA | ΔVCFS | ΔCCR | SAVIER hơn / bằng / kém (VCFS) |", "|---|---:|---:|---:|---|"]
    for name, us in (("refs-only", ur), ("keep-only", uk), ("I0", {p: {"diem": {"I1": um[p]["diem"]["I0"]}} for p in pids})):
        g = lambda k: ci([um[p]["diem"]["I1"][k] - us[p]["diem"]["I1"][k] for p in pids])
        dv, dc, dr = g("VQA"), g("VCFS"), g("CCR"); w = [um[p]["diem"]["I1"]["VCFS"] - us[p]["diem"]["I1"]["VCFS"] for p in pids]
        out.append(f"| {name} | {dv[0]:+.3f} [{dv[1]:+.2f}, {dv[2]:+.2f}] | {dc[0]:+.1f} [{dc[1]:+.1f}, {dc[2]:+.1f}] | {dr[0]:+.1f} [{dr[1]:+.1f}, {dr[2]:+.1f}] | {sum(1 for x in w if x > 0)} / {sum(1 for x in w if x == 0)} / {sum(1 for x in w if x < 0)} |")
    out.append("")
Path(a.o).parent.mkdir(parents=True, exist_ok=True); Path(a.o).write_text("\n".join(out), encoding="utf-8"); print("\n".join(out))
